#include "native_onnx_internal.hpp"

#include "dynamic_library.hpp"
#include "json_utility.hpp"
#include "model_asset.hpp"
#include "native_audio_query.hpp"
#include "streaming_audio.hpp"
#include "utility.hpp"

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

static std::string joinNativeOnnxStringValues(const std::vector<std::string> &values) {
    std::ostringstream valueStream;
    for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
        if (valueIndex > 0) {
            valueStream << ",";
        }
        valueStream << values[valueIndex];
    }
    return valueStream.str();
}

static std::string createNativeOnnxOperatorKey(const std::string &domainName, const std::string &operationName) {
    if (domainName.empty()) {
        return operationName;
    }
    return domainName + ":" + operationName;
}

static std::string createNativeOnnxTemporaryModelBaseName(const ModelAssetRecord &modelAsset, const std::vector<uint8_t> &modelBytes) {
    std::string fileName = modelAsset.archivePath.stem().string() + "-" + modelAsset.entryName;
    for (char &character : fileName) {
        if (!(std::isalnum(static_cast<unsigned char>(character)) || character == '-' || character == '_' || character == '.')) {
            character = '_';
        }
    }
    return fileName + "-" + createSha256Hex(modelBytes.data(), modelBytes.size()).substr(0, 16);
}

static fs::path createNativeOnnxTemporaryModelPath(const ModelAssetRecord &modelAsset, const std::vector<uint8_t> &modelBytes) {
    return createTemporaryFilePath(createNativeOnnxTemporaryModelBaseName(modelAsset, modelBytes), ".onnx");
}

std::vector<uint8_t> exportNativeOnnxOptimizedModelBytes(NativeOnnxApi &nativeOnnxApi, const ModelAssetRecord &modelAsset, const std::vector<uint8_t> &modelBytes, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    OrtEnv *env = nullptr;
    OrtSessionOptions *sessionOptions = nullptr;
    OrtSession *session = nullptr;
    fs::path optimizedModelPath = createNativeOnnxTemporaryModelPath(modelAsset, modelBytes);
    try {
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createEnv(ortLoggingLevelWarning, "litevox-native-export", &env), "OrtEnv 作成");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionOptions(&sessionOptions), "SessionOptions 作成");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setOptimizedModelFilePath(sessionOptions, optimizedModelPath.c_str()), "optimized model path 設定");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setSessionGraphOptimizationLevel(sessionOptions, ortGraphOptimizationLevelBasic), "graph optimization 設定");
        if (cpuThreadCount > 0) {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setIntraOpNumThreads(sessionOptions, cpuThreadCount), "intra op thread 設定");
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setInterOpNumThreads(sessionOptions, cpuThreadCount), "inter op thread 設定");
        }
        if (shouldUseVvBinConfig) {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.addSessionConfigEntry(sessionOptions, "session.use_vv_bin", "1"), "vv_bin session 設定");
        }
        applyNativeOnnxSeedIfConfigured(nativeOnnxApi);
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionFromArray(env, modelBytes.data(), modelBytes.size(), sessionOptions, &session), "ONNX session 作成");
        if (session) {
            nativeOnnxApi.releaseSession(session);
            session = nullptr;
        }
        if (sessionOptions) {
            nativeOnnxApi.releaseSessionOptions(sessionOptions);
            sessionOptions = nullptr;
        }
        if (env) {
            nativeOnnxApi.releaseEnv(env);
            env = nullptr;
        }
        std::vector<uint8_t> optimizedModelBytes = readNativeOnnxBinaryFile(optimizedModelPath);
        fs::remove(optimizedModelPath);
        return optimizedModelBytes;
    } catch (...) {
        if (session) {
            nativeOnnxApi.releaseSession(session);
        }
        if (sessionOptions) {
            nativeOnnxApi.releaseSessionOptions(sessionOptions);
        }
        if (env) {
            nativeOnnxApi.releaseEnv(env);
        }
        if (!optimizedModelPath.empty() && fs::exists(optimizedModelPath)) {
            fs::remove(optimizedModelPath);
        }
        throw;
    }
}

static uint64_t readNativeOnnxProtoVarint(const uint8_t *messageBytes, size_t messageSize, size_t &offset) {
    uint64_t value = 0;
    int shift = 0;
    while (offset < messageSize && shift < 64) {
        uint8_t byteValue = messageBytes[offset++];
        if (shift == 63 && (byteValue & 0xfe)) throw std::runtime_error("protobuf varint overflow");
        value |= static_cast<uint64_t>(byteValue & 0x7f) << shift;
        if ((byteValue & 0x80) == 0) {
            return value;
        }
        shift += 7;
    }
    throw std::runtime_error("protobuf varint が不正です");
}

static std::pair<size_t, size_t> readNativeOnnxProtoLengthDelimitedRange(const uint8_t *messageBytes, size_t messageSize, size_t &offset) {
    uint64_t fieldSize = readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
    if (fieldSize > messageSize - offset) {
        throw std::runtime_error("protobuf length-delimited field が不正です");
    }
    size_t fieldOffset = offset;
    offset += static_cast<size_t>(fieldSize);
    return {fieldOffset, static_cast<size_t>(fieldSize)};
}

static std::string readNativeOnnxProtoString(const uint8_t *messageBytes, size_t messageSize, size_t &offset) {
    std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
    return std::string(reinterpret_cast<const char *>(messageBytes + fieldRange.first), fieldRange.second);
}

static void skipNativeOnnxProtoField(const uint8_t *messageBytes, size_t messageSize, size_t &offset, uint32_t wireType) {
    if (wireType == 0) {
        readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
        return;
    }
    if (wireType == 1) {
        if (messageSize - offset < 8) {
            throw std::runtime_error("protobuf fixed64 が不正です");
        }
        offset += 8;
        return;
    }
    if (wireType == 2) {
        std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
        offset = fieldRange.first + fieldRange.second;
        return;
    }
    if (wireType == 5) {
        if (messageSize - offset < 4) {
            throw std::runtime_error("protobuf fixed32 が不正です");
        }
        offset += 4;
        return;
    }
    throw std::runtime_error("protobuf wire type に未対応です");
}

static void appendNativeOnnxProtoVarint(std::vector<uint8_t> &messageBytes, uint64_t value) {
    while (value >= 0x80) {
        messageBytes.push_back(static_cast<uint8_t>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    messageBytes.push_back(static_cast<uint8_t>(value));
}

static void appendNativeOnnxProtoTag(std::vector<uint8_t> &messageBytes, uint32_t fieldNumber, uint32_t wireType) {
    appendNativeOnnxProtoVarint(messageBytes, (static_cast<uint64_t>(fieldNumber) << 3) | wireType);
}

static void appendNativeOnnxProtoBytes(std::vector<uint8_t> &messageBytes, const uint8_t *fieldBytes, size_t fieldSize) {
    messageBytes.insert(messageBytes.end(), fieldBytes, fieldBytes + fieldSize);
}

static void appendNativeOnnxProtoLengthDelimitedField(std::vector<uint8_t> &messageBytes, uint32_t fieldNumber, const std::vector<uint8_t> &fieldValue) {
    appendNativeOnnxProtoTag(messageBytes, fieldNumber, 2);
    appendNativeOnnxProtoVarint(messageBytes, fieldValue.size());
    appendNativeOnnxProtoBytes(messageBytes, fieldValue.data(), fieldValue.size());
}

static void appendNativeOnnxProtoStringField(std::vector<uint8_t> &messageBytes, uint32_t fieldNumber, const std::string &fieldValue) {
    appendNativeOnnxProtoTag(messageBytes, fieldNumber, 2);
    appendNativeOnnxProtoVarint(messageBytes, fieldValue.size());
    appendNativeOnnxProtoBytes(messageBytes, reinterpret_cast<const uint8_t *>(fieldValue.data()), fieldValue.size());
}

static void appendNativeOnnxProtoFloatField(std::vector<uint8_t> &messageBytes, uint32_t fieldNumber, float fieldValue) {
    appendNativeOnnxProtoTag(messageBytes, fieldNumber, 5);
    uint32_t rawBits = 0;
    std::memcpy(&rawBits, &fieldValue, sizeof(rawBits));
    messageBytes.push_back(static_cast<uint8_t>(rawBits & 0xff));
    messageBytes.push_back(static_cast<uint8_t>((rawBits >> 8) & 0xff));
    messageBytes.push_back(static_cast<uint8_t>((rawBits >> 16) & 0xff));
    messageBytes.push_back(static_cast<uint8_t>((rawBits >> 24) & 0xff));
}

static void appendNativeOnnxProtoIntField(std::vector<uint8_t> &messageBytes, uint32_t fieldNumber, uint64_t fieldValue) {
    appendNativeOnnxProtoTag(messageBytes, fieldNumber, 0);
    appendNativeOnnxProtoVarint(messageBytes, fieldValue);
}

struct NativeOnnxProtoFieldRewrite {
    uint32_t fieldNumber = 0;
    std::string attributeName;
    std::vector<uint8_t> bytes;
};

std::vector<uint8_t> rewriteNativeOnnxModelRandomSeed(const uint8_t *messageBytes, size_t messageSize, float seedValue, size_t &rewrittenNodeCount);
static std::vector<uint8_t> rewriteNativeOnnxGraphRandomSeed(const uint8_t *messageBytes, size_t messageSize, float seedValue, size_t &rewrittenNodeCount);

static std::vector<uint8_t> createNativeOnnxSeedAttributeBytes(float seedValue) {
    std::vector<uint8_t> attributeBytes;
    appendNativeOnnxProtoStringField(attributeBytes, 1, "seed");
    appendNativeOnnxProtoFloatField(attributeBytes, 2, seedValue);
    appendNativeOnnxProtoIntField(attributeBytes, 20, 1);
    return attributeBytes;
}

static std::vector<uint8_t> rewriteNativeOnnxAttributeRandomSeed(const uint8_t *messageBytes, size_t messageSize, float seedValue, size_t &rewrittenNodeCount, std::string *attributeName) {
    size_t offset = 0;
    std::vector<uint8_t> rewrittenBytes;
    std::string parsedAttributeName;
    while (offset < messageSize) {
        size_t fieldStart = offset;
        uint64_t tagValue = readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
        uint32_t fieldNumber = static_cast<uint32_t>(tagValue >> 3);
        uint32_t wireType = static_cast<uint32_t>(tagValue & 0x7);
        if (fieldNumber == 1 && wireType == 2) {
            parsedAttributeName = readNativeOnnxProtoString(messageBytes, messageSize, offset);
            appendNativeOnnxProtoBytes(rewrittenBytes, messageBytes + fieldStart, offset - fieldStart);
            continue;
        }
        if ((fieldNumber == 6 || fieldNumber == 11) && wireType == 2) {
            std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
            std::vector<uint8_t> rewrittenGraphBytes = rewriteNativeOnnxGraphRandomSeed(messageBytes + fieldRange.first, fieldRange.second, seedValue, rewrittenNodeCount);
            appendNativeOnnxProtoLengthDelimitedField(rewrittenBytes, fieldNumber, rewrittenGraphBytes);
            continue;
        }
        skipNativeOnnxProtoField(messageBytes, messageSize, offset, wireType);
        appendNativeOnnxProtoBytes(rewrittenBytes, messageBytes + fieldStart, offset - fieldStart);
    }
    if (attributeName) {
        *attributeName = parsedAttributeName;
    }
    return rewrittenBytes;
}

static std::vector<uint8_t> rewriteNativeOnnxNodeRandomSeed(const uint8_t *messageBytes, size_t messageSize, float seedValue, size_t &rewrittenNodeCount) {
    size_t offset = 0;
    std::string operationName;
    std::vector<NativeOnnxProtoFieldRewrite> fieldRewrites;
    while (offset < messageSize) {
        size_t fieldStart = offset;
        uint64_t tagValue = readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
        uint32_t fieldNumber = static_cast<uint32_t>(tagValue >> 3);
        uint32_t wireType = static_cast<uint32_t>(tagValue & 0x7);
        if (fieldNumber == 4 && wireType == 2) {
            operationName = readNativeOnnxProtoString(messageBytes, messageSize, offset);
            NativeOnnxProtoFieldRewrite fieldRewrite;
            fieldRewrite.fieldNumber = fieldNumber;
            fieldRewrite.bytes.insert(fieldRewrite.bytes.end(), messageBytes + fieldStart, messageBytes + offset);
            fieldRewrites.push_back(std::move(fieldRewrite));
            continue;
        }
        if (fieldNumber == 5 && wireType == 2) {
            std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
            NativeOnnxProtoFieldRewrite fieldRewrite;
            fieldRewrite.fieldNumber = fieldNumber;
            std::vector<uint8_t> rewrittenAttributeBytes = rewriteNativeOnnxAttributeRandomSeed(messageBytes + fieldRange.first, fieldRange.second, seedValue, rewrittenNodeCount, &fieldRewrite.attributeName);
            appendNativeOnnxProtoLengthDelimitedField(fieldRewrite.bytes, fieldNumber, rewrittenAttributeBytes);
            fieldRewrites.push_back(std::move(fieldRewrite));
            continue;
        }
        skipNativeOnnxProtoField(messageBytes, messageSize, offset, wireType);
        NativeOnnxProtoFieldRewrite fieldRewrite;
        fieldRewrite.fieldNumber = fieldNumber;
        fieldRewrite.bytes.insert(fieldRewrite.bytes.end(), messageBytes + fieldStart, messageBytes + offset);
        fieldRewrites.push_back(std::move(fieldRewrite));
    }
    if (operationName != "RandomNormalLike") {
        std::vector<uint8_t> rewrittenBytes;
        for (const NativeOnnxProtoFieldRewrite &fieldRewrite : fieldRewrites) {
            appendNativeOnnxProtoBytes(rewrittenBytes, fieldRewrite.bytes.data(), fieldRewrite.bytes.size());
        }
        return rewrittenBytes;
    }
    rewrittenNodeCount++;
    std::vector<uint8_t> rewrittenBytes;
    bool hasSeedAttribute = false;
    for (const NativeOnnxProtoFieldRewrite &fieldRewrite : fieldRewrites) {
        if (fieldRewrite.fieldNumber == 5 && fieldRewrite.attributeName == "seed") {
            appendNativeOnnxProtoLengthDelimitedField(rewrittenBytes, 5, createNativeOnnxSeedAttributeBytes(seedValue));
            hasSeedAttribute = true;
            continue;
        }
        appendNativeOnnxProtoBytes(rewrittenBytes, fieldRewrite.bytes.data(), fieldRewrite.bytes.size());
    }
    if (!hasSeedAttribute) {
        appendNativeOnnxProtoLengthDelimitedField(rewrittenBytes, 5, createNativeOnnxSeedAttributeBytes(seedValue));
    }
    return rewrittenBytes;
}

static std::vector<uint8_t> rewriteNativeOnnxGraphRandomSeed(const uint8_t *messageBytes, size_t messageSize, float seedValue, size_t &rewrittenNodeCount) {
    size_t offset = 0;
    std::vector<uint8_t> rewrittenBytes;
    while (offset < messageSize) {
        size_t fieldStart = offset;
        uint64_t tagValue = readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
        uint32_t fieldNumber = static_cast<uint32_t>(tagValue >> 3);
        uint32_t wireType = static_cast<uint32_t>(tagValue & 0x7);
        if (fieldNumber == 1 && wireType == 2) {
            std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
            std::vector<uint8_t> rewrittenNodeBytes = rewriteNativeOnnxNodeRandomSeed(messageBytes + fieldRange.first, fieldRange.second, seedValue, rewrittenNodeCount);
            appendNativeOnnxProtoLengthDelimitedField(rewrittenBytes, fieldNumber, rewrittenNodeBytes);
            continue;
        }
        skipNativeOnnxProtoField(messageBytes, messageSize, offset, wireType);
        appendNativeOnnxProtoBytes(rewrittenBytes, messageBytes + fieldStart, offset - fieldStart);
    }
    return rewrittenBytes;
}

std::vector<uint8_t> rewriteNativeOnnxModelRandomSeed(const uint8_t *messageBytes, size_t messageSize, float seedValue, size_t &rewrittenNodeCount) {
    size_t offset = 0;
    std::vector<uint8_t> rewrittenBytes;
    while (offset < messageSize) {
        size_t fieldStart = offset;
        uint64_t tagValue = readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
        uint32_t fieldNumber = static_cast<uint32_t>(tagValue >> 3);
        uint32_t wireType = static_cast<uint32_t>(tagValue & 0x7);
        if (fieldNumber == 7 && wireType == 2) {
            std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
            std::vector<uint8_t> rewrittenGraphBytes = rewriteNativeOnnxGraphRandomSeed(messageBytes + fieldRange.first, fieldRange.second, seedValue, rewrittenNodeCount);
            appendNativeOnnxProtoLengthDelimitedField(rewrittenBytes, fieldNumber, rewrittenGraphBytes);
            continue;
        }
        skipNativeOnnxProtoField(messageBytes, messageSize, offset, wireType);
        appendNativeOnnxProtoBytes(rewrittenBytes, messageBytes + fieldStart, offset - fieldStart);
    }
    return rewrittenBytes;
}

static void collectNativeOnnxGraphOperatorCounts(const uint8_t *messageBytes, size_t messageSize, std::map<std::string, size_t> &operatorCounts);

static void collectNativeOnnxAttributeGraphOperatorCounts(const uint8_t *messageBytes, size_t messageSize, std::map<std::string, size_t> &operatorCounts) {
    size_t offset = 0;
    while (offset < messageSize) {
        uint64_t tagValue = readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
        uint32_t fieldNumber = static_cast<uint32_t>(tagValue >> 3);
        uint32_t wireType = static_cast<uint32_t>(tagValue & 0x7);
        if ((fieldNumber == 6 || fieldNumber == 11) && wireType == 2) {
            std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
            collectNativeOnnxGraphOperatorCounts(messageBytes + fieldRange.first, fieldRange.second, operatorCounts);
            continue;
        }
        skipNativeOnnxProtoField(messageBytes, messageSize, offset, wireType);
    }
}

static void collectNativeOnnxNodeOperatorCounts(const uint8_t *messageBytes, size_t messageSize, std::map<std::string, size_t> &operatorCounts) {
    size_t offset = 0;
    std::string operationName;
    std::string domainName;
    while (offset < messageSize) {
        uint64_t tagValue = readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
        uint32_t fieldNumber = static_cast<uint32_t>(tagValue >> 3);
        uint32_t wireType = static_cast<uint32_t>(tagValue & 0x7);
        if (fieldNumber == 4 && wireType == 2) {
            operationName = readNativeOnnxProtoString(messageBytes, messageSize, offset);
            continue;
        }
        if (fieldNumber == 7 && wireType == 2) {
            domainName = readNativeOnnxProtoString(messageBytes, messageSize, offset);
            continue;
        }
        if (fieldNumber == 5 && wireType == 2) {
            std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
            collectNativeOnnxAttributeGraphOperatorCounts(messageBytes + fieldRange.first, fieldRange.second, operatorCounts);
            continue;
        }
        skipNativeOnnxProtoField(messageBytes, messageSize, offset, wireType);
    }
    if (!operationName.empty()) {
        operatorCounts[createNativeOnnxOperatorKey(domainName, operationName)]++;
    }
}

static void collectNativeOnnxGraphOperatorCounts(const uint8_t *messageBytes, size_t messageSize, std::map<std::string, size_t> &operatorCounts) {
    size_t offset = 0;
    while (offset < messageSize) {
        uint64_t tagValue = readNativeOnnxProtoVarint(messageBytes, messageSize, offset);
        uint32_t fieldNumber = static_cast<uint32_t>(tagValue >> 3);
        uint32_t wireType = static_cast<uint32_t>(tagValue & 0x7);
        if (fieldNumber == 1 && wireType == 2) {
            std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(messageBytes, messageSize, offset);
            collectNativeOnnxNodeOperatorCounts(messageBytes + fieldRange.first, fieldRange.second, operatorCounts);
            continue;
        }
        skipNativeOnnxProtoField(messageBytes, messageSize, offset, wireType);
    }
}

std::map<std::string, size_t> collectNativeOnnxOperatorCounts(const std::vector<uint8_t> &optimizedModelBytes) {
    size_t offset = 0;
    std::map<std::string, size_t> operatorCounts;
    while (offset < optimizedModelBytes.size()) {
        uint64_t tagValue = readNativeOnnxProtoVarint(optimizedModelBytes.data(), optimizedModelBytes.size(), offset);
        uint32_t fieldNumber = static_cast<uint32_t>(tagValue >> 3);
        uint32_t wireType = static_cast<uint32_t>(tagValue & 0x7);
        if (fieldNumber == 7 && wireType == 2) {
            std::pair<size_t, size_t> fieldRange = readNativeOnnxProtoLengthDelimitedRange(optimizedModelBytes.data(), optimizedModelBytes.size(), offset);
            collectNativeOnnxGraphOperatorCounts(optimizedModelBytes.data() + fieldRange.first, fieldRange.second, operatorCounts);
            continue;
        }
        skipNativeOnnxProtoField(optimizedModelBytes.data(), optimizedModelBytes.size(), offset, wireType);
    }
    return operatorCounts;
}

static size_t countNativeOnnxOperators(const std::map<std::string, size_t> &operatorCounts) {
    size_t operatorCount = 0;
    for (const auto &operatorEntry : operatorCounts) {
        operatorCount += operatorEntry.second;
    }
    return operatorCount;
}

static std::string summarizeNativeOnnxOperatorCounts(const std::map<std::string, size_t> &operatorCounts) {
    std::vector<std::pair<std::string, size_t>> operatorEntries(operatorCounts.begin(), operatorCounts.end());
    std::sort(operatorEntries.begin(), operatorEntries.end(), [](const auto &leftEntry, const auto &rightEntry) {
        if (leftEntry.second != rightEntry.second) {
            return leftEntry.second > rightEntry.second;
        }
        return leftEntry.first < rightEntry.first;
    });
    std::ostringstream summaryStream;
    for (size_t operatorIndex = 0; operatorIndex < operatorEntries.size(); operatorIndex++) {
        if (operatorIndex > 0) {
            summaryStream << ",";
        }
        summaryStream << operatorEntries[operatorIndex].first << ":" << operatorEntries[operatorIndex].second;
    }
    return summaryStream.str();
}

static std::string extractNativeOnnxOperatorName(const std::string &operatorKey) {
    size_t separatorPosition = operatorKey.rfind(':');
    if (separatorPosition == std::string::npos) {
        return operatorKey;
    }
    return operatorKey.substr(separatorPosition + 1);
}

std::map<std::string, size_t> collectNativeOnnxRandomOperatorCounts(const std::map<std::string, size_t> &operatorCounts) {
    std::map<std::string, size_t> randomOperatorCounts;
    for (const auto &operatorEntry : operatorCounts) {
        if (extractNativeOnnxOperatorName(operatorEntry.first).rfind("Random", 0) != 0) {
            continue;
        }
        randomOperatorCounts.emplace(operatorEntry.first, operatorEntry.second);
    }
    return randomOperatorCounts;
}

static fs::path createNativeOnnxExportPath(const fs::path &outputDirectory, const ModelAssetRecord &modelAsset) {
    fs::path relativePath = fs::path(modelAsset.entryName);
    relativePath.replace_extension(".onnx");
    return outputDirectory / modelAsset.archivePath.stem() / relativePath;
}

std::map<std::string, ManifestModelRecord> createNativeOnnxManifestModelMap(const std::vector<VvmArchiveSummary> &archiveSummaries) {
    std::map<std::string, ManifestModelRecord> manifestModelMap;
    for (const VvmArchiveSummary &archiveSummary : archiveSummaries) {
        for (const ManifestModelRecord &manifestModel : collectManifestModels(archiveSummary)) {
            manifestModelMap[manifestModel.archivePath.string() + "\t" + manifestModel.entryName] = manifestModel;
        }
    }
    return manifestModelMap;
}

const ManifestModelRecord *findNativeOnnxManifestModelRecord(const std::map<std::string, ManifestModelRecord> &manifestModelMap, const ModelAssetRecord &modelAsset) {
    auto manifestIterator = manifestModelMap.find(modelAsset.archivePath.string() + "\t" + modelAsset.entryName);
    if (manifestIterator == manifestModelMap.end()) {
        return nullptr;
    }
    return &manifestIterator->second;
}

std::string summarizeNativeOnnxValueDescriptors(const std::vector<NativeOnnxValueDescriptor> &valueDescriptors) {
    std::ostringstream summaryStream;
    for (size_t valueIndex = 0; valueIndex < valueDescriptors.size(); valueIndex++) {
        if (valueIndex > 0) {
            summaryStream << ";";
        }
        summaryStream << valueDescriptors[valueIndex].name
                      << ":"
                      << formatNativeOnnxElementType(valueDescriptors[valueIndex].elementType)
                      << formatNativeOnnxShape(valueDescriptors[valueIndex].dimensions);
    }
    return summaryStream.str();
}


NativeOnnxPreparedInputSet createNativeOnnxPreparedInputs(const std::vector<NativeOnnxValueDescriptor> &inputDescriptors, const std::vector<NativeOnnxTraceInput> &traceInputs) {
    NativeOnnxPreparedInputSet preparedInputs;
    preparedInputs.tensors.reserve(inputDescriptors.size());
    for (size_t inputIndex = 0; inputIndex < inputDescriptors.size(); inputIndex++) {
        const NativeOnnxValueDescriptor &inputDescriptor = inputDescriptors[inputIndex];
        const NativeOnnxTraceInput *traceInput = findNativeOnnxTraceTensor(traceInputs, inputDescriptor.name);
        if (traceInput) {
            if (traceInput->elementType != inputDescriptor.elementType) {
                throw std::runtime_error("trace input dtype が一致しません: " + inputDescriptor.name);
            }
            if (!areNativeOnnxDimensionsEqual(inputDescriptor.dimensions, traceInput->dimensions)) {
                throw std::runtime_error("trace input shape が一致しません: " + inputDescriptor.name);
            }
            preparedInputs.tensors.push_back(*traceInput);
            preparedInputs.traceInputCount++;
            continue;
        }
        if (!canCreateNativeOnnxSmokeInput(inputDescriptor)) {
            throw std::runtime_error("trace input が足りず synthetic input も作れません: " + inputDescriptor.name);
        }
        size_t inputElementCount = calculatePositiveShapeElementCount(inputDescriptor.dimensions);
        NativeOnnxTraceInput syntheticInput;
        syntheticInput.name = inputDescriptor.name;
        syntheticInput.elementType = inputDescriptor.elementType;
        syntheticInput.dimensions = inputDescriptor.dimensions;
        syntheticInput.bytes = createNativeOnnxInputBytes(inputDescriptor, inputIndex, inputElementCount);
        preparedInputs.tensors.push_back(std::move(syntheticInput));
        preparedInputs.syntheticInputCount++;
    }
    return preparedInputs;
}

std::string createNativeOnnxInputModeText(const NativeOnnxPreparedInputSet &preparedInputs) {
    if (preparedInputs.traceInputCount > 0 && preparedInputs.syntheticInputCount == 0) {
        return "trace";
    }
    if (preparedInputs.traceInputCount == 0 && preparedInputs.syntheticInputCount > 0) {
        return "synthetic";
    }
    if (preparedInputs.traceInputCount == 0 && preparedInputs.syntheticInputCount == 0) {
        return "none";
    }
    return "mixed";
}

static std::string summarizeNativeOnnxTensorMatches(const std::vector<NativeOnnxTraceInput> &actualTensors, const std::vector<NativeOnnxTraceInput> &expectedTensors) {
    if (actualTensors.empty()) {
        return "empty";
    }
    std::vector<std::string> mismatchTexts;
    for (const NativeOnnxTraceInput &actualTensor : actualTensors) {
        std::string matchText = compareNativeOnnxTensors(actualTensor, findNativeOnnxTraceTensor(expectedTensors, actualTensor.name));
        if (matchText != "exact") {
            mismatchTexts.push_back(actualTensor.name + ":" + matchText);
        }
    }
    if (mismatchTexts.empty()) {
        return "exact";
    }
    std::ostringstream summaryStream;
    for (size_t mismatchIndex = 0; mismatchIndex < mismatchTexts.size(); mismatchIndex++) {
        if (mismatchIndex > 0) {
            summaryStream << ";";
        }
        summaryStream << mismatchTexts[mismatchIndex];
    }
    return summaryStream.str();
}

NativeOnnxCompareBenchmark benchmarkNativeOnnxSession(const std::shared_ptr<NativeOnnxCachedSession> &cachedSession, NativeOnnxApi &nativeOnnxApi, const std::vector<NativeOnnxTraceInput> &inputTensors, size_t runCount) {
    NativeOnnxCompareBenchmark benchmark;
    benchmark.inputCount = cachedSession->inputDescriptors.size();
    benchmark.outputCount = cachedSession->outputDescriptors.size();
    if (runCount == 0) {
        return benchmark;
    }
    double totalRunMilliseconds = 0.0;
    for (size_t runIndex = 0; runIndex < runCount; runIndex++) {
        auto runStartedAt = std::chrono::steady_clock::now();
        benchmark.outputs = runNativeOnnxPreparedSession(nativeOnnxApi, cachedSession, inputTensors);
        auto runFinishedAt = std::chrono::steady_clock::now();
        totalRunMilliseconds += std::chrono::duration<double, std::milli>(runFinishedAt - runStartedAt).count();
    }
    benchmark.averageRunMilliseconds = totalRunMilliseconds / static_cast<double>(runCount);
    return benchmark;
}

static std::string formatNativeOnnxMilliseconds(double milliseconds) {
    std::ostringstream valueStream;
    valueStream << std::fixed << std::setprecision(3) << milliseconds;
    return valueStream.str();
}

static bool canCreateNativeOnnxCompareAudioQueryInputs(const ModelAssetRecord &modelAsset) {
    return modelAsset.entryName == "models/pd.bin" || modelAsset.entryName == "models/pi.bin" || modelAsset.entryName == "models/d.bin";
}

std::vector<NativeOnnxTraceInput> createNativeOnnxCompareAudioQueryInputs(const std::vector<ModelAssetRecord> &modelAssets, const ModelAssetRecord &modelAsset, const std::string &audioQueryText, uint32_t styleId) {
    const ModelAssetRecord &durationAsset = requireNativeOnnxModelAsset(modelAssets, "models/pd.bin");
    NativeOnnxAudioQuerySettings audioQuerySettings = parseNativeOnnxAudioQuerySettings(audioQueryText);
    int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(durationAsset, "talk", styleId);
    std::vector<NativeOnnxTraceInput> frontendInputs = createNativeOnnxFrontendInputs(audioQueryText, innerVoiceId);
    if (modelAsset.entryName == "models/pd.bin" || modelAsset.entryName == "models/pi.bin") {
        return frontendInputs;
    }
    if (modelAsset.entryName == "models/d.bin") {
        return createNativeOnnxDecoderInputsFromAudioQuery(frontendInputs, audioQueryText, audioQuerySettings);
    }
    throw std::runtime_error("audio_query から入力を作れない asset です: " + modelAsset.entryName);
}

std::string createNativeOnnxVvmModelInfoText(const fs::path &onnxruntimeLibraryPath, const std::vector<VvmArchiveSummary> &archiveSummaries, uint16_t cpuThreadCount) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::vector<ModelAssetRecord> modelAssets = collectModelAssets(archiveSummaries);
        std::map<std::string, ManifestModelRecord> manifestModelMap = createNativeOnnxManifestModelMap(archiveSummaries);
        std::vector<std::string> providers = collectNativeOnnxAvailableProviders(nativeOnnxApi);
        std::ostringstream inspectStream;
        inspectStream << "field\tvalue\n";
        inspectStream << "onnxruntime\t" << onnxruntimeLibraryPath.string() << "\n";
        inspectStream << "api_version\t" << ortApiVersion << "\n";
        inspectStream << "ort_version\t" << getNativeOnnxVersion(nativeOnnxApi) << "\n";
        inspectStream << "cpu_threads\t" << cpuThreadCount << "\n";
        inspectStream << "providers\t" << joinNativeOnnxStringValues(providers) << "\n";
        inspectStream << "model_count\t" << modelAssets.size() << "\n";
        inspectStream << "vvm\tasset\tdomain\toperation\tmodel_type\tbytes\tinput_count\toutput_count\tinputs\toutputs\n";
        for (const ModelAssetRecord &modelAsset : modelAssets) {
            const ManifestModelRecord *manifestModel = findNativeOnnxManifestModelRecord(manifestModelMap, modelAsset);
            std::vector<uint8_t> modelBytes = extractVvmEntryBytesAt(
                modelAsset.archivePath,
                modelAsset.entryName,
                modelAsset.dataOffset,
                modelAsset.compressedSize,
                modelAsset.uncompressedSize,
                modelAsset.compressionMethod);
            std::string sessionCacheKey = modelAsset.archivePath.string() + "\t" + modelAsset.entryName + "\t" + std::to_string(cpuThreadCount);
            std::shared_ptr<NativeOnnxCachedSession> cachedSession = getNativeOnnxCachedSession(nativeOnnxApi, nullptr, modelBytes, cpuThreadCount, true, sessionCacheKey);
            std::string inputSummary = summarizeNativeOnnxValueDescriptors(cachedSession->inputDescriptors);
            std::string outputSummary = summarizeNativeOnnxValueDescriptors(cachedSession->outputDescriptors);
            sanitizeNativeOnnxTableCell(inputSummary);
            sanitizeNativeOnnxTableCell(outputSummary);
            inspectStream << modelAsset.archivePath.filename().string() << "\t"
                          << modelAsset.entryName << "\t"
                          << (manifestModel ? manifestModel->domainName : "-") << "\t"
                          << (manifestModel ? manifestModel->operationName : "-") << "\t"
                          << (manifestModel ? manifestModel->modelType : "-") << "\t"
                          << modelAsset.uncompressedSize << "\t"
                          << cachedSession->inputDescriptors.size() << "\t"
                          << cachedSession->outputDescriptors.size() << "\t"
                          << inputSummary << "\t"
                          << outputSummary << "\n";
        }
        closeNativeOnnxApi(nativeOnnxApi);
        return inspectStream.str();
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string createNativeOnnxVvmOperatorText(const fs::path &onnxruntimeLibraryPath, const std::vector<VvmArchiveSummary> &archiveSummaries, uint16_t cpuThreadCount) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::vector<ModelAssetRecord> modelAssets = collectModelAssets(archiveSummaries);
        std::map<std::string, ManifestModelRecord> manifestModelMap = createNativeOnnxManifestModelMap(archiveSummaries);
        std::vector<std::string> providers = collectNativeOnnxAvailableProviders(nativeOnnxApi);
        std::map<std::string, size_t> globalOperatorCounts;
        std::ostringstream inspectStream;
        inspectStream << "field\tvalue\n";
        inspectStream << "onnxruntime\t" << onnxruntimeLibraryPath.string() << "\n";
        inspectStream << "api_version\t" << ortApiVersion << "\n";
        inspectStream << "ort_version\t" << getNativeOnnxVersion(nativeOnnxApi) << "\n";
        inspectStream << "cpu_threads\t" << cpuThreadCount << "\n";
        inspectStream << "providers\t" << joinNativeOnnxStringValues(providers) << "\n";
        inspectStream << "model_count\t" << modelAssets.size() << "\n";
        inspectStream << "vvm\tasset\tdomain\toperation\tmodel_type\tonnx_bytes\tnode_count\toperator_kinds\toperators\n";
        for (const ModelAssetRecord &modelAsset : modelAssets) {
            const ManifestModelRecord *manifestModel = findNativeOnnxManifestModelRecord(manifestModelMap, modelAsset);
            std::vector<uint8_t> modelBytes = extractVvmEntryBytesAt(
                modelAsset.archivePath,
                modelAsset.entryName,
                modelAsset.dataOffset,
                modelAsset.compressedSize,
                modelAsset.uncompressedSize,
                modelAsset.compressionMethod);
            std::vector<uint8_t> optimizedModelBytes = exportNativeOnnxOptimizedModelBytes(nativeOnnxApi, modelAsset, modelBytes, cpuThreadCount, true);
            std::map<std::string, size_t> operatorCounts = collectNativeOnnxOperatorCounts(optimizedModelBytes);
            for (const auto &operatorEntry : operatorCounts) {
                globalOperatorCounts[operatorEntry.first] += operatorEntry.second;
            }
            std::string operatorSummary = summarizeNativeOnnxOperatorCounts(operatorCounts);
            sanitizeNativeOnnxTableCell(operatorSummary);
            inspectStream << modelAsset.archivePath.filename().string() << "\t"
                          << modelAsset.entryName << "\t"
                          << (manifestModel ? manifestModel->domainName : "-") << "\t"
                          << (manifestModel ? manifestModel->operationName : "-") << "\t"
                          << (manifestModel ? manifestModel->modelType : "-") << "\t"
                          << optimizedModelBytes.size() << "\t"
                          << countNativeOnnxOperators(operatorCounts) << "\t"
                          << operatorCounts.size() << "\t"
                          << operatorSummary << "\n";
        }
        inspectStream << "global_operator_kinds\t" << globalOperatorCounts.size() << "\n";
        inspectStream << "global_operator_counts\t" << summarizeNativeOnnxOperatorCounts(globalOperatorCounts) << "\n";
        closeNativeOnnxApi(nativeOnnxApi);
        return inspectStream.str();
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string exportNativeOnnxVvmOnnxFiles(const fs::path &onnxruntimeLibraryPath, const std::vector<VvmArchiveSummary> &archiveSummaries, const fs::path &outputDirectory, uint16_t cpuThreadCount, bool shouldPatchRandomSeed, float randomSeedValue) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::vector<ModelAssetRecord> modelAssets = collectModelAssets(archiveSummaries);
        std::map<std::string, ManifestModelRecord> manifestModelMap = createNativeOnnxManifestModelMap(archiveSummaries);
        std::ostringstream exportStream;
        exportStream << "vvm\tasset\tdomain\toperation\tmodel_type\tstatus\toutput\tbytes\trandom_seed\trewritten_nodes\n";
        for (const ModelAssetRecord &modelAsset : modelAssets) {
            const ManifestModelRecord *manifestModel = findNativeOnnxManifestModelRecord(manifestModelMap, modelAsset);
            std::vector<uint8_t> modelBytes = extractVvmEntryBytesAt(
                modelAsset.archivePath,
                modelAsset.entryName,
                modelAsset.dataOffset,
                modelAsset.compressedSize,
                modelAsset.uncompressedSize,
                modelAsset.compressionMethod);
            std::vector<uint8_t> optimizedModelBytes = exportNativeOnnxOptimizedModelBytes(nativeOnnxApi, modelAsset, modelBytes, cpuThreadCount, true);
            size_t rewrittenNodeCount = 0;
            if (shouldPatchRandomSeed) {
                optimizedModelBytes = rewriteNativeOnnxModelRandomSeed(optimizedModelBytes.data(), optimizedModelBytes.size(), randomSeedValue, rewrittenNodeCount);
            }
            fs::path outputPath = createNativeOnnxExportPath(outputDirectory, modelAsset);
            writeBinaryFile(outputPath, optimizedModelBytes);
            exportStream << modelAsset.archivePath.filename().string() << "\t"
                         << modelAsset.entryName << "\t"
                         << (manifestModel ? manifestModel->domainName : "-") << "\t"
                         << (manifestModel ? manifestModel->operationName : "-") << "\t"
                         << (manifestModel ? manifestModel->modelType : "-") << "\t"
                         << "exported\t"
                         << outputPath.string() << "\t"
                         << optimizedModelBytes.size() << "\t";
            if (shouldPatchRandomSeed) {
                exportStream << std::setprecision(9) << static_cast<double>(randomSeedValue);
            } else {
                exportStream << "-";
            }
            exportStream << "\t" << rewrittenNodeCount << "\n";
        }
        closeNativeOnnxApi(nativeOnnxApi);
        return exportStream.str();
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string createNativeOnnxVvmCompareText(const fs::path &vvBinOnnxruntimeLibraryPath, const fs::path &exportedOnnxruntimeLibraryPath, const std::vector<VvmArchiveSummary> &archiveSummaries, const fs::path &inputDirectory, const fs::path &audioQueryPath, uint32_t styleId, const std::string &assetFilter, uint16_t cpuThreadCount, size_t runCount) {
    NativeOnnxApi vvBinOnnxApi = loadNativeOnnxApi(vvBinOnnxruntimeLibraryPath);
    NativeOnnxApi exportedOnnxApi = loadNativeOnnxApi(exportedOnnxruntimeLibraryPath);
    try {
        std::vector<ModelAssetRecord> modelAssets = collectModelAssets(archiveSummaries);
        std::map<std::string, ManifestModelRecord> manifestModelMap = createNativeOnnxManifestModelMap(archiveSummaries);
        std::vector<NativeOnnxTraceInput> traceInputs = loadNativeOnnxTraceInputs(inputDirectory);
        std::vector<NativeOnnxTraceInput> traceOutputs = loadNativeOnnxTraceOutputs(inputDirectory);
        bool hasAudioQuery = false;
        std::string audioQueryText;
        if (!audioQueryPath.empty()) {
            audioQueryText = readNativeOnnxTextFile(resolveNativeOnnxAudioQueryPath(inputDirectory, audioQueryPath));
            hasAudioQuery = true;
        } else if (!inputDirectory.empty()) {
            fs::path tracedAudioQueryPath = inputDirectory.parent_path() / "audio_query.json";
            if (fs::exists(tracedAudioQueryPath)) {
                audioQueryText = readNativeOnnxTextFile(tracedAudioQueryPath);
                hasAudioQuery = true;
            }
        }
        std::ostringstream compareStream;
        compareStream << "vvm\tasset\tdomain\toperation\tmodel_type\tstochastic\tstochastic_operators\tinput_mode\tinput_count\toutput_count\truns\tvv_bin_onnxruntime\texported_onnx_onnxruntime\tvv_bin_session_ms\tvv_bin_run_avg_ms\texported_session_ms\texported_run_avg_ms\toutput_match\ttrace_match_vv_bin\ttrace_match_exported\tstatus\tdetail\n";
        for (const ModelAssetRecord &modelAsset : modelAssets) {
            if (!assetFilter.empty() && modelAsset.entryName != assetFilter) {
                continue;
            }
            const ManifestModelRecord *manifestModel = findNativeOnnxManifestModelRecord(manifestModelMap, modelAsset);
            std::string domainText = manifestModel ? manifestModel->domainName : "-";
            std::string operationText = manifestModel ? manifestModel->operationName : "-";
            std::string modelTypeText = manifestModel ? manifestModel->modelType : "-";
            std::string inputModeText = "-";
            std::string outputMatchText = "-";
            std::string traceMatchVvBinText = "-";
            std::string traceMatchExportedText = "-";
            std::string stochasticText = "false";
            std::string stochasticOperatorsText = "-";
            std::string statusText = "ok";
            std::string detailText = "ok";
            std::string vvBinSessionText = "-";
            std::string vvBinRunText = "-";
            std::string exportedSessionText = "-";
            std::string exportedRunText = "-";
            size_t inputCount = 0;
            size_t outputCount = 0;
            try {
                std::vector<uint8_t> modelBytes = extractVvmEntryBytesAt(
                    modelAsset.archivePath,
                    modelAsset.entryName,
                    modelAsset.dataOffset,
                    modelAsset.compressedSize,
                    modelAsset.uncompressedSize,
                    modelAsset.compressionMethod);
                auto vvBinSessionStartedAt = std::chrono::steady_clock::now();
                std::shared_ptr<NativeOnnxCachedSession> vvBinSession = createNativeOnnxCachedSession(vvBinOnnxApi, nullptr, modelBytes, cpuThreadCount, true);
                auto vvBinSessionFinishedAt = std::chrono::steady_clock::now();
                std::vector<NativeOnnxTraceInput> sourceInputs = traceInputs;
                if (hasAudioQuery && canCreateNativeOnnxCompareAudioQueryInputs(modelAsset)) {
                    sourceInputs = createNativeOnnxCompareAudioQueryInputs(modelAssets, modelAsset, audioQueryText, styleId);
                }
                NativeOnnxPreparedInputSet preparedInputs = createNativeOnnxPreparedInputs(vvBinSession->inputDescriptors, sourceInputs);
                std::vector<uint8_t> optimizedModelBytes = exportNativeOnnxOptimizedModelBytes(vvBinOnnxApi, modelAsset, modelBytes, cpuThreadCount, true);
                std::map<std::string, size_t> randomOperatorCounts = collectNativeOnnxRandomOperatorCounts(collectNativeOnnxOperatorCounts(optimizedModelBytes));
                stochasticText = randomOperatorCounts.empty() ? "false" : "true";
                stochasticOperatorsText = randomOperatorCounts.empty() ? "-" : summarizeNativeOnnxOperatorCounts(randomOperatorCounts);
                auto exportedSessionStartedAt = std::chrono::steady_clock::now();
                std::shared_ptr<NativeOnnxCachedSession> exportedSession = createNativeOnnxCachedSession(exportedOnnxApi, nullptr, optimizedModelBytes, cpuThreadCount, false);
                auto exportedSessionFinishedAt = std::chrono::steady_clock::now();
                NativeOnnxCompareBenchmark vvBinBenchmark = benchmarkNativeOnnxSession(vvBinSession, vvBinOnnxApi, preparedInputs.tensors, runCount);
                NativeOnnxCompareBenchmark exportedBenchmark = benchmarkNativeOnnxSession(exportedSession, exportedOnnxApi, preparedInputs.tensors, runCount);
                vvBinBenchmark.sessionCreateMilliseconds = std::chrono::duration<double, std::milli>(vvBinSessionFinishedAt - vvBinSessionStartedAt).count();
                exportedBenchmark.sessionCreateMilliseconds = std::chrono::duration<double, std::milli>(exportedSessionFinishedAt - exportedSessionStartedAt).count();
                inputModeText = createNativeOnnxInputModeText(preparedInputs);
                inputCount = vvBinBenchmark.inputCount;
                outputCount = vvBinBenchmark.outputCount;
                vvBinSessionText = formatNativeOnnxMilliseconds(vvBinBenchmark.sessionCreateMilliseconds);
                vvBinRunText = formatNativeOnnxMilliseconds(vvBinBenchmark.averageRunMilliseconds);
                exportedSessionText = formatNativeOnnxMilliseconds(exportedBenchmark.sessionCreateMilliseconds);
                exportedRunText = formatNativeOnnxMilliseconds(exportedBenchmark.averageRunMilliseconds);
                inputModeText = hasAudioQuery && canCreateNativeOnnxCompareAudioQueryInputs(modelAsset) ? "audio_query" : createNativeOnnxInputModeText(preparedInputs);
                outputMatchText = summarizeNativeOnnxTensorMatches(vvBinBenchmark.outputs, exportedBenchmark.outputs);
                if (traceOutputs.empty()) {
                    traceMatchVvBinText = "not_available";
                    traceMatchExportedText = "not_available";
                } else {
                    traceMatchVvBinText = summarizeNativeOnnxTensorMatches(vvBinBenchmark.outputs, traceOutputs);
                    traceMatchExportedText = summarizeNativeOnnxTensorMatches(exportedBenchmark.outputs, traceOutputs);
                }
                if (stochasticText == "true") {
                    detailText = "stochastic:" + stochasticOperatorsText;
                }
            } catch (const std::exception &caughtException) {
                statusText = "failed";
                detailText = caughtException.what();
            }
            sanitizeNativeOnnxTableCell(domainText);
            sanitizeNativeOnnxTableCell(operationText);
            sanitizeNativeOnnxTableCell(modelTypeText);
            sanitizeNativeOnnxTableCell(stochasticText);
            sanitizeNativeOnnxTableCell(stochasticOperatorsText);
            sanitizeNativeOnnxTableCell(inputModeText);
            sanitizeNativeOnnxTableCell(outputMatchText);
            sanitizeNativeOnnxTableCell(traceMatchVvBinText);
            sanitizeNativeOnnxTableCell(traceMatchExportedText);
            sanitizeNativeOnnxTableCell(statusText);
            sanitizeNativeOnnxTableCell(detailText);
            compareStream << modelAsset.archivePath.filename().string() << "\t"
                          << modelAsset.entryName << "\t"
                          << domainText << "\t"
                          << operationText << "\t"
                          << modelTypeText << "\t"
                          << stochasticText << "\t"
                          << stochasticOperatorsText << "\t"
                          << inputModeText << "\t"
                          << inputCount << "\t"
                          << outputCount << "\t"
                          << runCount << "\t"
                          << vvBinOnnxruntimeLibraryPath.string() << "\t"
                          << exportedOnnxruntimeLibraryPath.string() << "\t"
                          << vvBinSessionText << "\t"
                          << vvBinRunText << "\t"
                          << exportedSessionText << "\t"
                          << exportedRunText << "\t"
                          << outputMatchText << "\t"
                          << traceMatchVvBinText << "\t"
                          << traceMatchExportedText << "\t"
                          << statusText << "\t"
                          << detailText << "\n";
        }
        closeNativeOnnxApi(exportedOnnxApi);
        closeNativeOnnxApi(vvBinOnnxApi);
        return compareStream.str();
    } catch (...) {
        closeNativeOnnxApi(exportedOnnxApi);
        closeNativeOnnxApi(vvBinOnnxApi);
        throw;
    }
}

