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
#include <cstdio>
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

std::string getNativeOnnxVersion(const NativeOnnxApi &nativeOnnxApi) {
    if (!nativeOnnxApi.apiBase || !nativeOnnxApi.apiBase->getVersionString) {
        return "";
    }
    const char *versionText = nativeOnnxApi.apiBase->getVersionString();
    return versionText ? versionText : "";
}

std::string formatNativeOnnxElementType(int32_t elementType) {
    switch (elementType) {
        case 1:
            return "float32";
        case 2:
            return "uint8";
        case 3:
            return "int8";
        case 4:
            return "uint16";
        case 5:
            return "int16";
        case 6:
            return "int32";
        case 7:
            return "int64";
        case 9:
            return "bool";
        case 10:
            return "float16";
        case 11:
            return "float64";
        case 12:
            return "uint32";
        case 13:
            return "uint64";
        case 16:
            return "bfloat16";
        default:
            return "type_" + std::to_string(elementType);
    }
}

std::string formatNativeOnnxShape(const std::vector<int64_t> &dimensions) {
    std::ostringstream shapeStream;
    shapeStream << "[";
    for (size_t dimensionIndex = 0; dimensionIndex < dimensions.size(); dimensionIndex++) {
        if (dimensionIndex > 0) {
            shapeStream << ",";
        }
        if (dimensions[dimensionIndex] < 0) {
            shapeStream << "?";
        } else {
            shapeStream << dimensions[dimensionIndex];
        }
    }
    shapeStream << "]";
    return shapeStream.str();
}

static std::string takeNativeOnnxName(NativeOnnxApi &nativeOnnxApi, OrtAllocator *allocator, char *namePointer) {
    if (!namePointer) {
        return "";
    }
    std::string nameText(namePointer);
    ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.allocatorFree(allocator, namePointer), "name 解放");
    return nameText;
}

void appendNativeOnnxValueInfo(std::ostringstream &inspectStream, NativeOnnxApi &nativeOnnxApi, OrtSession *session, OrtAllocator *allocator, const std::string &prefixText, size_t valueIndex) {
    char *namePointer = nullptr;
    OrtTypeInfo *typeInfo = nullptr;
    try {
        if (prefixText == "input") {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetInputName(session, valueIndex, allocator, &namePointer), "input name 取得");
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetInputTypeInfo(session, valueIndex, &typeInfo), "input type 取得");
        } else {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetOutputName(session, valueIndex, allocator, &namePointer), "output name 取得");
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetOutputTypeInfo(session, valueIndex, &typeInfo), "output type 取得");
        }
        std::string nameText = takeNativeOnnxName(nativeOnnxApi, allocator, namePointer);
        namePointer = nullptr;
        const OrtTensorTypeAndShapeInfo *tensorInfo = nullptr;
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.castTypeInfoToTensorInfo(typeInfo, &tensorInfo), "tensor info 取得");
        int32_t elementType = 0;
        size_t dimensionCount = 0;
        std::vector<int64_t> dimensions;
        if (tensorInfo) {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorElementType(tensorInfo, &elementType), "tensor type 取得");
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getDimensionsCount(tensorInfo, &dimensionCount), "dimension count 取得");
            dimensions.resize(dimensionCount);
            if (!dimensions.empty()) {
                ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getDimensions(tensorInfo, dimensions.data(), dimensions.size()), "dimension 取得");
            }
        }
        inspectStream << prefixText << "_" << valueIndex << "\t"
                      << nameText << "\t"
                      << formatNativeOnnxElementType(elementType) << "\t"
                      << formatNativeOnnxShape(dimensions) << "\n";
        nativeOnnxApi.releaseTypeInfo(typeInfo);
    } catch (...) {
        if (namePointer) {
            nativeOnnxApi.allocatorFree(allocator, namePointer);
        }
        if (typeInfo) {
            nativeOnnxApi.releaseTypeInfo(typeInfo);
        }
        throw;
    }
}


std::mutex nativeOnnxSessionCacheMutex;
std::condition_variable nativeOnnxSessionCacheCondition;
std::map<std::string, std::shared_ptr<NativeOnnxCachedSession>> nativeOnnxSessionCache;
std::set<std::string> nativeOnnxSessionKeysInProgress;
uint64_t nativeOnnxSessionCacheHits = 0;
uint64_t nativeOnnxSessionCacheMisses = 0;
std::mutex nativeOnnxExportedModelCacheMutex;
std::condition_variable nativeOnnxExportedModelCacheCondition;
std::map<std::string, fs::path> nativeOnnxExportedModelCache;
std::set<std::string> nativeOnnxExportedModelKeysInProgress;
uint64_t nativeOnnxExportedModelMemoryHits = 0;
uint64_t nativeOnnxExportedModelMemoryMisses = 0;
uint64_t nativeOnnxExportedModelFileHits = 0;
uint64_t nativeOnnxExportedModelFileMisses = 0;
uint64_t nativeOnnxExportedModelFileWrites = 0;
uint64_t nativeOnnxExportedModelFileReadErrors = 0;
uint64_t nativeOnnxExportedModelFileWriteErrors = 0;
std::atomic_size_t nativeOnnxNextTraceId{0};

std::string createNativeOnnxSessionCacheInfoJson() {
    std::lock_guard<std::mutex> cacheLock(nativeOnnxSessionCacheMutex);
    std::ostringstream jsonStream;
    jsonStream << "{\"loaded\":" << (nativeOnnxSessionCache.empty() ? "false" : "true") << ",";
    jsonStream << "\"session_count\":" << nativeOnnxSessionCache.size() << ",";
    jsonStream << "\"cache_hits\":" << nativeOnnxSessionCacheHits << ",";
    jsonStream << "\"cache_misses\":" << nativeOnnxSessionCacheMisses << ",";
    jsonStream << "\"cache_accesses\":" << (nativeOnnxSessionCacheHits + nativeOnnxSessionCacheMisses) << "}";
    return jsonStream.str();
}

std::string createNativeOnnxExportedModelCacheInfoJson() {
    std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
    std::ostringstream jsonStream;
    jsonStream << "{\"loaded\":" << (nativeOnnxExportedModelCache.empty() ? "false" : "true") << ",";
    jsonStream << "\"model_count\":" << nativeOnnxExportedModelCache.size() << ",";
    jsonStream << "\"memory_hits\":" << nativeOnnxExportedModelMemoryHits << ",";
    jsonStream << "\"memory_misses\":" << nativeOnnxExportedModelMemoryMisses << ",";
    jsonStream << "\"memory_accesses\":" << (nativeOnnxExportedModelMemoryHits + nativeOnnxExportedModelMemoryMisses) << ",";
    jsonStream << "\"file_hits\":" << nativeOnnxExportedModelFileHits << ",";
    jsonStream << "\"file_misses\":" << nativeOnnxExportedModelFileMisses << ",";
    jsonStream << "\"file_writes\":" << nativeOnnxExportedModelFileWrites << ",";
    jsonStream << "\"file_read_errors\":" << nativeOnnxExportedModelFileReadErrors << ",";
    jsonStream << "\"file_write_errors\":" << nativeOnnxExportedModelFileWriteErrors << "}";
    return jsonStream.str();
}

std::string createNativeOnnxDeterministicSingTeacherInfoJson(bool shouldUseVvBinConfig) {
    NativeOnnxSingTeacherMode teacherMode = getNativeOnnxSingTeacherMode();
    std::ostringstream jsonStream;
    jsonStream << "{\"enabled\":true,";
    if (teacherMode == NativeOnnxSingTeacherMode::Deterministic) {
        float seedValue = getNativeOnnxDeterministicSingTeacherSeed();
        jsonStream << "\"mode\":\"seeded_exported_onnx\",";
        jsonStream << "\"deterministic\":true,";
        jsonStream << "\"seed\":" << std::setprecision(9) << static_cast<double>(seedValue) << "}";
        return jsonStream.str();
    }
    jsonStream << "\"mode\":\"" << (shouldUseVvBinConfig ? "vv_bin" : "exported_onnx") << "\",";
    jsonStream << "\"deterministic\":false,";
    jsonStream << "\"seed\":0}";
    return jsonStream.str();
}

int32_t parseNativeOnnxElementType(const std::string &typeText) {
    if (typeText == "float32" || typeText == "float") {
        return 1;
    }
    if (typeText == "int32") {
        return 6;
    }
    if (typeText == "int64") {
        return 7;
    }
    if (typeText == "bool") {
        return 9;
    }
    throw std::runtime_error("未対応の dtype です: " + typeText);
}

NativeOnnxValueDescriptor readNativeOnnxValueDescriptor(NativeOnnxApi &nativeOnnxApi, OrtSession *session, OrtAllocator *allocator, bool isInput, size_t valueIndex) {
    char *namePointer = nullptr;
    OrtTypeInfo *typeInfo = nullptr;
    try {
        NativeOnnxValueDescriptor valueDescriptor;
        if (isInput) {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetInputName(session, valueIndex, allocator, &namePointer), "input name 取得");
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetInputTypeInfo(session, valueIndex, &typeInfo), "input type 取得");
        } else {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetOutputName(session, valueIndex, allocator, &namePointer), "output name 取得");
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetOutputTypeInfo(session, valueIndex, &typeInfo), "output type 取得");
        }
        valueDescriptor.name = takeNativeOnnxName(nativeOnnxApi, allocator, namePointer);
        namePointer = nullptr;
        const OrtTensorTypeAndShapeInfo *tensorInfo = nullptr;
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.castTypeInfoToTensorInfo(typeInfo, &tensorInfo), "tensor info 取得");
        if (tensorInfo) {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorElementType(tensorInfo, &valueDescriptor.elementType), "tensor type 取得");
            size_t dimensionCount = 0;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getDimensionsCount(tensorInfo, &dimensionCount), "dimension count 取得");
            valueDescriptor.dimensions.resize(dimensionCount);
            if (!valueDescriptor.dimensions.empty()) {
                ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getDimensions(tensorInfo, valueDescriptor.dimensions.data(), valueDescriptor.dimensions.size()), "dimension 取得");
            }
        }
        nativeOnnxApi.releaseTypeInfo(typeInfo);
        return valueDescriptor;
    } catch (...) {
        if (namePointer) {
            nativeOnnxApi.allocatorFree(allocator, namePointer);
        }
        if (typeInfo) {
            nativeOnnxApi.releaseTypeInfo(typeInfo);
        }
        throw;
    }
}


static void CaptureProfile(NativeOnnxApi Api, NativeOnnxCachedSession &Session, OrtAllocator *Allocator, const NativeOnnxRuntimeState *State) {
    const char *Prefix = std::getenv("LITEVOX_ORT_PROFILE");
    if (!State || !Prefix || !*Prefix) return;
    auto End = Api.EndProfile;
    Session.FinishProfile = [Api, End, Allocator, Handle = Session.session] {
        char *Path = nullptr;
        OrtStatus *Status = End(Handle, Allocator, &Path);
        if (Status) {
            std::fprintf(stderr, "ORT profile 書き出し失敗: %s\n", Api.getErrorMessage(Status));
            Api.releaseStatus(Status);
        }
        if (Path) {
            Status = Api.allocatorFree(Allocator, Path);
            if (Status) Api.releaseStatus(Status);
        }
    };
}

std::shared_ptr<NativeOnnxCachedSession> createNativeOnnxCachedSession(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<uint8_t> &modelBytes, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    std::shared_ptr<NativeOnnxCachedSession> cachedSession = std::make_shared<NativeOnnxCachedSession>();
    OrtSessionOptions *sessionOptions = nullptr;
    OrtAllocator *allocator = nullptr;
    try {
        cachedSession->IsSerial = runtimeState && runtimeState->selectedExecutionProvider == "DmlExecutionProvider";
        cachedSession->releaseEnv = nativeOnnxApi.releaseEnv;
        cachedSession->releaseMemoryInfo = nativeOnnxApi.releaseMemoryInfo;
        cachedSession->releaseSession = nativeOnnxApi.releaseSession;
        cachedSession->libraryHandle = openDynamicLibrary(nativeOnnxApi.libraryPath);
        if (!cachedSession->libraryHandle) {
            throw std::runtime_error(std::string("ONNX Runtime を保持できません: ") + getDynamicLibraryErrorText());
        }
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createEnv(IsCoremlProfile() ? 0 : ortLoggingLevelWarning, "litevox-native-cache", &cachedSession->env), "OrtEnv 作成");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getAllocatorWithDefaultOptions(&allocator), "default allocator 取得");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionOptions(&sessionOptions), "SessionOptions 作成");
        configureNativeOnnxSessionOptions(nativeOnnxApi, runtimeState, sessionOptions, cpuThreadCount, shouldUseVvBinConfig);
        applyNativeOnnxSeedIfConfigured(nativeOnnxApi);
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionFromArray(cachedSession->env, modelBytes.data(), modelBytes.size(), sessionOptions, &cachedSession->session), "ONNX session 作成");
        size_t inputCount = 0;
        size_t outputCount = 0;
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetInputCount(cachedSession->session, &inputCount), "input count 取得");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetOutputCount(cachedSession->session, &outputCount), "output count 取得");
        cachedSession->inputDescriptors.reserve(inputCount);
        cachedSession->outputDescriptors.reserve(outputCount);
        for (size_t inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            cachedSession->inputDescriptors.push_back(readNativeOnnxValueDescriptor(nativeOnnxApi, cachedSession->session, allocator, true, inputIndex));
        }
        for (size_t outputIndex = 0; outputIndex < outputCount; outputIndex++) {
            cachedSession->outputDescriptors.push_back(readNativeOnnxValueDescriptor(nativeOnnxApi, cachedSession->session, allocator, false, outputIndex));
        }
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createCpuMemoryInfo(0, 0, &cachedSession->memoryInfo), "CPU memory info 作成");
        CaptureProfile(nativeOnnxApi, *cachedSession, allocator, runtimeState);
        nativeOnnxApi.releaseSessionOptions(sessionOptions);
        return cachedSession;
    } catch (...) {
        if (sessionOptions) {
            nativeOnnxApi.releaseSessionOptions(sessionOptions);
        }
        throw;
    }
}

std::shared_ptr<NativeOnnxCachedSession> createNativeOnnxCachedSession(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const fs::path &modelPath, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    std::shared_ptr<NativeOnnxCachedSession> cachedSession = std::make_shared<NativeOnnxCachedSession>();
    OrtSessionOptions *sessionOptions = nullptr;
    OrtAllocator *allocator = nullptr;
    try {
        cachedSession->IsSerial = runtimeState && runtimeState->selectedExecutionProvider == "DmlExecutionProvider";
        cachedSession->releaseEnv = nativeOnnxApi.releaseEnv;
        cachedSession->releaseMemoryInfo = nativeOnnxApi.releaseMemoryInfo;
        cachedSession->releaseSession = nativeOnnxApi.releaseSession;
        cachedSession->libraryHandle = openDynamicLibrary(nativeOnnxApi.libraryPath);
        if (!cachedSession->libraryHandle) {
            throw std::runtime_error(std::string("ONNX Runtime を保持できません: ") + getDynamicLibraryErrorText());
        }
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createEnv(IsCoremlProfile() ? 0 : ortLoggingLevelWarning, "litevox-native-cache", &cachedSession->env), "OrtEnv 作成");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getAllocatorWithDefaultOptions(&allocator), "default allocator 取得");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionOptions(&sessionOptions), "SessionOptions 作成");
        configureNativeOnnxSessionOptions(nativeOnnxApi, runtimeState, sessionOptions, cpuThreadCount, shouldUseVvBinConfig);
        applyNativeOnnxSeedIfConfigured(nativeOnnxApi);
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSession(cachedSession->env, modelPath.c_str(), sessionOptions, &cachedSession->session), "ONNX session 作成");
        size_t inputCount = 0;
        size_t outputCount = 0;
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetInputCount(cachedSession->session, &inputCount), "input count 取得");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetOutputCount(cachedSession->session, &outputCount), "output count 取得");
        cachedSession->inputDescriptors.reserve(inputCount);
        cachedSession->outputDescriptors.reserve(outputCount);
        for (size_t inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            cachedSession->inputDescriptors.push_back(readNativeOnnxValueDescriptor(nativeOnnxApi, cachedSession->session, allocator, true, inputIndex));
        }
        for (size_t outputIndex = 0; outputIndex < outputCount; outputIndex++) {
            cachedSession->outputDescriptors.push_back(readNativeOnnxValueDescriptor(nativeOnnxApi, cachedSession->session, allocator, false, outputIndex));
        }
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createCpuMemoryInfo(0, 0, &cachedSession->memoryInfo), "CPU memory info 作成");
        CaptureProfile(nativeOnnxApi, *cachedSession, allocator, runtimeState);
        nativeOnnxApi.releaseSessionOptions(sessionOptions);
        return cachedSession;
    } catch (...) {
        if (sessionOptions) {
            nativeOnnxApi.releaseSessionOptions(sessionOptions);
        }
        throw;
    }
}

std::shared_ptr<NativeOnnxCachedSession> CacheSession(const std::string &sessionCacheKey, const std::function<std::shared_ptr<NativeOnnxCachedSession>()> &Create) {
    {
        std::unique_lock<std::mutex> cacheLock(nativeOnnxSessionCacheMutex);
        while (true) {
            auto cacheIterator = nativeOnnxSessionCache.find(sessionCacheKey);
            if (cacheIterator != nativeOnnxSessionCache.end()) {
                nativeOnnxSessionCacheHits++;
                return cacheIterator->second;
            }
            if (nativeOnnxSessionKeysInProgress.insert(sessionCacheKey).second) {
                nativeOnnxSessionCacheMisses++;
                break;
            }
            nativeOnnxSessionCacheCondition.wait(cacheLock);
        }
    }
    try {
        std::shared_ptr<NativeOnnxCachedSession> cachedSession = Create();
        std::lock_guard<std::mutex> cacheLock(nativeOnnxSessionCacheMutex);
        nativeOnnxSessionCache[sessionCacheKey] = cachedSession;
        nativeOnnxSessionKeysInProgress.erase(sessionCacheKey);
        nativeOnnxSessionCacheCondition.notify_all();
        return cachedSession;
    } catch (...) {
        std::lock_guard<std::mutex> cacheLock(nativeOnnxSessionCacheMutex);
        nativeOnnxSessionKeysInProgress.erase(sessionCacheKey);
        nativeOnnxSessionCacheCondition.notify_all();
        throw;
    }
}

std::shared_ptr<NativeOnnxCachedSession> getNativeOnnxCachedSession(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<uint8_t> &modelBytes, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey) {
    return CacheSession(sessionCacheKey, [&] { return createNativeOnnxCachedSession(nativeOnnxApi, runtimeState, modelBytes, cpuThreadCount, shouldUseVvBinConfig); });
}

std::shared_ptr<NativeOnnxCachedSession> getNativeOnnxCachedSession(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const fs::path &modelPath, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey) {
    return CacheSession(sessionCacheKey, [&] { return createNativeOnnxCachedSession(nativeOnnxApi, runtimeState, modelPath, cpuThreadCount, shouldUseVvBinConfig); });
}

std::vector<NativeOnnxTraceInput> runNativeOnnxPreparedSession(NativeOnnxApi &nativeOnnxApi, const std::shared_ptr<NativeOnnxCachedSession> &cachedSession, const std::vector<NativeOnnxTraceInput> &inputTensors) {
    std::unique_lock<std::mutex> Lock(cachedSession->RunMutex, std::defer_lock);
    if (cachedSession->IsSerial) Lock.lock();
    std::vector<OrtValue *> inputValues;
    std::vector<OrtValue *> outputValues(cachedSession->outputDescriptors.size(), nullptr);
    try {
        applyNativeOnnxSeedIfConfigured(nativeOnnxApi);
        std::vector<const char *> inputNames;
        std::vector<const OrtValue *> inputPointers;
        inputNames.reserve(cachedSession->inputDescriptors.size());
        inputPointers.reserve(cachedSession->inputDescriptors.size());
        inputValues.reserve(cachedSession->inputDescriptors.size());
        for (const NativeOnnxValueDescriptor &inputDescriptor : cachedSession->inputDescriptors) {
            const NativeOnnxTraceInput &inputTensor = requireNativeOnnxTensor(inputTensors, inputDescriptor.name);
            if (inputTensor.elementType != inputDescriptor.elementType) {
                throw std::runtime_error("input dtype が一致しません: " + inputDescriptor.name);
            }
            if (!areNativeOnnxDimensionsEqual(inputDescriptor.dimensions, inputTensor.dimensions)) {
                throw std::runtime_error("input shape が一致しません: " + inputDescriptor.name);
            }
            OrtValue *inputValue = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createTensorWithDataAsOrtValue(cachedSession->memoryInfo, const_cast<uint8_t *>(inputTensor.bytes.data()), inputTensor.bytes.size(), inputTensor.dimensions.data(), inputTensor.dimensions.size(), inputDescriptor.elementType, &inputValue), "input tensor 作成");
            inputValues.push_back(inputValue);
            inputPointers.push_back(inputValue);
            inputNames.push_back(inputDescriptor.name.c_str());
        }
        std::vector<const char *> outputNames;
        outputNames.reserve(cachedSession->outputDescriptors.size());
        for (const NativeOnnxValueDescriptor &outputDescriptor : cachedSession->outputDescriptors) {
            outputNames.push_back(outputDescriptor.name.c_str());
        }
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.run(cachedSession->session, nullptr, inputNames.data(), inputPointers.data(), inputPointers.size(), outputNames.data(), outputNames.size(), outputValues.data()), "ONNX Run");
        std::vector<NativeOnnxTraceInput> outputTensors;
        outputTensors.reserve(cachedSession->outputDescriptors.size());
        for (size_t outputIndex = 0; outputIndex < outputValues.size(); outputIndex++) {
            OrtTensorTypeAndShapeInfo *outputShapeInfo = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorTypeAndShape(outputValues[outputIndex], &outputShapeInfo), "output tensor info 取得");
            size_t outputElementCount = 0;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorShapeElementCount(outputShapeInfo, &outputElementCount), "output element count 取得");
            std::vector<int64_t> outputDimensions = readNativeOnnxTensorDimensions(nativeOnnxApi, outputShapeInfo);
            void *outputPointer = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorMutableData(outputValues[outputIndex], &outputPointer), "output tensor data 取得");
            size_t outputByteCount = outputElementCount * getNativeOnnxElementByteCount(cachedSession->outputDescriptors[outputIndex].elementType);
            NativeOnnxTraceInput outputTensor;
            outputTensor.name = cachedSession->outputDescriptors[outputIndex].name;
            outputTensor.elementType = cachedSession->outputDescriptors[outputIndex].elementType;
            outputTensor.dimensions = outputDimensions;
            outputTensor.bytes.resize(outputByteCount);
            if (outputByteCount > 0) {
                std::memcpy(outputTensor.bytes.data(), outputPointer, outputByteCount);
            }
            outputTensors.push_back(std::move(outputTensor));
            nativeOnnxApi.releaseTensorTypeAndShapeInfo(outputShapeInfo);
        }
        for (OrtValue *outputValue : outputValues) {
            if (outputValue) {
                nativeOnnxApi.releaseValue(outputValue);
            }
        }
        for (OrtValue *inputValue : inputValues) {
            if (inputValue) {
                nativeOnnxApi.releaseValue(inputValue);
            }
        }
        return outputTensors;
    } catch (...) {
        for (OrtValue *outputValue : outputValues) {
            if (outputValue) {
                nativeOnnxApi.releaseValue(outputValue);
            }
        }
        for (OrtValue *inputValue : inputValues) {
            if (inputValue) {
                nativeOnnxApi.releaseValue(inputValue);
            }
        }
        throw;
    }
}

static std::vector<NativeOnnxTraceInput> runNativeOnnxCachedModelBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<uint8_t> &modelBytes, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey) {
    std::shared_ptr<NativeOnnxCachedSession> cachedSession = getNativeOnnxCachedSession(nativeOnnxApi, runtimeState, modelBytes, cpuThreadCount, shouldUseVvBinConfig, sessionCacheKey);
    return runNativeOnnxPreparedSession(nativeOnnxApi, cachedSession, inputTensors);
}

static std::vector<NativeOnnxTraceInput> runNativeOnnxCachedModelPath(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const fs::path &modelPath, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey) {
    std::shared_ptr<NativeOnnxCachedSession> cachedSession = getNativeOnnxCachedSession(nativeOnnxApi, runtimeState, modelPath, cpuThreadCount, shouldUseVvBinConfig, sessionCacheKey);
    return runNativeOnnxPreparedSession(nativeOnnxApi, cachedSession, inputTensors);
}

std::vector<NativeOnnxTraceInput> runNativeOnnxModelBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<uint8_t> &modelBytes, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey) {
    if (!sessionCacheKey.empty()) {
        return runNativeOnnxCachedModelBytes(nativeOnnxApi, runtimeState, modelBytes, inputTensors, cpuThreadCount, shouldUseVvBinConfig, sessionCacheKey);
    }
    OrtEnv *env = nullptr;
    OrtSessionOptions *sessionOptions = nullptr;
    OrtSession *session = nullptr;
    OrtAllocator *allocator = nullptr;
    OrtMemoryInfo *memoryInfo = nullptr;
    std::vector<OrtValue *> inputValues;
    std::vector<OrtValue *> outputValues;
    try {
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createEnv(ortLoggingLevelWarning, "litevox-native-chain", &env), "OrtEnv 作成");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getAllocatorWithDefaultOptions(&allocator), "default allocator 取得");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionOptions(&sessionOptions), "SessionOptions 作成");
        configureNativeOnnxSessionOptions(nativeOnnxApi, runtimeState, sessionOptions, cpuThreadCount, shouldUseVvBinConfig);
        applyNativeOnnxSeedIfConfigured(nativeOnnxApi);
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionFromArray(env, modelBytes.data(), modelBytes.size(), sessionOptions, &session), "ONNX session 作成");
        size_t inputCount = 0;
        size_t outputCount = 0;
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetInputCount(session, &inputCount), "input count 取得");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetOutputCount(session, &outputCount), "output count 取得");
        std::vector<NativeOnnxValueDescriptor> inputDescriptors;
        std::vector<NativeOnnxValueDescriptor> outputDescriptors;
        inputDescriptors.reserve(inputCount);
        outputDescriptors.reserve(outputCount);
        for (size_t inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            inputDescriptors.push_back(readNativeOnnxValueDescriptor(nativeOnnxApi, session, allocator, true, inputIndex));
        }
        for (size_t outputIndex = 0; outputIndex < outputCount; outputIndex++) {
            outputDescriptors.push_back(readNativeOnnxValueDescriptor(nativeOnnxApi, session, allocator, false, outputIndex));
        }
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createCpuMemoryInfo(0, 0, &memoryInfo), "CPU memory info 作成");
        std::vector<const char *> inputNames;
        std::vector<const OrtValue *> inputPointers;
        inputNames.reserve(inputDescriptors.size());
        inputPointers.reserve(inputDescriptors.size());
        inputValues.reserve(inputDescriptors.size());
        for (const NativeOnnxValueDescriptor &inputDescriptor : inputDescriptors) {
            const NativeOnnxTraceInput &inputTensor = requireNativeOnnxTensor(inputTensors, inputDescriptor.name);
            if (inputTensor.elementType != inputDescriptor.elementType) {
                throw std::runtime_error("input dtype が一致しません: " + inputDescriptor.name);
            }
            if (!areNativeOnnxDimensionsEqual(inputDescriptor.dimensions, inputTensor.dimensions)) {
                throw std::runtime_error("input shape が一致しません: " + inputDescriptor.name);
            }
            OrtValue *inputValue = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createTensorWithDataAsOrtValue(memoryInfo, const_cast<uint8_t *>(inputTensor.bytes.data()), inputTensor.bytes.size(), inputTensor.dimensions.data(), inputTensor.dimensions.size(), inputDescriptor.elementType, &inputValue), "input tensor 作成");
            inputValues.push_back(inputValue);
            inputPointers.push_back(inputValue);
            inputNames.push_back(inputDescriptor.name.c_str());
        }
        std::vector<const char *> outputNames;
        outputNames.reserve(outputDescriptors.size());
        for (const NativeOnnxValueDescriptor &outputDescriptor : outputDescriptors) {
            outputNames.push_back(outputDescriptor.name.c_str());
        }
        outputValues.resize(outputDescriptors.size(), nullptr);
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.run(session, nullptr, inputNames.data(), inputPointers.data(), inputPointers.size(), outputNames.data(), outputNames.size(), outputValues.data()), "ONNX Run");
        std::vector<NativeOnnxTraceInput> outputTensors;
        outputTensors.reserve(outputDescriptors.size());
        for (size_t outputIndex = 0; outputIndex < outputValues.size(); outputIndex++) {
            OrtTensorTypeAndShapeInfo *outputShapeInfo = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorTypeAndShape(outputValues[outputIndex], &outputShapeInfo), "output tensor info 取得");
            size_t outputElementCount = 0;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorShapeElementCount(outputShapeInfo, &outputElementCount), "output element count 取得");
            std::vector<int64_t> outputDimensions = readNativeOnnxTensorDimensions(nativeOnnxApi, outputShapeInfo);
            void *outputPointer = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorMutableData(outputValues[outputIndex], &outputPointer), "output tensor data 取得");
            size_t outputByteCount = outputElementCount * getNativeOnnxElementByteCount(outputDescriptors[outputIndex].elementType);
            NativeOnnxTraceInput outputTensor;
            outputTensor.name = outputDescriptors[outputIndex].name;
            outputTensor.elementType = outputDescriptors[outputIndex].elementType;
            outputTensor.dimensions = outputDimensions;
            outputTensor.bytes.resize(outputByteCount);
            if (outputByteCount > 0) {
                std::memcpy(outputTensor.bytes.data(), outputPointer, outputByteCount);
            }
            outputTensors.push_back(std::move(outputTensor));
            nativeOnnxApi.releaseTensorTypeAndShapeInfo(outputShapeInfo);
        }
        for (OrtValue *outputValue : outputValues) {
            if (outputValue) {
                nativeOnnxApi.releaseValue(outputValue);
            }
        }
        for (OrtValue *inputValue : inputValues) {
            if (inputValue) {
                nativeOnnxApi.releaseValue(inputValue);
            }
        }
        nativeOnnxApi.releaseMemoryInfo(memoryInfo);
        nativeOnnxApi.releaseSession(session);
        nativeOnnxApi.releaseSessionOptions(sessionOptions);
        nativeOnnxApi.releaseEnv(env);
        return outputTensors;
    } catch (...) {
        for (OrtValue *outputValue : outputValues) {
            if (outputValue) {
                nativeOnnxApi.releaseValue(outputValue);
            }
        }
        for (OrtValue *inputValue : inputValues) {
            if (inputValue) {
                nativeOnnxApi.releaseValue(inputValue);
            }
        }
        if (memoryInfo) {
            nativeOnnxApi.releaseMemoryInfo(memoryInfo);
        }
        if (session) {
            nativeOnnxApi.releaseSession(session);
        }
        if (sessionOptions) {
            nativeOnnxApi.releaseSessionOptions(sessionOptions);
        }
        if (env) {
            nativeOnnxApi.releaseEnv(env);
        }
        throw;
    }
}

std::vector<NativeOnnxTraceInput> runNativeOnnxModelPath(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const fs::path &modelPath, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey) {
    if (!sessionCacheKey.empty()) {
        return runNativeOnnxCachedModelPath(nativeOnnxApi, runtimeState, modelPath, inputTensors, cpuThreadCount, shouldUseVvBinConfig, sessionCacheKey);
    }
    std::shared_ptr<NativeOnnxCachedSession> cachedSession = createNativeOnnxCachedSession(nativeOnnxApi, runtimeState, modelPath, cpuThreadCount, shouldUseVvBinConfig);
    return runNativeOnnxPreparedSession(nativeOnnxApi, cachedSession, inputTensors);
}

