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

std::vector<uint8_t> extractNativeOnnxModelAssetBytes(const ModelAssetRecord &modelAsset) {
    return extractVvmEntryBytesAt(
        modelAsset.archivePath,
        modelAsset.entryName,
        modelAsset.dataOffset,
        modelAsset.compressedSize,
        modelAsset.uncompressedSize,
        modelAsset.compressionMethod);
}

static std::string createNativeOnnxModelAssetSessionCacheKey(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    std::ostringstream keyStream;
    keyStream << nativeOnnxApi.libraryPath.string() << "\t";
    keyStream << modelAsset.archivePath.string() << "\t";
    keyStream << modelAsset.entryName << "\t";
    keyStream << modelAsset.dataOffset << "\t";
    keyStream << modelAsset.compressedSize << "\t";
    keyStream << modelAsset.uncompressedSize << "\t";
    keyStream << modelAsset.crc32 << "\t";
    keyStream << modelAsset.compressionMethod << "\t";
    keyStream << cpuThreadCount << "\t";
    keyStream << (shouldUseVvBinConfig ? "vvbin" : "onnx") << "\t";
    keyStream << (runtimeState ? runtimeState->selectedExecutionProvider : "CPUExecutionProvider");
    return keyStream.str();
}

static std::vector<NativeOnnxTraceInput> RunVvBin(NativeOnnxApi &Api, const NativeOnnxRuntimeState *State, const ModelAssetRecord &Asset, const std::vector<NativeOnnxTraceInput> &Inputs, uint16_t Threads, const std::vector<uint8_t> *Bytes = nullptr) {
    auto Session = CacheSession(createNativeOnnxModelAssetSessionCacheKey(Api, State, Asset, Threads, true), [&] {
        // バイト列はセッション作成時だけ必要。キャッシュヒット時の展開・コピーを避ける。
        if (Bytes) return createNativeOnnxCachedSession(Api, State, *Bytes, Threads, true);
        return createNativeOnnxCachedSession(Api, State, extractNativeOnnxModelAssetBytes(Asset), Threads, true);
    });
    return runNativeOnnxPreparedSession(Api, Session, Inputs);
}

static fs::path getNativeOnnxVvBinExportLibraryPath(NativeOnnxApi &nativeOnnxApi) {
    const char *libraryPathText = std::getenv("LITEVOX_VV_BIN_ONNXRUNTIME");
    if (!libraryPathText || libraryPathText[0] == '\0') {
        return nativeOnnxApi.libraryPath;
    }
    return fs::path(libraryPathText);
}

static std::string createNativeOnnxFileIdentityText(const fs::path &filePath) {
    std::error_code canonicalError;
    fs::path stablePath = fs::weakly_canonical(filePath, canonicalError);
    std::error_code sizeError;
    uintmax_t fileSize = fs::file_size(filePath, sizeError);
    std::error_code timeError;
    fs::file_time_type lastWriteTime = fs::last_write_time(filePath, timeError);
    std::ostringstream identityStream;
    identityStream << (canonicalError ? filePath.lexically_normal().string() : stablePath.string()) << "\t";
    identityStream << (sizeError ? 0 : fileSize) << "\t";
    identityStream << (timeError ? 0 : static_cast<long long>(lastWriteTime.time_since_epoch().count()));
    return identityStream.str();
}

static std::string createNativeOnnxExportedModelCacheKey(const ModelAssetRecord &modelAsset, const fs::path &exportLibraryPath, uint16_t cpuThreadCount) {
    std::ostringstream keyStream;
    keyStream << "exported_onnx\t";
    keyStream << createNativeOnnxFileIdentityText(exportLibraryPath) << "\t";
    keyStream << createNativeOnnxFileIdentityText(modelAsset.archivePath) << "\t";
    keyStream << modelAsset.entryName << "\t";
    keyStream << modelAsset.dataOffset << "\t";
    keyStream << modelAsset.compressedSize << "\t";
    keyStream << modelAsset.uncompressedSize << "\t";
    keyStream << modelAsset.crc32 << "\t";
    keyStream << modelAsset.compressionMethod << "\t";
    keyStream << cpuThreadCount;
    return keyStream.str();
}

static std::string createNativeOnnxExportedModelCacheBaseName(const ModelAssetRecord &modelAsset) {
    std::string fileName = modelAsset.archivePath.stem().string() + "-" + modelAsset.entryName;
    for (char &character : fileName) {
        if (!(std::isalnum(static_cast<unsigned char>(character)) || character == '-' || character == '_' || character == '.')) {
            character = '_';
        }
    }
    return fileName;
}

static fs::path createNativeOnnxExportedModelCachePath(const ModelAssetRecord &modelAsset, const std::string &cacheKey) {
    std::string cacheDigest = createSha256Hex(reinterpret_cast<const uint8_t *>(cacheKey.data()), cacheKey.size());
    return modelAsset.archivePath.parent_path() / ".litevox-exported-onnx-cache" / (createNativeOnnxExportedModelCacheBaseName(modelAsset) + "-" + cacheDigest.substr(0, 16) + ".onnx");
}

static void storeNativeOnnxExportedModelCacheFileHit() {
    std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
    nativeOnnxExportedModelFileHits++;
}

static void storeNativeOnnxExportedModelCacheFileMiss() {
    std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
    nativeOnnxExportedModelFileMisses++;
}

static void storeNativeOnnxExportedModelCacheFileWrite() {
    std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
    nativeOnnxExportedModelFileWrites++;
}

static void storeNativeOnnxExportedModelCacheFileReadError() {
    std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
    nativeOnnxExportedModelFileReadErrors++;
}

static void storeNativeOnnxExportedModelCacheFileWriteError() {
    std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
    nativeOnnxExportedModelFileWriteErrors++;
}

static std::vector<uint8_t> createNativeOnnxExportedModelBytes(NativeOnnxApi &nativeOnnxApi, const fs::path &exportLibraryPath, const ModelAssetRecord &modelAsset, uint16_t cpuThreadCount) {
    std::vector<uint8_t> modelBytes = extractNativeOnnxModelAssetBytes(modelAsset);
    if (exportLibraryPath == nativeOnnxApi.libraryPath) {
        return exportNativeOnnxOptimizedModelBytes(nativeOnnxApi, modelAsset, modelBytes, cpuThreadCount, true);
    }
    NativeOnnxApi exportOnnxApi = loadNativeOnnxApi(exportLibraryPath);
    try {
        std::vector<uint8_t> optimizedModelBytes = exportNativeOnnxOptimizedModelBytes(exportOnnxApi, modelAsset, modelBytes, cpuThreadCount, true);
        closeNativeOnnxApi(exportOnnxApi);
        return optimizedModelBytes;
    } catch (...) {
        closeNativeOnnxApi(exportOnnxApi);
        throw;
    }
}

static bool canFallbackToNativeOnnxExportLibrary(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState) {
    // GPU 選択後の失敗を CPU 成功に偽装しない。EP 内の通常の部分 CPU 実行とは別。
    if (runtimeState && runtimeState->isGpuExecutionProviderSelected) return false;
    fs::path exportLibraryPath = getNativeOnnxVvBinExportLibraryPath(nativeOnnxApi);
    return !exportLibraryPath.empty() && exportLibraryPath != nativeOnnxApi.libraryPath && fs::exists(exportLibraryPath);
}

static std::vector<NativeOnnxTraceInput> runNativeOnnxModelAssetViaExportLibraryVvBin(NativeOnnxApi &nativeOnnxApi, const ModelAssetRecord &modelAsset, const std::vector<uint8_t> *modelBytes, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount) {
    fs::path exportLibraryPath = getNativeOnnxVvBinExportLibraryPath(nativeOnnxApi);
    NativeOnnxApi exportOnnxApi = loadNativeOnnxApi(exportLibraryPath);
    try {
        auto outputTensors = RunVvBin(exportOnnxApi, nullptr, modelAsset, inputTensors, cpuThreadCount, modelBytes);
        closeNativeOnnxApi(exportOnnxApi);
        return outputTensors;
    } catch (...) {
        closeNativeOnnxApi(exportOnnxApi);
        throw;
    }
}

static fs::path getNativeOnnxExportedModelCachePath(NativeOnnxApi &nativeOnnxApi, const ModelAssetRecord &modelAsset, uint16_t cpuThreadCount) {
    fs::path exportLibraryPath = getNativeOnnxVvBinExportLibraryPath(nativeOnnxApi);
    std::string cacheKey = createNativeOnnxExportedModelCacheKey(modelAsset, exportLibraryPath, cpuThreadCount);
    fs::path cachePath = createNativeOnnxExportedModelCachePath(modelAsset, cacheKey);
    {
        std::unique_lock<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
        while (true) {
            auto cacheIterator = nativeOnnxExportedModelCache.find(cacheKey);
            if (cacheIterator != nativeOnnxExportedModelCache.end()) {
                nativeOnnxExportedModelMemoryHits++;
                return cacheIterator->second;
            }
            if (nativeOnnxExportedModelKeysInProgress.find(cacheKey) == nativeOnnxExportedModelKeysInProgress.end()) {
                nativeOnnxExportedModelMemoryMisses++;
                nativeOnnxExportedModelKeysInProgress.insert(cacheKey);
                break;
            }
            nativeOnnxExportedModelCacheCondition.wait(cacheLock);
        }
    }
    try {
        if (fs::exists(cachePath)) {
            storeNativeOnnxExportedModelCacheFileHit();
        } else {
            storeNativeOnnxExportedModelCacheFileMiss();
            std::vector<uint8_t> optimizedModelBytes = createNativeOnnxExportedModelBytes(nativeOnnxApi, exportLibraryPath, modelAsset, cpuThreadCount);
            try {
                writeBinaryFile(cachePath, optimizedModelBytes);
                storeNativeOnnxExportedModelCacheFileWrite();
            } catch (...) {
                storeNativeOnnxExportedModelCacheFileWriteError();
                throw;
            }
        }
        {
            std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
            nativeOnnxExportedModelCache.emplace(cacheKey, cachePath);
            nativeOnnxExportedModelKeysInProgress.erase(cacheKey);
        }
        nativeOnnxExportedModelCacheCondition.notify_all();
        return cachePath;
    } catch (...) {
        {
            std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
            nativeOnnxExportedModelKeysInProgress.erase(cacheKey);
        }
        nativeOnnxExportedModelCacheCondition.notify_all();
        throw;
    }
}

static std::shared_ptr<const std::vector<uint8_t>> getNativeOnnxExportedModelBytes(NativeOnnxApi &nativeOnnxApi, const ModelAssetRecord &modelAsset, uint16_t cpuThreadCount) {
    fs::path cachePath = getNativeOnnxExportedModelCachePath(nativeOnnxApi, modelAsset, cpuThreadCount);
    try {
        std::vector<uint8_t> optimizedModelBytes = readBinaryFile(cachePath);
        if (optimizedModelBytes.empty()) {
            throw std::runtime_error("exported ONNX cache が空です");
        }
        return std::make_shared<const std::vector<uint8_t>>(std::move(optimizedModelBytes));
    } catch (...) {
        storeNativeOnnxExportedModelCacheFileReadError();
        std::error_code removeError;
        fs::remove(cachePath, removeError);
        {
            std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
            fs::path exportLibraryPath = getNativeOnnxVvBinExportLibraryPath(nativeOnnxApi);
            std::string cacheKey = createNativeOnnxExportedModelCacheKey(modelAsset, exportLibraryPath, cpuThreadCount);
            nativeOnnxExportedModelCache.erase(cacheKey);
        }
        fs::path regeneratedCachePath = getNativeOnnxExportedModelCachePath(nativeOnnxApi, modelAsset, cpuThreadCount);
        std::vector<uint8_t> optimizedModelBytes = readBinaryFile(regeneratedCachePath);
        if (optimizedModelBytes.empty()) {
            throw std::runtime_error("exported ONNX cache の再生成結果が空です");
        }
        return std::make_shared<const std::vector<uint8_t>>(std::move(optimizedModelBytes));
    }
}

NativeOnnxSingTeacherMode getNativeOnnxSingTeacherMode() {
    const char *modeText = std::getenv("LITEVOX_NATIVE_SING_TEACHER_MODE");
    if (!modeText || modeText[0] == '\0' || std::strcmp(modeText, "vv-bin") == 0 || std::strcmp(modeText, "vv_bin") == 0 || std::strcmp(modeText, "stochastic") == 0) {
        return NativeOnnxSingTeacherMode::VvBin;
    }
    if (std::strcmp(modeText, "deterministic") == 0 || std::strcmp(modeText, "seeded_exported_onnx") == 0) {
        return NativeOnnxSingTeacherMode::Deterministic;
    }
    throw std::runtime_error("LITEVOX_NATIVE_SING_TEACHER_MODE が不正です");
}

float getNativeOnnxDeterministicSingTeacherSeed() {
    const char *seedText = std::getenv("LITEVOX_NATIVE_SING_SEED");
    if (!seedText || seedText[0] == '\0') {
        return 1.0f;
    }
    char *endPointer = nullptr;
    float seedValue = std::strtof(seedText, &endPointer);
    if (!endPointer || endPointer == seedText || *endPointer != '\0' || !std::isfinite(seedValue)) {
        throw std::runtime_error("LITEVOX_NATIVE_SING_SEED が不正です");
    }
    return seedValue;
}

static std::string createNativeOnnxDeterministicSingTeacherCacheKey(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, uint16_t cpuThreadCount, float seedValue) {
    std::ostringstream keyStream;
    keyStream << createNativeOnnxModelAssetSessionCacheKey(nativeOnnxApi, runtimeState, modelAsset, cpuThreadCount, false) << "\tseeded_sing_teacher\t";
    keyStream << std::setprecision(9) << static_cast<double>(seedValue);
    return keyStream.str();
}

static std::mutex nativeOnnxDeterministicSingTeacherModelCacheMutex;
static std::condition_variable nativeOnnxDeterministicSingTeacherModelCacheCondition;
static std::map<std::string, fs::path> nativeOnnxDeterministicSingTeacherModelCache;
static std::set<std::string> nativeOnnxDeterministicSingTeacherModelKeysInProgress;

void clearNativeOnnxCaches() {
    {
        std::lock_guard<std::mutex> cacheLock(nativeOnnxSessionCacheMutex);
        nativeOnnxSessionCache.clear();
        nativeOnnxSessionKeysInProgress.clear();
        nativeOnnxSessionCacheHits = 0;
        nativeOnnxSessionCacheMisses = 0;
    }
    {
        std::lock_guard<std::mutex> cacheLock(nativeOnnxExportedModelCacheMutex);
        nativeOnnxExportedModelCache.clear();
        nativeOnnxExportedModelKeysInProgress.clear();
        nativeOnnxExportedModelMemoryHits = 0;
        nativeOnnxExportedModelMemoryMisses = 0;
        nativeOnnxExportedModelFileHits = 0;
        nativeOnnxExportedModelFileMisses = 0;
        nativeOnnxExportedModelFileWrites = 0;
        nativeOnnxExportedModelFileReadErrors = 0;
        nativeOnnxExportedModelFileWriteErrors = 0;
    }
    {
        std::lock_guard<std::mutex> cacheLock(nativeOnnxDeterministicSingTeacherModelCacheMutex);
        nativeOnnxDeterministicSingTeacherModelCache.clear();
        nativeOnnxDeterministicSingTeacherModelKeysInProgress.clear();
    }
}

static fs::path createNativeOnnxDeterministicSingTeacherModelPath(const ModelAssetRecord &modelAsset, const std::string &cacheKey) {
    std::string cacheDigest = createSha256Hex(reinterpret_cast<const uint8_t *>(cacheKey.data()), cacheKey.size());
    return modelAsset.archivePath.parent_path() / ".litevox-exported-onnx-cache" / (createNativeOnnxExportedModelCacheBaseName(modelAsset) + "-seeded-" + cacheDigest.substr(0, 16) + ".onnx");
}

static fs::path getNativeOnnxDeterministicSingTeacherModelPath(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, uint16_t cpuThreadCount, float seedValue) {
    std::string cacheKey = createNativeOnnxDeterministicSingTeacherCacheKey(nativeOnnxApi, runtimeState, modelAsset, cpuThreadCount, seedValue);
    fs::path cachePath = createNativeOnnxDeterministicSingTeacherModelPath(modelAsset, cacheKey);
    {
        std::unique_lock<std::mutex> cacheLock(nativeOnnxDeterministicSingTeacherModelCacheMutex);
        while (true) {
            auto cacheIterator = nativeOnnxDeterministicSingTeacherModelCache.find(cacheKey);
            if (cacheIterator != nativeOnnxDeterministicSingTeacherModelCache.end()) {
                return cacheIterator->second;
            }
            if (nativeOnnxDeterministicSingTeacherModelKeysInProgress.find(cacheKey) == nativeOnnxDeterministicSingTeacherModelKeysInProgress.end()) {
                nativeOnnxDeterministicSingTeacherModelKeysInProgress.insert(cacheKey);
                break;
            }
            nativeOnnxDeterministicSingTeacherModelCacheCondition.wait(cacheLock);
        }
    }
    try {
        if (!fs::exists(cachePath)) {
            std::shared_ptr<const std::vector<uint8_t>> optimizedModelBytes = getNativeOnnxExportedModelBytes(nativeOnnxApi, modelAsset, cpuThreadCount);
            size_t rewrittenNodeCount = 0;
            std::vector<uint8_t> seededModelBytes = rewriteNativeOnnxModelRandomSeed(optimizedModelBytes->data(), optimizedModelBytes->size(), seedValue, rewrittenNodeCount);
            writeBinaryFile(cachePath, seededModelBytes);
        }
        {
            std::lock_guard<std::mutex> cacheLock(nativeOnnxDeterministicSingTeacherModelCacheMutex);
            nativeOnnxDeterministicSingTeacherModelCache.emplace(cacheKey, cachePath);
            nativeOnnxDeterministicSingTeacherModelKeysInProgress.erase(cacheKey);
        }
        nativeOnnxDeterministicSingTeacherModelCacheCondition.notify_all();
        return cachePath;
    } catch (...) {
        {
            std::lock_guard<std::mutex> cacheLock(nativeOnnxDeterministicSingTeacherModelCacheMutex);
            nativeOnnxDeterministicSingTeacherModelKeysInProgress.erase(cacheKey);
        }
        nativeOnnxDeterministicSingTeacherModelCacheCondition.notify_all();
        throw;
    }
}

static std::vector<NativeOnnxTraceInput> runNativeOnnxDeterministicSingTeacherModelAssetBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount) {
    float seedValue = getNativeOnnxDeterministicSingTeacherSeed();
    fs::path seededModelPath = getNativeOnnxDeterministicSingTeacherModelPath(nativeOnnxApi, runtimeState, modelAsset, cpuThreadCount, seedValue);
    // RandomNormal の乱数状態は Run ごとに進む。固定 seed の再現性を保つためセッションは再利用しない。
    std::vector<NativeOnnxTraceInput> outputTensors = runNativeOnnxModelPath(nativeOnnxApi, runtimeState, seededModelPath, inputTensors, cpuThreadCount, false);
    writeNativeOnnxTensorTrace(modelAsset, inputTensors, outputTensors);
    return outputTensors;
}

const NativeOnnxRuntimeState *SelectModel(const NativeOnnxRuntimeState *State, const ModelAssetRecord &Asset, NativeOnnxRuntimeState &Cpu) {
    // 実測で小さな音長・音高・歌唱教師モデルは GPU 転送の方が高コスト。波形デコーダだけを GPU に置く。
    if (!State || !State->isGpuExecutionProviderSelected || Asset.entryName == "models/d.bin" || Asset.entryName == "models/sd.bin") return State;
    Cpu = *State; // libraryHandle は借用。所有元の runtimeState は変更しない。
    Cpu.selectedExecutionProvider = "CPUExecutionProvider";
    Cpu.isGpuExecutionProviderSelected = false;
    return &Cpu;
}

std::vector<NativeOnnxTraceInput> runNativeOnnxSingTeacherModelAssetBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxRuntimeState Cpu;
    runtimeState = SelectModel(runtimeState, modelAsset, Cpu);
    if (getNativeOnnxSingTeacherMode() == NativeOnnxSingTeacherMode::Deterministic) {
        return runNativeOnnxDeterministicSingTeacherModelAssetBytes(nativeOnnxApi, runtimeState, modelAsset, inputTensors, cpuThreadCount);
    }
    std::vector<NativeOnnxTraceInput> outputTensors;
    if (shouldUseVvBinConfig) {
        outputTensors = RunVvBin(nativeOnnxApi, runtimeState, modelAsset, inputTensors, cpuThreadCount);
    } else {
        try {
            fs::path exportedModelPath = getNativeOnnxExportedModelCachePath(nativeOnnxApi, modelAsset, cpuThreadCount);
            outputTensors = runNativeOnnxModelPath(
                nativeOnnxApi,
                runtimeState,
                exportedModelPath,
                inputTensors,
                cpuThreadCount,
                false,
                createNativeOnnxModelAssetSessionCacheKey(nativeOnnxApi, runtimeState, modelAsset, cpuThreadCount, false));
        } catch (...) {
            if (!canFallbackToNativeOnnxExportLibrary(nativeOnnxApi, runtimeState)) {
                throw;
            }
            outputTensors = runNativeOnnxModelAssetViaExportLibraryVvBin(nativeOnnxApi, modelAsset, nullptr, inputTensors, cpuThreadCount);
        }
    }
    writeNativeOnnxTensorTrace(modelAsset, inputTensors, outputTensors);
    return outputTensors;
}

std::vector<NativeOnnxTraceInput> runNativeOnnxModelAssetBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxRuntimeState Cpu;
    runtimeState = SelectModel(runtimeState, modelAsset, Cpu);
    if (shouldUseVvBinConfig) {
        auto outputTensors = RunVvBin(nativeOnnxApi, runtimeState, modelAsset, inputTensors, cpuThreadCount);
        writeNativeOnnxTensorTrace(modelAsset, inputTensors, outputTensors);
        return outputTensors;
    }
    std::vector<NativeOnnxTraceInput> outputTensors;
    try {
        fs::path exportedModelPath = getNativeOnnxExportedModelCachePath(nativeOnnxApi, modelAsset, cpuThreadCount);
        outputTensors = runNativeOnnxModelPath(
            nativeOnnxApi,
            runtimeState,
            exportedModelPath,
            inputTensors,
            cpuThreadCount,
            false,
            createNativeOnnxModelAssetSessionCacheKey(nativeOnnxApi, runtimeState, modelAsset, cpuThreadCount, false));
    } catch (...) {
        if (!canFallbackToNativeOnnxExportLibrary(nativeOnnxApi, runtimeState)) {
            throw;
        }
        outputTensors = runNativeOnnxModelAssetViaExportLibraryVvBin(nativeOnnxApi, modelAsset, nullptr, inputTensors, cpuThreadCount);
    }
    writeNativeOnnxTensorTrace(modelAsset, inputTensors, outputTensors);
    return outputTensors;
}

std::vector<NativeOnnxTraceInput> runNativeOnnxModelAssetBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, const std::vector<uint8_t> &modelBytes, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxRuntimeState Cpu;
    runtimeState = SelectModel(runtimeState, modelAsset, Cpu);
    std::vector<NativeOnnxTraceInput> outputTensors;
    if (shouldUseVvBinConfig) {
        outputTensors = RunVvBin(nativeOnnxApi, runtimeState, modelAsset, inputTensors, cpuThreadCount, &modelBytes);
    } else {
        try {
            fs::path exportedModelPath = getNativeOnnxExportedModelCachePath(nativeOnnxApi, modelAsset, cpuThreadCount);
            outputTensors = runNativeOnnxModelPath(nativeOnnxApi, runtimeState, exportedModelPath, inputTensors, cpuThreadCount, false, createNativeOnnxModelAssetSessionCacheKey(nativeOnnxApi, runtimeState, modelAsset, cpuThreadCount, false));
        } catch (...) {
            if (!canFallbackToNativeOnnxExportLibrary(nativeOnnxApi, runtimeState)) {
                throw;
            }
            outputTensors = runNativeOnnxModelAssetViaExportLibraryVvBin(nativeOnnxApi, modelAsset, &modelBytes, inputTensors, cpuThreadCount);
        }
    }
    writeNativeOnnxTensorTrace(modelAsset, inputTensors, outputTensors);
    return outputTensors;
}

std::string formatNativeOnnxFloat(float value) {
    if (std::fabs(value - 0.0f) < 0.0000005f) {
        return "0.0";
    }
    char valueBuffer[64];
    auto conversionResult = std::to_chars(valueBuffer, valueBuffer + sizeof(valueBuffer), static_cast<double>(value));
    if (conversionResult.ec == std::errc()) {
        return std::string(valueBuffer, conversionResult.ptr);
    }
    std::ostringstream valueStream;
    valueStream << std::setprecision(17) << static_cast<double>(value);
    return valueStream.str();
}

std::string formatNativeOnnxSettingFloat(float value) {
    if (std::fabs(value - 0.0f) < 0.0000005f) {
        return "0.0";
    }
    if (std::fabs(value - 0.1f) < 0.0000005f) {
        return "0.1";
    }
    if (std::fabs(value - 1.0f) < 0.0000005f) {
        return "1.0";
    }
    return formatNativeOnnxFloat(value);
}


std::vector<uint8_t> synthesizeNativeOnnxVvmAudioQuery(const fs::path &onnxruntimeLibraryPath, const std::vector<VvmArchiveSummary> &archiveSummaries, const fs::path &audioQueryPath, uint32_t styleId, uint16_t cpuThreadCount) {
    ensurePathExists(audioQueryPath, "audio query");
    std::string audioQueryText = readNativeOnnxTextFile(audioQueryPath);
    return synthesizeNativeOnnxModelAssetsAudioQuery(onnxruntimeLibraryPath, collectModelAssets(archiveSummaries), audioQueryText, styleId, cpuThreadCount);
}
