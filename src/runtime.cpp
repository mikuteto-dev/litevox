#include "runtime.hpp"

#include "runtime_internal.hpp"
#include "native_audio_query.hpp"
#include "native_audio_query_validation.hpp"
#include "native_text_query.hpp"
#include "native_user_dict.hpp"
#include "json_utility.hpp"
#include "preset_store.hpp"
#include "setting_store.hpp"
#include "utility.hpp"
#include "vvm_archive.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

static void buildStyleIndex(RuntimeState &runtimeState);
static void unloadLocalVoiceModel(RuntimeState &runtimeState, VoiceModelRecord &modelRecord);
static void syncAllStyleUnloadGenerations(RuntimeState &runtimeState);

RuntimePaths makeDefaultRuntimePaths(const fs::path &rootDirectory) {
    RuntimePaths runtimePaths;
    runtimePaths.rootDirectory = rootDirectory;
    runtimePaths.coreLibraryPath = getFirstNamedPath(rootDirectory, getVoicevoxCoreLibraryFileNames());
    runtimePaths.onnxruntimeLibraryPath = getFirstNamedPath(rootDirectory, getVoicevoxOnnxruntimeLibraryFileNames());
    runtimePaths.dictionaryDirectory = rootDirectory / "open_jtalk_dic_utf_8-1.11";
    runtimePaths.modelPaths = {rootDirectory / "model-vvm"};
    runtimePaths.manifestPath = rootDirectory / "engine_manifest.json";
    runtimePaths.userDictPath = rootDirectory / "user_dict.json";
    runtimePaths.presetPath = rootDirectory / "presets.json";
    runtimePaths.settingPath = rootDirectory / "setting.json";
    runtimePaths.characterResourceDirectory = rootDirectory / "resources" / "character_info";
    runtimePaths.libraryDirectory = rootDirectory / "core_libraries";
    runtimePaths.engineManifestAssetDirectory = rootDirectory / "resources" / "engine_manifest_assets";
    return runtimePaths;
}

static std::string createCombinedMetasJson(const std::vector<VoiceModelRecord> &voiceModels) {
    std::vector<std::string> metasJsonTexts;
    for (const VoiceModelRecord &modelRecord : voiceModels) {
        metasJsonTexts.push_back(modelRecord.metasJson);
    }
    return ::createCombinedMetasJson(metasJsonTexts);
}

static bool isPathInsideDirectory(const fs::path &path, const fs::path &directoryPath) {
    if (directoryPath.empty() || !fs::exists(directoryPath)) {
        return false;
    }
    fs::path canonicalPath = fs::weakly_canonical(path);
    fs::path canonicalDirectoryPath = fs::weakly_canonical(directoryPath);
    auto pathIterator = canonicalPath.begin();
    auto directoryIterator = canonicalDirectoryPath.begin();
    while (directoryIterator != canonicalDirectoryPath.end()) {
        if (pathIterator == canonicalPath.end() || *pathIterator != *directoryIterator) {
            return false;
        }
        pathIterator++;
        directoryIterator++;
    }
    return true;
}

static std::vector<VoiceModelRecord> collectVoiceModels(const std::vector<fs::path> &modelRoots, const fs::path &libraryDirectory) {
    std::vector<VoiceModelRecord> voiceModels;
    for (const fs::path &modelPath : collectVvmModelFiles(modelRoots)) {
        VvmArchiveSummary archiveSummary = inspectVvmArchive(modelPath);
        VoiceModelRecord modelRecord;
        modelRecord.modelPath = modelPath;
        modelRecord.manifestJson = archiveSummary.manifestJson;
        modelRecord.metasJson = archiveSummary.metasJson;
        modelRecord.modelBinNames = archiveSummary.modelBinNames;
        modelRecord.modelAssets = collectModelAssets(std::vector<VvmArchiveSummary>{archiveSummary});
        modelRecord.styles = extractStylesFromMetasJson(modelRecord.metasJson);
        modelRecord.styleIds = extractStyleIds(modelRecord.styles);
        modelRecord.isInstalledLibrary = isPathInsideDirectory(modelPath, libraryDirectory);
        voiceModels.push_back(std::move(modelRecord));
    }
    return voiceModels;
}

static std::vector<ModelAssetRecord> collectVoiceModelAssets(const std::vector<VoiceModelRecord> &voiceModels) {
    std::vector<ModelAssetRecord> modelAssets;
    for (const VoiceModelRecord &modelRecord : voiceModels) {
        modelAssets.insert(modelAssets.end(), modelRecord.modelAssets.begin(), modelRecord.modelAssets.end());
    }
    return modelAssets;
}

std::string getVoiceModelLibraryUuid(const VoiceModelRecord &modelRecord) {
    std::string libraryUuid = extractJsonStringField(modelRecord.manifestJson, "id");
    if (libraryUuid.empty()) {
        return modelRecord.modelPath.filename().string();
    }
    return libraryUuid;
}

static void rebuildRuntimeVoiceModelState(RuntimeState &runtimeState) {
    runtimeState.modelAssets = collectVoiceModelAssets(runtimeState.voiceModels);
    buildStyleIndex(runtimeState);
    runtimeState.combinedMetasJson = createCombinedMetasJson(runtimeState.voiceModels);
    if (runtimeState.hasManifestOverride && fs::exists(runtimeState.manifestPath)) {
        runtimeState.manifestJson = readTextFile(runtimeState.manifestPath);
    } else {
        runtimeState.manifestJson = createRuntimeManifestJson(runtimeState);
    }
}

static bool isLibraryArchivePath(const fs::path &filePath) {
    std::string extensionText = lowercaseAscii(filePath.extension().string());
    return extensionText == ".zip" || extensionText == ".vvm";
}

static bool isLibraryExtractionCachePath(const fs::path &filePath) {
    for (const fs::path &pathPart : filePath) {
        if (pathPart == ".litevox-model-vvm") {
            return true;
        }
        std::string pathText = pathPart.string();
        if (pathText.size() > 12 && pathText.compare(0, 12, ".installing-") == 0) {
            return true;
        }
    }
    return false;
}

static std::string createLibraryStateSignature(const fs::path &libraryDirectory) {
    if (libraryDirectory.empty() || !fs::exists(libraryDirectory)) {
        return "";
    }
    std::vector<std::string> signatureRecords;
    for (const fs::directory_entry &directoryEntry : fs::recursive_directory_iterator(libraryDirectory)) {
        if (!directoryEntry.is_regular_file() || isLibraryExtractionCachePath(directoryEntry.path()) || !isLibraryArchivePath(directoryEntry.path())) {
            continue;
        }
        fs::path relativePath = fs::relative(directoryEntry.path(), libraryDirectory);
        uint64_t fileBytes = static_cast<uint64_t>(directoryEntry.file_size());
        int64_t fileTime = static_cast<int64_t>(directoryEntry.last_write_time().time_since_epoch().count());
        signatureRecords.push_back(relativePath.string() + "\t" + std::to_string(fileBytes) + "\t" + std::to_string(fileTime));
    }
    std::sort(signatureRecords.begin(), signatureRecords.end());
    std::ostringstream signatureStream;
    for (const std::string &signatureRecord : signatureRecords) {
        signatureStream << signatureRecord << "\n";
    }
    return signatureStream.str();
}

static std::vector<fs::path> collectLibraryArchivePaths(const fs::path &libraryDirectory) {
    std::vector<fs::path> libraryArchivePaths;
    if (libraryDirectory.empty() || !fs::exists(libraryDirectory)) {
        return libraryArchivePaths;
    }
    for (const fs::directory_entry &directoryEntry : fs::recursive_directory_iterator(libraryDirectory)) {
        if (directoryEntry.is_regular_file() && !isLibraryExtractionCachePath(directoryEntry.path()) && isLibraryArchivePath(directoryEntry.path())) {
            libraryArchivePaths.push_back(directoryEntry.path());
        }
    }
    std::sort(libraryArchivePaths.begin(), libraryArchivePaths.end());
    return libraryArchivePaths;
}

static void unloadInstalledVoiceModels(RuntimeState &runtimeState) {
    for (VoiceModelRecord &modelRecord : runtimeState.voiceModels) {
        if (modelRecord.isInstalledLibrary) {
            unloadLocalVoiceModel(runtimeState, modelRecord);
        }
    }
}

static void removeInstalledVoiceModelRecords(RuntimeState &runtimeState) {
    std::vector<VoiceModelRecord> remainingVoiceModels;
    remainingVoiceModels.reserve(runtimeState.voiceModels.size());
    for (VoiceModelRecord &modelRecord : runtimeState.voiceModels) {
        if (!modelRecord.isInstalledLibrary) {
            remainingVoiceModels.push_back(std::move(modelRecord));
        }
    }
    runtimeState.voiceModels = std::move(remainingVoiceModels);
}

static void refreshInstalledVoiceLibraries(RuntimeState &runtimeState, const std::string &libraryStateSignature) {
    unloadInstalledVoiceModels(runtimeState);
    removeInstalledVoiceModelRecords(runtimeState);
    std::vector<fs::path> libraryArchivePaths = collectLibraryArchivePaths(runtimeState.libraryDirectory);
    if (!libraryArchivePaths.empty()) {
        std::vector<VoiceModelRecord> installedVoiceModels = collectVoiceModels(libraryArchivePaths, runtimeState.libraryDirectory);
        for (VoiceModelRecord &modelRecord : installedVoiceModels) {
            modelRecord.isInstalledLibrary = true;
            runtimeState.voiceModels.push_back(std::move(modelRecord));
        }
    }
    rebuildRuntimeVoiceModelState(runtimeState);
    runtimeState.libraryStateSignature = libraryStateSignature;
}

void syncInstalledVoiceLibraries(RuntimeState &runtimeState) {
    std::unique_lock<std::mutex> libraryLock;
    if (runtimeState.sharedLibraryMutex) {
        libraryLock = std::unique_lock<std::mutex>(*runtimeState.sharedLibraryMutex);
    }
    std::string libraryStateSignature = createLibraryStateSignature(runtimeState.libraryDirectory);
    if (libraryStateSignature == runtimeState.libraryStateSignature) {
        return;
    }
    refreshInstalledVoiceLibraries(runtimeState, libraryStateSignature);
}

uint64_t countModelAssetBytes(const std::vector<ModelAssetRecord> &modelAssets) {
    uint64_t totalBytes = 0;
    for (const ModelAssetRecord &modelAsset : modelAssets) {
        totalBytes += modelAsset.uncompressedSize;
    }
    return totalBytes;
}

struct RuntimeModelTypeCounts {
    size_t vvBinAssetCount = 0;
    size_t onnxAssetCount = 0;
    size_t unknownAssetCount = 0;
};

bool isNativeRuntimeBackend(const RuntimeState &runtimeState) {
    std::string backendMode = getCoreBackendMode(runtimeState.coreBackend);
    return backendMode == "native" || backendMode == "minimal-ort";
}

bool shouldUseNativeRuntimeVvBinConfig(const RuntimeState &runtimeState) {
    return getCoreBackendNativeModelMode(runtimeState.coreBackend) == "vv_bin";
}

static RuntimeModelTypeCounts countRuntimeModelTypes(const RuntimeState &runtimeState) {
    RuntimeModelTypeCounts modelTypeCounts;
    for (const VoiceModelRecord &voiceModel : runtimeState.voiceModels) {
        VvmArchiveSummary archiveSummary;
        archiveSummary.archivePath = voiceModel.modelPath;
        archiveSummary.manifestJson = voiceModel.manifestJson;
        for (const ManifestModelRecord &manifestModel : collectManifestModels(archiveSummary)) {
            if (manifestModel.modelType == "vv_bin") {
                modelTypeCounts.vvBinAssetCount++;
            } else if (manifestModel.modelType == "onnx") {
                modelTypeCounts.onnxAssetCount++;
            } else {
                modelTypeCounts.unknownAssetCount++;
            }
        }
    }
    return modelTypeCounts;
}

static bool requiresVoicevoxCoreDylib(const RuntimeState &runtimeState) {
    return !isNativeRuntimeBackend(runtimeState);
}

static bool requiresVoicevoxOnnxruntimeDylib(const RuntimeState &runtimeState) {
    return runtimeState.coreBackend.onnxruntime != nullptr || runtimeState.coreBackend.nativeOnnxRuntime.isLoaded;
}

static std::string createRuntimeDependencyStatusJson(RuntimeState &runtimeState) {
    RuntimeModelTypeCounts modelTypeCounts = countRuntimeModelTypes(runtimeState);
    bool requiresCoreDylib = requiresVoicevoxCoreDylib(runtimeState);
    bool requiresOnnxruntimeDylib = requiresVoicevoxOnnxruntimeDylib(runtimeState);
    std::ostringstream jsonStream;
    jsonStream << "{";
    jsonStream << "\"voicevox_core_dylib\":" << (requiresCoreDylib ? "true" : "false") << ",";
    jsonStream << "\"voicevox_onnxruntime_dylib\":" << (requiresOnnxruntimeDylib ? "true" : "false") << ",";
    jsonStream << "\"openjtalk_dictionary\":" << (!runtimeState.dictionaryDirectory.empty() ? "true" : "false") << ",";
    jsonStream << "\"vvm_archives\":" << (!runtimeState.voiceModels.empty() ? "true" : "false") << ",";
    jsonStream << "\"vv_bin_assets\":" << (modelTypeCounts.vvBinAssetCount > 0 ? "true" : "false") << ",";
    jsonStream << "\"onnx_assets\":" << (modelTypeCounts.onnxAssetCount > 0 ? "true" : "false") << ",";
    jsonStream << "\"core_independent\":" << (requiresCoreDylib ? "false" : "true") << ",";
    jsonStream << "\"onnx_independent\":" << (requiresOnnxruntimeDylib ? "false" : "true") << ",";
    jsonStream << "\"vv_bin_asset_count\":" << modelTypeCounts.vvBinAssetCount << ",";
    jsonStream << "\"onnx_asset_count\":" << modelTypeCounts.onnxAssetCount << ",";
    jsonStream << "\"unknown_asset_count\":" << modelTypeCounts.unknownAssetCount;
    jsonStream << "}";
    return jsonStream.str();
}

std::string createRuntimeDependencyInfoJson(RuntimeState &runtimeState) {
    syncAllStyleUnloadGenerations(runtimeState);
    return createRuntimeDependencyStatusJson(runtimeState);
}

std::string createRuntimeDependencyTable(RuntimeState &runtimeState) {
    syncAllStyleUnloadGenerations(runtimeState);
    RuntimeModelTypeCounts modelTypeCounts = countRuntimeModelTypes(runtimeState);
    bool requiresCoreDylib = requiresVoicevoxCoreDylib(runtimeState);
    bool requiresOnnxruntimeDylib = requiresVoicevoxOnnxruntimeDylib(runtimeState);
    std::string onnxruntimeValue = isCoreBackendNativeOnnxLoaded(runtimeState.coreBackend)
        ? getCoreBackendNativeOnnxVersion(runtimeState.coreBackend)
        : (requiresOnnxruntimeDylib ? "voicevox_core_runtime" : "");
    std::ostringstream tableStream;
    tableStream << "dependency\trequired\tvalue\n";
    tableStream << "voicevox_core_dylib\t" << (requiresCoreDylib ? "true" : "false") << "\t" << getCoreBackendMode(runtimeState.coreBackend) << "\n";
    tableStream << "voicevox_onnxruntime_dylib\t" << (requiresOnnxruntimeDylib ? "true" : "false") << "\t" << onnxruntimeValue << "\n";
    tableStream << "openjtalk_dictionary\t" << (!runtimeState.dictionaryDirectory.empty() ? "true" : "false") << "\t" << runtimeState.dictionaryDirectory.string() << "\n";
    tableStream << "vvm_archives\t" << (!runtimeState.voiceModels.empty() ? "true" : "false") << "\t" << runtimeState.voiceModels.size() << "\n";
    tableStream << "vv_bin_assets\t" << (modelTypeCounts.vvBinAssetCount > 0 ? "true" : "false") << "\t" << modelTypeCounts.vvBinAssetCount << "\n";
    tableStream << "onnx_assets\t" << (modelTypeCounts.onnxAssetCount > 0 ? "true" : "false") << "\t" << modelTypeCounts.onnxAssetCount << "\n";
    tableStream << "unknown_assets\t" << (modelTypeCounts.unknownAssetCount > 0 ? "true" : "false") << "\t" << modelTypeCounts.unknownAssetCount << "\n";
    tableStream << "core_independent\t" << (requiresCoreDylib ? "false" : "true") << "\t" << (requiresCoreDylib ? "core_required" : "native_backend") << "\n";
    tableStream << "onnx_independent\t" << (requiresOnnxruntimeDylib ? "false" : "true") << "\t" << (requiresOnnxruntimeDylib ? "onnxruntime_required" : "not_required") << "\n";
    return tableStream.str();
}

const VoiceModelRecord &getRuntimeVoiceModelForStyle(RuntimeState &runtimeState, uint32_t styleId) {
    auto styleIterator = runtimeState.styleToModelIndex.find(styleId);
    if (styleIterator == runtimeState.styleToModelIndex.end()) {
        throw std::runtime_error("未対応の speaker/style ID です: " + std::to_string(styleId));
    }
    return runtimeState.voiceModels.at(styleIterator->second);
}

std::string replaceNativeRuntimeAudioQueryMoraData(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t styleId) {
    const VoiceModelRecord &modelRecord = getRuntimeVoiceModelForStyle(runtimeState, styleId);
    return replaceNativeOnnxModelAssetsAudioQueryMoraData(runtimeState.coreBackend.nativeOnnxRuntime, modelRecord.modelAssets, audioQueryJson, styleId, getCoreBackendCpuNumThreads(runtimeState.coreBackend), shouldUseNativeRuntimeVvBinConfig(runtimeState));
}

std::string replaceNativeRuntimeMoraData(RuntimeState &runtimeState, const std::string &accentPhrasesJson, uint32_t styleId) {
    const VoiceModelRecord &modelRecord = getRuntimeVoiceModelForStyle(runtimeState, styleId);
    return replaceNativeOnnxModelAssetsMoraData(runtimeState.coreBackend.nativeOnnxRuntime, modelRecord.modelAssets, accentPhrasesJson, styleId, getCoreBackendCpuNumThreads(runtimeState.coreBackend), shouldUseNativeRuntimeVvBinConfig(runtimeState));
}

std::string replaceNativeRuntimePhonemeLength(RuntimeState &runtimeState, const std::string &accentPhrasesJson, uint32_t styleId) {
    const VoiceModelRecord &modelRecord = getRuntimeVoiceModelForStyle(runtimeState, styleId);
    return replaceNativeOnnxModelAssetsPhonemeLength(runtimeState.coreBackend.nativeOnnxRuntime, modelRecord.modelAssets, accentPhrasesJson, styleId, getCoreBackendCpuNumThreads(runtimeState.coreBackend), shouldUseNativeRuntimeVvBinConfig(runtimeState));
}

std::string replaceNativeRuntimeMoraPitch(RuntimeState &runtimeState, const std::string &accentPhrasesJson, uint32_t styleId) {
    const VoiceModelRecord &modelRecord = getRuntimeVoiceModelForStyle(runtimeState, styleId);
    return replaceNativeOnnxModelAssetsMoraPitch(runtimeState.coreBackend.nativeOnnxRuntime, modelRecord.modelAssets, accentPhrasesJson, styleId, getCoreBackendCpuNumThreads(runtimeState.coreBackend), shouldUseNativeRuntimeVvBinConfig(runtimeState));
}

void storeMaximumAtomicNumber(std::atomic<uint64_t> &targetNumber, uint64_t candidateNumber) {
    uint64_t storedNumber = targetNumber.load();
    while (candidateNumber > storedNumber && !targetNumber.compare_exchange_weak(storedNumber, candidateNumber)) {}
}

static void buildStyleIndex(RuntimeState &runtimeState) {
    runtimeState.styleToModelIndex.clear();
    for (size_t modelIndex = 0; modelIndex < runtimeState.voiceModels.size(); modelIndex++) {
        for (uint32_t styleId : runtimeState.voiceModels[modelIndex].styleIds) {
            runtimeState.styleToModelIndex[styleId] = modelIndex;
        }
    }
}

static void markSharedStylesLoaded(RuntimeState &runtimeState, const std::vector<uint32_t> &styleIds) {
    if (!runtimeState.sharedLoadedStyleCounts || !runtimeState.sharedLoadedStylesMutex) {
        return;
    }
    std::lock_guard<std::mutex> loadedStylesLock(*runtimeState.sharedLoadedStylesMutex);
    for (uint32_t styleId : styleIds) {
        (*runtimeState.sharedLoadedStyleCounts)[styleId]++;
    }
}

static uint64_t requestSharedStylesUnloaded(RuntimeState &runtimeState, const std::vector<uint32_t> &styleIds) {
    if (!runtimeState.sharedLoadedStyleCounts || !runtimeState.sharedLoadedStylesMutex || !runtimeState.sharedStyleUnloadGenerations || !runtimeState.sharedStyleUnloadMutex) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> loadedStylesLock(*runtimeState.sharedLoadedStylesMutex);
        for (uint32_t styleId : styleIds) {
            runtimeState.sharedLoadedStyleCounts->erase(styleId);
        }
    }
    uint64_t unloadGeneration = 0;
    {
        std::lock_guard<std::mutex> unloadLock(*runtimeState.sharedStyleUnloadMutex);
        for (uint32_t styleId : styleIds) {
            uint64_t nextGeneration = ++(*runtimeState.sharedStyleUnloadGenerations)[styleId];
            unloadGeneration = std::max(unloadGeneration, nextGeneration);
        }
    }
    return unloadGeneration;
}

static bool isSharedStyleLoaded(const RuntimeState &runtimeState, uint32_t styleId) {
    if (!runtimeState.sharedLoadedStyleCounts || !runtimeState.sharedLoadedStylesMutex) {
        return false;
    }
    std::lock_guard<std::mutex> loadedStylesLock(*runtimeState.sharedLoadedStylesMutex);
    return runtimeState.sharedLoadedStyleCounts->find(styleId) != runtimeState.sharedLoadedStyleCounts->end();
}

static uint64_t getSharedModelUnloadGeneration(RuntimeState &runtimeState, const VoiceModelRecord &modelRecord) {
    if (!runtimeState.sharedStyleUnloadGenerations || !runtimeState.sharedStyleUnloadMutex) {
        return 0;
    }
    uint64_t unloadGeneration = 0;
    std::lock_guard<std::mutex> unloadLock(*runtimeState.sharedStyleUnloadMutex);
    for (uint32_t styleId : modelRecord.styleIds) {
        auto generationIterator = runtimeState.sharedStyleUnloadGenerations->find(styleId);
        if (generationIterator != runtimeState.sharedStyleUnloadGenerations->end()) {
            unloadGeneration = std::max(unloadGeneration, generationIterator->second);
        }
    }
    return unloadGeneration;
}

static uint64_t getLocalModelUnloadGeneration(const RuntimeState &runtimeState, const VoiceModelRecord &modelRecord) {
    uint64_t unloadGeneration = 0;
    for (uint32_t styleId : modelRecord.styleIds) {
        auto generationIterator = runtimeState.localStyleUnloadGenerations.find(styleId);
        if (generationIterator != runtimeState.localStyleUnloadGenerations.end()) {
            unloadGeneration = std::max(unloadGeneration, generationIterator->second);
        }
    }
    return unloadGeneration;
}

static void storeLocalModelUnloadGeneration(RuntimeState &runtimeState, const VoiceModelRecord &modelRecord, uint64_t unloadGeneration) {
    for (uint32_t styleId : modelRecord.styleIds) {
        runtimeState.localStyleUnloadGenerations[styleId] = unloadGeneration;
    }
}

static void unloadLocalVoiceModel(RuntimeState &runtimeState, VoiceModelRecord &modelRecord) {
    if (modelRecord.isLoaded && modelRecord.hasModelId) {
        unloadCoreBackendVoiceModel(runtimeState.coreBackend, modelRecord.modelId);
    }
    modelRecord.isLoaded = false;
    modelRecord.hasModelId = false;
    modelRecord.modelId = {};
}

static void syncModelUnloadGeneration(RuntimeState &runtimeState, VoiceModelRecord &modelRecord) {
    uint64_t sharedGeneration = getSharedModelUnloadGeneration(runtimeState, modelRecord);
    if (sharedGeneration == 0 || sharedGeneration <= getLocalModelUnloadGeneration(runtimeState, modelRecord)) {
        return;
    }
    unloadLocalVoiceModel(runtimeState, modelRecord);
    storeLocalModelUnloadGeneration(runtimeState, modelRecord, sharedGeneration);
}

static void syncAllStyleUnloadGenerations(RuntimeState &runtimeState) {
    for (VoiceModelRecord &modelRecord : runtimeState.voiceModels) {
        syncModelUnloadGeneration(runtimeState, modelRecord);
    }
}

static void loadVoiceModel(RuntimeState &runtimeState, size_t modelIndex) {
    VoiceModelRecord &modelRecord = runtimeState.voiceModels.at(modelIndex);
    if (modelRecord.isLoaded) {
        return;
    }
    if (isNativeRuntimeBackend(runtimeState)) {
        modelRecord.isLoaded = true;
        markSharedStylesLoaded(runtimeState, modelRecord.styleIds);
        return;
    }
    bool shouldStoreModelBytes = !getCoreBackendCapabilities(runtimeState.coreBackend).supportsVvmAssetLoader;
    cacheModelAssets(runtimeState.modelSessionCache, modelRecord.modelAssets, shouldStoreModelBytes);
    runtimeState.hasModelSessionCache = true;
    modelRecord.modelId = loadCoreBackendVoiceModel(runtimeState.coreBackend, modelRecord.modelPath);
    modelRecord.hasModelId = true;
    modelRecord.isLoaded = true;
    markSharedStylesLoaded(runtimeState, modelRecord.styleIds);
}

RuntimeState createRuntimeState(const RuntimePaths &runtimePaths, bool shouldPreload) {
    RuntimeState runtimeState{};
    runtimeState.runtimePaths = runtimePaths;
    CoreBackendPaths backendPaths{};
    backendPaths.coreLibraryPath = runtimePaths.coreLibraryPath;
    backendPaths.onnxruntimeLibraryPath = runtimePaths.onnxruntimeLibraryPath;
    backendPaths.dictionaryDirectory = runtimePaths.dictionaryDirectory;
    backendPaths.userDictPath = runtimePaths.userDictPath;
    backendPaths.backendMode = runtimePaths.backendMode;
    backendPaths.coreProfile = runtimePaths.coreProfile;
    backendPaths.accelerationMode = runtimePaths.accelerationMode;
    backendPaths.nativeModelMode = runtimePaths.nativeModelMode;
    backendPaths.cpuNumThreads = runtimePaths.cpuNumThreads;
    runtimeState.coreBackend = createCoreBackendState(backendPaths);
    runtimeState.hasModelSessionCache = false;
    runtimeState.modelSessionCache = ModelSessionCache{};
    runtimeState.dictionaryDirectory = runtimePaths.dictionaryDirectory;
    runtimeState.presetPath = runtimePaths.presetPath;
    runtimeState.settingPath = runtimePaths.settingPath;
    runtimeState.manifestPath = runtimePaths.manifestPath;
    runtimeState.characterResourceDirectory = runtimePaths.characterResourceDirectory;
    runtimeState.libraryDirectory = runtimePaths.libraryDirectory;
    runtimeState.engineManifestAssetDirectory = runtimePaths.engineManifestAssetDirectory;
    runtimeState.enableCancellableSynthesis = runtimePaths.enableCancellableSynthesis;
    runtimeState.hasManifestOverride = runtimePaths.hasManifestOverride;
    runtimeState.characterResources = createCharacterResourceManager(runtimePaths.characterResourceDirectory);

    std::vector<fs::path> modelRoots;
    for (const fs::path &modelPath : runtimePaths.modelPaths) {
        if (!isPathInsideDirectory(modelPath, runtimePaths.libraryDirectory)) {
            modelRoots.push_back(modelPath);
        }
    }
    runtimeState.voiceModels = collectVoiceModels(modelRoots, runtimePaths.libraryDirectory);
    runtimeState.modelAssets = collectVoiceModelAssets(runtimeState.voiceModels);
    buildStyleIndex(runtimeState);
    runtimeState.combinedMetasJson = createCombinedMetasJson(runtimeState.voiceModels);
    if (runtimePaths.hasManifestOverride && fs::exists(runtimePaths.manifestPath)) {
        runtimeState.manifestJson = readTextFile(runtimePaths.manifestPath);
    } else {
        runtimeState.manifestJson = createRuntimeManifestJson(runtimeState);
    }
    syncInstalledVoiceLibraries(runtimeState);
    if (shouldPreload) {
        loadAllVoiceModels(runtimeState);
    }
    return runtimeState;
}

void destroyRuntimeState(RuntimeState &runtimeState) {
    if (runtimeState.segmentedTextPrefetchRuntime) {
        destroyRuntimeState(*runtimeState.segmentedTextPrefetchRuntime);
        runtimeState.segmentedTextPrefetchRuntime.reset();
    }
    clearNativeOnnxCaches();
    destroyCoreBackendState(runtimeState.coreBackend);
}

static std::vector<uint32_t> parseVersionNumbers(const std::string &versionText) {
    std::vector<uint32_t> versionNumbers;
    size_t versionOffset = 0;
    while (versionOffset < versionText.size()) {
        if (!std::isdigit(static_cast<unsigned char>(versionText[versionOffset]))) {
            return {};
        }
        uint32_t parsedNumber = 0;
        while (versionOffset < versionText.size() && std::isdigit(static_cast<unsigned char>(versionText[versionOffset]))) {
            parsedNumber = parsedNumber * 10 + static_cast<uint32_t>(versionText[versionOffset] - '0');
            versionOffset++;
        }
        versionNumbers.push_back(parsedNumber);
        if (versionOffset == versionText.size()) {
            break;
        }
        if (versionText[versionOffset] != '.') {
            return {};
        }
        versionOffset++;
    }
    return versionNumbers;
}

static bool isGreaterVersionText(const std::string &leftVersionText, const std::string &rightVersionText) {
    std::vector<uint32_t> leftVersionNumbers = parseVersionNumbers(leftVersionText);
    std::vector<uint32_t> rightVersionNumbers = parseVersionNumbers(rightVersionText);
    if (leftVersionNumbers.empty() || rightVersionNumbers.empty()) {
        return leftVersionText > rightVersionText;
    }
    size_t versionNumberCount = std::max(leftVersionNumbers.size(), rightVersionNumbers.size());
    for (size_t versionNumberIndex = 0; versionNumberIndex < versionNumberCount; versionNumberIndex++) {
        uint32_t leftVersionNumber = versionNumberIndex < leftVersionNumbers.size() ? leftVersionNumbers[versionNumberIndex] : 0;
        uint32_t rightVersionNumber = versionNumberIndex < rightVersionNumbers.size() ? rightVersionNumbers[versionNumberIndex] : 0;
        if (leftVersionNumber != rightVersionNumber) {
            return leftVersionNumber > rightVersionNumber;
        }
    }
    return false;
}

static std::string getMaximumVoiceModelVersion(const std::vector<VoiceModelRecord> &voiceModels) {
    std::string maximumVersionText;
    for (const VoiceModelRecord &modelRecord : voiceModels) {
        for (const std::string &speakerObject : splitJsonObjects(modelRecord.metasJson)) {
            std::string versionText = extractJsonStringField(speakerObject, "version");
            if (!versionText.empty() && (maximumVersionText.empty() || isGreaterVersionText(versionText, maximumVersionText))) {
                maximumVersionText = versionText;
            }
        }
    }
    return maximumVersionText;
}

std::string getRuntimeCoreVersion(RuntimeState &runtimeState) {
    if (isNativeRuntimeBackend(runtimeState)) {
        std::string modelVersionText = getMaximumVoiceModelVersion(runtimeState.voiceModels);
        if (!modelVersionText.empty()) {
            return modelVersionText;
        }
    }
    return getCoreBackendVersion(runtimeState.coreBackend);
}

void loadAllVoiceModels(RuntimeState &runtimeState) {
    for (size_t modelIndex = 0; modelIndex < runtimeState.voiceModels.size(); modelIndex++) {
        loadVoiceModel(runtimeState, modelIndex);
    }
}

void ensureStyleLoaded(RuntimeState &runtimeState, uint32_t styleId) {
    auto styleIterator = runtimeState.styleToModelIndex.find(styleId);
    if (styleIterator == runtimeState.styleToModelIndex.end()) {
        throw std::runtime_error("未対応の speaker/style ID です: " + std::to_string(styleId));
    }
    syncModelUnloadGeneration(runtimeState, runtimeState.voiceModels[styleIterator->second]);
    loadVoiceModel(runtimeState, styleIterator->second);
}

void unloadStyleModel(RuntimeState &runtimeState, uint32_t styleId) {
    auto styleIterator = runtimeState.styleToModelIndex.find(styleId);
    if (styleIterator == runtimeState.styleToModelIndex.end()) {
        throw std::runtime_error("未対応の speaker/style ID です: " + std::to_string(styleId));
    }
    VoiceModelRecord &modelRecord = runtimeState.voiceModels[styleIterator->second];
    uint64_t unloadGeneration = requestSharedStylesUnloaded(runtimeState, modelRecord.styleIds);
    unloadLocalVoiceModel(runtimeState, modelRecord);
    if (unloadGeneration > 0) {
        storeLocalModelUnloadGeneration(runtimeState, modelRecord, unloadGeneration);
    }
}

bool isStyleLoaded(RuntimeState &runtimeState, uint32_t styleId) {
    auto styleIterator = runtimeState.styleToModelIndex.find(styleId);
    if (styleIterator == runtimeState.styleToModelIndex.end()) {
        return false;
    }
    syncModelUnloadGeneration(runtimeState, runtimeState.voiceModels[styleIterator->second]);
    return runtimeState.voiceModels[styleIterator->second].isLoaded || isSharedStyleLoaded(runtimeState, styleId);
}

std::string createLoadedModelTable(RuntimeState &runtimeState) {
    syncAllStyleUnloadGenerations(runtimeState);
    std::ostringstream tableStream;
    tableStream << "style_id\tloaded\tshared_loaded\tmodel_id\tvvm\tasset_count\tasset_bytes\n";
    for (const VoiceModelRecord &modelRecord : runtimeState.voiceModels) {
        std::string modelName = modelRecord.modelPath.filename().string();
        std::string modelIdText = modelRecord.hasModelId ? formatUuidBytes(modelRecord.modelId) : "";
        uint64_t assetBytes = countModelAssetBytes(modelRecord.modelAssets);
        for (uint32_t styleId : modelRecord.styleIds) {
            bool coreLoaded = modelRecord.isLoaded;
            if (modelRecord.hasModelId) {
                coreLoaded = isCoreBackendVoiceModelLoaded(runtimeState.coreBackend, modelRecord.modelId);
            }
            tableStream << styleId << "\t"
                        << (coreLoaded ? "true" : "false") << "\t"
                        << (isSharedStyleLoaded(runtimeState, styleId) ? "true" : "false") << "\t"
                        << modelIdText << "\t"
                        << modelName << "\t"
                        << modelRecord.modelAssets.size() << "\t"
                        << assetBytes << "\n";
        }
    }
    return tableStream.str();
}

std::string createRuntimeInfoJson(RuntimeState &runtimeState) {
    syncAllStyleUnloadGenerations(runtimeState);
    std::ostringstream jsonStream;
    CoreBackendCapabilities backendCapabilities = getCoreBackendCapabilities(runtimeState.coreBackend);
    jsonStream << "{\"core_version\":\"" << getCoreBackendVersion(runtimeState.coreBackend) << "\",";
    jsonStream << "\"backend\":\"" << getCoreBackendMode(runtimeState.coreBackend) << "\",";
    jsonStream << "\"core_profile\":\"" << getCoreBackendProfile(runtimeState.coreBackend) << "\",";
    jsonStream << "\"acceleration_mode\":\"" << getCoreBackendAccelerationMode(runtimeState.coreBackend) << "\",";
    jsonStream << "\"onnxruntime_path\":" << quoteJsonString(runtimeState.coreBackend.onnxruntimeLibraryPath.string()) << ",";
    jsonStream << "\"native_onnx_loaded\":" << (isCoreBackendNativeOnnxLoaded(runtimeState.coreBackend) ? "true" : "false") << ",";
    jsonStream << "\"native_onnx_version\":\"" << getCoreBackendNativeOnnxVersion(runtimeState.coreBackend) << "\",";
    jsonStream << "\"native_onnx_api_version\":" << getCoreBackendNativeOnnxApiVersion(runtimeState.coreBackend) << ",";
    jsonStream << "\"native_onnx_providers\":" << createNativeOnnxProviderInfoJson(runtimeState.coreBackend.nativeOnnxRuntime) << ",";
    jsonStream << "\"dependency_status\":" << createRuntimeDependencyStatusJson(runtimeState) << ",";
    jsonStream << "\"worker_index\":" << runtimeState.workerIndex << ",";
    jsonStream << "\"worker_count\":" << runtimeState.workerCount << ",";
    std::string streamingMode = backendCapabilities.supportsTrueStreaming ? (isNativeRuntimeBackend(runtimeState) ? "native_decoder_chunk" : "decoder_frame") : "buffered_chunked";
    jsonStream << "\"streaming_mode\":\"" << streamingMode << "\",";
    jsonStream << "\"supports_true_streaming\":" << (backendCapabilities.supportsTrueStreaming ? "true" : "false") << ",";
    jsonStream << "\"supports_cancellation\":" << (backendCapabilities.supportsCancellation ? "true" : "false") << ",";
    jsonStream << "\"cancellation_mode\":\"" << (backendCapabilities.supportsCancellation ? "http_stream_disconnect" : "none") << "\",";
    jsonStream << "\"supports_sing\":" << (backendCapabilities.supportsSing ? "true" : "false") << ",";
    jsonStream << "\"supports_morphing\":" << (backendCapabilities.supportsMorphing ? "true" : "false") << ",";
    jsonStream << "\"morphing_mode\":\"" << (backendCapabilities.supportsNativeMorphing ? "backend" : "pcm_mix_fallback") << "\",";
    jsonStream << "\"supports_frame_synthesis\":" << (backendCapabilities.supportsFrameSynthesis ? "true" : "false") << ",";
    jsonStream << "\"supports_audio_query_validation\":" << (backendCapabilities.supportsAudioQueryValidation ? "true" : "false") << ",";
    jsonStream << "\"supports_frame_audio_query_validation\":" << (backendCapabilities.supportsFrameAudioQueryValidation ? "true" : "false") << ",";
    jsonStream << "\"supports_vvm_asset_loader\":" << (backendCapabilities.supportsVvmAssetLoader ? "true" : "false") << ",";
    jsonStream << "\"gpu_mode\":" << (isCoreBackendGpuMode(runtimeState.coreBackend) ? "true" : "false") << ",";
    jsonStream << "\"cpu_threads\":" << getCoreBackendCpuNumThreads(runtimeState.coreBackend) << ",";
    jsonStream << "\"character_resource_directory\":" << quoteJsonString(runtimeState.characterResourceDirectory.string()) << ",";
    jsonStream << "\"character_resource_count\":" << runtimeState.characterResources.pathHashes.size() << ",";
    jsonStream << "\"user_dict_path\":" << quoteJsonString(runtimeState.coreBackend.userDictPath.string()) << ",";
    jsonStream << "\"preset_path\":" << quoteJsonString(runtimeState.presetPath.string()) << ",";
    jsonStream << "\"setting_path\":" << quoteJsonString(runtimeState.settingPath.string()) << ",";
    jsonStream << "\"library_directory\":" << quoteJsonString(runtimeState.libraryDirectory.string()) << ",";
    jsonStream << "\"asset_count\":" << runtimeState.modelAssets.size() << ",";
    jsonStream << "\"asset_bytes\":" << countModelAssetBytes(runtimeState.modelAssets) << ",";
    jsonStream << "\"session_cache\":{\"loaded\":" << (runtimeState.hasModelSessionCache ? "true" : "false") << ",";
    jsonStream << "\"asset_count\":" << runtimeState.modelSessionCache.cachedModelAssets.size() << ",";
    jsonStream << "\"total_bytes\":" << runtimeState.modelSessionCache.totalBytes << ",";
    jsonStream << "\"stored_bytes\":" << runtimeState.modelSessionCache.storedBytes << ",";
    jsonStream << "\"cache_hits\":" << runtimeState.modelSessionCache.cacheHits << ",";
    jsonStream << "\"cache_misses\":" << runtimeState.modelSessionCache.cacheMisses << ",";
    jsonStream << "\"cache_accesses\":" << runtimeState.modelSessionCache.accessCount << ",";
    jsonStream << "\"partial_read\":" << (runtimeState.modelSessionCache.usedPartialRead ? "true" : "false") << "},";
    jsonStream << "\"native_onnx_session_cache\":" << createNativeOnnxSessionCacheInfoJson() << ",";
    jsonStream << "\"native_onnx_export_cache\":" << createNativeOnnxExportedModelCacheInfoJson() << ",";
    if (isNativeRuntimeBackend(runtimeState)) {
        jsonStream << "\"native_model_mode\":" << quoteJsonString(getCoreBackendNativeModelMode(runtimeState.coreBackend)) << ",";
        jsonStream << "\"native_sing_teacher\":" << createNativeOnnxDeterministicSingTeacherInfoJson(shouldUseNativeRuntimeVvBinConfig(runtimeState)) << ",";
    } else {
        jsonStream << "\"native_model_mode\":\"none\",";
        jsonStream << "\"native_sing_teacher\":{\"enabled\":false,\"mode\":\"none\",\"seed\":0},";
    }
    jsonStream << "\"voice_models\":[";
    for (size_t modelIndex = 0; modelIndex < runtimeState.voiceModels.size(); modelIndex++) {
        const VoiceModelRecord &modelRecord = runtimeState.voiceModels[modelIndex];
        if (modelIndex > 0) {
            jsonStream << ",";
        }
        jsonStream << "{\"vvm\":\"" << modelRecord.modelPath.filename().string() << "\",";
        jsonStream << "\"loaded\":" << (modelRecord.isLoaded ? "true" : "false") << ",";
        jsonStream << "\"asset_count\":" << modelRecord.modelAssets.size() << ",";
        jsonStream << "\"asset_bytes\":" << countModelAssetBytes(modelRecord.modelAssets) << ",";
        jsonStream << "\"model_id\":\"" << (modelRecord.hasModelId ? formatUuidBytes(modelRecord.modelId) : "") << "\"}";
    }
    jsonStream << "]}";
    return jsonStream.str();
}

static bool isSafeLibraryUuid(const std::string &libraryUuid) {
    if (libraryUuid.empty()) {
        return false;
    }
    for (char character : libraryUuid) {
        bool isSafeCharacter = std::isalnum(static_cast<unsigned char>(character)) || character == '-' || character == '_';
        if (!isSafeCharacter) {
            return false;
        }
    }
    return true;
}

static std::vector<VoiceModelRecord> collectInstalledLibraryVoiceModels(const fs::path &libraryZipPath, const fs::path &libraryDirectory) {
    return collectVoiceModels(std::vector<fs::path>{libraryZipPath}, libraryDirectory);
}

static bool hasVoiceModelLibraryUuid(const std::vector<VoiceModelRecord> &voiceModels, const std::string &libraryUuid) {
    for (const VoiceModelRecord &modelRecord : voiceModels) {
        if (getVoiceModelLibraryUuid(modelRecord) == libraryUuid) {
            return true;
        }
    }
    return false;
}

static bool hasBundledVoiceLibrary(RuntimeState &runtimeState, const std::string &libraryUuid) {
    for (const VoiceModelRecord &modelRecord : runtimeState.voiceModels) {
        if (getVoiceModelLibraryUuid(modelRecord) == libraryUuid && !modelRecord.isInstalledLibrary) {
            return true;
        }
    }
    return false;
}

static void eraseInstalledVoiceLibraryRecords(RuntimeState &runtimeState, const std::string &libraryUuid) {
    std::vector<VoiceModelRecord> remainingVoiceModels;
    remainingVoiceModels.reserve(runtimeState.voiceModels.size());
    for (VoiceModelRecord &modelRecord : runtimeState.voiceModels) {
        if (getVoiceModelLibraryUuid(modelRecord) == libraryUuid && modelRecord.isInstalledLibrary) {
            unloadLocalVoiceModel(runtimeState, modelRecord);
            continue;
        }
        remainingVoiceModels.push_back(std::move(modelRecord));
    }
    runtimeState.voiceModels = std::move(remainingVoiceModels);
}

void installVoiceLibrary(RuntimeState &runtimeState, const std::string &libraryUuid, const std::vector<uint8_t> &archiveBytes) {
    std::unique_lock<std::mutex> libraryLock;
    if (runtimeState.sharedLibraryMutex) {
        libraryLock = std::unique_lock<std::mutex>(*runtimeState.sharedLibraryMutex);
    }
    if (!isSafeLibraryUuid(libraryUuid)) {
        throw std::runtime_error("library_uuid が不正です");
    }
    if (archiveBytes.empty()) {
        throw std::runtime_error("library archive が空です");
    }
    if (hasBundledVoiceLibrary(runtimeState, libraryUuid)) {
        throw std::runtime_error("同じ uuid の標準ライブラリは上書きできません: " + libraryUuid);
    }
    fs::create_directories(runtimeState.libraryDirectory);
    fs::path temporaryDirectory = runtimeState.libraryDirectory / (".installing-" + libraryUuid);
    fs::path finalDirectory = runtimeState.libraryDirectory / libraryUuid;
    fs::remove_all(temporaryDirectory);
    fs::create_directories(temporaryDirectory);
    fs::path temporaryZipPath = temporaryDirectory / "library.zip";
    writeBinaryFile(temporaryZipPath, archiveBytes);
    std::vector<VoiceModelRecord> temporaryVoiceModels;
    try {
        temporaryVoiceModels = collectInstalledLibraryVoiceModels(temporaryZipPath, runtimeState.libraryDirectory);
        if (!hasVoiceModelLibraryUuid(temporaryVoiceModels, libraryUuid)) {
            throw std::runtime_error("library_uuid と VVM manifest id が一致しません: " + libraryUuid);
        }
        fs::remove_all(finalDirectory);
        fs::rename(temporaryDirectory, finalDirectory);
        std::vector<VoiceModelRecord> installedVoiceModels = collectInstalledLibraryVoiceModels(finalDirectory / "library.zip", runtimeState.libraryDirectory);
        eraseInstalledVoiceLibraryRecords(runtimeState, libraryUuid);
        for (VoiceModelRecord &modelRecord : installedVoiceModels) {
            modelRecord.isInstalledLibrary = true;
            runtimeState.voiceModels.push_back(std::move(modelRecord));
        }
        rebuildRuntimeVoiceModelState(runtimeState);
        runtimeState.libraryStateSignature = createLibraryStateSignature(runtimeState.libraryDirectory);
    } catch (...) {
        fs::remove_all(temporaryDirectory);
        throw;
    }
}

void uninstallVoiceLibrary(RuntimeState &runtimeState, const std::string &libraryUuid) {
    std::unique_lock<std::mutex> libraryLock;
    if (runtimeState.sharedLibraryMutex) {
        libraryLock = std::unique_lock<std::mutex>(*runtimeState.sharedLibraryMutex);
    }
    if (!isSafeLibraryUuid(libraryUuid)) {
        throw std::runtime_error("library_uuid が不正です");
    }
    if (hasBundledVoiceLibrary(runtimeState, libraryUuid)) {
        throw std::runtime_error("標準ライブラリは削除できません: " + libraryUuid);
    }
    bool hasInstalledLibrary = false;
    for (const VoiceModelRecord &modelRecord : runtimeState.voiceModels) {
        if (getVoiceModelLibraryUuid(modelRecord) == libraryUuid && modelRecord.isInstalledLibrary) {
            hasInstalledLibrary = true;
            break;
        }
    }
    if (!hasInstalledLibrary) {
        throw std::runtime_error("インストール済みライブラリが見つかりません: " + libraryUuid);
    }
    eraseInstalledVoiceLibraryRecords(runtimeState, libraryUuid);
    fs::remove_all(runtimeState.libraryDirectory / libraryUuid);
    rebuildRuntimeVoiceModelState(runtimeState);
    runtimeState.libraryStateSignature = createLibraryStateSignature(runtimeState.libraryDirectory);
}

std::string createRuntimeSpeakerInfoJson(RuntimeState &runtimeState, const std::string &speakerUuid, const std::string &styleGroup, const std::string &resourceFormat, const std::string &resourceBaseUrl) {
    return createCharacterInfoJson(runtimeState.combinedMetasJson, speakerUuid, styleGroup, runtimeState.characterResources, resourceBaseUrl, resourceFormat);
}

std::vector<uint8_t> readRuntimeCharacterResource(RuntimeState &runtimeState, const std::string &resourceHash) {
    fs::path resourcePath = getCharacterResourcePath(runtimeState.characterResources, resourceHash);
    std::string resourceBytes = readTextFile(resourcePath);
    return std::vector<uint8_t>(resourceBytes.begin(), resourceBytes.end());
}

std::string getRuntimeCharacterResourceContentType(RuntimeState &runtimeState, const std::string &resourceHash) {
    return getCharacterResourceContentType(getCharacterResourcePath(runtimeState.characterResources, resourceHash));
}
