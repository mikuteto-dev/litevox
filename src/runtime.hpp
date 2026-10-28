#pragma once

#include "core_backend.hpp"
#include "character_resource.hpp"
#include "model_asset.hpp"
#include "voicevox_types.hpp"
#include "model_metadata.hpp"
#include "model_session_cache.hpp"
#include "streaming_audio.hpp"

#include <atomic>
#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

struct RuntimePaths {
    std::filesystem::path rootDirectory;
    std::filesystem::path coreLibraryPath;
    std::filesystem::path onnxruntimeLibraryPath;
    std::filesystem::path dictionaryDirectory;
    std::vector<std::filesystem::path> modelPaths;
    std::filesystem::path manifestPath;
    std::filesystem::path userDictPath;
    std::filesystem::path presetPath;
    std::filesystem::path settingPath;
    std::filesystem::path characterResourceDirectory;
    std::filesystem::path libraryDirectory;
    std::filesystem::path engineManifestAssetDirectory;
    std::string backendMode = "native";
    std::string coreProfile = "auto";
    std::string accelerationMode = "auto";
    std::string nativeModelMode = "auto";
    bool enableCancellableSynthesis = false;
    bool hasManifestOverride = false;
    uint16_t cpuNumThreads = 0;
};

struct VoiceModelRecord {
    std::filesystem::path modelPath;
    std::string manifestJson;
    std::string metasJson;
    std::vector<std::string> modelBinNames;
    std::vector<ModelAssetRecord> modelAssets;
    std::vector<uint32_t> styleIds;
    std::vector<StyleRecord> styles;
    std::array<uint8_t, 16> modelId{};
    bool hasModelId = false;
    bool isLoaded = false;
    bool isInstalledLibrary = false;
};

struct RuntimeState {
    RuntimePaths runtimePaths;
    CoreBackendState coreBackend;
    size_t workerIndex = 0;
    size_t workerCount = 1;
    std::filesystem::path dictionaryDirectory;
    std::filesystem::path presetPath;
    std::filesystem::path settingPath;
    std::filesystem::path manifestPath;
    std::filesystem::path characterResourceDirectory;
    std::filesystem::path libraryDirectory;
    std::filesystem::path engineManifestAssetDirectory;
    std::vector<VoiceModelRecord> voiceModels;
    std::vector<ModelAssetRecord> modelAssets;
    mutable std::optional<CharacterResourceManager> characterResources;
    ModelSessionCache modelSessionCache;
    std::map<uint32_t, size_t> styleToModelIndex;
    std::map<uint32_t, size_t> *sharedLoadedStyleCounts = nullptr;
    std::mutex *sharedLoadedStylesMutex = nullptr;
    std::map<uint32_t, uint64_t> *sharedStyleUnloadGenerations = nullptr;
    std::mutex *sharedStyleUnloadMutex = nullptr;
    std::map<uint32_t, uint64_t> localStyleUnloadGenerations;
    std::mutex *sharedUserDictMutex = nullptr;
    std::mutex *sharedPresetMutex = nullptr;
    std::mutex *sharedSettingMutex = nullptr;
    std::mutex *sharedLibraryMutex = nullptr;
    std::string combinedMetasJson;
    std::optional<std::string> manifestJson;
    std::string libraryStateSignature;
    std::unique_ptr<RuntimeState> segmentedTextPrefetchRuntime;
    bool enableCancellableSynthesis = false;
    bool hasModelSessionCache = false;
    bool hasManifestOverride = false;
};

struct RuntimeAudioStreamOptions {
    AudioStreamFormat audioStreamFormat = AudioStreamFormat::Wav;
    size_t chunkSamples = 1024;
    size_t fallbackChunkBytes = 32768;
};

struct RuntimeRequestQueueMetrics {
    std::atomic<uint64_t> acceptedConnections{0};
    std::atomic<uint64_t> acceptedRequests{0};
    std::atomic<uint64_t> startedRequests{0};
    std::atomic<uint64_t> completedRequests{0};
    std::atomic<uint64_t> queuedConnections{0};
    std::atomic<uint64_t> maxQueuedConnections{0};
    std::atomic<uint64_t> totalWaitMilliseconds{0};
    std::atomic<uint64_t> maxWaitMilliseconds{0};
    std::atomic<uint64_t> totalServiceMilliseconds{0};
    std::atomic<uint64_t> maxServiceMilliseconds{0};
};

RuntimePaths makeDefaultRuntimePaths(const std::filesystem::path &rootDirectory);
RuntimeState createRuntimeState(const RuntimePaths &runtimePaths, bool shouldPreload);
const CharacterResourceManager &getCharacterResources(const RuntimeState &runtimeState);
const std::string &getRuntimeManifestJson(RuntimeState &runtimeState);
void destroyRuntimeState(RuntimeState &runtimeState);
std::string getRuntimeCoreVersion(RuntimeState &runtimeState);
void loadAllVoiceModels(RuntimeState &runtimeState);
void ensureStyleLoaded(RuntimeState &runtimeState, uint32_t styleId);
void unloadStyleModel(RuntimeState &runtimeState, uint32_t styleId);
bool isStyleLoaded(RuntimeState &runtimeState, uint32_t styleId);
std::string createLoadedModelTable(RuntimeState &runtimeState);
std::string createRuntimeInfoJson(RuntimeState &runtimeState);
std::string createRuntimeDependencyInfoJson(RuntimeState &runtimeState);
std::string createRuntimeDependencyTable(RuntimeState &runtimeState);
void syncInstalledVoiceLibraries(RuntimeState &runtimeState);
std::string createInstalledLibrariesJson(RuntimeState &runtimeState);
std::string createDownloadableLibrariesJson(RuntimeState &runtimeState);
void installVoiceLibrary(RuntimeState &runtimeState, const std::string &libraryUuid, const std::vector<uint8_t> &archiveBytes);
void uninstallVoiceLibrary(RuntimeState &runtimeState, const std::string &libraryUuid);
std::string createRuntimeSpeakerInfoJson(RuntimeState &runtimeState, const std::string &speakerUuid, const std::string &styleGroup, const std::string &resourceFormat, const std::string &resourceBaseUrl);
std::vector<uint8_t> readRuntimeCharacterResource(RuntimeState &runtimeState, const std::string &resourceHash);
std::string getRuntimeCharacterResourceContentType(RuntimeState &runtimeState, const std::string &resourceHash);
std::string createSettingPageHtml(RuntimeState &runtimeState);
void updateSetting(RuntimeState &runtimeState, const std::string &corsPolicyMode, const std::string &allowOrigin);
std::string createCorsAllowedOrigin(RuntimeState &runtimeState, const std::string &requestOrigin);
std::string createRuntimeRequestQueueMetricsJson(const RuntimeRequestQueueMetrics &requestQueueMetrics);
void markRuntimeConnectionAccepted(RuntimeRequestQueueMetrics &requestQueueMetrics);
void markRuntimeRequestAccepted(RuntimeRequestQueueMetrics &requestQueueMetrics);
uint64_t markRuntimeConnectionQueued(RuntimeRequestQueueMetrics &requestQueueMetrics);
void markRuntimeConnectionDequeued(RuntimeRequestQueueMetrics &requestQueueMetrics);
void markRuntimeRequestStarted(RuntimeRequestQueueMetrics &requestQueueMetrics, uint64_t waitMilliseconds);
void markRuntimeRequestCompleted(RuntimeRequestQueueMetrics &requestQueueMetrics, uint64_t serviceMilliseconds);
std::string createRuntimeModelAssetTable(const RuntimeState &runtimeState);
void loadRuntimeModelSessionCache(RuntimeState &runtimeState);
std::string createRuntimeModelSessionCacheSummary(const RuntimeState &runtimeState);
std::string analyzeText(RuntimeState &runtimeState, const std::string &text);
std::string createAudioQuery(RuntimeState &runtimeState, const std::string &text, uint32_t styleId);
std::string createAudioQueryFromKana(RuntimeState &runtimeState, const std::string &kana, uint32_t styleId);
void validateKana(RuntimeState &runtimeState, const std::string &kana);
std::string createAudioQueryFromPreset(RuntimeState &runtimeState, const std::string &text, int32_t presetId);
std::string createAudioQueryFromAccentPhrases(RuntimeState &runtimeState, const std::string &accentPhrasesJson);
void validateAudioQuery(RuntimeState &runtimeState, const std::string &audioQueryJson);
void validateFrameAudioQuery(RuntimeState &runtimeState, const std::string &frameAudioQueryJson);
std::string createAccentPhrases(RuntimeState &runtimeState, const std::string &text, uint32_t styleId);
std::string createAccentPhrasesFromKana(RuntimeState &runtimeState, const std::string &kana, uint32_t styleId);
std::string replaceMoraData(RuntimeState &runtimeState, const std::string &accentPhrasesJson, uint32_t styleId);
std::string replacePhonemeLength(RuntimeState &runtimeState, const std::string &accentPhrasesJson, uint32_t styleId);
std::string replaceMoraPitch(RuntimeState &runtimeState, const std::string &accentPhrasesJson, uint32_t styleId);
std::vector<uint8_t> synthesizeAudioQuery(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t styleId);
std::vector<uint8_t> synthesizeText(RuntimeState &runtimeState, const std::string &text, uint32_t styleId);
std::vector<uint8_t> synthesizeKana(RuntimeState &runtimeState, const std::string &kana, uint32_t styleId);
std::string createSingFrameAudioQuery(RuntimeState &runtimeState, const std::string &scoreJson, uint32_t styleId);
std::string createSingFrameF0(RuntimeState &runtimeState, const std::string &scoreJson, const std::string &frameAudioQueryJson, uint32_t styleId);
std::string createSingFrameVolume(RuntimeState &runtimeState, const std::string &scoreJson, const std::string &frameAudioQueryJson, uint32_t styleId);
std::vector<uint8_t> synthesizeFrameAudioQuery(RuntimeState &runtimeState, const std::string &frameAudioQueryJson, uint32_t styleId);
std::vector<uint8_t> synthesizeMorphingAudioQuery(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t baseStyleId, uint32_t targetStyleId, double morphRate);
void streamAudioQuery(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk);
void streamText(RuntimeState &runtimeState, const std::string &text, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk);
void streamKana(RuntimeState &runtimeState, const std::string &kana, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk);
std::string createSupportedDevicesJson(RuntimeState &runtimeState);
std::string createModelTable(const RuntimeState &runtimeState);
void reloadUserDict(RuntimeState &runtimeState);
std::string createUserDictJson(RuntimeState &runtimeState);
std::string addUserDictWord(RuntimeState &runtimeState, const std::string &surface, const std::string &pronunciation, uintptr_t accentType, VoicevoxUserDictWordType wordType, uint32_t priority);
void updateUserDictWord(RuntimeState &runtimeState, const std::string &wordUuid, const std::string &surface, const std::string &pronunciation, uintptr_t accentType, VoicevoxUserDictWordType wordType, uint32_t priority);
void removeUserDictWord(RuntimeState &runtimeState, const std::string &wordUuid);
void importUserDictJson(RuntimeState &runtimeState, const std::string &userDictJson, bool shouldOverride);
std::string createPresetsJson(const RuntimeState &runtimeState);
int32_t addPreset(RuntimeState &runtimeState, const std::string &presetJson);
int32_t updatePreset(RuntimeState &runtimeState, const std::string &presetJson);
void deletePreset(RuntimeState &runtimeState, int32_t presetId);
