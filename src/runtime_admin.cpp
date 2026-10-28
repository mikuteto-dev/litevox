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

static uint8_t hexToNibble(char character) {
    if (character >= '0' && character <= '9') {
        return static_cast<uint8_t>(character - '0');
    }
    if (character >= 'a' && character <= 'f') {
        return static_cast<uint8_t>(character - 'a' + 10);
    }
    if (character >= 'A' && character <= 'F') {
        return static_cast<uint8_t>(character - 'A' + 10);
    }
    throw std::runtime_error("UUID が不正です");
}

static std::array<uint8_t, 16> parseUuidBytes(const std::string &wordUuid) {
    std::string compactText;
    compactText.reserve(32);
    for (char character : wordUuid) {
        if (character != '-') {
            compactText.push_back(character);
        }
    }
    if (compactText.size() != 32) {
        throw std::runtime_error("UUID が不正です: " + wordUuid);
    }
    std::array<uint8_t, 16> uuidBytes{};
    for (size_t byteIndex = 0; byteIndex < uuidBytes.size(); byteIndex++) {
        uuidBytes[byteIndex] = static_cast<uint8_t>((hexToNibble(compactText[byteIndex * 2]) << 4) | hexToNibble(compactText[byteIndex * 2 + 1]));
    }
    return uuidBytes;
}

std::string formatUuidBytes(const uint8_t uuidBytes[16]) {
    std::ostringstream uuidStream;
    uuidStream << std::hex << std::setfill('0');
    for (size_t byteIndex = 0; byteIndex < 16; byteIndex++) {
        if (byteIndex == 4 || byteIndex == 6 || byteIndex == 8 || byteIndex == 10) {
            uuidStream << "-";
        }
        uuidStream << std::setw(2) << static_cast<unsigned int>(uuidBytes[byteIndex]);
    }
    return uuidStream.str();
}

std::string formatUuidBytes(const std::array<uint8_t, 16> &uuidBytes) {
    return formatUuidBytes(uuidBytes.data());
}

static VoicevoxUserDictWord createCoreUserDictWord(RuntimeState &runtimeState, const std::string &surface, const std::string &pronunciation, uintptr_t accentType, VoicevoxUserDictWordType wordType, uint32_t priority) {
    return makeCoreBackendUserDictWord(runtimeState.coreBackend, surface, pronunciation, accentType, wordType, priority);
}

static void saveUserDict(RuntimeState &runtimeState) {
    saveCoreBackendUserDict(runtimeState.coreBackend);
}

static void applyUserDict(RuntimeState &runtimeState) {
    applyCoreBackendUserDict(runtimeState.coreBackend);
}

static fs::path createTemporaryUserDictPath() {
    auto nowTicks = std::chrono::steady_clock::now().time_since_epoch().count();
    return fs::temp_directory_path() / ("litevox-import-user-dict-" + std::to_string(nowTicks) + ".json");
}


std::string createRuntimeRequestQueueMetricsJson(const RuntimeRequestQueueMetrics &requestQueueMetrics) {
    std::ostringstream jsonStream;
    uint64_t acceptedConnections = requestQueueMetrics.acceptedConnections.load();
    uint64_t acceptedRequests = requestQueueMetrics.acceptedRequests.load();
    uint64_t startedRequests = requestQueueMetrics.startedRequests.load();
    uint64_t completedRequests = requestQueueMetrics.completedRequests.load();
    uint64_t activeRequests = startedRequests >= completedRequests ? startedRequests - completedRequests : 0;
    uint64_t totalWaitMilliseconds = requestQueueMetrics.totalWaitMilliseconds.load();
    uint64_t totalServiceMilliseconds = requestQueueMetrics.totalServiceMilliseconds.load();
    uint64_t averageWaitMilliseconds = startedRequests == 0 ? 0 : totalWaitMilliseconds / startedRequests;
    uint64_t averageServiceMilliseconds = completedRequests == 0 ? 0 : totalServiceMilliseconds / completedRequests;
    jsonStream << "{\"accepted_connections\":" << acceptedConnections << ",";
    jsonStream << "\"accepted_requests\":" << acceptedRequests << ",";
    jsonStream << "\"started_requests\":" << startedRequests << ",";
    jsonStream << "\"completed_requests\":" << completedRequests << ",";
    jsonStream << "\"active_requests\":" << activeRequests << ",";
    jsonStream << "\"queued_connections\":" << requestQueueMetrics.queuedConnections.load() << ",";
    jsonStream << "\"max_queued_connections\":" << requestQueueMetrics.maxQueuedConnections.load() << ",";
    jsonStream << "\"queued_requests\":" << requestQueueMetrics.queuedConnections.load() << ",";
    jsonStream << "\"max_queued_requests\":" << requestQueueMetrics.maxQueuedConnections.load() << ",";
    jsonStream << "\"average_wait_ms\":" << averageWaitMilliseconds << ",";
    jsonStream << "\"max_wait_ms\":" << requestQueueMetrics.maxWaitMilliseconds.load() << ",";
    jsonStream << "\"average_service_ms\":" << averageServiceMilliseconds << ",";
    jsonStream << "\"max_service_ms\":" << requestQueueMetrics.maxServiceMilliseconds.load() << "}";
    return jsonStream.str();
}

std::string createSettingPageHtml(RuntimeState &runtimeState) {
    std::unique_lock<std::mutex> settingLock;
    if (runtimeState.sharedSettingMutex) {
        settingLock = std::unique_lock<std::mutex>(*runtimeState.sharedSettingMutex);
    }
    std::string brandName = extractJsonStringField(getRuntimeManifestJson(runtimeState), "brand_name");
    if (brandName.empty()) {
        brandName = "VOICEVOX";
    }
    std::string templateHtml = readOptionalTextFile(runtimeState.engineManifestAssetDirectory.parent_path() / "setting_ui_template.html", "");
    return createLiteVoxSettingPageHtml(loadLiteVoxSetting(runtimeState.settingPath), templateHtml, brandName);
}

void updateSetting(RuntimeState &runtimeState, const std::string &corsPolicyMode, const std::string &allowOrigin) {
    std::unique_lock<std::mutex> settingLock;
    if (runtimeState.sharedSettingMutex) {
        settingLock = std::unique_lock<std::mutex>(*runtimeState.sharedSettingMutex);
    }
    LiteVoxSetting setting;
    setting.corsPolicyMode = corsPolicyMode;
    setting.allowOrigin = allowOrigin;
    saveLiteVoxSetting(runtimeState.settingPath, setting);
}

std::string createCorsAllowedOrigin(RuntimeState &runtimeState, const std::string &requestOrigin) {
    std::unique_lock<std::mutex> settingLock;
    if (runtimeState.sharedSettingMutex) {
        settingLock = std::unique_lock<std::mutex>(*runtimeState.sharedSettingMutex);
    }
    LiteVoxSetting setting = loadLiteVoxSetting(runtimeState.settingPath);
    if (setting.corsPolicyMode == "all") {
        return "*";
    }
    if (!setting.allowOrigin.empty() && requestOrigin == setting.allowOrigin) {
        return requestOrigin;
    }
    if (!requestOrigin.empty() && isLiteVoxLocalAppOrigin(requestOrigin)) {
        return requestOrigin;
    }
    return "";
}

void markRuntimeConnectionAccepted(RuntimeRequestQueueMetrics &requestQueueMetrics) {
    requestQueueMetrics.acceptedConnections.fetch_add(1);
}

void markRuntimeRequestAccepted(RuntimeRequestQueueMetrics &requestQueueMetrics) {
    requestQueueMetrics.acceptedRequests.fetch_add(1);
}

uint64_t markRuntimeConnectionQueued(RuntimeRequestQueueMetrics &requestQueueMetrics) {
    uint64_t queuedConnectionCount = requestQueueMetrics.queuedConnections.fetch_add(1) + 1;
    storeMaximumAtomicNumber(requestQueueMetrics.maxQueuedConnections, queuedConnectionCount);
    return queuedConnectionCount;
}

void markRuntimeConnectionDequeued(RuntimeRequestQueueMetrics &requestQueueMetrics) {
    requestQueueMetrics.queuedConnections.fetch_sub(1);
}

void markRuntimeRequestStarted(RuntimeRequestQueueMetrics &requestQueueMetrics, uint64_t waitMilliseconds) {
    requestQueueMetrics.startedRequests.fetch_add(1);
    requestQueueMetrics.totalWaitMilliseconds.fetch_add(waitMilliseconds);
    storeMaximumAtomicNumber(requestQueueMetrics.maxWaitMilliseconds, waitMilliseconds);
}

void markRuntimeRequestCompleted(RuntimeRequestQueueMetrics &requestQueueMetrics, uint64_t serviceMilliseconds) {
    requestQueueMetrics.completedRequests.fetch_add(1);
    requestQueueMetrics.totalServiceMilliseconds.fetch_add(serviceMilliseconds);
    storeMaximumAtomicNumber(requestQueueMetrics.maxServiceMilliseconds, serviceMilliseconds);
}

std::string createRuntimeModelAssetTable(const RuntimeState &runtimeState) {
    return createModelAssetTable(runtimeState.modelAssets);
}

void loadRuntimeModelSessionCache(RuntimeState &runtimeState) {
    runtimeState.modelSessionCache = ModelSessionCache{};
    bool shouldStoreModelBytes = !getCoreBackendCapabilities(runtimeState.coreBackend).supportsVvmAssetLoader;
    cacheModelAssets(runtimeState.modelSessionCache, runtimeState.modelAssets, shouldStoreModelBytes);
    runtimeState.hasModelSessionCache = true;
}

std::string createRuntimeModelSessionCacheSummary(const RuntimeState &runtimeState) {
    std::ostringstream summaryStream;
    summaryStream << "cache_loaded\t" << (runtimeState.hasModelSessionCache ? "true" : "false") << "\n";
    summaryStream << createModelSessionCacheSummary(runtimeState.modelSessionCache);
    return summaryStream.str();
}

std::string createSupportedDevicesJson(RuntimeState &runtimeState) {
    return createCoreBackendSupportedDevicesJson(runtimeState.coreBackend);
}

std::string createModelTable(const RuntimeState &runtimeState) {
    std::ostringstream tableStream;
    tableStream << "style_id\ttype\tspeaker_uuid\tspeaker\tstyle\tvvm\n";
    for (const VoiceModelRecord &modelRecord : runtimeState.voiceModels) {
        std::string modelName = modelRecord.modelPath.filename().string();
        for (const StyleRecord &styleRecord : modelRecord.styles) {
            tableStream << styleRecord.styleId << "\t"
                        << styleRecord.styleType << "\t"
                        << styleRecord.speakerUuid << "\t"
                        << styleRecord.speakerName << "\t"
                        << styleRecord.styleName << "\t"
                        << modelName << "\n";
        }
    }
    return tableStream.str();
}

std::string createUserDictJson(RuntimeState &runtimeState) {
    std::unique_lock<std::mutex> userDictLock;
    if (runtimeState.sharedUserDictMutex) {
        userDictLock = std::unique_lock<std::mutex>(*runtimeState.sharedUserDictMutex);
    }
    if (isNativeRuntimeBackend(runtimeState)) {
        return createNativeUserDictJson(runtimeState.coreBackend.userDictPath);
    }
    return createCoreBackendUserDictJson(runtimeState.coreBackend);
}

std::string addUserDictWord(RuntimeState &runtimeState, const std::string &surface, const std::string &pronunciation, uintptr_t accentType, VoicevoxUserDictWordType wordType, uint32_t priority) {
    std::unique_lock<std::mutex> userDictLock;
    if (runtimeState.sharedUserDictMutex) {
        userDictLock = std::unique_lock<std::mutex>(*runtimeState.sharedUserDictMutex);
    }
    runtimeState.segmentedTextPrefetchRuntime.reset();
    if (isNativeRuntimeBackend(runtimeState)) {
        return addNativeUserDictWord(runtimeState.coreBackend.userDictPath, surface, pronunciation, accentType, wordType, priority);
    }
    VoicevoxUserDictWord userDictWord = createCoreUserDictWord(runtimeState, surface, pronunciation, accentType, wordType, priority);
    std::array<uint8_t, 16> uuidBytes = addCoreBackendUserDictWord(runtimeState.coreBackend, userDictWord);
    applyUserDict(runtimeState);
    saveUserDict(runtimeState);
    return formatUuidBytes(uuidBytes.data());
}

void updateUserDictWord(RuntimeState &runtimeState, const std::string &wordUuid, const std::string &surface, const std::string &pronunciation, uintptr_t accentType, VoicevoxUserDictWordType wordType, uint32_t priority) {
    std::unique_lock<std::mutex> userDictLock;
    if (runtimeState.sharedUserDictMutex) {
        userDictLock = std::unique_lock<std::mutex>(*runtimeState.sharedUserDictMutex);
    }
    runtimeState.segmentedTextPrefetchRuntime.reset();
    if (isNativeRuntimeBackend(runtimeState)) {
        updateNativeUserDictWord(runtimeState.coreBackend.userDictPath, wordUuid, surface, pronunciation, accentType, wordType, priority);
        return;
    }
    std::array<uint8_t, 16> parsedUuidBytes = parseUuidBytes(wordUuid);
    VoicevoxUserDictWord userDictWord = createCoreUserDictWord(runtimeState, surface, pronunciation, accentType, wordType, priority);
    updateCoreBackendUserDictWord(runtimeState.coreBackend, parsedUuidBytes, userDictWord);
    applyUserDict(runtimeState);
    saveUserDict(runtimeState);
}

void removeUserDictWord(RuntimeState &runtimeState, const std::string &wordUuid) {
    std::unique_lock<std::mutex> userDictLock;
    if (runtimeState.sharedUserDictMutex) {
        userDictLock = std::unique_lock<std::mutex>(*runtimeState.sharedUserDictMutex);
    }
    runtimeState.segmentedTextPrefetchRuntime.reset();
    if (isNativeRuntimeBackend(runtimeState)) {
        removeNativeUserDictWord(runtimeState.coreBackend.userDictPath, wordUuid);
        return;
    }
    std::array<uint8_t, 16> parsedUuidBytes = parseUuidBytes(wordUuid);
    removeCoreBackendUserDictWord(runtimeState.coreBackend, parsedUuidBytes);
    applyUserDict(runtimeState);
    saveUserDict(runtimeState);
}

void importUserDictJson(RuntimeState &runtimeState, const std::string &userDictJson, bool shouldOverride) {
    std::unique_lock<std::mutex> userDictLock;
    if (runtimeState.sharedUserDictMutex) {
        userDictLock = std::unique_lock<std::mutex>(*runtimeState.sharedUserDictMutex);
    }
    runtimeState.segmentedTextPrefetchRuntime.reset();
    if (isNativeRuntimeBackend(runtimeState)) {
        importNativeUserDictJson(runtimeState.coreBackend.userDictPath, userDictJson, shouldOverride);
        return;
    }
    fs::path temporaryPath = createTemporaryUserDictPath();
    writeTextFile(temporaryPath, userDictJson);
    VoicevoxUserDict *importedUserDict = nullptr;
    bool keepsImportedUserDict = false;
    try {
        importedUserDict = createCoreBackendUserDict(runtimeState.coreBackend);
        if (!shouldOverride) {
            reloadCoreBackendUserDict(runtimeState.coreBackend);
        }
        loadCoreBackendUserDictFile(runtimeState.coreBackend, importedUserDict, temporaryPath);
        if (shouldOverride) {
            deleteCoreBackendUserDict(runtimeState.coreBackend, runtimeState.coreBackend.userDict);
            runtimeState.coreBackend.userDict = importedUserDict;
            keepsImportedUserDict = true;
        } else {
            importCoreBackendUserDict(runtimeState.coreBackend, importedUserDict);
        }
        applyUserDict(runtimeState);
        saveUserDict(runtimeState);
        if (!keepsImportedUserDict) {
            deleteCoreBackendUserDict(runtimeState.coreBackend, importedUserDict);
        }
        fs::remove(temporaryPath);
    } catch (...) {
        if (!keepsImportedUserDict) {
            deleteCoreBackendUserDict(runtimeState.coreBackend, importedUserDict);
        }
        fs::remove(temporaryPath);
        throw;
    }
}

std::string createPresetsJson(const RuntimeState &runtimeState) {
    std::unique_lock<std::mutex> presetLock;
    if (runtimeState.sharedPresetMutex) {
        presetLock = std::unique_lock<std::mutex>(*runtimeState.sharedPresetMutex);
    }
    return loadPresetsJson(runtimeState.presetPath);
}

int32_t addPreset(RuntimeState &runtimeState, const std::string &presetJson) {
    std::unique_lock<std::mutex> presetLock;
    if (runtimeState.sharedPresetMutex) {
        presetLock = std::unique_lock<std::mutex>(*runtimeState.sharedPresetMutex);
    }
    return addPresetJson(runtimeState.presetPath, presetJson);
}

int32_t updatePreset(RuntimeState &runtimeState, const std::string &presetJson) {
    std::unique_lock<std::mutex> presetLock;
    if (runtimeState.sharedPresetMutex) {
        presetLock = std::unique_lock<std::mutex>(*runtimeState.sharedPresetMutex);
    }
    return updatePresetJson(runtimeState.presetPath, presetJson);
}

void deletePreset(RuntimeState &runtimeState, int32_t presetId) {
    std::unique_lock<std::mutex> presetLock;
    if (runtimeState.sharedPresetMutex) {
        presetLock = std::unique_lock<std::mutex>(*runtimeState.sharedPresetMutex);
    }
    deletePresetJson(runtimeState.presetPath, presetId);
}
