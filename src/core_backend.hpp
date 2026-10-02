#pragma once

#include "core_api.hpp"
#include "native_onnx.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

struct CoreBackendPaths {
    std::filesystem::path coreLibraryPath;
    std::filesystem::path onnxruntimeLibraryPath;
    std::filesystem::path dictionaryDirectory;
    std::filesystem::path userDictPath;
    std::string backendMode = "native";
    std::string coreProfile = "auto";
    std::string accelerationMode = "auto";
    std::string nativeModelMode = "auto";
    uint16_t cpuNumThreads = 0;
};

struct CoreBackendState {
    CoreApi coreApi;
    const VoicevoxOnnxruntime *onnxruntime = nullptr;
    VoicevoxOpenJtalkRc *openJtalk = nullptr;
    VoicevoxSynthesizer *synthesizer = nullptr;
    VoicevoxUserDict *userDict = nullptr;
    NativeOnnxRuntimeState nativeOnnxRuntime;
    std::filesystem::path onnxruntimeLibraryPath;
    std::filesystem::path userDictPath;
    std::string backendMode = "native";
    std::string coreProfile = "auto";
    std::string accelerationMode = "auto";
    std::string nativeModelMode = "auto";
    uint16_t cpuNumThreads = 0;
};


struct CoreBackendCapabilities {
    bool supportsCancellation = false;
    bool supportsNativeMorphing = false;
    bool supportsMorphing = true;
    bool supportsSing = false;
    bool supportsFrameSynthesis = false;
    bool supportsAudioQueryValidation = false;
    bool supportsFrameAudioQueryValidation = false;
    bool supportsTrueStreaming = false;
    bool supportsVvmAssetLoader = false;
};

CoreBackendState createCoreBackendState(const CoreBackendPaths &backendPaths);
void destroyCoreBackendState(CoreBackendState &backendState);
std::string getCoreBackendVersion(CoreBackendState &backendState);
bool isCoreBackendGpuMode(CoreBackendState &backendState);
uint16_t getCoreBackendCpuNumThreads(const CoreBackendState &backendState);
std::string getCoreBackendMode(const CoreBackendState &backendState);
std::string getCoreBackendProfile(const CoreBackendState &backendState);
std::string getCoreBackendAccelerationMode(const CoreBackendState &backendState);
std::string getCoreBackendNativeModelMode(const CoreBackendState &backendState);
bool isCoreBackendNativeOnnxLoaded(const CoreBackendState &backendState);
std::string getCoreBackendNativeOnnxVersion(const CoreBackendState &backendState);
uint32_t getCoreBackendNativeOnnxApiVersion(const CoreBackendState &backendState);
CoreBackendCapabilities getCoreBackendCapabilities(const CoreBackendState &backendState);
bool canCoreBackendCancellableSynthesis(const CoreBackendState &backendState);
bool canCoreBackendSynthesisMorphing(const CoreBackendState &backendState);
bool canCoreBackendSing(const CoreBackendState &backendState);
bool canCoreBackendFrameSynthesis(const CoreBackendState &backendState);
bool canCoreBackendValidateAudioQuery(const CoreBackendState &backendState);
bool canCoreBackendValidateFrameAudioQuery(const CoreBackendState &backendState);
std::array<uint8_t, 16> loadCoreBackendVoiceModel(CoreBackendState &backendState, const std::filesystem::path &modelPath);
void unloadCoreBackendVoiceModel(CoreBackendState &backendState, const std::array<uint8_t, 16> &modelId);
bool isCoreBackendVoiceModelLoaded(CoreBackendState &backendState, const std::array<uint8_t, 16> &modelId);
std::string analyzeCoreBackendText(CoreBackendState &backendState, const std::string &text);
std::string createCoreBackendAudioQuery(CoreBackendState &backendState, const std::string &text, uint32_t styleId);
std::string createCoreBackendAudioQueryFromKana(CoreBackendState &backendState, const std::string &kana, uint32_t styleId);
std::string createCoreBackendAudioQueryFromAccentPhrases(CoreBackendState &backendState, const std::string &accentPhrasesJson);
void validateCoreBackendAudioQuery(CoreBackendState &backendState, const std::string &audioQueryJson);
void validateCoreBackendFrameAudioQuery(CoreBackendState &backendState, const std::string &frameAudioQueryJson);
std::string createCoreBackendAccentPhrases(CoreBackendState &backendState, const std::string &text, uint32_t styleId);
std::string createCoreBackendAccentPhrasesFromKana(CoreBackendState &backendState, const std::string &kana, uint32_t styleId);
std::string replaceCoreBackendMoraData(CoreBackendState &backendState, const std::string &accentPhrasesJson, uint32_t styleId);
std::string replaceCoreBackendPhonemeLength(CoreBackendState &backendState, const std::string &accentPhrasesJson, uint32_t styleId);
std::string replaceCoreBackendMoraPitch(CoreBackendState &backendState, const std::string &accentPhrasesJson, uint32_t styleId);
std::vector<uint8_t> synthesizeCoreBackendAudioQuery(CoreBackendState &backendState, const std::string &audioQueryJson, uint32_t styleId);
std::vector<uint8_t> synthesizeCoreBackendText(CoreBackendState &backendState, const std::string &text, uint32_t styleId);
std::vector<uint8_t> synthesizeCoreBackendKana(CoreBackendState &backendState, const std::string &kana, uint32_t styleId);
std::string createCoreBackendSingFrameAudioQuery(CoreBackendState &backendState, const std::string &scoreJson, uint32_t styleId);
std::string createCoreBackendSingFrameF0(CoreBackendState &backendState, const std::string &scoreJson, const std::string &frameAudioQueryJson, uint32_t styleId);
std::string createCoreBackendSingFrameVolume(CoreBackendState &backendState, const std::string &scoreJson, const std::string &frameAudioQueryJson, uint32_t styleId);
std::vector<uint8_t> synthesizeCoreBackendFrameAudioQuery(CoreBackendState &backendState, const std::string &frameAudioQueryJson, uint32_t styleId);
std::string createCoreBackendSupportedDevicesJson(CoreBackendState &backendState);
void reloadCoreBackendUserDict(CoreBackendState &backendState);
void applyCoreBackendUserDict(CoreBackendState &backendState);
void saveCoreBackendUserDict(CoreBackendState &backendState);
VoicevoxUserDictWord makeCoreBackendUserDictWord(CoreBackendState &backendState, const std::string &surface, const std::string &pronunciation, uintptr_t accentType, VoicevoxUserDictWordType wordType, uint32_t priority);
std::string createCoreBackendUserDictJson(CoreBackendState &backendState);
VoicevoxUserDict *createCoreBackendUserDict(CoreBackendState &backendState);
void deleteCoreBackendUserDict(CoreBackendState &backendState, VoicevoxUserDict *userDict);
void loadCoreBackendUserDictFile(CoreBackendState &backendState, VoicevoxUserDict *userDict, const std::filesystem::path &userDictPath);
void importCoreBackendUserDict(CoreBackendState &backendState, VoicevoxUserDict *importedUserDict);
std::array<uint8_t, 16> addCoreBackendUserDictWord(CoreBackendState &backendState, VoicevoxUserDictWord userDictWord);
void updateCoreBackendUserDictWord(CoreBackendState &backendState, const std::array<uint8_t, 16> &wordUuid, VoicevoxUserDictWord userDictWord);
void removeCoreBackendUserDictWord(CoreBackendState &backendState, const std::array<uint8_t, 16> &wordUuid);
