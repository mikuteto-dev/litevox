#include "core_backend.hpp"

#include "utility.hpp"

#include <algorithm>
#include <stdexcept>

namespace fs = std::filesystem;

static constexpr VoicevoxAccelerationMode voicevoxAccelerationModeAuto = 0;
static constexpr VoicevoxAccelerationMode voicevoxAccelerationModeCpu = 1;
static constexpr VoicevoxAccelerationMode voicevoxAccelerationModeGpu = 2;

static std::string normalizeBackendMode(const std::string &backendMode) {
    if (backendMode.empty() || backendMode == "auto") {
        return "native";
    }
    if (backendMode == "voicevox-core" || backendMode == "voicevox_core" || backendMode == "vv") {
        return "voicevox-core";
    }
    if (backendMode == "minimal-ort" || backendMode == "minimal_ort") {
        return "minimal-ort";
    }
    if (backendMode == "native" || backendMode == "vvm-native" || backendMode == "vvm_native") {
        return "native";
    }
    throw std::runtime_error("未対応の backend です: " + backendMode);
}

static std::string normalizeCoreProfile(const std::string &coreProfile) {
    if (coreProfile.empty() || coreProfile == "auto") {
        return "auto";
    }
    if (coreProfile == "talk-only" || coreProfile == "talk_only" || coreProfile == "talk") {
        return "talk-only";
    }
    if (coreProfile == "full") {
        return "full";
    }
    throw std::runtime_error("未対応の core profile です: " + coreProfile);
}

static std::string normalizeAccelerationMode(const std::string &accelerationMode) {
    if (accelerationMode.empty() || accelerationMode == "auto") {
        return "auto";
    }
    if (accelerationMode == "cpu") {
        return "cpu";
    }
    if (accelerationMode == "gpu") {
        return "gpu";
    }
    throw std::runtime_error("未対応の acceleration mode です: " + accelerationMode);
}

static std::string normalizeNativeModelMode(const std::string &nativeModelMode) {
    if (nativeModelMode.empty() || nativeModelMode == "auto") {
        return "auto";
    }
    if (nativeModelMode == "vv-bin" || nativeModelMode == "vv_bin") {
        return "vv_bin";
    }
    if (nativeModelMode == "exported-onnx" || nativeModelMode == "exported_onnx" || nativeModelMode == "onnx") {
        return "exported_onnx";
    }
    throw std::runtime_error("未対応の native model mode です: " + nativeModelMode);
}

static VoicevoxAccelerationMode toVoicevoxAccelerationMode(const std::string &accelerationMode) {
    if (accelerationMode == "auto") {
        return voicevoxAccelerationModeAuto;
    }
    if (accelerationMode == "cpu") {
        return voicevoxAccelerationModeCpu;
    }
    if (accelerationMode == "gpu") {
        return voicevoxAccelerationModeGpu;
    }
    throw std::runtime_error("未対応の acceleration mode です: " + accelerationMode);
}

static bool isNativeBackendMode(const std::string &backendMode) {
    return backendMode == "native" || backendMode == "minimal-ort";
}

static bool isNativeBackendMode(const CoreBackendState &backendState) {
    return isNativeBackendMode(backendState.backendMode);
}

static fs::path getNativeBackendExportOnnxruntimePath(const fs::path &defaultOnnxruntimeLibraryPath) {
    const char *exportOnnxruntimeText = std::getenv("LITEVOX_VV_BIN_ONNXRUNTIME");
    if (!exportOnnxruntimeText || exportOnnxruntimeText[0] == '\0') {
        return defaultOnnxruntimeLibraryPath;
    }
    return fs::path(exportOnnxruntimeText);
}

static std::string resolveNativeModelMode(const CoreBackendPaths &backendPaths, const std::string &normalizedBackendMode, const std::string &normalizedAccelerationMode, const std::string &normalizedNativeModelMode) {
    if (normalizedBackendMode == "minimal-ort") {
        return "exported_onnx";
    }
    if (normalizedNativeModelMode != "auto") {
        return normalizedNativeModelMode;
    }
    if (normalizedAccelerationMode == "gpu") {
        return "exported_onnx";
    }
    fs::path exportOnnxruntimeLibraryPath = getNativeBackendExportOnnxruntimePath(backendPaths.onnxruntimeLibraryPath);
    if (!exportOnnxruntimeLibraryPath.empty() && exportOnnxruntimeLibraryPath != backendPaths.onnxruntimeLibraryPath) {
        return "exported_onnx";
    }
    return "vv_bin";
}

static void ensureImplementedBackendOperation(const CoreBackendState &backendState, const std::string &operationName) {
    if (isNativeBackendMode(backendState)) {
        throw std::runtime_error(operationName + " は native backend scaffold では未実装です");
    }
}

static std::array<uint8_t, 16> copyModelIdBytes(const uint8_t modelIdBytes[16]) {
    std::array<uint8_t, 16> modelId{};
    std::copy(modelIdBytes, modelIdBytes + modelId.size(), modelId.begin());
    return modelId;
}

static void copyModelIdBytes(const std::array<uint8_t, 16> &modelId, uint8_t modelIdBytes[16]) {
    std::copy(modelId.begin(), modelId.end(), modelIdBytes);
}

static std::vector<uint8_t> takeWavPointer(CoreBackendState &backendState, uintptr_t wavLength, uint8_t *wavPointer) {
    std::vector<uint8_t> wavBytes(wavPointer, wavPointer + wavLength);
    backendState.coreApi.wavFree(wavPointer);
    return wavBytes;
}

CoreBackendState createCoreBackendState(const CoreBackendPaths &backendPaths) {
    std::string normalizedBackendMode = normalizeBackendMode(backendPaths.backendMode);
    std::string normalizedCoreProfile = normalizeCoreProfile(backendPaths.coreProfile);
    std::string normalizedAccelerationMode = normalizeAccelerationMode(backendPaths.accelerationMode);
    std::string normalizedNativeModelMode = normalizeNativeModelMode(backendPaths.nativeModelMode);
    if (isNativeBackendMode(normalizedBackendMode)) {
        ensurePathExists(backendPaths.onnxruntimeLibraryPath, "onnxruntime library");
        CoreBackendState backendState{};
        backendState.backendMode = normalizedBackendMode;
        backendState.coreProfile = normalizedCoreProfile;
        backendState.onnxruntimeLibraryPath = backendPaths.onnxruntimeLibraryPath;
        backendState.cpuNumThreads = backendPaths.cpuNumThreads;
        backendState.userDictPath = backendPaths.userDictPath;
        backendState.nativeOnnxRuntime = createNativeOnnxRuntimeState(backendPaths.onnxruntimeLibraryPath, normalizedAccelerationMode);
        backendState.accelerationMode = normalizedAccelerationMode;
        backendState.nativeModelMode = resolveNativeModelMode(backendPaths, normalizedBackendMode, backendState.nativeOnnxRuntime.isGpuExecutionProviderSelected ? "gpu" : normalizedAccelerationMode, normalizedNativeModelMode);
        return backendState;
    }
    ensurePathExists(backendPaths.coreLibraryPath, "core library");
    ensurePathExists(backendPaths.onnxruntimeLibraryPath, "onnxruntime library");
    ensurePathExists(backendPaths.dictionaryDirectory, "OpenJTalk 辞書");

    CoreBackendState backendState{};
    backendState.backendMode = normalizedBackendMode;
    backendState.coreProfile = normalizedCoreProfile;
    backendState.accelerationMode = normalizedAccelerationMode;
    backendState.onnxruntimeLibraryPath = backendPaths.onnxruntimeLibraryPath;
    backendState.coreApi = loadCoreApi(backendPaths.coreLibraryPath);
    backendState.userDictPath = backendPaths.userDictPath;

    VoicevoxLoadOnnxruntimeOptions onnxruntimeOptions = backendState.coreApi.makeDefaultLoadOnnxruntimeOptions();
    std::string onnxruntimePathText = backendPaths.onnxruntimeLibraryPath.string();
    onnxruntimeOptions.filename = onnxruntimePathText.c_str();
    ensureCoreCall(backendState.coreApi, backendState.coreApi.onnxruntimeLoadOnce(onnxruntimeOptions, &backendState.onnxruntime), "ONNX Runtime ロード");

    std::string dictionaryPathText = backendPaths.dictionaryDirectory.string();
    ensureCoreCall(backendState.coreApi, backendState.coreApi.openJtalkRcNew(dictionaryPathText.c_str(), &backendState.openJtalk), "OpenJTalk 初期化");
    reloadCoreBackendUserDict(backendState);

    VoicevoxInitializeOptions initializeOptions = backendState.coreApi.makeDefaultInitializeOptions();
    initializeOptions.accelerationMode = toVoicevoxAccelerationMode(normalizedAccelerationMode);
    if (backendPaths.cpuNumThreads > 0) {
        initializeOptions.cpuNumThreads = backendPaths.cpuNumThreads;
    }
    backendState.cpuNumThreads = initializeOptions.cpuNumThreads;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerNew(backendState.onnxruntime, backendState.openJtalk, initializeOptions, &backendState.synthesizer), "Synthesizer 初期化");
    backendState.backendMode = normalizedBackendMode;
    backendState.coreProfile = normalizedCoreProfile;
    backendState.accelerationMode = normalizedAccelerationMode;
    backendState.nativeModelMode = "none";
    backendState.cpuNumThreads = initializeOptions.cpuNumThreads;
    return backendState;
}

void destroyCoreBackendState(CoreBackendState &backendState) {
    destroyNativeOnnxRuntimeState(backendState.nativeOnnxRuntime);
    backendState.synthesizer = nullptr;
    backendState.userDict = nullptr;
    backendState.openJtalk = nullptr;
    closeCoreApi(backendState.coreApi);
}

std::string getCoreBackendVersion(CoreBackendState &backendState) {
    if (isNativeBackendMode(backendState)) {
        std::string backendPrefix = getCoreBackendMode(backendState) == "minimal-ort" ? "minimal-ort-" : "native-ort-";
        return backendState.nativeOnnxRuntime.version.empty() ? backendPrefix + "unknown" : backendPrefix + backendState.nativeOnnxRuntime.version;
    }
    return backendState.coreApi.getVersion();
}

bool isCoreBackendGpuMode(CoreBackendState &backendState) {
    if (isNativeBackendMode(backendState)) {
        return backendState.nativeOnnxRuntime.isGpuExecutionProviderSelected;
    }
    return backendState.coreApi.synthesizerIsGpuMode(backendState.synthesizer);
}

uint16_t getCoreBackendCpuNumThreads(const CoreBackendState &backendState) {
    return backendState.cpuNumThreads;
}

std::string getCoreBackendMode(const CoreBackendState &backendState) {
    return backendState.backendMode.empty() ? "native" : backendState.backendMode;
}

std::string getCoreBackendProfile(const CoreBackendState &backendState) {
    return backendState.coreProfile.empty() ? "auto" : backendState.coreProfile;
}

std::string getCoreBackendAccelerationMode(const CoreBackendState &backendState) {
    if (backendState.accelerationMode == "auto" || backendState.accelerationMode == "cpu" || backendState.accelerationMode == "gpu") {
        return backendState.accelerationMode;
    }
    return "auto";
}

std::string getCoreBackendNativeModelMode(const CoreBackendState &backendState) {
    if (!isNativeBackendMode(backendState)) {
        return "none";
    }
    if (backendState.nativeModelMode == "vv_bin" || backendState.nativeModelMode == "exported_onnx") {
        return backendState.nativeModelMode;
    }
    return getCoreBackendMode(backendState) == "minimal-ort" ? "exported_onnx" : "vv_bin";
}

bool isCoreBackendNativeOnnxLoaded(const CoreBackendState &backendState) {
    return backendState.nativeOnnxRuntime.isLoaded;
}

std::string getCoreBackendNativeOnnxVersion(const CoreBackendState &backendState) {
    return backendState.nativeOnnxRuntime.version;
}

uint32_t getCoreBackendNativeOnnxApiVersion(const CoreBackendState &backendState) {
    return backendState.nativeOnnxRuntime.apiVersion;
}

CoreBackendCapabilities getCoreBackendCapabilities(const CoreBackendState &backendState) {
    CoreBackendCapabilities backendCapabilities;
    if (isNativeBackendMode(backendState)) {
        backendCapabilities.supportsMorphing = true;
        backendCapabilities.supportsCancellation = true;
        backendCapabilities.supportsSing = true;
        backendCapabilities.supportsFrameSynthesis = true;
        backendCapabilities.supportsAudioQueryValidation = true;
        backendCapabilities.supportsFrameAudioQueryValidation = true;
        backendCapabilities.supportsTrueStreaming = true;
        return backendCapabilities;
    }
    backendCapabilities.supportsCancellation = backendState.coreApi.hasCancellableSynthesis;
    backendCapabilities.supportsNativeMorphing = backendState.coreApi.hasSynthesisMorphing;
    backendCapabilities.supportsMorphing = true;
    backendCapabilities.supportsSing = backendState.coreApi.hasSingFrameAudioQuery && backendState.coreApi.hasSingFrameF0 && backendState.coreApi.hasSingFrameVolume;
    backendCapabilities.supportsFrameSynthesis = backendState.coreApi.hasFrameSynthesis;
    backendCapabilities.supportsAudioQueryValidation = backendState.coreApi.hasAudioQueryValidate;
    backendCapabilities.supportsFrameAudioQueryValidation = backendState.coreApi.hasFrameAudioQueryValidate;
    backendCapabilities.supportsTrueStreaming = false;
    backendCapabilities.supportsVvmAssetLoader = false;
    if (getCoreBackendProfile(backendState) == "talk-only") {
        backendCapabilities.supportsCancellation = false;
        backendCapabilities.supportsNativeMorphing = false;
        backendCapabilities.supportsMorphing = false;
        backendCapabilities.supportsSing = false;
        backendCapabilities.supportsFrameSynthesis = false;
    }
    return backendCapabilities;
}

bool canCoreBackendCancellableSynthesis(const CoreBackendState &backendState) {
    return getCoreBackendCapabilities(backendState).supportsCancellation;
}

bool canCoreBackendSynthesisMorphing(const CoreBackendState &backendState) {
    return getCoreBackendCapabilities(backendState).supportsNativeMorphing;
}

bool canCoreBackendSing(const CoreBackendState &backendState) {
    return getCoreBackendCapabilities(backendState).supportsSing;
}

bool canCoreBackendFrameSynthesis(const CoreBackendState &backendState) {
    return getCoreBackendCapabilities(backendState).supportsFrameSynthesis;
}

bool canCoreBackendValidateAudioQuery(const CoreBackendState &backendState) {
    return getCoreBackendCapabilities(backendState).supportsAudioQueryValidation;
}

bool canCoreBackendValidateFrameAudioQuery(const CoreBackendState &backendState) {
    return getCoreBackendCapabilities(backendState).supportsFrameAudioQueryValidation;
}

std::array<uint8_t, 16> loadCoreBackendVoiceModel(CoreBackendState &backendState, const fs::path &modelPath) {
    ensureImplementedBackendOperation(backendState, "モデルロード");
    VoicevoxVoiceModelFile *modelFile = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.voiceModelFileOpen(modelPath.c_str(), &modelFile), "モデルオープン");
    try {
        uint8_t modelIdBytes[16] = {};
        backendState.coreApi.voiceModelFileId(modelFile, &modelIdBytes);
        if (backendState.coreApi.synthesizerLoadVoiceModelWithOptions) {
            VoicevoxLoadVoiceModelOptions loadOptions = backendState.coreApi.makeDefaultLoadVoiceModelOptions();
            ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerLoadVoiceModelWithOptions(backendState.synthesizer, modelFile, loadOptions), "モデルロード");
        } else {
            ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerLoadVoiceModel(backendState.synthesizer, modelFile), "モデルロード");
        }
        backendState.coreApi.voiceModelFileDelete(modelFile);
        return copyModelIdBytes(modelIdBytes);
    } catch (...) {
        backendState.coreApi.voiceModelFileDelete(modelFile);
        throw;
    }
}

void unloadCoreBackendVoiceModel(CoreBackendState &backendState, const std::array<uint8_t, 16> &modelId) {
    ensureImplementedBackendOperation(backendState, "モデルアンロード");
    uint8_t modelIdBytes[16] = {};
    copyModelIdBytes(modelId, modelIdBytes);
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerUnloadVoiceModel(backendState.synthesizer, &modelIdBytes), "モデルアンロード");
}

bool isCoreBackendVoiceModelLoaded(CoreBackendState &backendState, const std::array<uint8_t, 16> &modelId) {
    ensureImplementedBackendOperation(backendState, "モデルロード状態確認");
    uint8_t modelIdBytes[16] = {};
    copyModelIdBytes(modelId, modelIdBytes);
    return backendState.coreApi.synthesizerIsLoadedVoiceModel(backendState.synthesizer, &modelIdBytes);
}

std::string analyzeCoreBackendText(CoreBackendState &backendState, const std::string &text) {
    ensureImplementedBackendOperation(backendState, "OpenJTalk 解析");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.openJtalkRcAnalyze(backendState.openJtalk, text.c_str(), &jsonPointer), "OpenJTalk 解析");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string createCoreBackendAudioQuery(CoreBackendState &backendState, const std::string &text, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "audio_query");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerCreateAudioQuery(backendState.synthesizer, text.c_str(), styleId, &jsonPointer), "audio_query");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string createCoreBackendAudioQueryFromKana(CoreBackendState &backendState, const std::string &kana, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "audio_query_from_kana");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerCreateAudioQueryFromKana(backendState.synthesizer, kana.c_str(), styleId, &jsonPointer), "audio_query_from_kana");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string createCoreBackendAudioQueryFromAccentPhrases(CoreBackendState &backendState, const std::string &accentPhrasesJson) {
    ensureImplementedBackendOperation(backendState, "accent phrases から audio_query 作成");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.audioQueryCreateFromAccentPhrases(accentPhrasesJson.c_str(), &jsonPointer), "accent phrases から audio_query 作成");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

void validateCoreBackendAudioQuery(CoreBackendState &backendState, const std::string &audioQueryJson) {
    ensureImplementedBackendOperation(backendState, "audio_query 検証");
    if (backendState.coreApi.audioQueryValidate) {
        ensureCoreCall(backendState.coreApi, backendState.coreApi.audioQueryValidate(audioQueryJson.c_str()), "audio_query 検証");
    }
}

void validateCoreBackendFrameAudioQuery(CoreBackendState &backendState, const std::string &frameAudioQueryJson) {
    ensureImplementedBackendOperation(backendState, "frame_audio_query 検証");
    if (backendState.coreApi.frameAudioQueryValidate) {
        ensureCoreCall(backendState.coreApi, backendState.coreApi.frameAudioQueryValidate(frameAudioQueryJson.c_str()), "frame_audio_query 検証");
    }
}

std::string createCoreBackendAccentPhrases(CoreBackendState &backendState, const std::string &text, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "accent_phrases");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerCreateAccentPhrases(backendState.synthesizer, text.c_str(), styleId, &jsonPointer), "accent_phrases");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string createCoreBackendAccentPhrasesFromKana(CoreBackendState &backendState, const std::string &kana, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "accent_phrases_from_kana");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerCreateAccentPhrasesFromKana(backendState.synthesizer, kana.c_str(), styleId, &jsonPointer), "accent_phrases_from_kana");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string replaceCoreBackendMoraData(CoreBackendState &backendState, const std::string &accentPhrasesJson, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "mora_data");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerReplaceMoraData(backendState.synthesizer, accentPhrasesJson.c_str(), styleId, &jsonPointer), "mora_data");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string replaceCoreBackendPhonemeLength(CoreBackendState &backendState, const std::string &accentPhrasesJson, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "mora_length");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerReplacePhonemeLength(backendState.synthesizer, accentPhrasesJson.c_str(), styleId, &jsonPointer), "mora_length");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string replaceCoreBackendMoraPitch(CoreBackendState &backendState, const std::string &accentPhrasesJson, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "mora_pitch");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerReplaceMoraPitch(backendState.synthesizer, accentPhrasesJson.c_str(), styleId, &jsonPointer), "mora_pitch");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::vector<uint8_t> synthesizeCoreBackendAudioQuery(CoreBackendState &backendState, const std::string &audioQueryJson, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "synthesis");
    uintptr_t wavLength = 0;
    uint8_t *wavPointer = nullptr;
    VoicevoxSynthesisOptions synthesisOptions = backendState.coreApi.makeDefaultSynthesisOptions();
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerSynthesis(backendState.synthesizer, audioQueryJson.c_str(), styleId, synthesisOptions, &wavLength, &wavPointer), "synthesis");
    return takeWavPointer(backendState, wavLength, wavPointer);
}

std::vector<uint8_t> synthesizeCoreBackendText(CoreBackendState &backendState, const std::string &text, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "tts");
    uintptr_t wavLength = 0;
    uint8_t *wavPointer = nullptr;
    VoicevoxTtsOptions ttsOptions = backendState.coreApi.makeDefaultTtsOptions();
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerTts(backendState.synthesizer, text.c_str(), styleId, ttsOptions, &wavLength, &wavPointer), "tts");
    return takeWavPointer(backendState, wavLength, wavPointer);
}

std::vector<uint8_t> synthesizeCoreBackendKana(CoreBackendState &backendState, const std::string &kana, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "tts_from_kana");
    uintptr_t wavLength = 0;
    uint8_t *wavPointer = nullptr;
    VoicevoxTtsOptions ttsOptions = backendState.coreApi.makeDefaultTtsOptions();
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerTtsFromKana(backendState.synthesizer, kana.c_str(), styleId, ttsOptions, &wavLength, &wavPointer), "tts_from_kana");
    return takeWavPointer(backendState, wavLength, wavPointer);
}

std::string createCoreBackendSingFrameAudioQuery(CoreBackendState &backendState, const std::string &scoreJson, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "sing_frame_audio_query");
    if (!backendState.coreApi.synthesizerCreateSingFrameAudioQuery) {
        throw std::runtime_error("sing_frame_audio_query は現在の backend では未対応です");
    }
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerCreateSingFrameAudioQuery(backendState.synthesizer, scoreJson.c_str(), styleId, &jsonPointer), "sing_frame_audio_query");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string createCoreBackendSingFrameF0(CoreBackendState &backendState, const std::string &scoreJson, const std::string &frameAudioQueryJson, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "sing_frame_f0");
    if (!backendState.coreApi.synthesizerCreateSingFrameF0) {
        throw std::runtime_error("sing_frame_f0 は現在の backend では未対応です");
    }
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerCreateSingFrameF0(backendState.synthesizer, scoreJson.c_str(), frameAudioQueryJson.c_str(), styleId, &jsonPointer), "sing_frame_f0");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::string createCoreBackendSingFrameVolume(CoreBackendState &backendState, const std::string &scoreJson, const std::string &frameAudioQueryJson, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "sing_frame_volume");
    if (!backendState.coreApi.synthesizerCreateSingFrameVolume) {
        throw std::runtime_error("sing_frame_volume は現在の backend では未対応です");
    }
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerCreateSingFrameVolume(backendState.synthesizer, scoreJson.c_str(), frameAudioQueryJson.c_str(), styleId, &jsonPointer), "sing_frame_volume");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

std::vector<uint8_t> synthesizeCoreBackendFrameAudioQuery(CoreBackendState &backendState, const std::string &frameAudioQueryJson, uint32_t styleId) {
    ensureImplementedBackendOperation(backendState, "frame_synthesis");
    if (!backendState.coreApi.synthesizerFrameSynthesis) {
        throw std::runtime_error("frame_synthesis は現在の backend では未対応です");
    }
    uintptr_t wavLength = 0;
    uint8_t *wavPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.synthesizerFrameSynthesis(backendState.synthesizer, frameAudioQueryJson.c_str(), styleId, &wavLength, &wavPointer), "frame_synthesis");
    return takeWavPointer(backendState, wavLength, wavPointer);
}

std::string createCoreBackendSupportedDevicesJson(CoreBackendState &backendState) {
    if (isNativeBackendMode(backendState)) {
        return createNativeOnnxSupportedDevicesJson(backendState.nativeOnnxRuntime);
    }
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.onnxruntimeCreateSupportedDevicesJson(backendState.onnxruntime, &jsonPointer), "supported_devices");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

void applyCoreBackendUserDict(CoreBackendState &backendState) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書適用");
    ensureCoreCall(backendState.coreApi, backendState.coreApi.openJtalkRcUseUserDict(backendState.openJtalk, backendState.userDict), "ユーザー辞書適用");
}

void saveCoreBackendUserDict(CoreBackendState &backendState) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書保存");
    if (backendState.userDictPath.empty()) {
        return;
    }
    fs::path parentPath = backendState.userDictPath.parent_path();
    if (!parentPath.empty()) {
        fs::create_directories(parentPath);
    }
    ensureCoreCall(backendState.coreApi, backendState.coreApi.userDictSave(backendState.userDict, backendState.userDictPath.c_str()), "ユーザー辞書保存");
}

void reloadCoreBackendUserDict(CoreBackendState &backendState) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書リロード");
    VoicevoxUserDict *loadedUserDict = backendState.coreApi.userDictNew();
    if (!loadedUserDict) {
        throw std::runtime_error("ユーザー辞書初期化に失敗しました");
    }
    try {
        if (!backendState.userDictPath.empty() && fs::exists(backendState.userDictPath)) {
            ensureCoreCall(backendState.coreApi, backendState.coreApi.userDictLoad(loadedUserDict, backendState.userDictPath.c_str()), "ユーザー辞書ロード");
        }
        VoicevoxUserDict *previousUserDict = backendState.userDict;
        backendState.userDict = loadedUserDict;
        applyCoreBackendUserDict(backendState);
        if (previousUserDict) {
            backendState.coreApi.userDictDelete(previousUserDict);
        }
    } catch (...) {
        backendState.coreApi.userDictDelete(loadedUserDict);
        throw;
    }
}

VoicevoxUserDictWord makeCoreBackendUserDictWord(CoreBackendState &backendState, const std::string &surface, const std::string &pronunciation, uintptr_t accentType, VoicevoxUserDictWordType wordType, uint32_t priority) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書単語作成");
    VoicevoxUserDictWord userDictWord = backendState.coreApi.userDictWordMake(surface.c_str(), pronunciation.c_str(), accentType);
    userDictWord.wordType = wordType;
    userDictWord.priority = priority;
    return userDictWord;
}

std::string createCoreBackendUserDictJson(CoreBackendState &backendState) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書 JSON 作成");
    char *jsonPointer = nullptr;
    ensureCoreCall(backendState.coreApi, backendState.coreApi.userDictToJson(backendState.userDict, &jsonPointer), "ユーザー辞書 JSON 作成");
    return takeJsonPointer(backendState.coreApi, jsonPointer);
}

VoicevoxUserDict *createCoreBackendUserDict(CoreBackendState &backendState) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書初期化");
    VoicevoxUserDict *userDict = backendState.coreApi.userDictNew();
    if (!userDict) {
        throw std::runtime_error("ユーザー辞書初期化に失敗しました");
    }
    return userDict;
}

void deleteCoreBackendUserDict(CoreBackendState &backendState, VoicevoxUserDict *userDict) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書削除");
    if (userDict) {
        backendState.coreApi.userDictDelete(userDict);
    }
}

void loadCoreBackendUserDictFile(CoreBackendState &backendState, VoicevoxUserDict *userDict, const fs::path &userDictPath) {
    ensureImplementedBackendOperation(backendState, "インポート用ユーザー辞書ロード");
    ensureCoreCall(backendState.coreApi, backendState.coreApi.userDictLoad(userDict, userDictPath.c_str()), "インポート用ユーザー辞書ロード");
}

void importCoreBackendUserDict(CoreBackendState &backendState, VoicevoxUserDict *importedUserDict) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書インポート");
    ensureCoreCall(backendState.coreApi, backendState.coreApi.userDictImport(backendState.userDict, importedUserDict), "ユーザー辞書インポート");
}

std::array<uint8_t, 16> addCoreBackendUserDictWord(CoreBackendState &backendState, VoicevoxUserDictWord userDictWord) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書単語追加");
    uint8_t uuidBytes[16] = {};
    ensureCoreCall(backendState.coreApi, backendState.coreApi.userDictAddWord(backendState.userDict, userDictWord, &uuidBytes), "ユーザー辞書単語追加");
    return copyModelIdBytes(uuidBytes);
}

void updateCoreBackendUserDictWord(CoreBackendState &backendState, const std::array<uint8_t, 16> &wordUuid, VoicevoxUserDictWord userDictWord) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書単語更新");
    uint8_t uuidBytes[16] = {};
    copyModelIdBytes(wordUuid, uuidBytes);
    ensureCoreCall(backendState.coreApi, backendState.coreApi.userDictUpdateWord(backendState.userDict, &uuidBytes, userDictWord), "ユーザー辞書単語更新");
}

void removeCoreBackendUserDictWord(CoreBackendState &backendState, const std::array<uint8_t, 16> &wordUuid) {
    ensureImplementedBackendOperation(backendState, "ユーザー辞書単語削除");
    uint8_t uuidBytes[16] = {};
    copyModelIdBytes(wordUuid, uuidBytes);
    ensureCoreCall(backendState.coreApi, backendState.coreApi.userDictRemoveWord(backendState.userDict, &uuidBytes), "ユーザー辞書単語削除");
}
