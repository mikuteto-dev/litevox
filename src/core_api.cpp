#include "core_api.hpp"

#include "dynamic_library.hpp"

#include <stdexcept>

template <typename FunctionType>
static FunctionType loadSymbol(void *libraryHandle, const char *symbolName) {
    void *symbolPointer = loadDynamicLibrarySymbol(libraryHandle, symbolName);
    if (!symbolPointer) {
        throw std::runtime_error(std::string("シンボルを読めません: ") + symbolName);
    }
    return reinterpret_cast<FunctionType>(symbolPointer);
}

template <typename FunctionType>
static FunctionType loadOptionalSymbol(void *libraryHandle, const char *symbolName) {
    void *symbolPointer = loadDynamicLibrarySymbol(libraryHandle, symbolName);
    if (!symbolPointer) {
        return nullptr;
    }
    return reinterpret_cast<FunctionType>(symbolPointer);
}

static bool hasSymbol(void *libraryHandle, const char *symbolName) {
    return loadDynamicLibrarySymbol(libraryHandle, symbolName) != nullptr;
}

CoreApi loadCoreApi(const std::filesystem::path &coreLibraryPath) {
    CoreApi coreApi;
    coreApi.libraryHandle = openDynamicLibrary(coreLibraryPath);
    if (!coreApi.libraryHandle) {
        throw std::runtime_error(std::string("core を開けません: ") + getDynamicLibraryErrorText());
    }
    coreApi.hasCancellableSynthesis = hasSymbol(coreApi.libraryHandle, "voicevox_synthesizer_cancellable_synthesis");
    coreApi.hasSynthesisMorphing = hasSymbol(coreApi.libraryHandle, "voicevox_synthesizer_synthesis_morphing");
    coreApi.getVersion = loadSymbol<const char *(*)()>(coreApi.libraryHandle, "voicevox_get_version");
    coreApi.makeDefaultLoadOnnxruntimeOptions = loadSymbol<VoicevoxLoadOnnxruntimeOptions (*)()>(coreApi.libraryHandle, "voicevox_make_default_load_onnxruntime_options");
    coreApi.makeDefaultInitializeOptions = loadSymbol<VoicevoxInitializeOptions (*)()>(coreApi.libraryHandle, "voicevox_make_default_initialize_options");
    coreApi.makeDefaultTtsOptions = loadSymbol<VoicevoxTtsOptions (*)()>(coreApi.libraryHandle, "voicevox_make_default_tts_options");
    coreApi.makeDefaultSynthesisOptions = loadSymbol<VoicevoxSynthesisOptions (*)()>(coreApi.libraryHandle, "voicevox_make_default_synthesis_options");
    coreApi.makeDefaultLoadVoiceModelOptions = loadOptionalSymbol<VoicevoxLoadVoiceModelOptions (*)()>(coreApi.libraryHandle, "voicevox_make_default_load_voice_model_options");
    coreApi.onnxruntimeLoadOnce = loadSymbol<VoicevoxResultCode (*)(VoicevoxLoadOnnxruntimeOptions, const VoicevoxOnnxruntime **)>(coreApi.libraryHandle, "voicevox_onnxruntime_load_once");
    coreApi.onnxruntimeCreateSupportedDevicesJson = loadSymbol<VoicevoxResultCode (*)(const VoicevoxOnnxruntime *, char **)>(coreApi.libraryHandle, "voicevox_onnxruntime_create_supported_devices_json");
    coreApi.openJtalkRcNew = loadSymbol<VoicevoxResultCode (*)(const char *, VoicevoxOpenJtalkRc **)>(coreApi.libraryHandle, "voicevox_open_jtalk_rc_new");
    coreApi.openJtalkRcUseUserDict = loadSymbol<VoicevoxResultCode (*)(const VoicevoxOpenJtalkRc *, const VoicevoxUserDict *)>(coreApi.libraryHandle, "voicevox_open_jtalk_rc_use_user_dict");
    coreApi.openJtalkRcAnalyze = loadSymbol<VoicevoxResultCode (*)(const VoicevoxOpenJtalkRc *, const char *, char **)>(coreApi.libraryHandle, "voicevox_open_jtalk_rc_analyze");
    coreApi.openJtalkRcDelete = loadSymbol<void (*)(VoicevoxOpenJtalkRc *)>(coreApi.libraryHandle, "voicevox_open_jtalk_rc_delete");
    coreApi.synthesizerNew = loadSymbol<VoicevoxResultCode (*)(const VoicevoxOnnxruntime *, const VoicevoxOpenJtalkRc *, VoicevoxInitializeOptions, VoicevoxSynthesizer **)>(coreApi.libraryHandle, "voicevox_synthesizer_new");
    coreApi.synthesizerDelete = loadSymbol<void (*)(VoicevoxSynthesizer *)>(coreApi.libraryHandle, "voicevox_synthesizer_delete");
    if (coreApi.makeDefaultLoadVoiceModelOptions) {
        coreApi.synthesizerLoadVoiceModelWithOptions = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const VoicevoxVoiceModelFile *, VoicevoxLoadVoiceModelOptions)>(coreApi.libraryHandle, "voicevox_synthesizer_load_voice_model");
    } else {
        coreApi.synthesizerLoadVoiceModel = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const VoicevoxVoiceModelFile *)>(coreApi.libraryHandle, "voicevox_synthesizer_load_voice_model");
    }
    coreApi.synthesizerUnloadVoiceModel = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const uint8_t (*)[16])>(coreApi.libraryHandle, "voicevox_synthesizer_unload_voice_model");
    coreApi.synthesizerIsGpuMode = loadSymbol<bool (*)(const VoicevoxSynthesizer *)>(coreApi.libraryHandle, "voicevox_synthesizer_is_gpu_mode");
    coreApi.synthesizerIsLoadedVoiceModel = loadSymbol<bool (*)(const VoicevoxSynthesizer *, const uint8_t (*)[16])>(coreApi.libraryHandle, "voicevox_synthesizer_is_loaded_voice_model");
    coreApi.synthesizerCreateAudioQuery = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_create_audio_query");
    coreApi.synthesizerCreateAudioQueryFromKana = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_create_audio_query_from_kana");
    coreApi.synthesizerCreateAccentPhrases = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_create_accent_phrases");
    coreApi.synthesizerCreateAccentPhrasesFromKana = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_create_accent_phrases_from_kana");
    coreApi.synthesizerReplaceMoraData = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_replace_mora_data");
    coreApi.synthesizerReplacePhonemeLength = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_replace_phoneme_length");
    coreApi.synthesizerReplaceMoraPitch = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_replace_mora_pitch");
    coreApi.synthesizerSynthesis = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, VoicevoxSynthesisOptions, uintptr_t *, uint8_t **)>(coreApi.libraryHandle, "voicevox_synthesizer_synthesis");
    coreApi.synthesizerTts = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, VoicevoxTtsOptions, uintptr_t *, uint8_t **)>(coreApi.libraryHandle, "voicevox_synthesizer_tts");
    coreApi.synthesizerTtsFromKana = loadSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, VoicevoxTtsOptions, uintptr_t *, uint8_t **)>(coreApi.libraryHandle, "voicevox_synthesizer_tts_from_kana");
    coreApi.synthesizerCreateSingFrameAudioQuery = loadOptionalSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_create_sing_frame_audio_query");
    coreApi.synthesizerCreateSingFrameF0 = loadOptionalSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_create_sing_frame_f0");
    if (!coreApi.synthesizerCreateSingFrameF0) {
        coreApi.synthesizerCreateSingFrameF0 = loadOptionalSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_predict_sing_frame_f0");
    }
    coreApi.synthesizerCreateSingFrameVolume = loadOptionalSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_create_sing_frame_volume");
    if (!coreApi.synthesizerCreateSingFrameVolume) {
        coreApi.synthesizerCreateSingFrameVolume = loadOptionalSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, const char *, uint32_t, char **)>(coreApi.libraryHandle, "voicevox_synthesizer_predict_sing_frame_volume");
    }
    coreApi.synthesizerFrameSynthesis = loadOptionalSymbol<VoicevoxResultCode (*)(const VoicevoxSynthesizer *, const char *, uint32_t, uintptr_t *, uint8_t **)>(coreApi.libraryHandle, "voicevox_synthesizer_frame_synthesis");
    coreApi.hasSingFrameAudioQuery = coreApi.synthesizerCreateSingFrameAudioQuery != nullptr;
    coreApi.hasSingFrameF0 = coreApi.synthesizerCreateSingFrameF0 != nullptr;
    coreApi.hasSingFrameVolume = coreApi.synthesizerCreateSingFrameVolume != nullptr;
    coreApi.hasFrameSynthesis = coreApi.synthesizerFrameSynthesis != nullptr;
    coreApi.audioQueryCreateFromAccentPhrases = loadSymbol<VoicevoxResultCode (*)(const char *, char **)>(coreApi.libraryHandle, "voicevox_audio_query_create_from_accent_phrases");
    coreApi.audioQueryValidate = loadOptionalSymbol<VoicevoxResultCode (*)(const char *)>(coreApi.libraryHandle, "voicevox_audio_query_validate");
    coreApi.frameAudioQueryValidate = loadOptionalSymbol<VoicevoxResultCode (*)(const char *)>(coreApi.libraryHandle, "voicevox_frame_audio_query_validate");
    coreApi.hasAudioQueryValidate = coreApi.audioQueryValidate != nullptr;
    coreApi.hasFrameAudioQueryValidate = coreApi.frameAudioQueryValidate != nullptr;
    coreApi.voiceModelFileOpen = loadSymbol<VoicevoxResultCode (*)(const char *, VoicevoxVoiceModelFile **)>(coreApi.libraryHandle, "voicevox_voice_model_file_open");
    coreApi.voiceModelFileId = loadSymbol<void (*)(const VoicevoxVoiceModelFile *, uint8_t (*)[16])>(coreApi.libraryHandle, "voicevox_voice_model_file_id");
    coreApi.voiceModelFileDelete = loadSymbol<void (*)(VoicevoxVoiceModelFile *)>(coreApi.libraryHandle, "voicevox_voice_model_file_delete");
    coreApi.userDictWordMake = loadSymbol<VoicevoxUserDictWord (*)(const char *, const char *, uintptr_t)>(coreApi.libraryHandle, "voicevox_user_dict_word_make");
    coreApi.userDictNew = loadSymbol<VoicevoxUserDict *(*)()>(coreApi.libraryHandle, "voicevox_user_dict_new");
    coreApi.userDictDelete = loadSymbol<void (*)(VoicevoxUserDict *)>(coreApi.libraryHandle, "voicevox_user_dict_delete");
    coreApi.userDictLoad = loadSymbol<VoicevoxResultCode (*)(const VoicevoxUserDict *, const char *)>(coreApi.libraryHandle, "voicevox_user_dict_load");
    coreApi.userDictSave = loadSymbol<VoicevoxResultCode (*)(const VoicevoxUserDict *, const char *)>(coreApi.libraryHandle, "voicevox_user_dict_save");
    coreApi.userDictToJson = loadSymbol<VoicevoxResultCode (*)(const VoicevoxUserDict *, char **)>(coreApi.libraryHandle, "voicevox_user_dict_to_json");
    coreApi.userDictAddWord = loadSymbol<VoicevoxResultCode (*)(const VoicevoxUserDict *, VoicevoxUserDictWord, uint8_t (*)[16])>(coreApi.libraryHandle, "voicevox_user_dict_add_word");
    coreApi.userDictUpdateWord = loadSymbol<VoicevoxResultCode (*)(const VoicevoxUserDict *, const uint8_t (*)[16], VoicevoxUserDictWord)>(coreApi.libraryHandle, "voicevox_user_dict_update_word");
    coreApi.userDictRemoveWord = loadSymbol<VoicevoxResultCode (*)(const VoicevoxUserDict *, const uint8_t (*)[16])>(coreApi.libraryHandle, "voicevox_user_dict_remove_word");
    coreApi.userDictImport = loadSymbol<VoicevoxResultCode (*)(const VoicevoxUserDict *, const VoicevoxUserDict *)>(coreApi.libraryHandle, "voicevox_user_dict_import");
    coreApi.jsonFree = loadSymbol<void (*)(char *)>(coreApi.libraryHandle, "voicevox_json_free");
    coreApi.wavFree = loadSymbol<void (*)(uint8_t *)>(coreApi.libraryHandle, "voicevox_wav_free");
    coreApi.errorResultToMessage = loadSymbol<const char *(*)(VoicevoxResultCode)>(coreApi.libraryHandle, "voicevox_error_result_to_message");
    coreApi.lastErrorMessage = loadSymbol<const char *(*)()>(coreApi.libraryHandle, "last_error_message");
    return coreApi;
}

void closeCoreApi(CoreApi &coreApi) {
    if (coreApi.libraryHandle) {
        closeDynamicLibrary(coreApi.libraryHandle);
        coreApi.libraryHandle = nullptr;
    }
}

std::string describeCoreError(const CoreApi &coreApi, VoicevoxResultCode callStatus) {
    std::string messageText;
    if (coreApi.errorResultToMessage) {
        const char *errorMessage = coreApi.errorResultToMessage(callStatus);
        if (errorMessage && *errorMessage) {
            messageText += errorMessage;
        }
    }
    if (coreApi.lastErrorMessage) {
        const char *lastMessage = coreApi.lastErrorMessage();
        if (lastMessage && *lastMessage) {
            if (!messageText.empty()) {
                messageText += ": ";
            }
            messageText += lastMessage;
        }
    }
    if (messageText.empty()) {
        messageText = "VOICEVOX core error " + std::to_string(callStatus);
    }
    return messageText;
}

void ensureCoreCall(const CoreApi &coreApi, VoicevoxResultCode callStatus, const std::string &actionName) {
    if (callStatus != 0) {
        throw std::runtime_error(actionName + " に失敗しました: " + describeCoreError(coreApi, callStatus));
    }
}

std::string takeJsonPointer(CoreApi &coreApi, char *jsonPointer) {
    if (!jsonPointer) {
        return "";
    }
    std::string jsonText(jsonPointer);
    coreApi.jsonFree(jsonPointer);
    return jsonText;
}
