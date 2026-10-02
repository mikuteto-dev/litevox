#pragma once

#include <cstdint>

using VoicevoxResultCode = int32_t;
using VoicevoxAccelerationMode = int32_t;
using VoicevoxUserDictWordType = int32_t;
using VoicevoxOnExistingVoiceModelId = int32_t;

struct VoicevoxLoadOnnxruntimeOptions {
    const char *filename;
};

struct VoicevoxInitializeOptions {
    VoicevoxAccelerationMode accelerationMode;
    uint16_t cpuNumThreads;
};

struct VoicevoxTtsOptions {
    bool enableInterrogativeUpspeak;
};

struct VoicevoxSynthesisOptions {
    bool enableInterrogativeUpspeak;
};

struct VoicevoxLoadVoiceModelOptions {
    VoicevoxOnExistingVoiceModelId onExisting;
};

struct VoicevoxUserDictWord {
    const char *surface;
    const char *pronunciation;
    uintptr_t accentType;
    VoicevoxUserDictWordType wordType;
    uint32_t priority;
};

struct VoicevoxOnnxruntime;
struct VoicevoxOpenJtalkRc;
struct VoicevoxSynthesizer;
struct VoicevoxVoiceModelFile;
struct VoicevoxUserDict;

struct CoreApi {
    void *libraryHandle = nullptr;
    bool hasCancellableSynthesis = false;
    bool hasSynthesisMorphing = false;
    bool hasSingFrameAudioQuery = false;
    bool hasSingFrameF0 = false;
    bool hasSingFrameVolume = false;
    bool hasFrameSynthesis = false;
    bool hasAudioQueryValidate = false;
    bool hasFrameAudioQueryValidate = false;
    const char *(*getVersion)() = nullptr;
    VoicevoxLoadOnnxruntimeOptions (*makeDefaultLoadOnnxruntimeOptions)() = nullptr;
    VoicevoxInitializeOptions (*makeDefaultInitializeOptions)() = nullptr;
    VoicevoxTtsOptions (*makeDefaultTtsOptions)() = nullptr;
    VoicevoxSynthesisOptions (*makeDefaultSynthesisOptions)() = nullptr;
    VoicevoxLoadVoiceModelOptions (*makeDefaultLoadVoiceModelOptions)() = nullptr;
    VoicevoxResultCode (*onnxruntimeLoadOnce)(VoicevoxLoadOnnxruntimeOptions, const VoicevoxOnnxruntime **) = nullptr;
    VoicevoxResultCode (*onnxruntimeCreateSupportedDevicesJson)(const VoicevoxOnnxruntime *, char **) = nullptr;
    VoicevoxResultCode (*openJtalkRcNew)(const char *, VoicevoxOpenJtalkRc **) = nullptr;
    VoicevoxResultCode (*openJtalkRcUseUserDict)(const VoicevoxOpenJtalkRc *, const VoicevoxUserDict *) = nullptr;
    VoicevoxResultCode (*openJtalkRcAnalyze)(const VoicevoxOpenJtalkRc *, const char *, char **) = nullptr;
    void (*openJtalkRcDelete)(VoicevoxOpenJtalkRc *) = nullptr;
    VoicevoxResultCode (*synthesizerNew)(const VoicevoxOnnxruntime *, const VoicevoxOpenJtalkRc *, VoicevoxInitializeOptions, VoicevoxSynthesizer **) = nullptr;
    void (*synthesizerDelete)(VoicevoxSynthesizer *) = nullptr;
    VoicevoxResultCode (*synthesizerLoadVoiceModel)(const VoicevoxSynthesizer *, const VoicevoxVoiceModelFile *) = nullptr;
    VoicevoxResultCode (*synthesizerLoadVoiceModelWithOptions)(const VoicevoxSynthesizer *, const VoicevoxVoiceModelFile *, VoicevoxLoadVoiceModelOptions) = nullptr;
    VoicevoxResultCode (*synthesizerUnloadVoiceModel)(const VoicevoxSynthesizer *, const uint8_t (*)[16]) = nullptr;
    bool (*synthesizerIsGpuMode)(const VoicevoxSynthesizer *) = nullptr;
    bool (*synthesizerIsLoadedVoiceModel)(const VoicevoxSynthesizer *, const uint8_t (*)[16]) = nullptr;
    VoicevoxResultCode (*synthesizerCreateAudioQuery)(const VoicevoxSynthesizer *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerCreateAudioQueryFromKana)(const VoicevoxSynthesizer *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerCreateAccentPhrases)(const VoicevoxSynthesizer *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerCreateAccentPhrasesFromKana)(const VoicevoxSynthesizer *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerReplaceMoraData)(const VoicevoxSynthesizer *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerReplacePhonemeLength)(const VoicevoxSynthesizer *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerReplaceMoraPitch)(const VoicevoxSynthesizer *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerSynthesis)(const VoicevoxSynthesizer *, const char *, uint32_t, VoicevoxSynthesisOptions, uintptr_t *, uint8_t **) = nullptr;
    VoicevoxResultCode (*synthesizerTts)(const VoicevoxSynthesizer *, const char *, uint32_t, VoicevoxTtsOptions, uintptr_t *, uint8_t **) = nullptr;
    VoicevoxResultCode (*synthesizerTtsFromKana)(const VoicevoxSynthesizer *, const char *, uint32_t, VoicevoxTtsOptions, uintptr_t *, uint8_t **) = nullptr;
    VoicevoxResultCode (*synthesizerCreateSingFrameAudioQuery)(const VoicevoxSynthesizer *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerCreateSingFrameF0)(const VoicevoxSynthesizer *, const char *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerCreateSingFrameVolume)(const VoicevoxSynthesizer *, const char *, const char *, uint32_t, char **) = nullptr;
    VoicevoxResultCode (*synthesizerFrameSynthesis)(const VoicevoxSynthesizer *, const char *, uint32_t, uintptr_t *, uint8_t **) = nullptr;
    VoicevoxResultCode (*audioQueryCreateFromAccentPhrases)(const char *, char **) = nullptr;
    VoicevoxResultCode (*audioQueryValidate)(const char *) = nullptr;
    VoicevoxResultCode (*frameAudioQueryValidate)(const char *) = nullptr;
    VoicevoxResultCode (*voiceModelFileOpen)(const char *, VoicevoxVoiceModelFile **) = nullptr;
    void (*voiceModelFileId)(const VoicevoxVoiceModelFile *, uint8_t (*)[16]) = nullptr;
    void (*voiceModelFileDelete)(VoicevoxVoiceModelFile *) = nullptr;
    VoicevoxUserDictWord (*userDictWordMake)(const char *, const char *, uintptr_t) = nullptr;
    VoicevoxUserDict *(*userDictNew)() = nullptr;
    void (*userDictDelete)(VoicevoxUserDict *) = nullptr;
    VoicevoxResultCode (*userDictLoad)(const VoicevoxUserDict *, const char *) = nullptr;
    VoicevoxResultCode (*userDictSave)(const VoicevoxUserDict *, const char *) = nullptr;
    VoicevoxResultCode (*userDictToJson)(const VoicevoxUserDict *, char **) = nullptr;
    VoicevoxResultCode (*userDictAddWord)(const VoicevoxUserDict *, VoicevoxUserDictWord, uint8_t (*)[16]) = nullptr;
    VoicevoxResultCode (*userDictUpdateWord)(const VoicevoxUserDict *, const uint8_t (*)[16], VoicevoxUserDictWord) = nullptr;
    VoicevoxResultCode (*userDictRemoveWord)(const VoicevoxUserDict *, const uint8_t (*)[16]) = nullptr;
    VoicevoxResultCode (*userDictImport)(const VoicevoxUserDict *, const VoicevoxUserDict *) = nullptr;
    void (*jsonFree)(char *) = nullptr;
    void (*wavFree)(uint8_t *) = nullptr;
    const char *(*errorResultToMessage)(VoicevoxResultCode) = nullptr;
    const char *(*lastErrorMessage)() = nullptr;
};
