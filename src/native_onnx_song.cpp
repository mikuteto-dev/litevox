#include "native_onnx_internal.hpp"

#include "streaming_audio.hpp"

#include <cmath>
#include <limits>
#include <utility>

static std::vector<int64_t> runNativeOnnxSingConsonantLengths(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::vector<NativeOnnxScoreNote> &scoreNotes, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    const ModelAssetRecord &consonantLengthAsset = requireNativeOnnxModelAsset(modelAssets, "models/pscl.bin");
    int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(consonantLengthAsset, "singing_teacher", styleId);
    std::vector<int64_t> consonantValues;
    std::vector<int64_t> vowelValues;
    std::vector<int64_t> noteDurationValues;
    consonantValues.reserve(scoreNotes.size());
    vowelValues.reserve(scoreNotes.size());
    noteDurationValues.reserve(scoreNotes.size());
    for (const NativeOnnxScoreNote &scoreNote : scoreNotes) {
        if (scoreNote.frameLength > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("frame_length が大きすぎます");
        }
        noteDurationValues.push_back(static_cast<int64_t>(scoreNote.frameLength));
        if (!scoreNote.hasKey) {
            consonantValues.push_back(-1);
            vowelValues.push_back(0);
        } else {
            consonantValues.push_back(scoreNote.mora.hasConsonant ? parseNativeOnnxPhonemeCode(scoreNote.mora.consonant) : -1);
            vowelValues.push_back(parseNativeOnnxPhonemeCode(scoreNote.mora.vowel));
        }
    }
    std::vector<NativeOnnxTraceInput> consonantLengthInputs;
    consonantLengthInputs.push_back(createNativeOnnxInt64Tensor("consonants", {1, static_cast<int64_t>(consonantValues.size())}, consonantValues));
    consonantLengthInputs.push_back(createNativeOnnxInt64Tensor("vowels", {1, static_cast<int64_t>(vowelValues.size())}, vowelValues));
    consonantLengthInputs.push_back(createNativeOnnxInt64Tensor("note_durations", {1, static_cast<int64_t>(noteDurationValues.size())}, noteDurationValues));
    consonantLengthInputs.push_back(createNativeOnnxInt64Tensor("speaker_id", {1}, {innerVoiceId}));
    std::vector<NativeOnnxTraceInput> consonantLengthOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, runtimeState, consonantLengthAsset, consonantLengthInputs, cpuThreadCount, shouldUseVvBinConfig);
    std::vector<int64_t> consonantLengths = readNativeOnnxTensorValues<int64_t>(requireNativeOnnxTensor(consonantLengthOutputs, "consonant_lengths"), 7);
    if (consonantLengths.size() != scoreNotes.size()) {
        throw std::runtime_error("consonant_lengths の長さが一致しません");
    }
    for (size_t noteIndex = 0; noteIndex < scoreNotes.size(); noteIndex++) {
        bool hasConsonant = scoreNotes[noteIndex].hasKey && scoreNotes[noteIndex].mora.hasConsonant;
        if (consonantLengths[noteIndex] == 0 && hasConsonant) {
            throw std::runtime_error("子音あり音符の consonant_length が 0 です");
        }
        if (consonantLengths[noteIndex] != 0 && !hasConsonant) {
            throw std::runtime_error("子音なし音符の consonant_length が 0 ではありません");
        }
    }
    return consonantLengths;
}

static std::vector<float> runNativeOnnxSingF0Values(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const NativeOnnxSongFrameInputs &frameInputs, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    const ModelAssetRecord &f0Asset = requireNativeOnnxModelAsset(modelAssets, "models/psf.bin");
    int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(f0Asset, "singing_teacher", styleId);
    std::vector<NativeOnnxTraceInput> f0Inputs;
    f0Inputs.push_back(createNativeOnnxInt64Tensor("phonemes", {1, static_cast<int64_t>(frameInputs.phonemeValues.size())}, frameInputs.phonemeValues));
    f0Inputs.push_back(createNativeOnnxInt64Tensor("notes", {1, static_cast<int64_t>(frameInputs.keyValues.size())}, frameInputs.keyValues));
    f0Inputs.push_back(createNativeOnnxInt64Tensor("speaker_id", {1}, {innerVoiceId}));
    std::vector<NativeOnnxTraceInput> f0Outputs = runNativeOnnxSingTeacherModelAssetBytes(nativeOnnxApi, runtimeState, f0Asset, f0Inputs, cpuThreadCount, shouldUseVvBinConfig);
    std::vector<float> f0Values = readNativeOnnxTensorValues<float>(requireNativeOnnxTensor(f0Outputs, "f0s"), 1);
    if (f0Values.size() != frameInputs.phonemeValues.size()) {
        throw std::runtime_error("f0s の長さが一致しません");
    }
    for (float f0Value : f0Values) {
        if (!std::isfinite(f0Value) || f0Value < 0.0f) {
            throw std::runtime_error("f0s に不正な値があります");
        }
    }
    return f0Values;
}

static std::vector<float> runNativeOnnxSingVolumeValues(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const NativeOnnxSongFrameInputs &frameInputs, const std::vector<float> &f0Values, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    if (f0Values.size() != frameInputs.phonemeValues.size()) {
        throw std::invalid_argument("f0 の長さが frame 数と一致しません");
    }
    const ModelAssetRecord &volumeAsset = requireNativeOnnxModelAsset(modelAssets, "models/psv.bin");
    int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(volumeAsset, "singing_teacher", styleId);
    std::vector<NativeOnnxTraceInput> volumeInputs;
    volumeInputs.push_back(createNativeOnnxInt64Tensor("phonemes", {1, static_cast<int64_t>(frameInputs.phonemeValues.size())}, frameInputs.phonemeValues));
    volumeInputs.push_back(createNativeOnnxInt64Tensor("notes", {1, static_cast<int64_t>(frameInputs.keyValues.size())}, frameInputs.keyValues));
    volumeInputs.push_back(createNativeOnnxFloatTensor("frame_f0s", {1, static_cast<int64_t>(f0Values.size())}, f0Values));
    volumeInputs.push_back(createNativeOnnxInt64Tensor("speaker_id", {1}, {innerVoiceId}));
    std::vector<NativeOnnxTraceInput> volumeOutputs = runNativeOnnxSingTeacherModelAssetBytes(nativeOnnxApi, runtimeState, volumeAsset, volumeInputs, cpuThreadCount, shouldUseVvBinConfig);
    std::vector<float> volumeValues = readNativeOnnxTensorValues<float>(requireNativeOnnxTensor(volumeOutputs, "volumes"), 1);
    if (volumeValues.size() != frameInputs.phonemeValues.size()) {
        throw std::runtime_error("volumes の長さが一致しません");
    }
    for (float volumeValue : volumeValues) {
        if (!std::isfinite(volumeValue)) {
            throw std::runtime_error("volumes に不正な値があります");
        }
    }
    return volumeValues;
}

static NativeOnnxFrameAudioQuery createNativeOnnxSingFrameAudioQueryWithApi(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &scoreText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    std::vector<NativeOnnxScoreNote> scoreNotes = parseNativeOnnxScore(scoreText);
    std::vector<int64_t> consonantLengths = runNativeOnnxSingConsonantLengths(nativeOnnxApi, runtimeState, modelAssets, scoreNotes, styleId, cpuThreadCount, shouldUseVvBinConfig);
    std::vector<NativeOnnxSongPhonemeFeature> phonemeFeatures = createNativeOnnxSongPhonemeFeatures(scoreNotes);
    std::vector<uint64_t> phonemeLengths = createNativeOnnxSongPhonemeLengths(scoreNotes, consonantLengths);
    if (phonemeFeatures.size() != phonemeLengths.size()) {
        throw std::runtime_error("歌唱 phoneme_length の長さが一致しません");
    }
    NativeOnnxFrameAudioQuery frameAudioQuery;
    for (size_t phonemeIndex = 0; phonemeIndex < phonemeFeatures.size(); phonemeIndex++) {
        NativeOnnxFramePhoneme framePhoneme;
        framePhoneme.phoneme = phonemeFeatures[phonemeIndex].phoneme;
        framePhoneme.frameLength = phonemeLengths[phonemeIndex];
        framePhoneme.noteId = phonemeFeatures[phonemeIndex].noteId;
        framePhoneme.hasNoteId = phonemeFeatures[phonemeIndex].hasNoteId;
        frameAudioQuery.phonemes.push_back(std::move(framePhoneme));
    }
    NativeOnnxSongFrameInputs frameInputs = createNativeOnnxSongFrameInputs(phonemeFeatures, frameAudioQuery.phonemes);
    frameAudioQuery.f0Values = runNativeOnnxSingF0Values(nativeOnnxApi, runtimeState, modelAssets, frameInputs, styleId, cpuThreadCount, shouldUseVvBinConfig);
    frameAudioQuery.volumeValues = runNativeOnnxSingVolumeValues(nativeOnnxApi, runtimeState, modelAssets, frameInputs, frameAudioQuery.f0Values, styleId, cpuThreadCount, shouldUseVvBinConfig);
    frameAudioQuery.volumeScale = 1.0f;
    frameAudioQuery.outputSamplingRate = nativeOnnxDefaultSamplingRate;
    frameAudioQuery.outputStereo = false;
    validateNativeOnnxParsedFrameAudioQuery(frameAudioQuery);
    return frameAudioQuery;
}

static NativeOnnxSongFrameInputs createNativeOnnxSongFrameInputsFromScoreAndQuery(const std::string &scoreText, const NativeOnnxFrameAudioQuery &frameAudioQuery) {
    std::vector<NativeOnnxScoreNote> scoreNotes = parseNativeOnnxScore(scoreText);
    std::vector<NativeOnnxSongPhonemeFeature> phonemeFeatures = createNativeOnnxSongPhonemeFeatures(scoreNotes);
    return createNativeOnnxSongFrameInputs(phonemeFeatures, frameAudioQuery.phonemes);
}

void validateNativeOnnxFrameAudioQuery(const std::string &frameAudioQueryText) {
    validateNativeOnnxParsedFrameAudioQuery(parseNativeOnnxFrameAudioQuery(frameAudioQueryText));
}

std::string createNativeOnnxModelAssetsSingFrameAudioQuery(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &scoreText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        NativeOnnxFrameAudioQuery frameAudioQuery = createNativeOnnxSingFrameAudioQueryWithApi(nativeOnnxApi, nullptr, modelAssets, scoreText, styleId, cpuThreadCount, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return createNativeOnnxFrameAudioQueryJson(frameAudioQuery);
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string createNativeOnnxModelAssetsSingFrameF0(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &scoreText, const std::string &frameAudioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxFrameAudioQuery frameAudioQuery = parseNativeOnnxFrameAudioQuery(frameAudioQueryText);
    NativeOnnxSongFrameInputs frameInputs = createNativeOnnxSongFrameInputsFromScoreAndQuery(scoreText, frameAudioQuery);
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::vector<float> f0Values = runNativeOnnxSingF0Values(nativeOnnxApi, nullptr, modelAssets, frameInputs, styleId, cpuThreadCount, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return createNativeOnnxFloatArrayJson(f0Values);
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string createNativeOnnxModelAssetsSingFrameVolume(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &scoreText, const std::string &frameAudioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxFrameAudioQuery frameAudioQuery = parseNativeOnnxFrameAudioQuery(frameAudioQueryText);
    NativeOnnxSongFrameInputs frameInputs = createNativeOnnxSongFrameInputsFromScoreAndQuery(scoreText, frameAudioQuery);
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::vector<float> volumeValues = runNativeOnnxSingVolumeValues(nativeOnnxApi, nullptr, modelAssets, frameInputs, frameAudioQuery.f0Values, styleId, cpuThreadCount, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return createNativeOnnxFloatArrayJson(volumeValues);
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::vector<uint8_t> synthesizeNativeOnnxModelAssetsFrameAudioQuery(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &frameAudioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxFrameAudioQuery frameAudioQuery = parseNativeOnnxFrameAudioQuery(frameAudioQueryText);
    validateNativeOnnxParsedFrameAudioQuery(frameAudioQuery);
    const ModelAssetRecord &decodeAsset = requireNativeOnnxModelAsset(modelAssets, "models/sd.bin");
    int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(decodeAsset, "frame_decode", styleId);
    std::vector<int64_t> framePhonemeValues;
    framePhonemeValues.reserve(frameAudioQuery.f0Values.size());
    for (const NativeOnnxFramePhoneme &framePhoneme : frameAudioQuery.phonemes) {
        int64_t phonemeCode = parseNativeOnnxPhonemeCode(framePhoneme.phoneme);
        for (uint64_t frameIndex = 0; frameIndex < framePhoneme.frameLength; frameIndex++) {
            framePhonemeValues.push_back(phonemeCode);
        }
    }
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::vector<NativeOnnxTraceInput> decodeInputs;
        decodeInputs.push_back(createNativeOnnxInt64Tensor("frame_phonemes", {1, static_cast<int64_t>(framePhonemeValues.size())}, framePhonemeValues));
        decodeInputs.push_back(createNativeOnnxFloatTensor("frame_f0s", {1, static_cast<int64_t>(frameAudioQuery.f0Values.size())}, frameAudioQuery.f0Values));
        decodeInputs.push_back(createNativeOnnxFloatTensor("frame_volumes", {1, static_cast<int64_t>(frameAudioQuery.volumeValues.size())}, frameAudioQuery.volumeValues));
        decodeInputs.push_back(createNativeOnnxInt64Tensor("speaker_id", {1}, {innerVoiceId}));
        std::vector<NativeOnnxTraceInput> decodeOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, nullptr, decodeAsset, decodeInputs, cpuThreadCount, shouldUseVvBinConfig);
        std::vector<float> waveValues = readNativeOnnxTensorValues<float>(requireNativeOnnxTensor(decodeOutputs, "wav"), 1);
        NativeOnnxAudioQuerySettings audioQuerySettings = createNativeOnnxSettingsFromFrameAudioQuery(frameAudioQuery);
        std::vector<uint8_t> pcmBytes = createNativeOnnxPcmBytes(waveValues, audioQuerySettings);
        uint16_t channels = frameAudioQuery.outputStereo ? 2 : 1;
        std::vector<uint8_t> wavBytes = createPcmWaveHeader(frameAudioQuery.outputSamplingRate, channels, 16, pcmBytes.size());
        wavBytes.insert(wavBytes.end(), pcmBytes.begin(), pcmBytes.end());
        closeNativeOnnxApi(nativeOnnxApi);
        return wavBytes;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}



std::string createNativeOnnxModelAssetsSingFrameAudioQuery(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &scoreText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        NativeOnnxFrameAudioQuery frameAudioQuery = createNativeOnnxSingFrameAudioQueryWithApi(nativeOnnxApi, &runtimeState, modelAssets, scoreText, styleId, cpuThreadCount, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return createNativeOnnxFrameAudioQueryJson(frameAudioQuery);
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string createNativeOnnxModelAssetsSingFrameF0(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &scoreText, const std::string &frameAudioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxFrameAudioQuery frameAudioQuery = parseNativeOnnxFrameAudioQuery(frameAudioQueryText);
    NativeOnnxSongFrameInputs frameInputs = createNativeOnnxSongFrameInputsFromScoreAndQuery(scoreText, frameAudioQuery);
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        std::vector<float> f0Values = runNativeOnnxSingF0Values(nativeOnnxApi, &runtimeState, modelAssets, frameInputs, styleId, cpuThreadCount, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return createNativeOnnxFloatArrayJson(f0Values);
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string createNativeOnnxModelAssetsSingFrameVolume(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &scoreText, const std::string &frameAudioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxFrameAudioQuery frameAudioQuery = parseNativeOnnxFrameAudioQuery(frameAudioQueryText);
    NativeOnnxSongFrameInputs frameInputs = createNativeOnnxSongFrameInputsFromScoreAndQuery(scoreText, frameAudioQuery);
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        std::vector<float> volumeValues = runNativeOnnxSingVolumeValues(nativeOnnxApi, &runtimeState, modelAssets, frameInputs, frameAudioQuery.f0Values, styleId, cpuThreadCount, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return createNativeOnnxFloatArrayJson(volumeValues);
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::vector<uint8_t> synthesizeNativeOnnxModelAssetsFrameAudioQuery(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &frameAudioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxFrameAudioQuery frameAudioQuery = parseNativeOnnxFrameAudioQuery(frameAudioQueryText);
    validateNativeOnnxParsedFrameAudioQuery(frameAudioQuery);
    const ModelAssetRecord &decodeAsset = requireNativeOnnxModelAsset(modelAssets, "models/sd.bin");
    int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(decodeAsset, "frame_decode", styleId);
    std::vector<int64_t> framePhonemeValues;
    framePhonemeValues.reserve(frameAudioQuery.f0Values.size());
    for (const NativeOnnxFramePhoneme &framePhoneme : frameAudioQuery.phonemes) {
        int64_t phonemeCode = parseNativeOnnxPhonemeCode(framePhoneme.phoneme);
        for (uint64_t frameIndex = 0; frameIndex < framePhoneme.frameLength; frameIndex++) {
            framePhonemeValues.push_back(phonemeCode);
        }
    }
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        std::vector<NativeOnnxTraceInput> decodeInputs;
        decodeInputs.push_back(createNativeOnnxInt64Tensor("frame_phonemes", {1, static_cast<int64_t>(framePhonemeValues.size())}, framePhonemeValues));
        decodeInputs.push_back(createNativeOnnxFloatTensor("frame_f0s", {1, static_cast<int64_t>(frameAudioQuery.f0Values.size())}, frameAudioQuery.f0Values));
        decodeInputs.push_back(createNativeOnnxFloatTensor("frame_volumes", {1, static_cast<int64_t>(frameAudioQuery.volumeValues.size())}, frameAudioQuery.volumeValues));
        decodeInputs.push_back(createNativeOnnxInt64Tensor("speaker_id", {1}, {innerVoiceId}));
        std::vector<NativeOnnxTraceInput> decodeOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, &runtimeState, decodeAsset, decodeInputs, cpuThreadCount, shouldUseVvBinConfig);
        std::vector<float> waveValues = readNativeOnnxTensorValues<float>(requireNativeOnnxTensor(decodeOutputs, "wav"), 1);
        NativeOnnxAudioQuerySettings audioQuerySettings = createNativeOnnxSettingsFromFrameAudioQuery(frameAudioQuery);
        std::vector<uint8_t> pcmBytes = createNativeOnnxPcmBytes(waveValues, audioQuerySettings);
        uint16_t channels = frameAudioQuery.outputStereo ? 2 : 1;
        std::vector<uint8_t> wavBytes = createPcmWaveHeader(frameAudioQuery.outputSamplingRate, channels, 16, pcmBytes.size());
        wavBytes.insert(wavBytes.end(), pcmBytes.begin(), pcmBytes.end());
        closeNativeOnnxApi(nativeOnnxApi);
        return wavBytes;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}
