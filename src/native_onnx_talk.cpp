#include "native_onnx_internal.hpp"

#include "json_utility.hpp"
#include "streaming_audio.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <limits>
#include <soxr.h>

static std::string createNativeOnnxMoraJson(const NativeOnnxMora &mora) {
    std::string jsonText = "{\"text\":" + quoteJsonString(mora.text) + ",";
    jsonText += "\"consonant\":";
    jsonText += mora.hasConsonant ? quoteJsonString(mora.consonant) : "null";
    jsonText += ",\"consonant_length\":";
    jsonText += mora.hasConsonant ? formatNativeOnnxFloat(mora.consonantLength) : "null";
    jsonText += ",\"vowel\":" + quoteJsonString(mora.vowel);
    jsonText += ",\"vowel_length\":" + formatNativeOnnxFloat(mora.vowelLength);
    jsonText += ",\"pitch\":" + formatNativeOnnxFloat(mora.pitch);
    jsonText += "}";
    return jsonText;
}

static std::string createNativeOnnxAccentPhrasesJson(const std::vector<NativeOnnxAccentPhrase> &accentPhrases) {
    std::string jsonText = "[";
    for (size_t phraseIndex = 0; phraseIndex < accentPhrases.size(); phraseIndex++) {
        const NativeOnnxAccentPhrase &accentPhrase = accentPhrases[phraseIndex];
        if (phraseIndex > 0) {
            jsonText += ",";
        }
        jsonText += "{\"moras\":[";
        for (size_t moraIndex = 0; moraIndex < accentPhrase.moras.size(); moraIndex++) {
            if (moraIndex > 0) {
                jsonText += ",";
            }
            jsonText += createNativeOnnxMoraJson(accentPhrase.moras[moraIndex]);
        }
        jsonText += "],\"accent\":" + std::to_string(accentPhrase.accent);
        jsonText += ",\"pause_mora\":";
        jsonText += accentPhrase.hasPauseMora ? createNativeOnnxMoraJson(accentPhrase.pauseMora) : "null";
        jsonText += ",\"is_interrogative\":";
        jsonText += accentPhrase.isInterrogative ? "true" : "false";
        jsonText += "}";
    }
    jsonText += "]";
    return jsonText;
}

static std::string createNativeOnnxAudioQueryTextFromAccentPhrasesJson(const std::string &accentPhrasesJson) {
    std::string jsonText = "{\"accent_phrases\":" + accentPhrasesJson;
    jsonText += ",\"speedScale\":1.0";
    jsonText += ",\"pitchScale\":0.0";
    jsonText += ",\"intonationScale\":1.0";
    jsonText += ",\"volumeScale\":1.0";
    jsonText += ",\"prePhonemeLength\":0.1";
    jsonText += ",\"postPhonemeLength\":0.1";
    jsonText += ",\"pauseLength\":null";
    jsonText += ",\"pauseLengthScale\":1.0";
    jsonText += ",\"outputSamplingRate\":24000";
    jsonText += ",\"outputStereo\":false";
    jsonText += ",\"kana\":\"\"";
    jsonText += "}";
    return jsonText;
}

static std::string createNativeOnnxAudioQueryTextWithAccentPhrases(const std::string &audioQueryText, const std::string &accentPhrasesJson) {
    NativeOnnxAudioQuerySettings audioQuerySettings = parseNativeOnnxAudioQuerySettings(audioQueryText);
    std::string kana = extractJsonStringField(audioQueryText, "kana");
    std::string jsonText = "{\"accent_phrases\":" + accentPhrasesJson;
    jsonText += ",\"speedScale\":" + formatNativeOnnxSettingFloat(audioQuerySettings.speedScale);
    jsonText += ",\"pitchScale\":" + formatNativeOnnxSettingFloat(audioQuerySettings.pitchScale);
    jsonText += ",\"intonationScale\":" + formatNativeOnnxSettingFloat(audioQuerySettings.intonationScale);
    jsonText += ",\"volumeScale\":" + formatNativeOnnxSettingFloat(audioQuerySettings.volumeScale);
    jsonText += ",\"prePhonemeLength\":" + formatNativeOnnxSettingFloat(audioQuerySettings.prePhonemeLength);
    jsonText += ",\"postPhonemeLength\":" + formatNativeOnnxSettingFloat(audioQuerySettings.postPhonemeLength);
    jsonText += ",\"pauseLength\":null";
    jsonText += ",\"pauseLengthScale\":1.0";
    jsonText += ",\"outputSamplingRate\":" + std::to_string(audioQuerySettings.outputSamplingRate);
    jsonText += ",\"outputStereo\":";
    jsonText += audioQuerySettings.outputStereo ? "true" : "false";
    jsonText += ",\"kana\":" + quoteJsonString(kana);
    jsonText += "}";
    return jsonText;
}

static bool hasNativeOnnxCompleteMoraData(const std::vector<NativeOnnxAccentPhrase> &accentPhrases) {
    for (const NativeOnnxAccentPhrase &accentPhrase : accentPhrases) {
        for (const NativeOnnxMora &mora : accentPhrase.moras) {
            if (!mora.hasVowelLength || !mora.hasPitch || mora.vowelLength <= 0.0f) {
                return false;
            }
            if (mora.hasConsonant && (!mora.hasConsonantLength || mora.consonantLength <= 0.0f)) {
                return false;
            }
        }
        if (accentPhrase.hasPauseMora && (!accentPhrase.pauseMora.hasVowelLength || !accentPhrase.pauseMora.hasPitch || accentPhrase.pauseMora.vowelLength <= 0.0f)) {
            return false;
        }
    }
    return true;
}

static void ensureNativeOnnxMinimumPhonemeLengths(std::vector<float> &phonemeLengthValues) {
    static constexpr float minimumPhonemeLength = 0.01f;
    for (float &phonemeLength : phonemeLengthValues) {
        if (phonemeLength < minimumPhonemeLength) {
            phonemeLength = minimumPhonemeLength;
        }
    }
}

static std::vector<float> createNativeOnnxPhonemeLengthValuesFromAccentPhrases(const NativeOnnxAudioQuerySettings &audioQuerySettings, const std::vector<NativeOnnxAccentPhrase> &accentPhrases) {
    std::vector<float> phonemeLengthValues;
    phonemeLengthValues.push_back(audioQuerySettings.prePhonemeLength);
    for (const NativeOnnxAccentPhrase &accentPhrase : accentPhrases) {
        for (const NativeOnnxMora &mora : accentPhrase.moras) {
            if (mora.hasConsonant) {
                phonemeLengthValues.push_back(mora.consonantLength);
            }
            phonemeLengthValues.push_back(mora.vowelLength);
        }
        if (accentPhrase.hasPauseMora) {
            if (accentPhrase.pauseMora.hasConsonant) {
                phonemeLengthValues.push_back(accentPhrase.pauseMora.consonantLength);
            }
            phonemeLengthValues.push_back(accentPhrase.pauseMora.vowelLength);
        }
    }
    phonemeLengthValues.push_back(audioQuerySettings.postPhonemeLength);
    return phonemeLengthValues;
}

static std::vector<float> createNativeOnnxF0ValuesFromAccentPhrases(const std::vector<NativeOnnxAccentPhrase> &accentPhrases) {
    std::vector<float> f0Values;
    f0Values.push_back(0.0f);
    for (const NativeOnnxAccentPhrase &accentPhrase : accentPhrases) {
        for (const NativeOnnxMora &mora : accentPhrase.moras) {
            f0Values.push_back(mora.pitch);
        }
        if (accentPhrase.hasPauseMora) {
            f0Values.push_back(accentPhrase.pauseMora.pitch);
        }
    }
    f0Values.push_back(0.0f);
    return f0Values;
}

std::vector<NativeOnnxTraceInput> createNativeOnnxDecoderInputsFromAudioQuery(const std::vector<NativeOnnxTraceInput> &frontendInputs, const std::string &audioQueryText, const NativeOnnxAudioQuerySettings &audioQuerySettings) {
    std::vector<NativeOnnxAccentPhrase> accentPhrases = parseNativeOnnxAccentPhrases(audioQueryText);
    std::vector<float> phonemeLengthValues = createNativeOnnxPhonemeLengthValuesFromAccentPhrases(audioQuerySettings, accentPhrases);
    std::vector<float> f0Values = createNativeOnnxF0ValuesFromAccentPhrases(accentPhrases);
    NativeOnnxTraceInput durationTensor;
    durationTensor.name = "phoneme_length";
    durationTensor.elementType = 1;
    durationTensor.dimensions = {static_cast<int64_t>(phonemeLengthValues.size())};
    durationTensor.bytes = createNativeOnnxTensorBytes(phonemeLengthValues);
    NativeOnnxTraceInput intonationTensor;
    intonationTensor.name = "f0_list";
    intonationTensor.elementType = 1;
    intonationTensor.dimensions = {static_cast<int64_t>(f0Values.size())};
    intonationTensor.bytes = createNativeOnnxTensorBytes(f0Values);
    return createNativeOnnxDecoderInputs(frontendInputs, {durationTensor}, {intonationTensor}, audioQuerySettings, false);
}

static std::vector<float> runNativeOnnxDurationValues(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::vector<NativeOnnxTraceInput> &frontendInputs, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    const ModelAssetRecord &durationAsset = requireNativeOnnxModelAsset(modelAssets, "models/pd.bin");
    std::vector<NativeOnnxTraceInput> durationOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, runtimeState, durationAsset, frontendInputs, cpuThreadCount, shouldUseVvBinConfig);
    std::vector<float> phonemeLengthValues = readNativeOnnxTensorValues<float>(requireNativeOnnxTensor(durationOutputs, "phoneme_length"), 1);
    ensureNativeOnnxMinimumPhonemeLengths(phonemeLengthValues);
    return phonemeLengthValues;
}

static std::vector<float> runNativeOnnxF0Values(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::vector<NativeOnnxTraceInput> &frontendInputs, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    const ModelAssetRecord &intonationAsset = requireNativeOnnxModelAsset(modelAssets, "models/pi.bin");
    std::vector<NativeOnnxTraceInput> intonationOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, runtimeState, intonationAsset, frontendInputs, cpuThreadCount, shouldUseVvBinConfig);
    std::vector<float> f0Values = readNativeOnnxTensorValues<float>(requireNativeOnnxTensor(intonationOutputs, "f0_list"), 1);
    const NativeOnnxTraceInput &vowelPhonemeListTensor = requireNativeOnnxTensor(frontendInputs, "vowel_phoneme_list");
    std::vector<int64_t> vowelValues = readNativeOnnxTensorValues<int64_t>(vowelPhonemeListTensor, 7);
    if (f0Values.size() != vowelValues.size()) {
        throw std::runtime_error("f0_list の長さが一致しません");
    }
    for (size_t vowelIndex = 0; vowelIndex < vowelValues.size(); vowelIndex++) {
        if (isNativeOnnxUnvoicedVowel(vowelValues[vowelIndex])) {
            f0Values[vowelIndex] = 0.0f;
        }
    }
    return f0Values;
}

static void applyNativeOnnxPhonemeLengthsToAccentPhrases(std::vector<NativeOnnxAccentPhrase> &accentPhrases, const std::vector<NativeOnnxTraceInput> &frontendInputs, const std::vector<float> &phonemeLengthValues) {
    const NativeOnnxTraceInput &phonemeListTensor = requireNativeOnnxTensor(frontendInputs, "phoneme_list");
    const NativeOnnxTraceInput &vowelPhonemeListTensor = requireNativeOnnxTensor(frontendInputs, "vowel_phoneme_list");
    std::vector<int64_t> phonemeValues = readNativeOnnxTensorValues<int64_t>(phonemeListTensor, 7);
    std::vector<int64_t> vowelValues = readNativeOnnxTensorValues<int64_t>(vowelPhonemeListTensor, 7);
    std::vector<int64_t> vowelIndexes = findNativeOnnxVowelIndexes(phonemeValues, vowelValues);
    if (phonemeLengthValues.size() != phonemeValues.size()) {
        throw std::runtime_error("phoneme_length の長さが一致しません");
    }
    size_t moraIndex = 0;
    auto applyLength = [&](NativeOnnxMora &mora) {
        size_t vowelIndex = static_cast<size_t>(vowelIndexes.at(moraIndex + 1));
        if (mora.hasConsonant) {
            mora.consonantLength = phonemeLengthValues.at(vowelIndex - 1);
            mora.hasConsonantLength = true;
        } else {
            mora.consonantLength = 0.0f;
            mora.hasConsonantLength = false;
        }
        mora.vowelLength = phonemeLengthValues.at(vowelIndex);
        mora.hasVowelLength = true;
        moraIndex++;
    };
    for (NativeOnnxAccentPhrase &accentPhrase : accentPhrases) {
        for (NativeOnnxMora &mora : accentPhrase.moras) {
            applyLength(mora);
        }
        if (accentPhrase.hasPauseMora) {
            applyLength(accentPhrase.pauseMora);
        }
    }
}

static void applyNativeOnnxF0ToAccentPhrases(std::vector<NativeOnnxAccentPhrase> &accentPhrases, const std::vector<float> &f0Values) {
    size_t moraIndex = 0;
    auto applyPitch = [&](NativeOnnxMora &mora) {
        mora.pitch = f0Values.at(moraIndex + 1);
        mora.hasPitch = true;
        moraIndex++;
    };
    for (NativeOnnxAccentPhrase &accentPhrase : accentPhrases) {
        for (NativeOnnxMora &mora : accentPhrase.moras) {
            applyPitch(mora);
        }
        if (accentPhrase.hasPauseMora) {
            applyPitch(accentPhrase.pauseMora);
        }
    }
}

static std::string replaceNativeOnnxMoraDataWithApi(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &accentPhrasesJson, uint32_t styleId, uint16_t cpuThreadCount, bool shouldReplaceLength, bool shouldReplacePitch, bool shouldUseVvBinConfig) {
    std::string audioQueryText = createNativeOnnxAudioQueryTextFromAccentPhrasesJson(accentPhrasesJson);
    std::vector<NativeOnnxAccentPhrase> accentPhrases = parseNativeOnnxAccentPhrases(audioQueryText);
    const ModelAssetRecord &durationAsset = requireNativeOnnxModelAsset(modelAssets, "models/pd.bin");
    int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(durationAsset, "talk", styleId);
    std::vector<NativeOnnxTraceInput> frontendInputs = createNativeOnnxFrontendInputs(audioQueryText, innerVoiceId);
    if (shouldReplaceLength) {
        applyNativeOnnxPhonemeLengthsToAccentPhrases(accentPhrases, frontendInputs, runNativeOnnxDurationValues(nativeOnnxApi, runtimeState, modelAssets, frontendInputs, cpuThreadCount, shouldUseVvBinConfig));
    }
    if (shouldReplacePitch) {
        applyNativeOnnxF0ToAccentPhrases(accentPhrases, runNativeOnnxF0Values(nativeOnnxApi, runtimeState, modelAssets, frontendInputs, cpuThreadCount, shouldUseVvBinConfig));
    }
    return createNativeOnnxAccentPhrasesJson(accentPhrases);
}

NativeOnnxTraceInput createNativeOnnxFloatTensor(const std::string &tensorName, const std::vector<int64_t> &dimensions, const std::vector<float> &values) {
    NativeOnnxTraceInput tensor;
    tensor.name = tensorName;
    tensor.elementType = 1;
    tensor.dimensions = dimensions;
    tensor.bytes = createNativeOnnxTensorBytes(values);
    return tensor;
}

static std::vector<NativeOnnxTraceInput> createNativeOnnxModelAssetDecoderInputs(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &audioQueryText, uint32_t styleId, uint16_t cpuThreadCount, const NativeOnnxAudioQuerySettings &audioQuerySettings, bool shouldUseVvBinConfig) {
    const ModelAssetRecord &durationAsset = requireNativeOnnxModelAsset(modelAssets, "models/pd.bin");
    const ModelAssetRecord &intonationAsset = requireNativeOnnxModelAsset(modelAssets, "models/pi.bin");
    int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(durationAsset, "talk", styleId);
    std::vector<NativeOnnxTraceInput> frontendInputs = createNativeOnnxFrontendInputs(audioQueryText, innerVoiceId);
    if (hasNativeOnnxCompleteMoraData(parseNativeOnnxAccentPhrases(audioQueryText))) {
        return createNativeOnnxDecoderInputsFromAudioQuery(frontendInputs, audioQueryText, audioQuerySettings);
    }
    std::vector<NativeOnnxTraceInput> durationOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, runtimeState, durationAsset, frontendInputs, cpuThreadCount, shouldUseVvBinConfig);
    std::vector<NativeOnnxTraceInput> intonationOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, runtimeState, intonationAsset, frontendInputs, cpuThreadCount, shouldUseVvBinConfig);
    return createNativeOnnxDecoderInputs(frontendInputs, durationOutputs, intonationOutputs, audioQuerySettings, true);
}

static NativeOnnxTraceInput runNativeOnnxModelAssetChainWave(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &audioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    const ModelAssetRecord &decodeAsset = requireNativeOnnxModelAsset(modelAssets, "models/d.bin");
    NativeOnnxAudioQuerySettings audioQuerySettings = parseNativeOnnxAudioQuerySettings(audioQueryText);
    std::vector<NativeOnnxTraceInput> decoderInputs = createNativeOnnxModelAssetDecoderInputs(nativeOnnxApi, runtimeState, modelAssets, audioQueryText, styleId, cpuThreadCount, audioQuerySettings, shouldUseVvBinConfig);
    std::vector<NativeOnnxTraceInput> decodeOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, runtimeState, decodeAsset, decoderInputs, cpuThreadCount, shouldUseVvBinConfig);
    return requireNativeOnnxTensor(decodeOutputs, "wave");
}

static std::vector<float> createNativeOnnxWaveValuesWithoutDecoderFramePadding(const NativeOnnxTraceInput &waveTensor, size_t frontCropFrames, size_t backCropFrames) {
    std::vector<float> waveValues = readNativeOnnxTensorValues<float>(waveTensor, 1);
    size_t frontCropSamples = frontCropFrames * nativeOnnxSamplesPerFrame;
    size_t backCropSamples = backCropFrames * nativeOnnxSamplesPerFrame;
    if (waveValues.size() < frontCropSamples + backCropSamples) {
        throw std::runtime_error("decoder wave crop が不正です");
    }
    waveValues = std::vector<float>(waveValues.begin() + static_cast<std::vector<float>::difference_type>(frontCropSamples), waveValues.end() - static_cast<std::vector<float>::difference_type>(backCropSamples));
    return waveValues;
}

static std::vector<float> createNativeOnnxWaveValuesWithoutDecoderPadding(const NativeOnnxTraceInput &waveTensor) {
    return createNativeOnnxWaveValuesWithoutDecoderFramePadding(waveTensor, nativeOnnxDecoderPaddingFrames, nativeOnnxDecoderPaddingFrames);
}

NativeOnnxPcm::NativeOnnxPcm(const NativeOnnxAudioQuerySettings &Settings) : Settings(Settings) {
    if (Settings.outputSamplingRate == 0 || !std::isfinite(Settings.volumeScale) || Settings.volumeScale < 0) {
        throw std::invalid_argument("PCM 出力設定が不正です");
    }
    if (Settings.outputSamplingRate != nativeOnnxDefaultSamplingRate) {
        soxr_error_t Error = nullptr;
        Resampler = soxr_create(nativeOnnxDefaultSamplingRate, Settings.outputSamplingRate, 1, &Error, nullptr, nullptr, nullptr);
        if (Error) {
            std::string Message = Error;
            soxr_delete(Resampler);
            throw std::runtime_error("resampler: " + Message);
        }
    }
}

NativeOnnxPcm::~NativeOnnxPcm() {
    soxr_delete(Resampler);
}

std::vector<uint8_t> NativeOnnxPcm::Convert(const std::vector<float> &Wave, bool Final) {
    std::vector<float> Output;
    const std::vector<float> *Values = &Wave;
    if (Resampler) {
        double Count = std::ceil(Wave.size() * static_cast<double>(Settings.outputSamplingRate) / nativeOnnxDefaultSamplingRate + soxr_delay(Resampler)) + 1;
        uint32_t Channels = Settings.outputStereo ? 2 : 1;
        if (Count > (std::numeric_limits<uint32_t>::max() - 44) / (Channels * sizeof(int16_t))) {
            throw std::invalid_argument("PCM 出力が WAV のサイズ上限を超えます");
        }
        Output.resize(static_cast<size_t>(Count));
        size_t Used = 0;
        size_t Written = 0;
        if (!Wave.empty()) {
            soxr_error_t Error = soxr_process(Resampler, Wave.data(), Wave.size(), &Used, Output.data(), Output.size(), &Written);
            if (Error || Used != Wave.size()) {
                throw std::runtime_error(std::string("resampler: ") + (Error ? Error : "入力を処理しきれません"));
            }
        }
        if (Final) {
            size_t Tail = 0;
            soxr_error_t Error = soxr_process(Resampler, nullptr, 0, nullptr, Output.data() + Written, Output.size() - Written, &Tail);
            if (Error) {
                throw std::runtime_error(std::string("resampler: ") + Error);
            }
            Written += Tail;
        }
        Output.resize(Written);
        Values = &Output;
    }
    uint32_t Channels = Settings.outputStereo ? 2 : 1;
    std::vector<uint8_t> Bytes;
    Bytes.reserve(Values->size() * Channels * sizeof(int16_t));
    for (float Value : *Values) {
        float Scaled = Value * Settings.volumeScale;
        if (!std::isfinite(Scaled)) {
            throw std::runtime_error("PCM に有限でない値があります");
        }
        int16_t Sample = static_cast<int16_t>(std::clamp(Scaled, -1.0f, 1.0f) * 32767.0f);
        for (uint32_t Channel = 0; Channel < Channels; Channel++) {
            Bytes.push_back(static_cast<uint8_t>(Sample & 0xff));
            Bytes.push_back(static_cast<uint8_t>((static_cast<uint16_t>(Sample) >> 8) & 0xff));
        }
    }
    return Bytes;
}

std::vector<uint8_t> createNativeOnnxPcmBytes(const std::vector<float> &Wave, const NativeOnnxAudioQuerySettings &Settings) {
    return NativeOnnxPcm(Settings).Convert(Wave, true);
}

static std::vector<uint8_t> createNativeOnnxWavBytes(const NativeOnnxTraceInput &waveTensor, const NativeOnnxAudioQuerySettings &audioQuerySettings) {
    std::vector<float> waveValues = createNativeOnnxWaveValuesWithoutDecoderPadding(waveTensor);
    std::vector<uint8_t> pcmBytes = createNativeOnnxPcmBytes(waveValues, audioQuerySettings);
    uint16_t channels = audioQuerySettings.outputStereo ? 2 : 1;
    std::vector<uint8_t> wavBytes = createPcmWaveHeader(audioQuerySettings.outputSamplingRate, channels, 16, pcmBytes.size());
    wavBytes.insert(wavBytes.end(), pcmBytes.begin(), pcmBytes.end());
    return wavBytes;
}

static size_t getNativeOnnxPaddedDecoderFrameCount(const NativeOnnxTraceInput &f0Tensor) {
    std::vector<float> paddedF0Values = readNativeOnnxTensorValues<float>(f0Tensor, 1);
    if (!f0Tensor.dimensions.empty() && f0Tensor.dimensions[0] >= 0 && static_cast<size_t>(f0Tensor.dimensions[0]) != paddedF0Values.size()) {
        throw std::runtime_error("decoder f0 frame 数が一致しません");
    }
    return paddedF0Values.size();
}

static std::vector<float> sliceNativeOnnxFrameValues(const std::vector<float> &frameValues, size_t frameWidth, size_t startFrame, size_t endFrame) {
    if (frameWidth == 0 || startFrame > endFrame || endFrame * frameWidth > frameValues.size()) {
        throw std::runtime_error("decoder chunk frame が不正です");
    }
    auto sliceStartIterator = frameValues.begin() + static_cast<std::vector<float>::difference_type>(startFrame * frameWidth);
    auto sliceEndIterator = frameValues.begin() + static_cast<std::vector<float>::difference_type>(endFrame * frameWidth);
    return std::vector<float>(sliceStartIterator, sliceEndIterator);
}

static NativeOnnxDecoderChunkInputSet createNativeOnnxDecoderChunkInputs(const std::vector<NativeOnnxTraceInput> &decoderInputs, size_t coreStartFrame, size_t coreEndFrame, size_t contextFrames) {
    const NativeOnnxTraceInput &f0Tensor = requireNativeOnnxTensor(decoderInputs, "f0");
    const NativeOnnxTraceInput &phonemeTensor = requireNativeOnnxTensor(decoderInputs, "phoneme");
    NativeOnnxTraceInput speakerIdTensor = requireNativeOnnxTensor(decoderInputs, "speaker_id");
    std::vector<float> paddedF0Values = readNativeOnnxTensorValues<float>(f0Tensor, 1);
    std::vector<float> paddedPhonemeValues = readNativeOnnxTensorValues<float>(phonemeTensor, 1);
    size_t paddedFrameCount = paddedF0Values.size();
    size_t paddingFrameCount = nativeOnnxDecoderPaddingFrames * 2;
    if (paddedFrameCount < paddingFrameCount || paddedPhonemeValues.size() != paddedFrameCount * static_cast<size_t>(nativeOnnxPhonemeSize)) {
        throw std::runtime_error("decoder 入力 frame 数が不正です");
    }
    size_t coreFrameCount = paddedFrameCount - paddingFrameCount;
    if (coreStartFrame > coreEndFrame || coreEndFrame > coreFrameCount) {
        throw std::runtime_error("decoder chunk 範囲が不正です");
    }
    size_t targetStartFrame = nativeOnnxDecoderPaddingFrames + coreStartFrame;
    size_t targetEndFrame = nativeOnnxDecoderPaddingFrames + coreEndFrame;
    size_t sliceStartFrame = coreStartFrame > contextFrames ? coreStartFrame - contextFrames : 0;
    size_t sliceEndFrame = std::min(paddedFrameCount, coreEndFrame + paddingFrameCount + contextFrames);
    size_t sliceFrameCount = sliceEndFrame - sliceStartFrame;
    std::vector<float> f0Values = sliceNativeOnnxFrameValues(paddedF0Values, 1, sliceStartFrame, sliceEndFrame);
    std::vector<float> phonemeValues = sliceNativeOnnxFrameValues(paddedPhonemeValues, static_cast<size_t>(nativeOnnxPhonemeSize), sliceStartFrame, sliceEndFrame);
    NativeOnnxDecoderChunkInputSet inputSet;
    inputSet.tensors = {
        createNativeOnnxFloatTensor("f0", {static_cast<int64_t>(sliceFrameCount), 1}, f0Values),
        createNativeOnnxFloatTensor("phoneme", {static_cast<int64_t>(sliceFrameCount), nativeOnnxPhonemeSize}, phonemeValues),
        speakerIdTensor,
    };
    inputSet.frontCropFrames = targetStartFrame - sliceStartFrame;
    inputSet.backCropFrames = sliceEndFrame - targetEndFrame;
    return inputSet;
}

static NativeOnnxPcmStreamInfo createNativeOnnxPcmStreamInfo(const NativeOnnxAudioQuerySettings &audioQuerySettings, size_t coreFrameCount) {
    NativeOnnxPcmStreamInfo streamInfo;
    streamInfo.sampleRate = audioQuerySettings.outputSamplingRate;
    streamInfo.channels = audioQuerySettings.outputStereo ? 2 : 1;
    streamInfo.bitsPerSample = 16;
    double Samples = std::round(static_cast<double>(coreFrameCount) * nativeOnnxSamplesPerFrame * streamInfo.sampleRate / nativeOnnxDefaultSamplingRate);
    double Bytes = Samples * streamInfo.channels * sizeof(int16_t);
    if (Bytes > std::numeric_limits<uint32_t>::max() - 44) {
        throw std::invalid_argument("PCM 出力が WAV のサイズ上限を超えます");
    }
    streamInfo.pcmBytes = static_cast<uintptr_t>(Bytes);
    return streamInfo;
}

std::string createNativeOnnxVvmChainText(const fs::path &onnxruntimeLibraryPath, const std::vector<VvmArchiveSummary> &archiveSummaries, const fs::path &inputDirectory, const fs::path &audioQueryPath, uint32_t styleId, uint16_t cpuThreadCount) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::vector<ModelAssetRecord> modelAssets = collectModelAssets(archiveSummaries);
        const ModelAssetRecord &decodeAsset = requireNativeOnnxModelAsset(modelAssets, "models/d.bin");
        const ModelAssetRecord &durationAsset = requireNativeOnnxModelAsset(modelAssets, "models/pd.bin");
        const ModelAssetRecord &intonationAsset = requireNativeOnnxModelAsset(modelAssets, "models/pi.bin");
        std::vector<NativeOnnxTraceInput> traceInputs = loadNativeOnnxTraceInputs(inputDirectory);
        std::vector<NativeOnnxTraceInput> traceOutputs = loadNativeOnnxTraceOutputs(inputDirectory);
        fs::path resolvedAudioQueryPath = resolveNativeOnnxAudioQueryPath(inputDirectory, audioQueryPath);
        std::string audioQueryText = readNativeOnnxTextFile(resolvedAudioQueryPath);
        NativeOnnxAudioQuerySettings audioQuerySettings = parseNativeOnnxAudioQuerySettings(audioQueryText);
        int64_t innerVoiceId = resolveNativeOnnxInnerVoiceId(durationAsset, "talk", styleId);
        std::vector<NativeOnnxTraceInput> frontendInputs = createNativeOnnxFrontendInputs(audioQueryText, innerVoiceId);
        std::vector<uint8_t> durationBytes = extractNativeOnnxModelAssetBytes(durationAsset);
        std::vector<uint8_t> intonationBytes = extractNativeOnnxModelAssetBytes(intonationAsset);
        std::vector<uint8_t> decodeBytes = extractNativeOnnxModelAssetBytes(decodeAsset);
        std::vector<NativeOnnxTraceInput> durationOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, nullptr, durationAsset, durationBytes, frontendInputs, cpuThreadCount, true);
        const NativeOnnxTraceInput &durationTensor = requireNativeOnnxTensor(durationOutputs, "phoneme_length");
        std::vector<NativeOnnxTraceInput> intonationOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, nullptr, intonationAsset, intonationBytes, frontendInputs, cpuThreadCount, true);
        const NativeOnnxTraceInput &intonationTensor = requireNativeOnnxTensor(intonationOutputs, "f0_list");
        std::vector<NativeOnnxTraceInput> decoderInputs = createNativeOnnxDecoderInputs(frontendInputs, durationOutputs, intonationOutputs, audioQuerySettings, true);
        const NativeOnnxTraceInput &generatedF0Tensor = requireNativeOnnxTensor(decoderInputs, "f0");
        const NativeOnnxTraceInput &generatedPhonemeTensor = requireNativeOnnxTensor(decoderInputs, "phoneme");
        std::vector<NativeOnnxTraceInput> decodeOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, nullptr, decodeAsset, decodeBytes, decoderInputs, cpuThreadCount, true);
        const NativeOnnxTraceInput &waveTensor = requireNativeOnnxTensor(decodeOutputs, "wave");
        std::ostringstream chainStream;
        chainStream << "field\tvalue\n";
        chainStream << "onnxruntime\t" << onnxruntimeLibraryPath.string() << "\n";
        chainStream << "api_version\t" << ortApiVersion << "\n";
        chainStream << "ort_version\t" << getNativeOnnxVersion(nativeOnnxApi) << "\n";
        chainStream << "cpu_threads\t" << cpuThreadCount << "\n";
        chainStream << "audio_query\t" << resolvedAudioQueryPath.string() << "\n";
        chainStream << "style_id\t" << styleId << "\n";
        if (!inputDirectory.empty()) {
            chainStream << "trace_inputs\t" << inputDirectory.string() << "\n";
        }
        chainStream << "trace_input_count\t" << traceInputs.size() << "\n";
        chainStream << "trace_output_count\t" << traceOutputs.size() << "\n";
        chainStream << "speed_scale\t" << audioQuerySettings.speedScale << "\n";
        chainStream << "pre_phoneme_length\t" << audioQuerySettings.prePhonemeLength << "\n";
        chainStream << "post_phoneme_length\t" << audioQuerySettings.postPhonemeLength << "\n";
        chainStream << "duration_asset\t" << durationAsset.archivePath.filename().string() << ":" << durationAsset.entryName << "\n";
        chainStream << "intonation_asset\t" << intonationAsset.archivePath.filename().string() << ":" << intonationAsset.entryName << "\n";
        chainStream << "decode_asset\t" << decodeAsset.archivePath.filename().string() << ":" << decodeAsset.entryName << "\n";
        for (const NativeOnnxTraceInput &frontendInput : frontendInputs) {
            chainStream << "frontend_" << frontendInput.name << "\t" << compareNativeOnnxTensors(frontendInput, findNativeOnnxTraceTensor(traceInputs, frontendInput.name)) << "\n";
        }
        chainStream << "duration_run\tok\n";
        chainStream << "duration_output\t" << compareNativeOnnxTensors(durationTensor, findNativeOnnxTraceTensor(traceOutputs, "phoneme_length")) << "\n";
        chainStream << "intonation_run\tok\n";
        chainStream << "intonation_output\t" << compareNativeOnnxTensors(intonationTensor, findNativeOnnxTraceTensor(traceOutputs, "f0_list")) << "\n";
        chainStream << "decoder_generated_f0\t" << compareNativeOnnxTensors(generatedF0Tensor, findNativeOnnxTraceTensor(traceInputs, "f0")) << "\n";
        chainStream << "decoder_generated_phoneme\t" << compareNativeOnnxTensors(generatedPhonemeTensor, findNativeOnnxTraceTensor(traceInputs, "phoneme")) << "\n";
        chainStream << "decoder_frames\t" << (generatedF0Tensor.dimensions.empty() ? 0 : generatedF0Tensor.dimensions[0]) << "\n";
        chainStream << "decode_run\tok\n";
        chainStream << "decode_output\t" << compareNativeOnnxTensors(waveTensor, findNativeOnnxTraceTensor(traceOutputs, "wave")) << "\n";
        closeNativeOnnxApi(nativeOnnxApi);
        return chainStream.str();
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string replaceNativeOnnxModelAssetsMoraData(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &accentPhrasesJson, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::string replacedJson = replaceNativeOnnxMoraDataWithApi(nativeOnnxApi, nullptr, modelAssets, accentPhrasesJson, styleId, cpuThreadCount, true, true, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return replacedJson;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string replaceNativeOnnxModelAssetsPhonemeLength(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &accentPhrasesJson, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::string replacedJson = replaceNativeOnnxMoraDataWithApi(nativeOnnxApi, nullptr, modelAssets, accentPhrasesJson, styleId, cpuThreadCount, true, false, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return replacedJson;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string replaceNativeOnnxModelAssetsMoraPitch(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &accentPhrasesJson, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        std::string replacedJson = replaceNativeOnnxMoraDataWithApi(nativeOnnxApi, nullptr, modelAssets, accentPhrasesJson, styleId, cpuThreadCount, false, true, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return replacedJson;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string replaceNativeOnnxModelAssetsAudioQueryMoraData(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &audioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    std::string accentPhrasesJson = extractJsonArrayField(audioQueryText, "accent_phrases");
    if (accentPhrasesJson.empty()) {
        throw std::runtime_error("accent_phrases がありません");
    }
    std::string replacedAccentPhrasesJson = replaceNativeOnnxModelAssetsMoraData(onnxruntimeLibraryPath, modelAssets, accentPhrasesJson, styleId, cpuThreadCount, shouldUseVvBinConfig);
    return createNativeOnnxAudioQueryTextWithAccentPhrases(audioQueryText, replacedAccentPhrasesJson);
}

std::vector<uint8_t> synthesizeNativeOnnxModelAssetsAudioQuery(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &audioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxAudioQuerySettings audioQuerySettings = parseNativeOnnxAudioQuerySettings(audioQueryText);
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        NativeOnnxTraceInput waveTensor = runNativeOnnxModelAssetChainWave(nativeOnnxApi, nullptr, modelAssets, audioQueryText, styleId, cpuThreadCount, shouldUseVvBinConfig);
        std::vector<uint8_t> wavBytes = createNativeOnnxWavBytes(waveTensor, audioQuerySettings);
        closeNativeOnnxApi(nativeOnnxApi);
        return wavBytes;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

void streamNativeOnnxModelAssetsAudioQueryPcm(const fs::path &onnxruntimeLibraryPath, const std::vector<ModelAssetRecord> &modelAssets, const std::string &audioQueryText, uint32_t styleId, uint16_t cpuThreadCount, size_t chunkFrames, const std::function<void(const NativeOnnxPcmStreamInfo &)> &startStream, const std::function<void(const uint8_t *, size_t)> &writeChunk, bool shouldUseVvBinConfig) {
    NativeOnnxAudioQuerySettings audioQuerySettings = parseNativeOnnxAudioQuerySettings(audioQueryText);
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        const ModelAssetRecord &decodeAsset = requireNativeOnnxModelAsset(modelAssets, "models/d.bin");
        std::vector<uint8_t> decodeBytes = extractNativeOnnxModelAssetBytes(decodeAsset);
        std::vector<NativeOnnxTraceInput> decoderInputs = createNativeOnnxModelAssetDecoderInputs(nativeOnnxApi, nullptr, modelAssets, audioQueryText, styleId, cpuThreadCount, audioQuerySettings, shouldUseVvBinConfig);
        size_t paddedFrameCount = getNativeOnnxPaddedDecoderFrameCount(requireNativeOnnxTensor(decoderInputs, "f0"));
        size_t paddingFrameCount = nativeOnnxDecoderPaddingFrames * 2;
        size_t coreFrameCount = paddedFrameCount > paddingFrameCount ? paddedFrameCount - paddingFrameCount : 0;
        NativeOnnxPcmStreamInfo streamInfo = createNativeOnnxPcmStreamInfo(audioQuerySettings, coreFrameCount);
        NativeOnnxPcm Pcm(audioQuerySettings);
        startStream(streamInfo);
        size_t safeChunkFrames = std::max({static_cast<size_t>(1), chunkFrames, nativeOnnxDecoderMinimumChunkFrames});
        size_t contextFrames = safeChunkFrames;
        for (size_t coreStartFrame = 0; coreStartFrame < coreFrameCount;) {
            size_t coreEndFrame = std::min(coreStartFrame + safeChunkFrames, coreFrameCount);
            if (coreEndFrame < coreFrameCount && coreFrameCount - coreEndFrame < nativeOnnxDecoderMinimumChunkFrames) {
                coreEndFrame = coreFrameCount;
            }
            NativeOnnxDecoderChunkInputSet chunkInputSet = createNativeOnnxDecoderChunkInputs(decoderInputs, coreStartFrame, coreEndFrame, contextFrames);
            std::vector<NativeOnnxTraceInput> decodeOutputs = runNativeOnnxModelAssetBytes(nativeOnnxApi, nullptr, decodeAsset, decodeBytes, chunkInputSet.tensors, cpuThreadCount, shouldUseVvBinConfig);
            std::vector<float> waveValues = createNativeOnnxWaveValuesWithoutDecoderFramePadding(requireNativeOnnxTensor(decodeOutputs, "wave"), chunkInputSet.frontCropFrames, chunkInputSet.backCropFrames);
            std::vector<uint8_t> pcmBytes = Pcm.Convert(waveValues, coreEndFrame == coreFrameCount);
            if (!pcmBytes.empty()) {
                writeChunk(pcmBytes.data(), pcmBytes.size());
            }
            coreStartFrame = coreEndFrame;
        }
        closeNativeOnnxApi(nativeOnnxApi);
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}
std::string replaceNativeOnnxModelAssetsPhonemeLength(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &accentPhrasesJson, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        std::string replacedJson = replaceNativeOnnxMoraDataWithApi(nativeOnnxApi, &runtimeState, modelAssets, accentPhrasesJson, styleId, cpuThreadCount, true, false, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return replacedJson;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string replaceNativeOnnxModelAssetsMoraPitch(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &accentPhrasesJson, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        std::string replacedJson = replaceNativeOnnxMoraDataWithApi(nativeOnnxApi, &runtimeState, modelAssets, accentPhrasesJson, styleId, cpuThreadCount, false, true, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return replacedJson;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

std::string replaceNativeOnnxModelAssetsAudioQueryMoraData(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &audioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    std::string accentPhrasesJson = extractJsonArrayField(audioQueryText, "accent_phrases");
    if (accentPhrasesJson.empty()) {
        throw std::runtime_error("accent_phrases がありません");
    }
    std::string replacedAccentPhrasesJson = replaceNativeOnnxModelAssetsMoraData(runtimeState, modelAssets, accentPhrasesJson, styleId, cpuThreadCount, shouldUseVvBinConfig);
    return createNativeOnnxAudioQueryTextWithAccentPhrases(audioQueryText, replacedAccentPhrasesJson);
}

std::vector<uint8_t> synthesizeNativeOnnxModelAssetsAudioQuery(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &audioQueryText, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxAudioQuerySettings audioQuerySettings = parseNativeOnnxAudioQuerySettings(audioQueryText);
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        NativeOnnxTraceInput waveTensor = runNativeOnnxModelAssetChainWave(nativeOnnxApi, &runtimeState, modelAssets, audioQueryText, styleId, cpuThreadCount, shouldUseVvBinConfig);
        std::vector<uint8_t> wavBytes = createNativeOnnxWavBytes(waveTensor, audioQuerySettings);
        closeNativeOnnxApi(nativeOnnxApi);
        return wavBytes;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

void streamNativeOnnxModelAssetsAudioQueryPcm(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &audioQueryText, uint32_t styleId, uint16_t cpuThreadCount, size_t chunkFrames, const std::function<void(const NativeOnnxPcmStreamInfo &)> &startStream, const std::function<void(const uint8_t *, size_t)> &writeChunk, bool shouldUseVvBinConfig) {
    NativeOnnxAudioQuerySettings audioQuerySettings = parseNativeOnnxAudioQuerySettings(audioQueryText);
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        const ModelAssetRecord &decodeAsset = requireNativeOnnxModelAsset(modelAssets, "models/d.bin");
        std::vector<uint8_t> decodeBytes;
        if (shouldUseVvBinConfig) {
            decodeBytes = extractNativeOnnxModelAssetBytes(decodeAsset);
        }
        std::vector<NativeOnnxTraceInput> decoderInputs = createNativeOnnxModelAssetDecoderInputs(nativeOnnxApi, &runtimeState, modelAssets, audioQueryText, styleId, cpuThreadCount, audioQuerySettings, shouldUseVvBinConfig);
        size_t paddedFrameCount = getNativeOnnxPaddedDecoderFrameCount(requireNativeOnnxTensor(decoderInputs, "f0"));
        size_t paddingFrameCount = nativeOnnxDecoderPaddingFrames * 2;
        size_t coreFrameCount = paddedFrameCount > paddingFrameCount ? paddedFrameCount - paddingFrameCount : 0;
        NativeOnnxPcmStreamInfo streamInfo = createNativeOnnxPcmStreamInfo(audioQuerySettings, coreFrameCount);
        NativeOnnxPcm Pcm(audioQuerySettings);
        startStream(streamInfo);
        size_t safeChunkFrames = std::max({static_cast<size_t>(1), chunkFrames, nativeOnnxDecoderMinimumChunkFrames});
        size_t contextFrames = safeChunkFrames;
        for (size_t coreStartFrame = 0; coreStartFrame < coreFrameCount;) {
            size_t coreEndFrame = std::min(coreStartFrame + safeChunkFrames, coreFrameCount);
            if (coreEndFrame < coreFrameCount && coreFrameCount - coreEndFrame < nativeOnnxDecoderMinimumChunkFrames) {
                coreEndFrame = coreFrameCount;
            }
            NativeOnnxDecoderChunkInputSet chunkInputSet = createNativeOnnxDecoderChunkInputs(decoderInputs, coreStartFrame, coreEndFrame, contextFrames);
            std::vector<NativeOnnxTraceInput> decodeOutputs = shouldUseVvBinConfig
                ? runNativeOnnxModelAssetBytes(nativeOnnxApi, &runtimeState, decodeAsset, decodeBytes, chunkInputSet.tensors, cpuThreadCount, true)
                : runNativeOnnxModelAssetBytes(nativeOnnxApi, &runtimeState, decodeAsset, chunkInputSet.tensors, cpuThreadCount, false);
            std::vector<float> waveValues = createNativeOnnxWaveValuesWithoutDecoderFramePadding(requireNativeOnnxTensor(decodeOutputs, "wave"), chunkInputSet.frontCropFrames, chunkInputSet.backCropFrames);
            std::vector<uint8_t> pcmBytes = Pcm.Convert(waveValues, coreEndFrame == coreFrameCount);
            if (!pcmBytes.empty()) {
                writeChunk(pcmBytes.data(), pcmBytes.size());
            }
            coreStartFrame = coreEndFrame;
        }
        closeNativeOnnxApi(nativeOnnxApi);
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}
std::string replaceNativeOnnxModelAssetsMoraData(const NativeOnnxRuntimeState &runtimeState, const std::vector<ModelAssetRecord> &modelAssets, const std::string &accentPhrasesJson, uint32_t styleId, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(runtimeState.libraryPath);
    try {
        std::string replacedJson = replaceNativeOnnxMoraDataWithApi(nativeOnnxApi, &runtimeState, modelAssets, accentPhrasesJson, styleId, cpuThreadCount, true, true, shouldUseVvBinConfig);
        closeNativeOnnxApi(nativeOnnxApi);
        return replacedJson;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}
