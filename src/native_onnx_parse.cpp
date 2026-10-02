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

int64_t parseNativeOnnxPhonemeCode(const std::string &phonemeText) {
    if (phonemeText == "pau") return 0;
    if (phonemeText == "A") return 1;
    if (phonemeText == "E") return 2;
    if (phonemeText == "I") return 3;
    if (phonemeText == "N") return 4;
    if (phonemeText == "O") return 5;
    if (phonemeText == "U") return 6;
    if (phonemeText == "a") return 7;
    if (phonemeText == "b") return 8;
    if (phonemeText == "by") return 9;
    if (phonemeText == "ch") return 10;
    if (phonemeText == "cl") return 11;
    if (phonemeText == "d") return 12;
    if (phonemeText == "dy") return 13;
    if (phonemeText == "e") return 14;
    if (phonemeText == "f") return 15;
    if (phonemeText == "g") return 16;
    if (phonemeText == "gw") return 17;
    if (phonemeText == "gy") return 18;
    if (phonemeText == "h") return 19;
    if (phonemeText == "hy") return 20;
    if (phonemeText == "i") return 21;
    if (phonemeText == "j") return 22;
    if (phonemeText == "k") return 23;
    if (phonemeText == "kw") return 24;
    if (phonemeText == "ky") return 25;
    if (phonemeText == "m") return 26;
    if (phonemeText == "my") return 27;
    if (phonemeText == "n") return 28;
    if (phonemeText == "ny") return 29;
    if (phonemeText == "o") return 30;
    if (phonemeText == "p") return 31;
    if (phonemeText == "py") return 32;
    if (phonemeText == "r") return 33;
    if (phonemeText == "ry") return 34;
    if (phonemeText == "s") return 35;
    if (phonemeText == "sh") return 36;
    if (phonemeText == "t") return 37;
    if (phonemeText == "ts") return 38;
    if (phonemeText == "ty") return 39;
    if (phonemeText == "u") return 40;
    if (phonemeText == "v") return 41;
    if (phonemeText == "w") return 42;
    if (phonemeText == "y") return 43;
    if (phonemeText == "z") return 44;
    throw std::invalid_argument("未対応の phoneme です: " + phonemeText);
}

static bool isNativeOnnxJsonNumberStart(char character) {
    return character == '-' || character == '+' || std::isdigit(static_cast<unsigned char>(character));
}

static size_t skipNativeOnnxJsonSpaces(const std::string &jsonText, size_t position) {
    while (position < jsonText.size() && std::isspace(static_cast<unsigned char>(jsonText[position]))) {
        position++;
    }
    return position;
}

static bool isNativeOnnxJsonNullAt(const std::string &jsonText, size_t position) {
    return jsonText.compare(position, 4, "null") == 0;
}

static size_t findNativeOnnxJsonNumberEnd(const std::string &jsonText, size_t position) {
    size_t endPosition = position;
    while (endPosition < jsonText.size()) {
        char character = jsonText[endPosition];
        if (!(std::isdigit(static_cast<unsigned char>(character)) || character == '-' || character == '+' || character == '.' || character == 'e' || character == 'E')) {
            break;
        }
        endPosition++;
    }
    return endPosition;
}

static int64_t parseNativeOnnxJsonInt64At(const std::string &jsonText, size_t position, const std::string &fieldName) {
    position = skipNativeOnnxJsonSpaces(jsonText, position);
    if (position >= jsonText.size() || !isNativeOnnxJsonNumberStart(jsonText[position])) {
        throw std::invalid_argument(fieldName + " が数値ではありません");
    }
    size_t endPosition = findNativeOnnxJsonNumberEnd(jsonText, position);
    int64_t Value = 0;
    auto Result = std::from_chars(jsonText.data() + position, jsonText.data() + endPosition, Value);
    if (Result.ec != std::errc() || Result.ptr != jsonText.data() + endPosition) {
        throw std::invalid_argument(fieldName + " が整数ではありません、または範囲外です");
    }
    return Value;
}

static uint64_t parseNativeOnnxJsonUint64At(const std::string &jsonText, size_t position, const std::string &fieldName) {
    position = skipNativeOnnxJsonSpaces(jsonText, position);
    if (position >= jsonText.size() || jsonText[position] == '-' || !isNativeOnnxJsonNumberStart(jsonText[position])) {
        throw std::invalid_argument(fieldName + " が非負整数ではありません");
    }
    int64_t Value = parseNativeOnnxJsonInt64At(jsonText, position, fieldName);
    if (Value < 0) {
        throw std::invalid_argument(fieldName + " が負数です");
    }
    return static_cast<uint64_t>(Value);
}

static float parseNativeOnnxJsonFloatAt(const std::string &jsonText, size_t position, const std::string &fieldName) {
    position = skipNativeOnnxJsonSpaces(jsonText, position);
    if (position >= jsonText.size() || !isNativeOnnxJsonNumberStart(jsonText[position])) {
        throw std::invalid_argument(fieldName + " が数値ではありません");
    }
    size_t endPosition = findNativeOnnxJsonNumberEnd(jsonText, position);
    size_t End = 0;
    float parsedNumber = std::stof(jsonText.substr(position, endPosition - position), &End);
    if (End != endPosition - position || !std::isfinite(parsedNumber)) {
        throw std::invalid_argument(fieldName + " が有限値ではありません");
    }
    return parsedNumber;
}

static uint64_t requireNativeOnnxJsonUint64Field(const std::string &jsonText, const std::string &fieldName) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos) {
        throw std::invalid_argument(fieldName + " がありません");
    }
    return parseNativeOnnxJsonUint64At(jsonText, valuePosition, fieldName);
}

static bool extractNativeOnnxJsonOptionalInt64Field(const std::string &jsonText, const std::string &fieldName, int64_t &numberValue) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos) {
        return false;
    }
    valuePosition = skipNativeOnnxJsonSpaces(jsonText, valuePosition);
    if (isNativeOnnxJsonNullAt(jsonText, valuePosition)) {
        return false;
    }
    numberValue = parseNativeOnnxJsonInt64At(jsonText, valuePosition, fieldName);
    return true;
}

static std::string requireNativeOnnxJsonStringField(const std::string &jsonText, const std::string &fieldName) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size() || jsonText[valuePosition] != '"') {
        throw std::invalid_argument(fieldName + " がありません");
    }
    return decodeJsonString(jsonText, valuePosition);
}

static bool extractNativeOnnxJsonOptionalStringField(const std::string &jsonText, const std::string &fieldName, std::string &stringValue) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos) {
        return false;
    }
    valuePosition = skipNativeOnnxJsonSpaces(jsonText, valuePosition);
    if (isNativeOnnxJsonNullAt(jsonText, valuePosition)) {
        return false;
    }
    if (valuePosition >= jsonText.size() || jsonText[valuePosition] != '"') {
        throw std::invalid_argument(fieldName + " が文字列ではありません");
    }
    stringValue = decodeJsonString(jsonText, valuePosition);
    return true;
}

static std::vector<float> parseNativeOnnxFloatArrayText(const std::string &arrayText, const std::string &fieldName) {
    size_t position = skipNativeOnnxJsonSpaces(arrayText, 0);
    if (position >= arrayText.size() || arrayText[position] != '[') {
        throw std::invalid_argument(fieldName + " が配列ではありません");
    }
    position++;
    std::vector<float> numberValues;
    bool isFirstValue = true;
    while (true) {
        position = skipNativeOnnxJsonSpaces(arrayText, position);
        if (position >= arrayText.size()) {
            throw std::invalid_argument(fieldName + " 配列が閉じていません");
        }
        if (arrayText[position] == ']') {
            return numberValues;
        }
        if (!isFirstValue) {
            if (arrayText[position] != ',') {
                throw std::invalid_argument(fieldName + " 配列の区切りが不正です");
            }
            position = skipNativeOnnxJsonSpaces(arrayText, position + 1);
        }
        numberValues.push_back(parseNativeOnnxJsonFloatAt(arrayText, position, fieldName));
        position = findNativeOnnxJsonNumberEnd(arrayText, position);
        isFirstValue = false;
    }
}

static std::vector<std::string> ParseObjects(const std::string &Json) {
    std::vector<std::string> Objects;
    size_t Position = skipNativeOnnxJsonSpaces(Json, 1);
    if (Position < Json.size() && Json[Position] == ']') {
        return Objects;
    }
    while (Position < Json.size()) {
        if (Json[Position] != '{') {
            throw std::invalid_argument("歌唱配列には object が必要です");
        }
        size_t End = findJsonMatchingToken(Json, Position, '{', '}');
        if (End == std::string::npos) {
            throw std::invalid_argument("歌唱 object が閉じていません");
        }
        Objects.push_back(Json.substr(Position, End - Position + 1));
        Position = skipNativeOnnxJsonSpaces(Json, End + 1);
        if (Position < Json.size() && Json[Position] == ']') {
            return Objects;
        }
        if (Position >= Json.size() || Json[Position] != ',') {
            break;
        }
        Position = skipNativeOnnxJsonSpaces(Json, Position + 1);
    }
    throw std::invalid_argument("歌唱配列が不正です");
}

std::vector<NativeOnnxScoreNote> parseNativeOnnxScore(const std::string &scoreText) {
    std::string notesJson = extractJsonArrayField(scoreText, "notes");
    if (notesJson.empty()) {
        throw std::invalid_argument("notes がありません");
    }
    std::vector<NativeOnnxScoreNote> scoreNotes;
    for (const std::string &noteObject : ParseObjects(notesJson)) {
        NativeOnnxScoreNote scoreNote;
        scoreNote.hasNoteId = extractNativeOnnxJsonOptionalStringField(noteObject, "id", scoreNote.noteId);
        scoreNote.hasKey = extractNativeOnnxJsonOptionalInt64Field(noteObject, "key", scoreNote.key);
        if (scoreNote.hasKey && (scoreNote.key < 0 || scoreNote.key > 127)) {
            throw std::invalid_argument("key が範囲外です");
        }
        scoreNote.frameLength = requireNativeOnnxJsonUint64Field(noteObject, "frame_length");
        scoreNote.lyric = requireNativeOnnxJsonStringField(noteObject, "lyric");
        if (!scoreNote.hasKey && !scoreNote.lyric.empty()) {
            throw std::invalid_argument("休符の lyric は空文字が必要です");
        }
        if (scoreNote.hasKey && scoreNote.lyric.empty()) {
            throw std::invalid_argument("音符の lyric がありません");
        }
        if (scoreNote.hasKey) {
            scoreNote.mora = createNativeAudioQueryMoraFromText(scoreNote.lyric);
            scoreNote.hasMora = true;
        }
        scoreNotes.push_back(std::move(scoreNote));
    }
    if (scoreNotes.empty()) {
        throw std::invalid_argument("notes が空です");
    }
    if (scoreNotes.front().hasKey) {
        throw std::invalid_argument("score の先頭は休符が必要です");
    }
    return scoreNotes;
}

static std::vector<NativeOnnxFramePhoneme> parseNativeOnnxFramePhonemes(const std::string &phonemesJson) {
    std::vector<NativeOnnxFramePhoneme> framePhonemes;
    for (const std::string &phonemeObject : ParseObjects(phonemesJson)) {
        NativeOnnxFramePhoneme framePhoneme;
        framePhoneme.phoneme = requireNativeOnnxJsonStringField(phonemeObject, "phoneme");
        framePhoneme.frameLength = requireNativeOnnxJsonUint64Field(phonemeObject, "frame_length");
        framePhoneme.hasNoteId = extractNativeOnnxJsonOptionalStringField(phonemeObject, "note_id", framePhoneme.noteId);
        framePhonemes.push_back(std::move(framePhoneme));
    }
    return framePhonemes;
}

NativeOnnxFrameAudioQuery parseNativeOnnxFrameAudioQuery(const std::string &frameAudioQueryText) {
    NativeOnnxFrameAudioQuery frameAudioQuery;
    std::string f0Json = extractJsonArrayField(frameAudioQueryText, "f0");
    std::string volumeJson = extractJsonArrayField(frameAudioQueryText, "volume");
    std::string phonemesJson = extractJsonArrayField(frameAudioQueryText, "phonemes");
    if (f0Json.empty()) {
        throw std::invalid_argument("f0 がありません");
    }
    if (volumeJson.empty()) {
        throw std::invalid_argument("volume がありません");
    }
    if (phonemesJson.empty()) {
        throw std::invalid_argument("phonemes がありません");
    }
    frameAudioQuery.f0Values = parseNativeOnnxFloatArrayText(f0Json, "f0");
    frameAudioQuery.volumeValues = parseNativeOnnxFloatArrayText(volumeJson, "volume");
    frameAudioQuery.phonemes = parseNativeOnnxFramePhonemes(phonemesJson);
    frameAudioQuery.volumeScale = static_cast<float>(extractNativeOnnxJsonNumberField(frameAudioQueryText, "volumeScale", frameAudioQuery.volumeScale));
    if (findJsonFieldValuePosition(frameAudioQueryText, "outputSamplingRate") != std::string::npos) {
        uint64_t Rate = requireNativeOnnxJsonUint64Field(frameAudioQueryText, "outputSamplingRate");
        if (Rate == 0 || Rate > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("outputSamplingRate が不正です");
        }
        frameAudioQuery.outputSamplingRate = static_cast<uint32_t>(Rate);
    }
    frameAudioQuery.outputStereo = extractNativeOnnxJsonBoolField(frameAudioQueryText, "outputStereo", frameAudioQuery.outputStereo);
    if (!std::isfinite(frameAudioQuery.volumeScale) || frameAudioQuery.volumeScale < 0.0f) {
        throw std::invalid_argument("volumeScale が不正です");
    }
    if (frameAudioQuery.outputSamplingRate == 0) {
        throw std::invalid_argument("outputSamplingRate が不正です");
    }
    return frameAudioQuery;
}

static size_t calculateNativeOnnxFrameAudioQueryFrameCount(const NativeOnnxFrameAudioQuery &frameAudioQuery) {
    uint64_t frameCount = 0;
    uint64_t Limit = std::min<uint64_t>(std::numeric_limits<size_t>::max(), std::numeric_limits<int64_t>::max());
    for (const NativeOnnxFramePhoneme &framePhoneme : frameAudioQuery.phonemes) {
        if (framePhoneme.frameLength > Limit - frameCount) {
            throw std::invalid_argument("frame_length の合計が大きすぎます");
        }
        frameCount += framePhoneme.frameLength;
    }
    return static_cast<size_t>(frameCount);
}

void validateNativeOnnxParsedFrameAudioQuery(const NativeOnnxFrameAudioQuery &frameAudioQuery) {
    size_t frameCount = calculateNativeOnnxFrameAudioQueryFrameCount(frameAudioQuery);
    if (frameAudioQuery.f0Values.size() != frameCount) {
        throw std::invalid_argument("f0 の長さが frame_length 合計と一致しません");
    }
    if (frameAudioQuery.volumeValues.size() != frameCount) {
        throw std::invalid_argument("volume の長さが frame_length 合計と一致しません");
    }
    for (const NativeOnnxFramePhoneme &framePhoneme : frameAudioQuery.phonemes) {
        parseNativeOnnxPhonemeCode(framePhoneme.phoneme);
    }
}

std::vector<NativeOnnxSongPhonemeFeature> createNativeOnnxSongPhonemeFeatures(const std::vector<NativeOnnxScoreNote> &scoreNotes) {
    std::vector<NativeOnnxSongPhonemeFeature> phonemeFeatures;
    for (const NativeOnnxScoreNote &scoreNote : scoreNotes) {
        if (!scoreNote.hasKey) {
            NativeOnnxSongPhonemeFeature phonemeFeature;
            phonemeFeature.phoneme = "pau";
            phonemeFeature.phonemeCode = 0;
            phonemeFeature.key = -1;
            phonemeFeature.noteId = scoreNote.noteId;
            phonemeFeature.hasNoteId = scoreNote.hasNoteId;
            phonemeFeatures.push_back(std::move(phonemeFeature));
            continue;
        }
        if (scoreNote.mora.hasConsonant) {
            NativeOnnxSongPhonemeFeature consonantFeature;
            consonantFeature.phoneme = scoreNote.mora.consonant;
            consonantFeature.phonemeCode = parseNativeOnnxPhonemeCode(scoreNote.mora.consonant);
            consonantFeature.key = scoreNote.key;
            consonantFeature.noteId = scoreNote.noteId;
            consonantFeature.hasNoteId = scoreNote.hasNoteId;
            phonemeFeatures.push_back(std::move(consonantFeature));
        }
        NativeOnnxSongPhonemeFeature vowelFeature;
        vowelFeature.phoneme = scoreNote.mora.vowel;
        vowelFeature.phonemeCode = parseNativeOnnxPhonemeCode(scoreNote.mora.vowel);
        vowelFeature.key = scoreNote.key;
        vowelFeature.noteId = scoreNote.noteId;
        vowelFeature.hasNoteId = scoreNote.hasNoteId;
        phonemeFeatures.push_back(std::move(vowelFeature));
    }
    return phonemeFeatures;
}

std::vector<uint64_t> createNativeOnnxSongPhonemeLengths(const std::vector<NativeOnnxScoreNote> &scoreNotes, const std::vector<int64_t> &consonantLengths) {
    if (scoreNotes.empty() || scoreNotes.size() != consonantLengths.size()) {
        throw std::runtime_error("consonant_lengths の長さが一致しません");
    }
    if (consonantLengths.front() != 0) {
        throw std::runtime_error("consonant_lengths の先頭が 0 ではありません");
    }
    std::vector<uint64_t> phonemeLengths;
    for (size_t noteIndex = 0; noteIndex + 1 < scoreNotes.size(); noteIndex++) {
        uint64_t noteDuration = scoreNotes[noteIndex].frameLength;
        int64_t nextConsonantLength = consonantLengths[noteIndex + 1];
        bool hasNextConsonant = nextConsonantLength != 0;
        uint64_t adjustedConsonantLength = 0;
        if (hasNextConsonant) {
            if (nextConsonantLength < 0 || static_cast<uint64_t>(nextConsonantLength) > noteDuration) {
                adjustedConsonantLength = noteDuration / 2;
            } else {
                adjustedConsonantLength = static_cast<uint64_t>(nextConsonantLength);
            }
        }
        phonemeLengths.push_back(noteDuration - adjustedConsonantLength);
        if (hasNextConsonant) {
            phonemeLengths.push_back(adjustedConsonantLength);
        }
    }
    phonemeLengths.push_back(scoreNotes.back().frameLength);
    return phonemeLengths;
}

NativeOnnxSongFrameInputs createNativeOnnxSongFrameInputs(const std::vector<NativeOnnxSongPhonemeFeature> &phonemeFeatures, const std::vector<NativeOnnxFramePhoneme> &framePhonemes) {
    if (phonemeFeatures.size() != framePhonemes.size()) {
        throw std::invalid_argument("score と frame_audio_query の phoneme 数が一致しません");
    }
    NativeOnnxSongFrameInputs frameInputs;
    for (size_t phonemeIndex = 0; phonemeIndex < phonemeFeatures.size(); phonemeIndex++) {
        const NativeOnnxSongPhonemeFeature &phonemeFeature = phonemeFeatures[phonemeIndex];
        const NativeOnnxFramePhoneme &framePhoneme = framePhonemes[phonemeIndex];
        if (parseNativeOnnxPhonemeCode(framePhoneme.phoneme) != phonemeFeature.phonemeCode) {
            throw std::invalid_argument("score と frame_audio_query の phoneme が一致しません");
        }
        if (framePhoneme.frameLength > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::invalid_argument("frame_length が大きすぎます");
        }
        for (uint64_t frameIndex = 0; frameIndex < framePhoneme.frameLength; frameIndex++) {
            frameInputs.phonemeValues.push_back(phonemeFeature.phonemeCode);
            frameInputs.keyValues.push_back(phonemeFeature.key);
        }
    }
    return frameInputs;
}

NativeOnnxAudioQuerySettings createNativeOnnxSettingsFromFrameAudioQuery(const NativeOnnxFrameAudioQuery &frameAudioQuery) {
    NativeOnnxAudioQuerySettings audioQuerySettings;
    audioQuerySettings.volumeScale = frameAudioQuery.volumeScale;
    audioQuerySettings.outputSamplingRate = frameAudioQuery.outputSamplingRate;
    audioQuerySettings.outputStereo = frameAudioQuery.outputStereo;
    return audioQuerySettings;
}

std::string createNativeOnnxFloatArrayJson(const std::vector<float> &numberValues) {
    std::ostringstream jsonStream;
    jsonStream << "[";
    jsonStream << std::setprecision(9);
    for (size_t numberIndex = 0; numberIndex < numberValues.size(); numberIndex++) {
        if (numberIndex > 0) {
            jsonStream << ",";
        }
        if (!std::isfinite(numberValues[numberIndex])) {
            throw std::runtime_error("float 配列に有限でない値があります");
        }
        jsonStream << numberValues[numberIndex];
    }
    jsonStream << "]";
    return jsonStream.str();
}

std::string createNativeOnnxFrameAudioQueryJson(const NativeOnnxFrameAudioQuery &frameAudioQuery) {
    std::ostringstream jsonStream;
    jsonStream << std::setprecision(9);
    jsonStream << "{\"f0\":" << createNativeOnnxFloatArrayJson(frameAudioQuery.f0Values) << ",";
    jsonStream << "\"volume\":" << createNativeOnnxFloatArrayJson(frameAudioQuery.volumeValues) << ",";
    jsonStream << "\"phonemes\":[";
    for (size_t phonemeIndex = 0; phonemeIndex < frameAudioQuery.phonemes.size(); phonemeIndex++) {
        const NativeOnnxFramePhoneme &framePhoneme = frameAudioQuery.phonemes[phonemeIndex];
        if (phonemeIndex > 0) {
            jsonStream << ",";
        }
        jsonStream << "{\"phoneme\":" << quoteJsonString(framePhoneme.phoneme) << ",";
        jsonStream << "\"frame_length\":" << framePhoneme.frameLength << ",";
        jsonStream << "\"note_id\":";
        if (framePhoneme.hasNoteId) {
            jsonStream << quoteJsonString(framePhoneme.noteId);
        } else {
            jsonStream << "null";
        }
        jsonStream << "}";
    }
    jsonStream << "],";
    jsonStream << "\"volumeScale\":" << frameAudioQuery.volumeScale << ",";
    jsonStream << "\"outputSamplingRate\":" << frameAudioQuery.outputSamplingRate << ",";
    jsonStream << "\"outputStereo\":" << (frameAudioQuery.outputStereo ? "true" : "false") << "}";
    return jsonStream.str();
}

int64_t resolveNativeOnnxInnerVoiceId(const ModelAssetRecord &modelAsset, const std::string &domainName, uint32_t styleId) {
    std::vector<uint8_t> manifestBytes = extractVvmEntryBytes(modelAsset.archivePath, "manifest.json");
    std::string manifestText(manifestBytes.begin(), manifestBytes.end());
    std::string domainObject = extractJsonObjectField(manifestText, domainName);
    if (domainObject.empty()) {
        throw std::runtime_error(domainName + " が manifest にありません");
    }
    std::string mappingObject = extractJsonObjectField(domainObject, "style_id_to_inner_voice_id");
    if (mappingObject.empty()) {
        return static_cast<int64_t>(styleId);
    }
    size_t valuePosition = findJsonFieldValuePosition(mappingObject, std::to_string(styleId));
    if (valuePosition == std::string::npos) {
        throw std::runtime_error(domainName + " の style ID が未対応です: " + std::to_string(styleId));
    }
    return parseNativeOnnxJsonInt64At(mappingObject, valuePosition, "style_id_to_inner_voice_id");
}

static bool isNativeOnnxMoraTailPhoneme(int64_t phonemeValue) {
    return phonemeValue == 0 || phonemeValue == 1 || phonemeValue == 2 || phonemeValue == 3 || phonemeValue == 4 || phonemeValue == 5 || phonemeValue == 6 || phonemeValue == 7 || phonemeValue == 11 || phonemeValue == 14 || phonemeValue == 21 || phonemeValue == 30 || phonemeValue == 40;
}

static NativeOnnxMora parseNativeOnnxMora(const std::string &moraJson) {
    NativeOnnxMora mora;
    mora.text = extractJsonStringField(moraJson, "text");
    mora.consonant = extractJsonStringField(moraJson, "consonant");
    mora.hasConsonant = !mora.consonant.empty();
    mora.hasConsonantLength = extractNativeOnnxJsonFloatField(moraJson, "consonant_length", mora.consonantLength);
    mora.vowel = extractJsonStringField(moraJson, "vowel");
    if (mora.vowel.empty()) {
        throw std::runtime_error("mora vowel がありません");
    }
    mora.hasVowelLength = extractNativeOnnxJsonFloatField(moraJson, "vowel_length", mora.vowelLength);
    mora.hasPitch = extractNativeOnnxJsonFloatField(moraJson, "pitch", mora.pitch);
    return mora;
}

std::vector<NativeOnnxAccentPhrase> parseNativeOnnxAccentPhrases(const std::string &audioQueryText) {
    std::string accentPhrasesJson = extractJsonArrayField(audioQueryText, "accent_phrases");
    if (accentPhrasesJson.empty()) {
        throw std::runtime_error("accent_phrases がありません");
    }
    std::vector<NativeOnnxAccentPhrase> accentPhrases;
    for (const std::string &phraseJson : splitJsonObjects(accentPhrasesJson)) {
        NativeOnnxAccentPhrase accentPhrase;
        uint32_t accentValue = 1;
        extractJsonNumberField(phraseJson, "accent", accentValue);
        accentPhrase.accent = static_cast<int64_t>(accentValue);
        std::string morasJson = extractJsonArrayField(phraseJson, "moras");
        for (const std::string &moraJson : splitJsonObjects(morasJson)) {
            accentPhrase.moras.push_back(parseNativeOnnxMora(moraJson));
        }
        std::string pauseMoraJson = extractJsonObjectField(phraseJson, "pause_mora");
        if (!pauseMoraJson.empty()) {
            accentPhrase.pauseMora = parseNativeOnnxMora(pauseMoraJson);
            accentPhrase.hasPauseMora = true;
        }
        accentPhrase.isInterrogative = extractNativeOnnxJsonBoolField(phraseJson, "is_interrogative", false);
        accentPhrases.push_back(std::move(accentPhrase));
    }
    return accentPhrases;
}

NativeOnnxTraceInput createNativeOnnxInt64Tensor(const std::string &tensorName, const std::vector<int64_t> &dimensions, const std::vector<int64_t> &values) {
    NativeOnnxTraceInput tensor;
    tensor.name = tensorName;
    tensor.elementType = 7;
    tensor.dimensions = dimensions;
    tensor.bytes = createNativeOnnxTensorBytes(values);
    return tensor;
}

static void appendNativeOnnxAccentValues(std::vector<int64_t> &accentValues, const NativeOnnxAccentPhrase &accentPhrase, int64_t point) {
    for (size_t moraIndex = 0; moraIndex < accentPhrase.moras.size(); moraIndex++) {
        bool isPoint = static_cast<int64_t>(moraIndex) == point || (point < 0 && static_cast<int64_t>(moraIndex) == static_cast<int64_t>(accentPhrase.moras.size()) + point);
        int64_t value = isPoint ? 1 : 0;
        accentValues.push_back(value);
        if (accentPhrase.moras[moraIndex].hasConsonant) {
            accentValues.push_back(value);
        }
    }
    if (accentPhrase.hasPauseMora) {
        accentValues.push_back(0);
    }
}

static void splitNativeOnnxMoraPhonemes(const std::vector<int64_t> &phonemeValues, std::vector<int64_t> &consonantValues, std::vector<int64_t> &vowelValues, std::vector<int64_t> &vowelIndexes) {
    for (size_t phonemeIndex = 0; phonemeIndex < phonemeValues.size(); phonemeIndex++) {
        if (isNativeOnnxMoraTailPhoneme(phonemeValues[phonemeIndex])) {
            vowelValues.push_back(phonemeValues[phonemeIndex]);
            vowelIndexes.push_back(static_cast<int64_t>(phonemeIndex));
        }
    }
    if (vowelIndexes.empty()) {
        throw std::runtime_error("vowel phoneme がありません");
    }
    consonantValues.push_back(-1);
    for (size_t vowelIndex = 0; vowelIndex + 1 < vowelIndexes.size(); vowelIndex++) {
        int64_t previousIndex = vowelIndexes[vowelIndex];
        int64_t nextIndex = vowelIndexes[vowelIndex + 1];
        if (nextIndex - previousIndex == 1) {
            consonantValues.push_back(-1);
        } else {
            consonantValues.push_back(phonemeValues[static_cast<size_t>(nextIndex - 1)]);
        }
    }
}

std::vector<NativeOnnxTraceInput> createNativeOnnxFrontendInputs(const std::string &audioQueryText, int64_t innerVoiceId) {
    std::vector<NativeOnnxAccentPhrase> accentPhrases = parseNativeOnnxAccentPhrases(audioQueryText);
    std::vector<int64_t> phonemeValues;
    phonemeValues.push_back(0);
    for (const NativeOnnxAccentPhrase &accentPhrase : accentPhrases) {
        for (const NativeOnnxMora &mora : accentPhrase.moras) {
            if (mora.hasConsonant) {
                phonemeValues.push_back(parseNativeOnnxPhonemeCode(mora.consonant));
            }
            phonemeValues.push_back(parseNativeOnnxPhonemeCode(mora.vowel));
        }
        if (accentPhrase.hasPauseMora) {
            if (accentPhrase.pauseMora.hasConsonant) {
                phonemeValues.push_back(parseNativeOnnxPhonemeCode(accentPhrase.pauseMora.consonant));
            }
            phonemeValues.push_back(parseNativeOnnxPhonemeCode(accentPhrase.pauseMora.vowel));
        }
    }
    phonemeValues.push_back(0);
    std::vector<int64_t> consonantValues;
    std::vector<int64_t> vowelValues;
    std::vector<int64_t> vowelIndexes;
    splitNativeOnnxMoraPhonemes(phonemeValues, consonantValues, vowelValues, vowelIndexes);
    std::vector<int64_t> baseStartAccentValues{0};
    std::vector<int64_t> baseEndAccentValues{0};
    std::vector<int64_t> baseStartAccentPhraseValues{0};
    std::vector<int64_t> baseEndAccentPhraseValues{0};
    for (const NativeOnnxAccentPhrase &accentPhrase : accentPhrases) {
        appendNativeOnnxAccentValues(baseStartAccentValues, accentPhrase, accentPhrase.accent != 1 ? 1 : 0);
        appendNativeOnnxAccentValues(baseEndAccentValues, accentPhrase, accentPhrase.accent - 1);
        appendNativeOnnxAccentValues(baseStartAccentPhraseValues, accentPhrase, 0);
        appendNativeOnnxAccentValues(baseEndAccentPhraseValues, accentPhrase, -1);
    }
    baseStartAccentValues.push_back(0);
    baseEndAccentValues.push_back(0);
    baseStartAccentPhraseValues.push_back(0);
    baseEndAccentPhraseValues.push_back(0);
    std::vector<int64_t> startAccentValues;
    std::vector<int64_t> endAccentValues;
    std::vector<int64_t> startAccentPhraseValues;
    std::vector<int64_t> endAccentPhraseValues;
    for (int64_t vowelIndex : vowelIndexes) {
        size_t valueIndex = static_cast<size_t>(vowelIndex);
        startAccentValues.push_back(baseStartAccentValues.at(valueIndex));
        endAccentValues.push_back(baseEndAccentValues.at(valueIndex));
        startAccentPhraseValues.push_back(baseStartAccentPhraseValues.at(valueIndex));
        endAccentPhraseValues.push_back(baseEndAccentPhraseValues.at(valueIndex));
    }
    std::vector<NativeOnnxTraceInput> frontendInputs;
    frontendInputs.push_back(createNativeOnnxInt64Tensor("phoneme_list", {static_cast<int64_t>(phonemeValues.size())}, phonemeValues));
    frontendInputs.push_back(createNativeOnnxInt64Tensor("speaker_id", {1}, {innerVoiceId}));
    frontendInputs.push_back(createNativeOnnxInt64Tensor("length", {}, {static_cast<int64_t>(vowelValues.size())}));
    frontendInputs.push_back(createNativeOnnxInt64Tensor("vowel_phoneme_list", {static_cast<int64_t>(vowelValues.size())}, vowelValues));
    frontendInputs.push_back(createNativeOnnxInt64Tensor("consonant_phoneme_list", {static_cast<int64_t>(consonantValues.size())}, consonantValues));
    frontendInputs.push_back(createNativeOnnxInt64Tensor("start_accent_list", {static_cast<int64_t>(startAccentValues.size())}, startAccentValues));
    frontendInputs.push_back(createNativeOnnxInt64Tensor("end_accent_list", {static_cast<int64_t>(endAccentValues.size())}, endAccentValues));
    frontendInputs.push_back(createNativeOnnxInt64Tensor("start_accent_phrase_list", {static_cast<int64_t>(startAccentPhraseValues.size())}, startAccentPhraseValues));
    frontendInputs.push_back(createNativeOnnxInt64Tensor("end_accent_phrase_list", {static_cast<int64_t>(endAccentPhraseValues.size())}, endAccentPhraseValues));
    return frontendInputs;
}

std::vector<int64_t> findNativeOnnxVowelIndexes(const std::vector<int64_t> &phonemeValues, const std::vector<int64_t> &vowelValues) {
    std::vector<int64_t> vowelIndexes;
    vowelIndexes.reserve(vowelValues.size());
    size_t searchOffset = 0;
    for (int64_t vowelValue : vowelValues) {
        bool hasFound = false;
        for (size_t phonemeIndex = searchOffset; phonemeIndex < phonemeValues.size(); phonemeIndex++) {
            if (phonemeValues[phonemeIndex] == vowelValue) {
                vowelIndexes.push_back(static_cast<int64_t>(phonemeIndex));
                searchOffset = phonemeIndex + 1;
                hasFound = true;
                break;
            }
        }
        if (!hasFound) {
            throw std::runtime_error("vowel index を復元できません");
        }
    }
    return vowelIndexes;
}

static size_t calculateNativeOnnxFrameCount(float phonemeLength, float speedScale) {
    static constexpr float samplingRate = 24000.0f;
    static constexpr float frameHop = 256.0f;
    float adjustedLength = phonemeLength < 0.01f ? 0.01f : phonemeLength;
    double roundedSamples = std::nearbyint(static_cast<double>(adjustedLength * samplingRate / frameHop));
    double roundedFrames = std::nearbyint(roundedSamples / static_cast<double>(speedScale));
    if (roundedFrames <= 0.0) {
        return 0;
    }
    return static_cast<size_t>(roundedFrames);
}

bool isNativeOnnxUnvoicedVowel(int64_t phonemeValue) {
    return phonemeValue == 0 || phonemeValue == 1 || phonemeValue == 2 || phonemeValue == 3 || phonemeValue == 5 || phonemeValue == 6 || phonemeValue == 11;
}

static void applyNativeOnnxPitchSettings(std::vector<float> &f0Values, const NativeOnnxAudioQuerySettings &audioQuerySettings) {
    float f0Sum = 0.0f;
    size_t voicedCount = 0;
    for (float &f0Value : f0Values) {
        f0Value *= std::pow(2.0f, audioQuerySettings.pitchScale);
        if (f0Value > 0.0f) {
            f0Sum += f0Value;
            voicedCount++;
        }
    }
    if (voicedCount == 0) {
        return;
    }
    float meanF0 = f0Sum / static_cast<float>(voicedCount);
    for (float &f0Value : f0Values) {
        if (f0Value > 0.0f) {
            f0Value = (f0Value - meanF0) * audioQuerySettings.intonationScale + meanF0;
        }
    }
}

std::vector<NativeOnnxTraceInput> createNativeOnnxDecoderInputs(const std::vector<NativeOnnxTraceInput> &traceInputs, const std::vector<NativeOnnxTraceInput> &durationOutputs, const std::vector<NativeOnnxTraceInput> &intonationOutputs, const NativeOnnxAudioQuerySettings &audioQuerySettings, bool shouldZeroUnvoicedVowels) {
    const NativeOnnxTraceInput &phonemeListTensor = requireNativeOnnxTensor(traceInputs, "phoneme_list");
    const NativeOnnxTraceInput &vowelPhonemeListTensor = requireNativeOnnxTensor(traceInputs, "vowel_phoneme_list");
    const NativeOnnxTraceInput &speakerIdTensor = requireNativeOnnxTensor(traceInputs, "speaker_id");
    const NativeOnnxTraceInput &phonemeLengthTensor = requireNativeOnnxTensor(durationOutputs, "phoneme_length");
    const NativeOnnxTraceInput &f0ListTensor = requireNativeOnnxTensor(intonationOutputs, "f0_list");
    std::vector<int64_t> phonemeValues = readNativeOnnxTensorValues<int64_t>(phonemeListTensor, 7);
    std::vector<int64_t> vowelValues = readNativeOnnxTensorValues<int64_t>(vowelPhonemeListTensor, 7);
    std::vector<float> phonemeLengthValues = readNativeOnnxTensorValues<float>(phonemeLengthTensor, 1);
    std::vector<float> f0ListValues = readNativeOnnxTensorValues<float>(f0ListTensor, 1);
    if (phonemeValues.size() != phonemeLengthValues.size()) {
        throw std::runtime_error("phoneme_length の長さが一致しません");
    }
    if (!phonemeLengthValues.empty()) {
        phonemeLengthValues.front() = audioQuerySettings.prePhonemeLength;
        phonemeLengthValues.back() = audioQuerySettings.postPhonemeLength;
    }
    std::vector<int64_t> vowelIndexes = findNativeOnnxVowelIndexes(phonemeValues, vowelValues);
    if (vowelIndexes.size() != f0ListValues.size()) {
        throw std::runtime_error("f0_list の長さが一致しません");
    }
    std::vector<float> decoderF0Values = f0ListValues;
    if (!decoderF0Values.empty()) {
        decoderF0Values.front() = 0.0f;
        decoderF0Values.back() = 0.0f;
    }
    for (size_t vowelIndex = 0; vowelIndex < vowelValues.size(); vowelIndex++) {
        if (shouldZeroUnvoicedVowels && isNativeOnnxUnvoicedVowel(vowelValues[vowelIndex])) {
            decoderF0Values[vowelIndex] = 0.0f;
        }
    }
    applyNativeOnnxPitchSettings(decoderF0Values, audioQuerySettings);
    std::vector<float> f0FrameValues;
    std::vector<float> phonemeFrameValues;
    size_t pendingF0FrameCount = 0;
    size_t f0Index = 0;
    size_t vowelIndexCursor = 0;
    for (size_t phonemeIndex = 0; phonemeIndex < phonemeValues.size(); phonemeIndex++) {
        int64_t phonemeValue = phonemeValues[phonemeIndex];
        if (phonemeValue < 0 || phonemeValue >= nativeOnnxPhonemeSize) {
            throw std::runtime_error("phoneme id が範囲外です");
        }
        size_t frameCount = calculateNativeOnnxFrameCount(phonemeLengthValues[phonemeIndex], audioQuerySettings.speedScale);
        for (size_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
            size_t frameOffset = phonemeFrameValues.size();
            phonemeFrameValues.resize(frameOffset + static_cast<size_t>(nativeOnnxPhonemeSize), 0.0f);
            phonemeFrameValues[frameOffset + static_cast<size_t>(phonemeValue)] = 1.0f;
        }
        pendingF0FrameCount += frameCount;
        if (vowelIndexCursor < vowelIndexes.size() && static_cast<int64_t>(phonemeIndex) == vowelIndexes[vowelIndexCursor]) {
            for (size_t frameIndex = 0; frameIndex < pendingF0FrameCount; frameIndex++) {
                f0FrameValues.push_back(decoderF0Values[f0Index]);
            }
            pendingF0FrameCount = 0;
            f0Index++;
            vowelIndexCursor++;
        }
    }
    if (pendingF0FrameCount != 0 || f0FrameValues.size() * static_cast<size_t>(nativeOnnxPhonemeSize) != phonemeFrameValues.size()) {
        throw std::runtime_error("decoder feature の frame 数が一致しません");
    }
    std::vector<float> paddedF0FrameValues;
    std::vector<float> paddedPhonemeFrameValues;
    paddedF0FrameValues.reserve(f0FrameValues.size() + nativeOnnxDecoderPaddingFrames * 2);
    paddedPhonemeFrameValues.reserve(phonemeFrameValues.size() + nativeOnnxDecoderPaddingFrames * 2 * static_cast<size_t>(nativeOnnxPhonemeSize));
    for (size_t frameIndex = 0; frameIndex < nativeOnnxDecoderPaddingFrames; frameIndex++) {
        paddedF0FrameValues.push_back(0.0f);
        size_t frameOffset = paddedPhonemeFrameValues.size();
        paddedPhonemeFrameValues.resize(frameOffset + static_cast<size_t>(nativeOnnxPhonemeSize), 0.0f);
        paddedPhonemeFrameValues[frameOffset] = 1.0f;
    }
    paddedF0FrameValues.insert(paddedF0FrameValues.end(), f0FrameValues.begin(), f0FrameValues.end());
    paddedPhonemeFrameValues.insert(paddedPhonemeFrameValues.end(), phonemeFrameValues.begin(), phonemeFrameValues.end());
    for (size_t frameIndex = 0; frameIndex < nativeOnnxDecoderPaddingFrames; frameIndex++) {
        paddedF0FrameValues.push_back(0.0f);
        size_t frameOffset = paddedPhonemeFrameValues.size();
        paddedPhonemeFrameValues.resize(frameOffset + static_cast<size_t>(nativeOnnxPhonemeSize), 0.0f);
        paddedPhonemeFrameValues[frameOffset] = 1.0f;
    }
    NativeOnnxTraceInput f0Tensor;
    f0Tensor.name = "f0";
    f0Tensor.elementType = 1;
    f0Tensor.dimensions = {static_cast<int64_t>(paddedF0FrameValues.size()), 1};
    f0Tensor.bytes = createNativeOnnxTensorBytes(paddedF0FrameValues);
    NativeOnnxTraceInput phonemeTensor;
    phonemeTensor.name = "phoneme";
    phonemeTensor.elementType = 1;
    phonemeTensor.dimensions = {static_cast<int64_t>(paddedF0FrameValues.size()), nativeOnnxPhonemeSize};
    phonemeTensor.bytes = createNativeOnnxTensorBytes(paddedPhonemeFrameValues);
    return {f0Tensor, phonemeTensor, speakerIdTensor};
}

