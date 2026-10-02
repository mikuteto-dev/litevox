#include "native_audio_query_validation.hpp"

#include "json_utility.hpp"
#include "utility.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

enum class NativeJsonValueKind {
    Missing,
    Null,
    String,
    Number,
    Bool,
    Array,
    Object
};

static bool startsWithNativeValidationText(const std::string &text, size_t position, const std::string &prefixText) {
    return position + prefixText.size() <= text.size() && text.compare(position, prefixText.size(), prefixText) == 0;
}

static NativeJsonValueKind getNativeJsonFieldKind(const std::string &jsonText, const std::string &fieldName) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size()) {
        return NativeJsonValueKind::Missing;
    }
    char valueToken = jsonText[valuePosition];
    if (valueToken == '"') {
        return NativeJsonValueKind::String;
    }
    if (valueToken == '[') {
        return NativeJsonValueKind::Array;
    }
    if (valueToken == '{') {
        return NativeJsonValueKind::Object;
    }
    if (startsWithNativeValidationText(jsonText, valuePosition, "null")) {
        return NativeJsonValueKind::Null;
    }
    if (startsWithNativeValidationText(jsonText, valuePosition, "true") || startsWithNativeValidationText(jsonText, valuePosition, "false")) {
        return NativeJsonValueKind::Bool;
    }
    if ((valueToken >= '0' && valueToken <= '9') || valueToken == '-') {
        return NativeJsonValueKind::Number;
    }
    return NativeJsonValueKind::Missing;
}

static void ensureNativeJsonObject(const std::string &jsonText, const std::string &labelText) {
    std::string trimmedJson = trimAscii(jsonText);
    if (trimmedJson.size() < 2 || trimmedJson.front() != '{' || trimmedJson.back() != '}') {
        throw std::runtime_error(labelText + " は JSON object が必要です");
    }
}

static std::string requireNativeJsonStringField(const std::string &jsonText, const std::string &fieldName, const std::string &labelText) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size() || jsonText[valuePosition] != '"') {
        throw std::runtime_error(labelText + "." + fieldName + " は string が必要です");
    }
    return decodeJsonString(jsonText, valuePosition);
}

static double requireNativeJsonNumberField(const std::string &jsonText, const std::string &fieldName, const std::string &labelText) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size()) {
        throw std::runtime_error(labelText + "." + fieldName + " は number が必要です");
    }
    errno = 0;
    char *numberEnd = nullptr;
    const char *numberStart = jsonText.c_str() + valuePosition;
    double numberValue = std::strtod(numberStart, &numberEnd);
    if (numberStart == numberEnd || errno == ERANGE || !std::isfinite(numberValue)) {
        throw std::runtime_error(labelText + "." + fieldName + " は有限の number が必要です");
    }
    return numberValue;
}

static bool requireNativeJsonBoolField(const std::string &jsonText, const std::string &fieldName, const std::string &labelText) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size()) {
        throw std::runtime_error(labelText + "." + fieldName + " は bool が必要です");
    }
    if (startsWithNativeValidationText(jsonText, valuePosition, "true")) {
        return true;
    }
    if (startsWithNativeValidationText(jsonText, valuePosition, "false")) {
        return false;
    }
    throw std::runtime_error(labelText + "." + fieldName + " は bool が必要です");
}

static bool tryGetNativeJsonNullableStringField(const std::string &jsonText, const std::string &fieldName, std::string &fieldText, bool &hasFieldValue) {
    NativeJsonValueKind valueKind = getNativeJsonFieldKind(jsonText, fieldName);
    if (valueKind == NativeJsonValueKind::Missing || valueKind == NativeJsonValueKind::Null) {
        hasFieldValue = false;
        fieldText.clear();
        return true;
    }
    if (valueKind != NativeJsonValueKind::String) {
        return false;
    }
    fieldText = extractJsonStringField(jsonText, fieldName);
    hasFieldValue = true;
    return true;
}

static bool tryGetNativeJsonNullableNumberField(const std::string &jsonText, const std::string &fieldName, double &numberValue, bool &hasFieldValue) {
    NativeJsonValueKind valueKind = getNativeJsonFieldKind(jsonText, fieldName);
    if (valueKind == NativeJsonValueKind::Missing || valueKind == NativeJsonValueKind::Null) {
        hasFieldValue = false;
        numberValue = 0.0;
        return true;
    }
    if (valueKind != NativeJsonValueKind::Number) {
        return false;
    }
    numberValue = requireNativeJsonNumberField(jsonText, fieldName, "mora");
    hasFieldValue = true;
    return true;
}

static bool isNativeValidationNonConsonantPhoneme(const std::string &phonemeText) {
    return phonemeText == "pau" || phonemeText == "A" || phonemeText == "E" || phonemeText == "I" || phonemeText == "N" || phonemeText == "O" || phonemeText == "U" || phonemeText == "a" || phonemeText == "cl" || phonemeText == "e" || phonemeText == "i" || phonemeText == "o" || phonemeText == "u";
}

static bool isNativeValidationConsonantPhoneme(const std::string &phonemeText) {
    return phonemeText == "b" || phonemeText == "by" || phonemeText == "ch" || phonemeText == "d" || phonemeText == "dy" || phonemeText == "f" || phonemeText == "g" || phonemeText == "gw" || phonemeText == "gy" || phonemeText == "h" || phonemeText == "hy" || phonemeText == "j" || phonemeText == "k" || phonemeText == "kw" || phonemeText == "ky" || phonemeText == "m" || phonemeText == "my" || phonemeText == "n" || phonemeText == "ny" || phonemeText == "p" || phonemeText == "py" || phonemeText == "r" || phonemeText == "ry" || phonemeText == "s" || phonemeText == "sh" || phonemeText == "t" || phonemeText == "ts" || phonemeText == "ty" || phonemeText == "v" || phonemeText == "w" || phonemeText == "y" || phonemeText == "z";
}

static void validateNativeMoraJson(const std::string &moraJson, const std::string &labelText) {
    ensureNativeJsonObject(moraJson, labelText);
    requireNativeJsonStringField(moraJson, "text", labelText);
    std::string consonantText;
    bool hasConsonant = false;
    if (!tryGetNativeJsonNullableStringField(moraJson, "consonant", consonantText, hasConsonant)) {
        throw std::runtime_error(labelText + ".consonant は string または null が必要です");
    }
    double consonantLength = 0.0;
    bool hasConsonantLength = false;
    if (!tryGetNativeJsonNullableNumberField(moraJson, "consonant_length", consonantLength, hasConsonantLength)) {
        throw std::runtime_error(labelText + ".consonant_length は number または null が必要です");
    }
    if (hasConsonant != hasConsonantLength) {
        throw std::runtime_error(labelText + ".consonant と consonant_length の有無が一致しません");
    }
    if (hasConsonant && !isNativeValidationConsonantPhoneme(consonantText)) {
        throw std::runtime_error(labelText + ".consonant が子音ではありません: " + consonantText);
    }
    std::string vowelText = requireNativeJsonStringField(moraJson, "vowel", labelText);
    if (!isNativeValidationNonConsonantPhoneme(vowelText)) {
        throw std::runtime_error(labelText + ".vowel が母音または特殊音素ではありません: " + vowelText);
    }
    requireNativeJsonNumberField(moraJson, "vowel_length", labelText);
    requireNativeJsonNumberField(moraJson, "pitch", labelText);
}

static void validateNativeAccentPhraseJson(const std::string &phraseJson, const std::string &labelText) {
    ensureNativeJsonObject(phraseJson, labelText);
    double accentValue = requireNativeJsonNumberField(phraseJson, "accent", labelText);
    if (accentValue < 1.0 || std::floor(accentValue) != accentValue) {
        throw std::runtime_error(labelText + ".accent は 1 以上の整数が必要です");
    }
    std::string morasJson = extractJsonArrayField(phraseJson, "moras");
    if (morasJson.empty()) {
        throw std::runtime_error(labelText + ".moras は array が必要です");
    }
    std::vector<std::string> moraJsons = splitJsonObjects(morasJson);
    for (size_t moraIndex = 0; moraIndex < moraJsons.size(); moraIndex++) {
        validateNativeMoraJson(moraJsons[moraIndex], labelText + ".moras[" + std::to_string(moraIndex) + "]");
    }
    NativeJsonValueKind pauseMoraKind = getNativeJsonFieldKind(phraseJson, "pause_mora");
    if (pauseMoraKind == NativeJsonValueKind::Object) {
        validateNativeMoraJson(extractJsonObjectField(phraseJson, "pause_mora"), labelText + ".pause_mora");
    } else if (pauseMoraKind != NativeJsonValueKind::Null && pauseMoraKind != NativeJsonValueKind::Missing) {
        throw std::runtime_error(labelText + ".pause_mora は object または null が必要です");
    }
    requireNativeJsonBoolField(phraseJson, "is_interrogative", labelText);
}

void validateNativeAudioQuery(const std::string &audioQueryJson) {
    ensureNativeJsonObject(audioQueryJson, "audio_query");
    std::string accentPhrasesJson = extractJsonArrayField(audioQueryJson, "accent_phrases");
    if (accentPhrasesJson.empty()) {
        throw std::runtime_error("audio_query.accent_phrases は array が必要です");
    }
    std::vector<std::string> phraseJsons = splitJsonObjects(accentPhrasesJson);
    for (size_t phraseIndex = 0; phraseIndex < phraseJsons.size(); phraseIndex++) {
        validateNativeAccentPhraseJson(phraseJsons[phraseIndex], "audio_query.accent_phrases[" + std::to_string(phraseIndex) + "]");
    }
    requireNativeJsonNumberField(audioQueryJson, "speedScale", "audio_query");
    requireNativeJsonNumberField(audioQueryJson, "pitchScale", "audio_query");
    requireNativeJsonNumberField(audioQueryJson, "intonationScale", "audio_query");
    requireNativeJsonNumberField(audioQueryJson, "volumeScale", "audio_query");
    requireNativeJsonNumberField(audioQueryJson, "prePhonemeLength", "audio_query");
    requireNativeJsonNumberField(audioQueryJson, "postPhonemeLength", "audio_query");
    double samplingRate = requireNativeJsonNumberField(audioQueryJson, "outputSamplingRate", "audio_query");
    if (samplingRate < 1.0 || std::floor(samplingRate) != samplingRate || samplingRate > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("audio_query.outputSamplingRate は正の uint32 整数が必要です");
    }
    requireNativeJsonBoolField(audioQueryJson, "outputStereo", "audio_query");
    NativeJsonValueKind kanaKind = getNativeJsonFieldKind(audioQueryJson, "kana");
    if (kanaKind != NativeJsonValueKind::Missing && kanaKind != NativeJsonValueKind::Null && kanaKind != NativeJsonValueKind::String) {
        throw std::runtime_error("audio_query.kana は string または null が必要です");
    }
}
