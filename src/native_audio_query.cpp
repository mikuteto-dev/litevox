#include "native_audio_query.hpp"

#include "json_utility.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct NativeMoraPhoneme {
    const char *text;
    const char *consonant;
    const char *vowel;
};

static constexpr NativeMoraPhoneme nativeMoraPhonemes[] = {
    {"ヴォ", "v", "o"},
    {"ヴェ", "v", "e"},
    {"ヴィ", "v", "i"},
    {"ヴァ", "v", "a"},
    {"ヴ", "v", "u"},
    {"ン", "", "N"},
    {"ワ", "w", "a"},
    {"ロ", "r", "o"},
    {"レ", "r", "e"},
    {"ル", "r", "u"},
    {"リョ", "ry", "o"},
    {"リュ", "ry", "u"},
    {"リャ", "ry", "a"},
    {"リェ", "ry", "e"},
    {"リ", "r", "i"},
    {"ラ", "r", "a"},
    {"ヨ", "y", "o"},
    {"ユ", "y", "u"},
    {"ヤ", "y", "a"},
    {"モ", "m", "o"},
    {"メ", "m", "e"},
    {"ム", "m", "u"},
    {"ミョ", "my", "o"},
    {"ミュ", "my", "u"},
    {"ミャ", "my", "a"},
    {"ミェ", "my", "e"},
    {"ミ", "m", "i"},
    {"マ", "m", "a"},
    {"ポ", "p", "o"},
    {"ボ", "b", "o"},
    {"ホ", "h", "o"},
    {"ペ", "p", "e"},
    {"ベ", "b", "e"},
    {"ヘ", "h", "e"},
    {"プ", "p", "u"},
    {"ブ", "b", "u"},
    {"フォ", "f", "o"},
    {"フェ", "f", "e"},
    {"フィ", "f", "i"},
    {"ファ", "f", "a"},
    {"フ", "f", "u"},
    {"ピョ", "py", "o"},
    {"ピュ", "py", "u"},
    {"ピャ", "py", "a"},
    {"ピェ", "py", "e"},
    {"ピ", "p", "i"},
    {"ビョ", "by", "o"},
    {"ビュ", "by", "u"},
    {"ビャ", "by", "a"},
    {"ビェ", "by", "e"},
    {"ビ", "b", "i"},
    {"ヒョ", "hy", "o"},
    {"ヒュ", "hy", "u"},
    {"ヒャ", "hy", "a"},
    {"ヒェ", "hy", "e"},
    {"ヒ", "h", "i"},
    {"パ", "p", "a"},
    {"バ", "b", "a"},
    {"ハ", "h", "a"},
    {"ノ", "n", "o"},
    {"ネ", "n", "e"},
    {"ヌ", "n", "u"},
    {"ニョ", "ny", "o"},
    {"ニュ", "ny", "u"},
    {"ニャ", "ny", "a"},
    {"ニェ", "ny", "e"},
    {"ニ", "n", "i"},
    {"ナ", "n", "a"},
    {"ドゥ", "d", "u"},
    {"ド", "d", "o"},
    {"トゥ", "t", "u"},
    {"ト", "t", "o"},
    {"デョ", "dy", "o"},
    {"デュ", "dy", "u"},
    {"デャ", "dy", "a"},
    {"ディ", "d", "i"},
    {"デ", "d", "e"},
    {"テョ", "ty", "o"},
    {"テュ", "ty", "u"},
    {"テャ", "ty", "a"},
    {"ティ", "t", "i"},
    {"テ", "t", "e"},
    {"ツォ", "ts", "o"},
    {"ツェ", "ts", "e"},
    {"ツィ", "ts", "i"},
    {"ツァ", "ts", "a"},
    {"ツ", "ts", "u"},
    {"ッ", "", "cl"},
    {"チョ", "ch", "o"},
    {"チュ", "ch", "u"},
    {"チャ", "ch", "a"},
    {"チェ", "ch", "e"},
    {"チ", "ch", "i"},
    {"ダ", "d", "a"},
    {"タ", "t", "a"},
    {"ゾ", "z", "o"},
    {"ソ", "s", "o"},
    {"ゼ", "z", "e"},
    {"セ", "s", "e"},
    {"ズィ", "z", "i"},
    {"ズ", "z", "u"},
    {"スィ", "s", "i"},
    {"ス", "s", "u"},
    {"ジョ", "j", "o"},
    {"ジュ", "j", "u"},
    {"ジャ", "j", "a"},
    {"ジェ", "j", "e"},
    {"ジ", "j", "i"},
    {"ショ", "sh", "o"},
    {"シュ", "sh", "u"},
    {"シャ", "sh", "a"},
    {"シェ", "sh", "e"},
    {"シ", "sh", "i"},
    {"ザ", "z", "a"},
    {"サ", "s", "a"},
    {"ゴ", "g", "o"},
    {"コ", "k", "o"},
    {"ゲ", "g", "e"},
    {"ケ", "k", "e"},
    {"グヮ", "gw", "a"},
    {"グ", "g", "u"},
    {"クヮ", "kw", "a"},
    {"ク", "k", "u"},
    {"ギョ", "gy", "o"},
    {"ギュ", "gy", "u"},
    {"ギャ", "gy", "a"},
    {"ギェ", "gy", "e"},
    {"ギ", "g", "i"},
    {"キョ", "ky", "o"},
    {"キュ", "ky", "u"},
    {"キャ", "ky", "a"},
    {"キェ", "ky", "e"},
    {"キ", "k", "i"},
    {"ガ", "g", "a"},
    {"カ", "k", "a"},
    {"オ", "", "o"},
    {"エ", "", "e"},
    {"ウォ", "w", "o"},
    {"ウェ", "w", "e"},
    {"ウィ", "w", "i"},
    {"ウ", "", "u"},
    {"イェ", "y", "e"},
    {"イ", "", "i"},
    {"ア", "", "a"},
};

static bool startsWithNativeText(const std::string &text, size_t position, const std::string &prefix) {
    return position + prefix.size() <= text.size() && text.compare(position, prefix.size(), prefix) == 0;
}

static bool readNativeUtf8Codepoint(const std::string &text, size_t &position, uint32_t &codepoint) {
    if (position >= text.size()) {
        return false;
    }
    unsigned char firstByte = static_cast<unsigned char>(text[position]);
    if (firstByte < 0x80) {
        codepoint = firstByte;
        position++;
        return true;
    }
    size_t byteCount = 0;
    uint32_t codepointMask = 0;
    if ((firstByte & 0xe0) == 0xc0) {
        byteCount = 2;
        codepointMask = firstByte & 0x1f;
    } else if ((firstByte & 0xf0) == 0xe0) {
        byteCount = 3;
        codepointMask = firstByte & 0x0f;
    } else if ((firstByte & 0xf8) == 0xf0) {
        byteCount = 4;
        codepointMask = firstByte & 0x07;
    } else {
        return false;
    }
    if (position + byteCount > text.size()) {
        return false;
    }
    uint32_t decodedCodepoint = codepointMask;
    for (size_t byteIndex = 1; byteIndex < byteCount; byteIndex++) {
        unsigned char nextByte = static_cast<unsigned char>(text[position + byteIndex]);
        if ((nextByte & 0xc0) != 0x80) {
            return false;
        }
        decodedCodepoint = (decodedCodepoint << 6) | (nextByte & 0x3f);
    }
    position += byteCount;
    codepoint = decodedCodepoint;
    return true;
}

static void appendNativeUtf8Codepoint(std::string &text, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        text.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        text.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        text.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        text.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

static std::string normalizeNativeMoraText(const std::string &moraText) {
    std::string normalizedText;
    size_t position = 0;
    while (position < moraText.size()) {
        uint32_t codepoint = 0;
        if (!readNativeUtf8Codepoint(moraText, position, codepoint)) {
            throw std::runtime_error("歌詞の UTF-8 が不正です");
        }
        if (codepoint >= 0x3041 && codepoint <= 0x3096) {
            codepoint += 0x60;
        }
        appendNativeUtf8Codepoint(normalizedText, codepoint);
    }
    return normalizedText;
}

static bool canUnvoiceNativeVowel(const std::string &vowel) {
    return vowel == "a" || vowel == "e" || vowel == "i" || vowel == "o" || vowel == "u";
}

static std::string unvoiceNativeVowel(const std::string &vowel) {
    if (vowel == "a") return "A";
    if (vowel == "e") return "E";
    if (vowel == "i") return "I";
    if (vowel == "o") return "O";
    if (vowel == "u") return "U";
    throw std::runtime_error("無声化できない vowel です: " + vowel);
}

static std::string normalizeNativeMoraVowelText(const std::string &vowel) {
    if (vowel == "A") return "a";
    if (vowel == "E") return "e";
    if (vowel == "I") return "i";
    if (vowel == "O") return "o";
    if (vowel == "U") return "u";
    return vowel;
}

NativeAudioQueryMora createNativeAudioQueryMora(const std::string &consonant, const std::string &vowel) {
    std::string moraKey = consonant + normalizeNativeMoraVowelText(vowel);
    for (const NativeMoraPhoneme &moraPhoneme : nativeMoraPhonemes) {
        if (std::string(moraPhoneme.consonant) + std::string(moraPhoneme.vowel) == moraKey) {
            NativeAudioQueryMora mora;
            mora.text = moraPhoneme.text;
            mora.consonant = consonant;
            mora.vowel = vowel;
            mora.hasConsonant = !consonant.empty();
            return mora;
        }
    }
    return NativeAudioQueryMora{moraKey, consonant, vowel, !consonant.empty()};
}

static bool findNativeMoraAt(const std::string &phrase, size_t position, NativeAudioQueryMora &mora, size_t &matchedBytes) {
    bool shouldUnvoice = false;
    size_t searchPosition = position;
    if (searchPosition < phrase.size() && phrase[searchPosition] == '_') {
        shouldUnvoice = true;
        searchPosition++;
    }
    const NativeMoraPhoneme *matchedPhoneme = nullptr;
    size_t matchedLength = 0;
    for (const NativeMoraPhoneme &moraPhoneme : nativeMoraPhonemes) {
        std::string moraText = moraPhoneme.text;
        if (moraText.size() > matchedLength && startsWithNativeText(phrase, searchPosition, moraText)) {
            matchedPhoneme = &moraPhoneme;
            matchedLength = moraText.size();
        }
    }
    if (!matchedPhoneme) {
        return false;
    }
    std::string vowel = matchedPhoneme->vowel;
    if (shouldUnvoice) {
        if (!canUnvoiceNativeVowel(vowel)) {
            return false;
        }
        vowel = unvoiceNativeVowel(vowel);
    }
    mora = createNativeAudioQueryMora(matchedPhoneme->consonant, vowel);
    matchedBytes = matchedLength + (shouldUnvoice ? 1 : 0);
    return true;
}

NativeAudioQueryMora createNativeAudioQueryMoraFromText(const std::string &moraText) {
    std::string normalizedText = normalizeNativeMoraText(moraText);
    NativeAudioQueryMora mora;
    size_t matchedBytes = 0;
    if (!findNativeMoraAt(normalizedText, 0, mora, matchedBytes) || matchedBytes != normalizedText.size()) {
        throw std::invalid_argument("未知の歌詞モーラです: " + moraText);
    }
    return mora;
}

static NativeAudioQueryAccentPhrase parseNativeKanaAccentPhrase(const std::string &phrase) {
    NativeAudioQueryAccentPhrase accentPhrase;
    bool hasAccent = false;
    size_t position = 0;
    while (position < phrase.size()) {
        if (phrase[position] == '\'') {
            if (accentPhrase.moras.empty()) {
                throw std::runtime_error("accent をアクセント句の先頭に置けません: " + phrase);
            }
            if (hasAccent) {
                throw std::runtime_error("accent が複数あります: " + phrase);
            }
            accentPhrase.accent = accentPhrase.moras.size();
            hasAccent = true;
            position++;
            continue;
        }
        NativeAudioQueryMora mora;
        size_t matchedBytes = 0;
        if (!findNativeMoraAt(phrase, position, mora, matchedBytes)) {
            throw std::runtime_error("未知の kana です: " + phrase.substr(position));
        }
        accentPhrase.moras.push_back(std::move(mora));
        position += matchedBytes;
    }
    if (!hasAccent) {
        throw std::runtime_error("accent がありません: " + phrase);
    }
    return accentPhrase;
}

static std::vector<NativeAudioQueryAccentPhrase> parseNativeKana(const std::string &kana) {
    if (kana.empty()) {
        return {};
    }
    std::vector<NativeAudioQueryAccentPhrase> accentPhrases;
    std::string phrase;
    for (size_t position = 0; position <= kana.size();) {
        bool isEnd = position == kana.size();
        bool isPauseDelimiter = !isEnd && startsWithNativeText(kana, position, "、");
        bool isNoPauseDelimiter = !isEnd && kana[position] == '/';
        if (isEnd || isPauseDelimiter || isNoPauseDelimiter) {
            if (phrase.empty()) {
                throw std::runtime_error("空のアクセント句です: " + std::to_string(accentPhrases.size() + 1));
            }
            bool isInterrogative = false;
            if (phrase.size() >= std::string("？").size() && phrase.compare(phrase.size() - std::string("？").size(), std::string("？").size(), "？") == 0) {
                isInterrogative = true;
                phrase.resize(phrase.size() - std::string("？").size());
            }
            NativeAudioQueryAccentPhrase accentPhrase = parseNativeKanaAccentPhrase(phrase);
            accentPhrase.isInterrogative = isInterrogative;
            if (isPauseDelimiter) {
                accentPhrase.pauseMora.text = "、";
                accentPhrase.pauseMora.vowel = "pau";
                accentPhrase.hasPauseMora = true;
            }
            accentPhrases.push_back(std::move(accentPhrase));
            phrase.clear();
            if (isPauseDelimiter) {
                position += std::string("、").size();
            } else if (isNoPauseDelimiter) {
                position++;
            } else {
                position++;
            }
            continue;
        }
        if (startsWithNativeText(kana, position, "？")) {
            size_t nextPosition = position + std::string("？").size();
            bool isPhraseEnd = nextPosition == kana.size() || startsWithNativeText(kana, nextPosition, "、") || kana[nextPosition] == '/';
            if (!isPhraseEnd) {
                size_t phraseEndPosition = nextPosition;
                while (phraseEndPosition < kana.size() && !startsWithNativeText(kana, phraseEndPosition, "、") && kana[phraseEndPosition] != '/') {
                    phraseEndPosition++;
                }
                throw std::runtime_error("疑問符が末尾以外です: " + phrase + kana.substr(position, phraseEndPosition - position));
            }
            phrase += "？";
            position += std::string("？").size();
        } else {
            phrase.push_back(kana[position]);
            position++;
        }
    }
    return accentPhrases;
}

static std::string createNativeMoraJson(const NativeAudioQueryMora &mora) {
    std::string jsonText = "{\"text\":" + quoteJsonString(mora.text) + ",";
    jsonText += "\"consonant\":";
    jsonText += mora.hasConsonant ? quoteJsonString(mora.consonant) : "null";
    jsonText += ",\"consonant_length\":null";
    jsonText += ",\"vowel\":" + quoteJsonString(mora.vowel);
    jsonText += ",\"vowel_length\":0.0";
    jsonText += ",\"pitch\":0.0}";
    return jsonText;
}

std::string createNativeAccentPhrasesJson(const std::vector<NativeAudioQueryAccentPhrase> &accentPhrases) {
    std::string jsonText = "[";
    for (size_t phraseIndex = 0; phraseIndex < accentPhrases.size(); phraseIndex++) {
        const NativeAudioQueryAccentPhrase &accentPhrase = accentPhrases[phraseIndex];
        if (phraseIndex > 0) {
            jsonText += ",";
        }
        jsonText += "{\"moras\":[";
        for (size_t moraIndex = 0; moraIndex < accentPhrase.moras.size(); moraIndex++) {
            if (moraIndex > 0) {
                jsonText += ",";
            }
            jsonText += createNativeMoraJson(accentPhrase.moras[moraIndex]);
        }
        jsonText += "],\"accent\":" + std::to_string(accentPhrase.accent);
        jsonText += ",\"pause_mora\":";
        jsonText += accentPhrase.hasPauseMora ? createNativeMoraJson(accentPhrase.pauseMora) : "null";
        jsonText += ",\"is_interrogative\":";
        jsonText += accentPhrase.isInterrogative ? "true" : "false";
        jsonText += "}";
    }
    jsonText += "]";
    return jsonText;
}

std::string createNativeKanaFromAccentPhrases(const std::vector<NativeAudioQueryAccentPhrase> &accentPhrases) {
    std::string kana;
    for (const NativeAudioQueryAccentPhrase &accentPhrase : accentPhrases) {
        for (size_t moraIndex = 0; moraIndex < accentPhrase.moras.size(); moraIndex++) {
            const NativeAudioQueryMora &mora = accentPhrase.moras[moraIndex];
            if (mora.vowel == "A" || mora.vowel == "E" || mora.vowel == "I" || mora.vowel == "O" || mora.vowel == "U") {
                kana += "_";
            }
            kana += mora.text;
            if (moraIndex + 1 == accentPhrase.accent) {
                kana += "'";
            }
        }
        if (accentPhrase.isInterrogative) {
            kana += "？";
        }
        kana += accentPhrase.hasPauseMora ? "、" : "/";
    }
    if (!kana.empty()) {
        kana.pop_back();
    }
    return kana;
}

std::string createNativeAudioQueryJson(const std::vector<NativeAudioQueryAccentPhrase> &accentPhrases, const std::string &kana) {
    std::string jsonText = "{\"accent_phrases\":" + createNativeAccentPhrasesJson(accentPhrases);
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
    jsonText += ",\"kana\":" + quoteJsonString(kana);
    jsonText += "}";
    return jsonText;
}

std::string createNativeAccentPhrasesJsonFromKana(const std::string &kana) {
    return createNativeAccentPhrasesJson(parseNativeKana(kana));
}

std::string createNativeAudioQueryFromAccentPhrasesJson(const std::string &accentPhrasesJson) {
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

std::string createNativeAudioQueryFromKana(const std::string &kana) {
    return createNativeAudioQueryJson(parseNativeKana(kana), kana);
}
