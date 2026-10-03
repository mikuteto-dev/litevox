#include "model_metadata.hpp"

#include "json_utility.hpp"
#include "utility.hpp"

#include <charconv>
#include <stdexcept>

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <utility>

struct MergedSpeakerRecord {
    std::string speakerUuid;
    std::string speakerName;
    std::string versionText;
    std::vector<std::string> talkStyleObjects;
    std::vector<std::string> singerStyleObjects;
    std::set<uint32_t> talkStyleIds;
    std::set<uint32_t> singerStyleIds;
    uint64_t speakerOrderKey = std::numeric_limits<uint32_t>::max() + 1ULL;
};

static std::vector<MergedSpeakerRecord> createMergedSpeakerRecords(const std::string &metasJson);
static std::vector<std::string> createSortedStyleObjects(const std::vector<std::string> &styleObjects);

static std::string quoteMetadataJsonString(const std::string &text) {
    std::string quotedText = "\"";
    for (char character : text) {
        if (character == '"' || character == '\\') {
            quotedText.push_back('\\');
        }
        quotedText.push_back(character);
    }
    quotedText.push_back('"');
    return quotedText;
}

static size_t findTopLevelJsonFieldValuePosition(const std::string &jsonText, const std::string &fieldName) {
    std::string fieldPattern = "\"" + fieldName + "\"";
    bool isString = false;
    bool isEscaped = false;
    int depth = 0;
    for (size_t position = 0; position < jsonText.size(); position++) {
        char character = jsonText[position];
        if (isString) {
            if (isEscaped) {
                isEscaped = false;
            } else if (character == '\\') {
                isEscaped = true;
            } else if (character == '"') {
                isString = false;
            }
            continue;
        }
        if (character == '"') {
            if (depth == 1 && jsonText.compare(position, fieldPattern.size(), fieldPattern) == 0) {
                size_t colonPosition = jsonText.find(':', position + fieldPattern.size());
                if (colonPosition == std::string::npos) {
                    return std::string::npos;
                }
                size_t valuePosition = colonPosition + 1;
                while (valuePosition < jsonText.size() && std::isspace(static_cast<unsigned char>(jsonText[valuePosition]))) {
                    valuePosition++;
                }
                return valuePosition;
            }
            isString = true;
        } else if (character == '{' || character == '[') {
            depth++;
        } else if (character == '}' || character == ']') {
            depth--;
        }
    }
    return std::string::npos;
}

static std::string extractTopLevelJsonStringField(const std::string &jsonText, const std::string &fieldName) {
    size_t valuePosition = findTopLevelJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size() || jsonText[valuePosition] != '"') {
        return "";
    }
    return decodeJsonString(jsonText, valuePosition);
}

static bool extractTopLevelJsonNumberField(const std::string &jsonText, const std::string &fieldName, uint32_t &numberValue) {
    size_t valuePosition = findTopLevelJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size() || !std::isdigit(static_cast<unsigned char>(jsonText[valuePosition]))) {
        return false;
    }
    uint32_t parsedNumber = 0;
    while (valuePosition < jsonText.size() && std::isdigit(static_cast<unsigned char>(jsonText[valuePosition]))) {
        parsedNumber = parsedNumber * 10 + static_cast<uint32_t>(jsonText[valuePosition] - '0');
        valuePosition++;
    }
    numberValue = parsedNumber;
    return true;
}

std::vector<StyleRecord> extractStylesFromMetasJson(const std::string &metasJson) {
    std::vector<StyleRecord> styleRecords;
    for (const std::string &speakerObject : splitJsonObjects(metasJson)) {
        size_t stylesFieldPosition = speakerObject.find("\"styles\"");
        std::string speakerScope = stylesFieldPosition == std::string::npos ? speakerObject : speakerObject.substr(0, stylesFieldPosition);
        std::string speakerName = extractJsonStringField(speakerScope, "name");
        std::string speakerUuid = extractJsonStringField(speakerObject, "speaker_uuid");
        std::string stylesArrayText = extractJsonArrayField(speakerObject, "styles");
        for (const std::string &styleObject : splitJsonObjects(stylesArrayText)) {
            uint32_t styleId = 0;
            if (!extractJsonNumberField(styleObject, "id", styleId)) {
                continue;
            }
            StyleRecord styleRecord;
            styleRecord.styleId = styleId;
            styleRecord.speakerUuid = speakerUuid;
            styleRecord.speakerName = speakerName;
            styleRecord.styleName = extractJsonStringField(styleObject, "name");
            styleRecord.styleType = extractJsonStringField(styleObject, "type");
            if (styleRecord.styleType.empty()) {
                styleRecord.styleType = "talk";
            }
            styleRecords.push_back(std::move(styleRecord));
        }
    }
    return styleRecords;
}

static StyleRecord createStyleRecordFromObject(const MergedSpeakerRecord &speakerRecord, const std::string &styleObject) {
    StyleRecord styleRecord;
    extractJsonNumberField(styleObject, "id", styleRecord.styleId);
    styleRecord.speakerUuid = speakerRecord.speakerUuid;
    styleRecord.speakerName = speakerRecord.speakerName;
    styleRecord.styleName = extractJsonStringField(styleObject, "name");
    styleRecord.styleType = extractJsonStringField(styleObject, "type");
    if (styleRecord.styleType.empty()) {
        styleRecord.styleType = "talk";
    }
    return styleRecord;
}

std::vector<StyleRecord> extractOrderedStylesFromMetasJson(const std::string &metasJson) {
    std::vector<StyleRecord> styleRecords;
    for (const MergedSpeakerRecord &speakerRecord : createMergedSpeakerRecords(metasJson)) {
        for (const std::string &styleObject : createSortedStyleObjects(speakerRecord.talkStyleObjects)) {
            uint32_t styleId = 0;
            if (!extractJsonNumberField(styleObject, "id", styleId)) {
                continue;
            }
            styleRecords.push_back(createStyleRecordFromObject(speakerRecord, styleObject));
        }
        for (const std::string &styleObject : createSortedStyleObjects(speakerRecord.singerStyleObjects)) {
            uint32_t styleId = 0;
            if (!extractJsonNumberField(styleObject, "id", styleId)) {
                continue;
            }
            styleRecords.push_back(createStyleRecordFromObject(speakerRecord, styleObject));
        }
    }
    return styleRecords;
}

std::vector<uint32_t> extractStyleIds(const std::vector<StyleRecord> &styleRecords) {
    std::set<uint32_t> uniqueStyleIds;
    for (const StyleRecord &styleRecord : styleRecords) {
        uniqueStyleIds.insert(styleRecord.styleId);
    }
    return std::vector<uint32_t>(uniqueStyleIds.begin(), uniqueStyleIds.end());
}

const std::map<std::string, std::string> &GetAliases() {
    static const std::map<std::string, std::string> Aliases{
        {"zunda", "ずんだもん"}, {"zundamon", "ずんだもん"}, {"metan", "四国めたん"},
        {"tsumugi", "春日部つむぎ"}, {"hau", "雨晴はう"}, {"ritsu", "波音リツ"},
        {"takehiro", "玄野武宏"}, {"kotaro", "白上虎太郎"}, {"ryusei", "青山龍星"},
        {"himari", "冥鳴ひまり"}, {"sora", "九州そら"}, {"mochiko", "もち子さん"},
        {"chibi", "ちび式じい"}, {"miko", "櫻歌ミコ"}, {"nurse-t", "ナースロボ＿タイプＴ"},
        {"benizakura", "†聖騎士 紅桜†"}, {"akashi", "雀松朱司"}, {"sorin", "麒ヶ島宗麟"},
        {"nana", "春歌ナナ"}, {"aru", "猫使アル"}, {"bii", "猫使ビィ"}, {"maron", "栗田まろん"},
        {"aierutan", "あいえるたん"}, {"hanamaru", "満別花丸"}, {"nia", "琴詠ニア"},
        {"sayo", "小夜/SAYO"}, {"goki", "後鬼"}, {"voidoll", "Voidoll"}, {"zonko", "ぞん子"},
        {"tsurugi", "中部つるぎ"}, {"rito", "離途"}, {"kurosawa", "黒沢冴白"},
        {"yurei", "ユーレイちゃん"}, {"zunko", "東北ずん子"}, {"kiritan", "東北きりたん"},
        {"itako", "東北イタコ"}, {"ankomon", "あんこもん"}, {"tobari", "夜語トバリ"},
        {"mitama", "暁記ミタマ"}, {"yuka", "里石ユカ"}, {"usagi", "中国うさぎ"},
        {"mesuo", "剣崎雌雄"}, {"no7", "No.7"}, {"whitecul", "WhiteCUL"}
    };
    return Aliases;
}

uint32_t ResolveStyle(const std::string &Text, const std::vector<StyleRecord> &Styles, const std::string &Type) {
    if (Text.empty()) throw std::invalid_argument("話者指定が空です");
    if (Text[0] == '-' || Text[0] == '+' || (Text[0] >= '0' && Text[0] <= '9')) {
        uint32_t Id = 0;
        auto Result = std::from_chars(Text.data(), Text.data() + Text.size(), Id);
        if (Result.ec != std::errc() || Result.ptr != Text.data() + Text.size()) {
            throw std::invalid_argument("style ID が不正です: " + Text);
        }
        return Id;
    }
    size_t Separator = Text.find(':');
    std::string Name = lowercaseAscii(Text.substr(0, Separator));
    std::string Style = Separator == std::string::npos ? "" : lowercaseAscii(Text.substr(Separator + 1));
    if (Name.empty() || (Separator != std::string::npos && Style.empty())) {
        throw std::invalid_argument("話者指定は NAME または NAME:STYLE です: " + Text);
    }
    const auto &Aliases = GetAliases();
    auto Alias = Aliases.find(Name);
    if (Alias != Aliases.end()) Name = lowercaseAscii(Alias->second);
    std::map<uint32_t, const StyleRecord *> Matches, Defaults;
    for (const auto &Entry : Styles) {
        bool IsStyleId = Style == std::to_string(Entry.styleId);
        bool IsSong = Type == "frame_decode" && Entry.styleType == "sing" && IsStyleId;
        if ((Entry.styleType != Type && !IsSong) || lowercaseAscii(Entry.speakerName) != Name) continue;
        std::string StyleName = lowercaseAscii(Entry.styleName);
        bool IsDefault = StyleName == "ノーマル" || StyleName == "ふつう" || StyleName == "normal";
        if (!Style.empty() && !IsStyleId && StyleName != Style && !(Style == "normal" && IsDefault)) continue;
        Matches.emplace(Entry.styleId, &Entry);
        if (IsDefault) Defaults.emplace(Entry.styleId, &Entry);
    }
    if (Matches.empty()) throw std::invalid_argument("話者が見つかりません (" + Type + "): " + Text + "。models / singers で確認してください");
    if (Style.empty() && Defaults.size() == 1) return Defaults.begin()->first;
    if (Matches.size() == 1) return Matches.begin()->first;
    std::string Choices;
    for (const auto &Match : Matches) {
        if (!Choices.empty()) Choices += ", ";
        Choices += Match.second->speakerName + ":" + Match.second->styleName + "=" + std::to_string(Match.first);
    }
    throw std::invalid_argument("話者指定が曖昧です: " + Text + "。スタイルを指定してください: " + Choices);
}

std::string createCombinedMetasJson(const std::vector<std::string> &metasJsonTexts) {
    std::string combinedText = "[";
    bool hasPrevious = false;
    for (const std::string &metasJson : metasJsonTexts) {
        std::string bodyText = stripJsonArrayEnvelope(metasJson);
        if (bodyText.empty()) {
            continue;
        }
        if (hasPrevious) {
            combinedText += ",";
        }
        combinedText += bodyText;
        hasPrevious = true;
    }
    combinedText += "]";
    return combinedText;
}

static bool isTalkStyleType(const std::string &styleType) {
    return styleType.empty() || styleType == "talk";
}

static bool isSingerStyleType(const std::string &styleType) {
    return !styleType.empty() && styleType != "talk";
}

static uint64_t getJsonOrderKey(const std::string &jsonObject) {
    uint32_t orderNumber = 0;
    if (!extractTopLevelJsonNumberField(jsonObject, "order", orderNumber)) {
        return std::numeric_limits<uint32_t>::max() + 1ULL;
    }
    return orderNumber;
}

static std::string createSupportedFeaturesJson(const std::string &speakerUuid, bool hasTalkStyle, bool supportsMorphing, const std::map<std::string, std::string> &speakerSupportedFeaturesJsons) {
    auto featureEntry = speakerSupportedFeaturesJsons.find(speakerUuid);
    if (featureEntry != speakerSupportedFeaturesJsons.end()) {
        return featureEntry->second;
    }
    return std::string("{\"permitted_synthesis_morphing\":\"") + (hasTalkStyle && supportsMorphing ? "ALL" : "NOTHING") + "\"}";
}

static std::string createSpeakerStyleJson(const std::string &styleObject) {
    uint32_t styleId = 0;
    if (!extractJsonNumberField(styleObject, "id", styleId)) {
        return "";
    }
    std::string styleType = extractJsonStringField(styleObject, "type");
    if (styleType.empty()) {
        styleType = "talk";
    }
    std::ostringstream styleStream;
    styleStream << "{\"name\":" << quoteMetadataJsonString(extractJsonStringField(styleObject, "name")) << ",";
    styleStream << "\"id\":" << styleId << ",";
    styleStream << "\"type\":" << quoteMetadataJsonString(styleType);
    styleStream << "}";
    return styleStream.str();
}

static void appendStyleObject(std::vector<std::string> &styleObjects, std::set<uint32_t> &styleIds, const std::string &styleObject) {
    uint32_t styleId = 0;
    if (!extractJsonNumberField(styleObject, "id", styleId)) {
        return;
    }
    if (styleIds.find(styleId) != styleIds.end()) {
        return;
    }
    styleIds.insert(styleId);
    styleObjects.push_back(styleObject);
}

static std::vector<MergedSpeakerRecord> createMergedSpeakerRecords(const std::string &metasJson) {
    std::vector<MergedSpeakerRecord> speakerRecords;
    std::map<std::string, size_t> speakerIndexes;
    for (const std::string &speakerObject : splitJsonObjects(metasJson)) {
        std::string speakerUuid = extractJsonStringField(speakerObject, "speaker_uuid");
        if (speakerUuid.empty()) {
            continue;
        }
        size_t speakerIndex = 0;
        auto speakerEntry = speakerIndexes.find(speakerUuid);
        if (speakerEntry == speakerIndexes.end()) {
            speakerIndex = speakerRecords.size();
            speakerIndexes[speakerUuid] = speakerIndex;
            MergedSpeakerRecord speakerRecord;
            speakerRecord.speakerUuid = speakerUuid;
            speakerRecord.speakerName = extractJsonStringField(speakerObject, "name");
            speakerRecord.versionText = extractJsonStringField(speakerObject, "version");
            speakerRecord.speakerOrderKey = getJsonOrderKey(speakerObject);
            speakerRecords.push_back(std::move(speakerRecord));
        } else {
            speakerIndex = speakerEntry->second;
        }
        MergedSpeakerRecord &speakerRecord = speakerRecords[speakerIndex];
        for (const std::string &styleObject : splitJsonObjects(extractJsonArrayField(speakerObject, "styles"))) {
            std::string styleType = extractJsonStringField(styleObject, "type");
            if (isTalkStyleType(styleType)) {
                appendStyleObject(speakerRecord.talkStyleObjects, speakerRecord.talkStyleIds, styleObject);
            } else if (isSingerStyleType(styleType)) {
                appendStyleObject(speakerRecord.singerStyleObjects, speakerRecord.singerStyleIds, styleObject);
            }
        }
    }
    std::stable_sort(speakerRecords.begin(), speakerRecords.end(), [](const MergedSpeakerRecord &leftSpeakerRecord, const MergedSpeakerRecord &rightSpeakerRecord) {
        return leftSpeakerRecord.speakerOrderKey < rightSpeakerRecord.speakerOrderKey;
    });
    return speakerRecords;
}

static std::vector<std::string> createSortedStyleObjects(const std::vector<std::string> &styleObjects) {
    std::vector<std::string> sortedStyleObjects = styleObjects;
    std::stable_sort(sortedStyleObjects.begin(), sortedStyleObjects.end(), [](const std::string &leftStyleObject, const std::string &rightStyleObject) {
        return getJsonOrderKey(leftStyleObject) < getJsonOrderKey(rightStyleObject);
    });
    return sortedStyleObjects;
}

static void appendSpeakerMetadataJson(std::ostringstream &speakerStream, const MergedSpeakerRecord &speakerRecord, const std::vector<std::string> &styleObjects, const std::string &versionText, bool supportsMorphing, const std::map<std::string, std::string> &speakerSupportedFeaturesJsons) {
    speakerStream << "{\"name\":" << quoteMetadataJsonString(speakerRecord.speakerName) << ",";
    speakerStream << "\"speaker_uuid\":" << quoteMetadataJsonString(speakerRecord.speakerUuid) << ",";
    speakerStream << "\"styles\":[";
    bool hasPreviousStyle = false;
    for (const std::string &styleObject : styleObjects) {
        std::string styleJson = createSpeakerStyleJson(styleObject);
        if (styleJson.empty()) {
            continue;
        }
        if (hasPreviousStyle) {
            speakerStream << ",";
        }
        speakerStream << styleJson;
        hasPreviousStyle = true;
    }
    speakerStream << "]";
    if (!versionText.empty()) {
        speakerStream << ",\"version\":" << quoteMetadataJsonString(versionText);
    }
    speakerStream << ",\"supported_features\":" << createSupportedFeaturesJson(speakerRecord.speakerUuid, !speakerRecord.talkStyleObjects.empty(), supportsMorphing, speakerSupportedFeaturesJsons);
    speakerStream << "}";
}

std::string createSpeakersJson(const std::string &metasJson, bool supportsMorphing) {
    return createSpeakersJson(metasJson, supportsMorphing, {});
}

std::string createSpeakersJson(const std::string &metasJson, bool supportsMorphing, const std::map<std::string, std::string> &speakerSupportedFeaturesJsons) {
    std::ostringstream speakersStream;
    speakersStream << "[";
    bool hasPreviousSpeaker = false;
    for (const MergedSpeakerRecord &speakerRecord : createMergedSpeakerRecords(metasJson)) {
        if (speakerRecord.talkStyleObjects.empty()) {
            continue;
        }
        if (hasPreviousSpeaker) {
            speakersStream << ",";
        }
        appendSpeakerMetadataJson(speakersStream, speakerRecord, createSortedStyleObjects(speakerRecord.talkStyleObjects), speakerRecord.versionText, supportsMorphing, speakerSupportedFeaturesJsons);
        hasPreviousSpeaker = true;
    }
    speakersStream << "]";
    return speakersStream.str();
}

std::string createSingersJson(const std::string &metasJson) {
    return createSingersJson(metasJson, {});
}

std::string createSingersJson(const std::string &metasJson, const std::map<std::string, std::string> &speakerSupportedFeaturesJsons) {
    std::ostringstream singersStream;
    singersStream << "[";
    bool hasPreviousSpeaker = false;
    for (const MergedSpeakerRecord &speakerRecord : createMergedSpeakerRecords(metasJson)) {
        if (speakerRecord.singerStyleObjects.empty()) {
            continue;
        }
        if (hasPreviousSpeaker) {
            singersStream << ",";
        }
        appendSpeakerMetadataJson(singersStream, speakerRecord, createSortedStyleObjects(speakerRecord.singerStyleObjects), speakerRecord.versionText, true, speakerSupportedFeaturesJsons);
        hasPreviousSpeaker = true;
    }
    singersStream << "]";
    return singersStream.str();
}

std::string createStyleTable(const std::vector<std::pair<std::filesystem::path, std::vector<StyleRecord>>> &modelStyleGroups) {
    std::ostringstream tableStream;
    tableStream << "style_id\ttype\tspeaker_uuid\tspeaker\tstyle\tvvm\n";
    for (const auto &modelStyleGroup : modelStyleGroups) {
        std::string modelName = modelStyleGroup.first.filename().string();
        for (const StyleRecord &styleRecord : modelStyleGroup.second) {
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

std::string createSpeakerInfoJson(const std::string &metasJson, const std::string &speakerUuid) {
    for (const std::string &speakerObject : splitJsonObjects(metasJson)) {
        if (extractTopLevelJsonStringField(speakerObject, "speaker_uuid") != speakerUuid) {
            continue;
        }
        std::string stylesArrayText = extractJsonArrayField(speakerObject, "styles");
        std::ostringstream speakerInfoStream;
        speakerInfoStream << "{\"speaker_uuid\":" << quoteMetadataJsonString(speakerUuid) << ",";
        speakerInfoStream << "\"name\":" << quoteMetadataJsonString(extractTopLevelJsonStringField(speakerObject, "name")) << ",";
        speakerInfoStream << "\"version\":" << quoteMetadataJsonString(extractTopLevelJsonStringField(speakerObject, "version")) << ",";
        speakerInfoStream << "\"policy\":\"\",\"portrait\":\"\",\"style_infos\":[";
        bool hasPrevious = false;
        for (const std::string &styleObject : splitJsonObjects(stylesArrayText)) {
            uint32_t styleId = 0;
            if (!extractJsonNumberField(styleObject, "id", styleId)) {
                continue;
            }
            std::string styleType = extractJsonStringField(styleObject, "type");
            if (styleType.empty()) {
                styleType = "talk";
            }
            if (hasPrevious) {
                speakerInfoStream << ",";
            }
            speakerInfoStream << "{\"id\":" << styleId << ",";
            speakerInfoStream << "\"name\":" << quoteMetadataJsonString(extractJsonStringField(styleObject, "name")) << ",";
            speakerInfoStream << "\"type\":" << quoteMetadataJsonString(styleType) << ",";
            speakerInfoStream << "\"icon\":\"\",\"portrait\":\"\",\"voice_samples\":[]}";
            hasPrevious = true;
        }
        speakerInfoStream << "]}";
        return speakerInfoStream.str();
    }
    throw std::runtime_error("speaker_uuid が見つかりません: " + speakerUuid);
}
