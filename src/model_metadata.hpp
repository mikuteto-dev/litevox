#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

struct StyleRecord {
    uint32_t styleId = 0;
    std::string speakerUuid;
    std::string speakerName;
    std::string styleName;
    std::string styleType;
};

std::vector<StyleRecord> extractStylesFromMetasJson(const std::string &metasJson);
std::vector<StyleRecord> extractOrderedStylesFromMetasJson(const std::string &metasJson);
std::vector<uint32_t> extractStyleIds(const std::vector<StyleRecord> &styleRecords);
const std::map<std::string, std::string> &GetAliases();
uint32_t ResolveStyle(const std::string &Text, const std::vector<StyleRecord> &Styles, const std::string &Type = "talk");
std::string createCombinedMetasJson(const std::vector<std::string> &metasJsonTexts);
std::string createSpeakersJson(const std::string &metasJson, bool supportsMorphing = true);
std::string createSpeakersJson(const std::string &metasJson, bool supportsMorphing, const std::map<std::string, std::string> &speakerSupportedFeaturesJsons);
std::string createSingersJson(const std::string &metasJson);
std::string createSingersJson(const std::string &metasJson, const std::map<std::string, std::string> &speakerSupportedFeaturesJsons);
std::string createStyleTable(const std::vector<std::pair<std::filesystem::path, std::vector<StyleRecord>>> &modelStyleGroups);
std::string createSpeakerInfoJson(const std::string &metasJson, const std::string &speakerUuid);
