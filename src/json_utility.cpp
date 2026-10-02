#include "json_utility.hpp"

#include "utility.hpp"

#include <cctype>
#include <charconv>
#include <codecvt>
#include <locale>
#include <stdexcept>
#include <iomanip>
#include <sstream>

size_t findJsonMatchingToken(const std::string &jsonText, size_t openPosition, char openToken, char closeToken) {
    bool isString = false;
    bool isEscaped = false;
    int depth = 0;
    for (size_t position = openPosition; position < jsonText.size(); position++) {
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
            isString = true;
        } else if (character == openToken) {
            depth++;
        } else if (character == closeToken) {
            depth--;
            if (depth == 0) {
                return position;
            }
        }
    }
    return std::string::npos;
}

std::vector<std::string> splitJsonObjects(const std::string &jsonArrayText) {
    std::vector<std::string> objectTexts;
    size_t searchPosition = 0;
    while (true) {
        size_t openPosition = jsonArrayText.find('{', searchPosition);
        if (openPosition == std::string::npos) {
            break;
        }
        size_t closePosition = findJsonMatchingToken(jsonArrayText, openPosition, '{', '}');
        if (closePosition == std::string::npos) {
            break;
        }
        objectTexts.push_back(jsonArrayText.substr(openPosition, closePosition - openPosition + 1));
        searchPosition = closePosition + 1;
    }
    return objectTexts;
}

static char16_t ReadUnicode(const std::string &Json, size_t &Position) {
    if (Position + 4 >= Json.size()) {
        throw std::invalid_argument("JSON Unicode escape が不完全です");
    }
    uint32_t Value = 0;
    const char *Start = Json.data() + Position + 1;
    auto Result = std::from_chars(Start, Start + 4, Value, 16);
    if (Result.ec != std::errc() || Result.ptr != Start + 4) {
        throw std::invalid_argument("JSON Unicode escape が不正です");
    }
    Position += 4;
    return static_cast<char16_t>(Value);
}

std::string decodeJsonString(const std::string &Json, size_t QuotePosition) {
    if (QuotePosition >= Json.size() || Json[QuotePosition] != '"') {
        throw std::invalid_argument("JSON string が必要です");
    }
    std::string Text;
    for (size_t Position = QuotePosition + 1; Position < Json.size(); Position++) {
        char Character = Json[Position];
        if (Character == '"') {
            return Text;
        }
        if (static_cast<unsigned char>(Character) < 0x20) {
            throw std::invalid_argument("JSON string に制御文字があります");
        }
        if (Character != '\\') {
            Text.push_back(Character);
            continue;
        }
        if (++Position >= Json.size()) {
            break;
        }
        switch (Json[Position]) {
        case '"': case '\\': case '/': Text.push_back(Json[Position]); break;
        case 'b': Text.push_back('\b'); break;
        case 'f': Text.push_back('\f'); break;
        case 'n': Text.push_back('\n'); break;
        case 'r': Text.push_back('\r'); break;
        case 't': Text.push_back('\t'); break;
        case 'u': {
            std::u16string Units(1, ReadUnicode(Json, Position));
            if (Units[0] >= 0xd800 && Units[0] <= 0xdbff) {
                if (Json.compare(Position + 1, 2, "\\u") != 0) {
                    throw std::invalid_argument("JSON surrogate pair が不完全です");
                }
                Position += 2;
                Units.push_back(ReadUnicode(Json, Position));
                if (Units[1] < 0xdc00 || Units[1] > 0xdfff) {
                    throw std::invalid_argument("JSON surrogate pair が不正です");
                }
            } else if (Units[0] >= 0xdc00 && Units[0] <= 0xdfff) {
                throw std::invalid_argument("JSON surrogate pair が不正です");
            }
            Text += std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>().to_bytes(Units);
            break;
        }
        default: throw std::invalid_argument("JSON escape が不正です");
        }
    }
    throw std::invalid_argument("JSON string が閉じていません");
}

size_t findJsonFieldValuePosition(const std::string &jsonText, const std::string &fieldName) {
    std::string fieldPattern = "\"" + fieldName + "\"";
    for (size_t Position = jsonText.find(fieldPattern); Position != std::string::npos; Position = jsonText.find(fieldPattern, Position + fieldPattern.size())) {
        size_t Colon = Position + fieldPattern.size();
        while (Colon < jsonText.size() && std::isspace(static_cast<unsigned char>(jsonText[Colon]))) {
            Colon++;
        }
        if (Colon >= jsonText.size() || jsonText[Colon] != ':') {
            continue;
        }
        size_t Value = Colon + 1;
        while (Value < jsonText.size() && std::isspace(static_cast<unsigned char>(jsonText[Value]))) {
            Value++;
        }
        return Value;
    }
    return std::string::npos;
}

std::string extractJsonStringField(const std::string &jsonText, const std::string &fieldName) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size() || jsonText[valuePosition] != '"') {
        return "";
    }
    return decodeJsonString(jsonText, valuePosition);
}

bool extractJsonNumberField(const std::string &jsonText, const std::string &fieldName, uint32_t &numberValue) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
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

std::string extractJsonArrayField(const std::string &jsonText, const std::string &fieldName) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size() || jsonText[valuePosition] != '[') {
        return "";
    }
    size_t closePosition = findJsonMatchingToken(jsonText, valuePosition, '[', ']');
    if (closePosition == std::string::npos) {
        return "";
    }
    return jsonText.substr(valuePosition, closePosition - valuePosition + 1);
}

std::string extractJsonObjectField(const std::string &jsonText, const std::string &fieldName) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size() || jsonText[valuePosition] != '{') {
        return "";
    }
    size_t closePosition = findJsonMatchingToken(jsonText, valuePosition, '{', '}');
    if (closePosition == std::string::npos) {
        return "";
    }
    return jsonText.substr(valuePosition, closePosition - valuePosition + 1);
}

std::string stripJsonArrayEnvelope(const std::string &jsonText) {
    std::string trimmedText = trimAscii(jsonText);
    if (trimmedText.size() < 2 || trimmedText.front() != '[' || trimmedText.back() != ']') {
        return "";
    }
    return trimAscii(trimmedText.substr(1, trimmedText.size() - 2));
}

std::string quoteJsonString(const std::string &text) {
    std::ostringstream quotedStream;
    quotedStream << "\"";
    for (unsigned char character : text) {
        if (character == '\\' || character == '"') {
            quotedStream << "\\" << static_cast<char>(character);
        } else if (character == '\n') {
            quotedStream << "\\n";
        } else if (character == '\r') {
            quotedStream << "\\r";
        } else if (character == '\t') {
            quotedStream << "\\t";
        } else if (character < 0x20) {
            quotedStream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned int>(character) << std::dec;
        } else {
            quotedStream << static_cast<char>(character);
        }
    }
    quotedStream << "\"";
    return quotedStream.str();
}
