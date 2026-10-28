#include "utility.hpp"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <sys/resource.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <bcrypt.h>
#include <processthreadsapi.h>
#include <psapi.h>
#include <windows.h>
#else
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;

std::string readTextFile(const fs::path &filePath) {
    std::ifstream inputStream(filePath, std::ios::binary);
    if (!inputStream) {
        throw std::runtime_error("ファイルを読めません: " + filePath.string());
    }
    std::ostringstream bufferStream;
    bufferStream << inputStream.rdbuf();
    return bufferStream.str();
}

std::vector<uint8_t> readBinaryFile(const fs::path &filePath) {
    std::ifstream inputStream(filePath, std::ios::binary);
    if (!inputStream) {
        throw std::runtime_error("ファイルを読めません: " + filePath.string());
    }
    inputStream.seekg(0, std::ios::end);
    std::streamoff fileSize = inputStream.tellg();
    if (fileSize < 0) {
        throw std::runtime_error("ファイルサイズを取得できません: " + filePath.string());
    }
    inputStream.seekg(0, std::ios::beg);
    std::vector<uint8_t> fileBytes(static_cast<size_t>(fileSize));
    if (!fileBytes.empty()) {
        inputStream.read(reinterpret_cast<char *>(fileBytes.data()), fileSize);
        if (!inputStream) {
            throw std::runtime_error("ファイルを読めません: " + filePath.string());
        }
    }
    return fileBytes;
}

void writeTextFile(const fs::path &filePath, const std::string &bodyText) {
    fs::path parentPath = filePath.parent_path();
    if (!parentPath.empty()) {
        fs::create_directories(parentPath);
    }
    std::ofstream outputStream(filePath, std::ios::binary);
    if (!outputStream) {
        throw std::runtime_error("ファイルを書けません: " + filePath.string());
    }
    outputStream << bodyText;
}

void writeBinaryFile(const fs::path &filePath, const std::vector<uint8_t> &fileBytes) {
    fs::path parentPath = filePath.parent_path();
    if (!parentPath.empty()) {
        fs::create_directories(parentPath);
    }
    std::ofstream outputStream(filePath, std::ios::binary);
    if (!outputStream) {
        throw std::runtime_error("ファイルを書けません: " + filePath.string());
    }
    outputStream.write(reinterpret_cast<const char *>(fileBytes.data()), static_cast<std::streamsize>(fileBytes.size()));
    outputStream.close();
    if (!outputStream) {
        throw std::runtime_error("ファイルを書けません: " + filePath.string());
    }
}

void writeBinaryStdout(const std::vector<uint8_t> &fileBytes) {
    std::cout.write(reinterpret_cast<const char *>(fileBytes.data()), static_cast<std::streamsize>(fileBytes.size()));
}

std::string trimAscii(const std::string &text) {
    size_t startPosition = 0;
    while (startPosition < text.size() && std::isspace(static_cast<unsigned char>(text[startPosition]))) {
        startPosition++;
    }
    size_t endPosition = text.size();
    while (endPosition > startPosition && std::isspace(static_cast<unsigned char>(text[endPosition - 1]))) {
        endPosition--;
    }
    return text.substr(startPosition, endPosition - startPosition);
}

std::string lowercaseAscii(std::string text) {
    for (char &character : text) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return text;
}

std::vector<uint8_t> makeBodyBytes(const std::string &bodyText) {
    return std::vector<uint8_t>(bodyText.begin(), bodyText.end());
}

std::string createSha256Hex(const uint8_t *bytes, size_t byteCount) {
#if defined(__APPLE__)
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes, static_cast<CC_LONG>(byteCount), digest);
#elif defined(_WIN32)
    BCRYPT_ALG_HANDLE algorithmHandle = nullptr;
    BCRYPT_HASH_HANDLE hashHandle = nullptr;
    DWORD objectLength = 0;
    DWORD digestLength = 0;
    DWORD resultLength = 0;
    std::vector<uint8_t> hashObject;
    std::vector<uint8_t> digest;
    NTSTATUS status = BCryptOpenAlgorithmProvider(&algorithmHandle, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
    if (status < 0) {
        throw std::runtime_error("SHA-256 provider を開けません");
    }
    status = BCryptGetProperty(algorithmHandle, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&objectLength), sizeof(objectLength), &resultLength, 0);
    if (status < 0) {
        BCryptCloseAlgorithmProvider(algorithmHandle, 0);
        throw std::runtime_error("SHA-256 object length を取得できません");
    }
    status = BCryptGetProperty(algorithmHandle, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&digestLength), sizeof(digestLength), &resultLength, 0);
    if (status < 0) {
        BCryptCloseAlgorithmProvider(algorithmHandle, 0);
        throw std::runtime_error("SHA-256 hash length を取得できません");
    }
    hashObject.resize(objectLength);
    digest.resize(digestLength);
    status = BCryptCreateHash(algorithmHandle, &hashHandle, hashObject.data(), objectLength, nullptr, 0, 0);
    if (status < 0) {
        BCryptCloseAlgorithmProvider(algorithmHandle, 0);
        throw std::runtime_error("SHA-256 hash を作れません");
    }
    status = BCryptHashData(hashHandle, const_cast<PUCHAR>(reinterpret_cast<const UCHAR *>(bytes)), static_cast<ULONG>(byteCount), 0);
    if (status < 0) {
        BCryptDestroyHash(hashHandle);
        BCryptCloseAlgorithmProvider(algorithmHandle, 0);
        throw std::runtime_error("SHA-256 data を処理できません");
    }
    status = BCryptFinishHash(hashHandle, digest.data(), digestLength, 0);
    BCryptDestroyHash(hashHandle);
    BCryptCloseAlgorithmProvider(algorithmHandle, 0);
    if (status < 0) {
        throw std::runtime_error("SHA-256 hash を確定できません");
    }
#else
    throw std::runtime_error("SHA-256 はこの環境では未対応です");
#endif
    std::ostringstream hashStream;
#if defined(__APPLE__)
    for (unsigned char digestByte : digest) {
        hashStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(digestByte);
    }
#else
    for (uint8_t digestByte : digest) {
        hashStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(digestByte);
    }
#endif
    return hashStream.str();
}

fs::path getExecutableDirectory() {
#if defined(__APPLE__)
    uint32_t bufferSize = 0;
    _NSGetExecutablePath(nullptr, &bufferSize);
    std::string executablePathText(bufferSize, '\0');
    if (_NSGetExecutablePath(executablePathText.data(), &bufferSize) != 0) {
        throw std::runtime_error("実行ファイルパスを取得できません");
    }
    executablePathText.resize(std::strlen(executablePathText.c_str()));
    return fs::weakly_canonical(fs::path(executablePathText)).parent_path();
#elif defined(_WIN32)
    std::wstring executablePathText(MAX_PATH, L'\0');
    DWORD pathLength = GetModuleFileNameW(nullptr, executablePathText.data(), static_cast<DWORD>(executablePathText.size()));
    if (pathLength == 0) {
        throw std::runtime_error("実行ファイルパスを取得できません");
    }
    while (pathLength >= executablePathText.size() - 1) {
        executablePathText.resize(executablePathText.size() * 2);
        pathLength = GetModuleFileNameW(nullptr, executablePathText.data(), static_cast<DWORD>(executablePathText.size()));
        if (pathLength == 0) {
            throw std::runtime_error("実行ファイルパスを取得できません");
        }
    }
    executablePathText.resize(pathLength);
    return fs::weakly_canonical(fs::path(executablePathText)).parent_path();
#else
    return fs::current_path();
#endif
}

void ensurePathExists(const fs::path &pathToCheck, const std::string &label) {
    if (!fs::exists(pathToCheck)) {
        throw std::runtime_error(label + " がありません: " + pathToCheck.string());
    }
}

fs::path findFirstExistingPath(const fs::path &directoryPath, const std::vector<std::string> &fileNames) {
    if (directoryPath.empty()) {
        return {};
    }
    for (const std::string &fileName : fileNames) {
        fs::path candidatePath = directoryPath / fileName;
        if (fs::exists(candidatePath)) {
            return candidatePath;
        }
    }
    return {};
}

fs::path getFirstNamedPath(const fs::path &directoryPath, const std::vector<std::string> &fileNames) {
    fs::path existingPath = findFirstExistingPath(directoryPath, fileNames);
    if (!existingPath.empty()) {
        return existingPath;
    }
    if (fileNames.empty()) {
        return directoryPath;
    }
    return directoryPath / fileNames.front();
}

std::string getPlatformExecutableFilename(const std::string &baseName) {
#if defined(_WIN32)
    return baseName + ".exe";
#else
    return baseName;
#endif
}

std::vector<std::string> getVoicevoxCoreLibraryFileNames() {
#if defined(_WIN32)
    return {"voicevox_core.dll", "libvoicevox_core.dylib"};
#else
    return {"libvoicevox_core.dylib", "voicevox_core.dll"};
#endif
}

std::vector<std::string> getVoicevoxOnnxruntimeLibraryFileNames() {
#if defined(_WIN32)
    return {"voicevox_onnxruntime.dll", "libvoicevox_onnxruntime.dylib"};
#else
    return {"libvoicevox_onnxruntime.dylib", "voicevox_onnxruntime.dll"};
#endif
}

void setEnvironmentVariable(const std::string &name, const std::string &value) {
#if defined(_WIN32)
    if (_putenv_s(name.c_str(), value.c_str()) != 0) {
        throw std::runtime_error("環境変数を設定できません: " + name);
    }
#else
    if (setenv(name.c_str(), value.c_str(), 1) != 0) {
        throw std::runtime_error("環境変数を設定できません: " + name);
    }
#endif
}

void clearEnvironmentVariable(const std::string &name) {
#if defined(_WIN32)
    if (_putenv_s(name.c_str(), "") != 0) {
        throw std::runtime_error("環境変数を削除できません: " + name);
    }
#else
    unsetenv(name.c_str());
#endif
}

uint64_t getPeakResidentBytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX memoryCounters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&memoryCounters), sizeof(memoryCounters))) {
        return 0;
    }
    return static_cast<uint64_t>(memoryCounters.PeakWorkingSetSize);
#elif defined(__APPLE__)
    struct rusage resourceUsage{};
    if (getrusage(RUSAGE_SELF, &resourceUsage) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(resourceUsage.ru_maxrss);
#else
    struct rusage resourceUsage{};
    if (getrusage(RUSAGE_SELF, &resourceUsage) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(resourceUsage.ru_maxrss) * 1024;
#endif
}

fs::path createTemporaryFilePath(const std::string &baseName, const std::string &extensionText) {
    static std::atomic<uint64_t> counter{0};
    uint64_t serial = counter.fetch_add(1, std::memory_order_relaxed);
    uint64_t timePart = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937_64 randomEngine(timePart ^ (serial << 1));
    for (size_t attemptIndex = 0; attemptIndex < 1024; attemptIndex++) {
        std::ostringstream fileNameStream;
        fileNameStream << baseName << "-" << std::hex << randomEngine();
        if (!extensionText.empty()) {
            fileNameStream << extensionText;
        }
        fs::path filePath = fs::temp_directory_path() / fileNameStream.str();
        if (!fs::exists(filePath)) {
            return filePath;
        }
    }
    throw std::runtime_error("一時ファイル path を作れません");
}
