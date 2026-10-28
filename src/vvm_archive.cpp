#include "vvm_archive.hpp"

#include "json_utility.hpp"
#include "utility.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <zlib.h>

namespace fs = std::filesystem;

static uint16_t readLittleUint16(const std::vector<uint8_t> &archiveBytes, size_t offset) {
    if (offset + 2 > archiveBytes.size()) {
        throw std::runtime_error("ZIP を読めません");
    }
    return static_cast<uint16_t>(archiveBytes[offset]) |
           static_cast<uint16_t>(archiveBytes[offset + 1] << 8);
}

static uint32_t readLittleUint32(const std::vector<uint8_t> &archiveBytes, size_t offset) {
    if (offset + 4 > archiveBytes.size()) {
        throw std::runtime_error("ZIP を読めません");
    }
    return static_cast<uint32_t>(archiveBytes[offset]) |
           (static_cast<uint32_t>(archiveBytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(archiveBytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(archiveBytes[offset + 3]) << 24);
}

struct EndOfCentralDirectoryRecord {
    uint16_t entryCount = 0;
    uint32_t directorySize = 0;
    uint32_t directoryOffset = 0;
};

static uint64_t readArchiveSize(const fs::path &archivePath) {
    std::ifstream inputStream(archivePath, std::ios::binary);
    if (!inputStream) {
        throw std::runtime_error("VVM を読めません: " + archivePath.string());
    }
    inputStream.seekg(0, std::ios::end);
    std::streamoff fileSize = inputStream.tellg();
    if (fileSize < 0) {
        throw std::runtime_error("VVM サイズを読めません: " + archivePath.string());
    }
    return static_cast<uint64_t>(fileSize);
}

static std::ifstream openArchive(const fs::path &archivePath) {
    std::ifstream inputStream(archivePath, std::ios::binary);
    if (!inputStream) {
        throw std::runtime_error("VVM を読めません: " + archivePath.string());
    }
    return inputStream;
}

static std::vector<uint8_t> readArchiveBytesAt(std::ifstream &inputStream, const fs::path &archivePath, uint64_t offset, uint64_t byteCount) {
    if (byteCount > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("VVM 読み込みサイズが大きすぎます: " + archivePath.string());
    }
    inputStream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!inputStream) {
        throw std::runtime_error("VVM seek に失敗しました: " + archivePath.string());
    }
    std::vector<uint8_t> archiveBytes(static_cast<size_t>(byteCount));
    if (!archiveBytes.empty()) {
        inputStream.read(reinterpret_cast<char *>(archiveBytes.data()), static_cast<std::streamsize>(archiveBytes.size()));
        if (!inputStream) {
            throw std::runtime_error("VVM 部分読み込みに失敗しました: " + archivePath.string());
        }
    }
    return archiveBytes;
}

static std::vector<uint8_t> readArchiveBytesAt(const fs::path &archivePath, uint64_t offset, uint64_t byteCount) {
    std::ifstream inputStream = openArchive(archivePath);
    return readArchiveBytesAt(inputStream, archivePath, offset, byteCount);
}

static EndOfCentralDirectoryRecord readEndOfCentralDirectory(const fs::path &archivePath, uint64_t archiveSize) {
    const size_t minimumEndSize = 22;
    const size_t maximumCommentSize = 65535;
    const uint32_t endSignature = 0x06054b50;
    if (archiveSize < minimumEndSize) {
        throw std::runtime_error("ZIP 終端がありません");
    }
    uint64_t tailSize = archiveSize > minimumEndSize + maximumCommentSize ? minimumEndSize + maximumCommentSize : archiveSize;
    uint64_t tailOffset = archiveSize - tailSize;
    std::vector<uint8_t> tailBytes = readArchiveBytesAt(archivePath, tailOffset, tailSize);
    size_t position = tailBytes.size() - minimumEndSize;
    while (true) {
        if (readLittleUint32(tailBytes, position) == endSignature) {
            EndOfCentralDirectoryRecord endRecord;
            endRecord.entryCount = readLittleUint16(tailBytes, position + 10);
            endRecord.directorySize = readLittleUint32(tailBytes, position + 12);
            endRecord.directoryOffset = readLittleUint32(tailBytes, position + 16);
            return endRecord;
        }
        if (position == 0) {
            break;
        }
        position--;
    }
    throw std::runtime_error("ZIP 終端がありません");
}

static bool hasPrefix(const std::string &text, const std::string &prefixText) {
    return text.size() >= prefixText.size() && text.compare(0, prefixText.size(), prefixText) == 0;
}

static bool hasSuffix(const std::string &text, const std::string &suffixText) {
    return text.size() >= suffixText.size() && text.compare(text.size() - suffixText.size(), suffixText.size(), suffixText) == 0;
}

static uint64_t readArchiveEntryDataOffset(std::ifstream &inputStream, const fs::path &archivePath, uint64_t archiveSize, const VvmEntryRecord &archiveEntry) {
    const uint32_t localHeaderSignature = 0x04034b50;
    if (archiveEntry.localHeaderOffset + 30 > archiveSize) {
        throw std::runtime_error("ZIP local header が壊れています: " + archiveEntry.name);
    }
    std::vector<uint8_t> localHeaderBytes = readArchiveBytesAt(inputStream, archivePath, archiveEntry.localHeaderOffset, 30);
    if (readLittleUint32(localHeaderBytes, 0) != localHeaderSignature) {
        throw std::runtime_error("ZIP local header が壊れています: " + archiveEntry.name);
    }
    if ((archiveEntry.generalPurposeFlags & 1) != 0) {
        throw std::runtime_error("暗号化 VVM entry は未対応です: " + archiveEntry.name);
    }
    uint16_t nameLength = readLittleUint16(localHeaderBytes, 26);
    uint16_t extraLength = readLittleUint16(localHeaderBytes, 28);
    uint64_t bodyPosition = archiveEntry.localHeaderOffset + 30 + nameLength + extraLength;
    if (bodyPosition + archiveEntry.compressedSize > archiveSize) {
        throw std::runtime_error("ZIP entry 本体が壊れています: " + archiveEntry.name);
    }
    return bodyPosition;
}

static std::vector<VvmEntryRecord> readCentralDirectory(const fs::path &archivePath, uint64_t archiveSize) {
    const uint32_t centralHeaderSignature = 0x02014b50;
    EndOfCentralDirectoryRecord endRecord = readEndOfCentralDirectory(archivePath, archiveSize);
    if (endRecord.directoryOffset == 0xffffffff || endRecord.directorySize == 0xffffffff || endRecord.entryCount == 0xffff) {
        throw std::runtime_error("ZIP64 VVM は未対応です");
    }
    if (static_cast<uint64_t>(endRecord.directoryOffset) + endRecord.directorySize > archiveSize) {
        throw std::runtime_error("ZIP central directory が壊れています");
    }
    std::ifstream inputStream = openArchive(archivePath);
    std::vector<uint8_t> directoryBytes = readArchiveBytesAt(inputStream, archivePath, endRecord.directoryOffset, endRecord.directorySize);
    std::vector<VvmEntryRecord> entries;
    entries.reserve(endRecord.entryCount);
    size_t position = 0;
    size_t directoryEndPosition = directoryBytes.size();
    for (uint16_t entryIndex = 0; entryIndex < endRecord.entryCount; entryIndex++) {
        if (position + 46 > directoryEndPosition || readLittleUint32(directoryBytes, position) != centralHeaderSignature) {
            throw std::runtime_error("ZIP central directory entry が壊れています");
        }
        uint16_t nameLength = readLittleUint16(directoryBytes, position + 28);
        uint16_t extraLength = readLittleUint16(directoryBytes, position + 30);
        uint16_t commentLength = readLittleUint16(directoryBytes, position + 32);
        size_t namePosition = position + 46;
        size_t nextPosition = namePosition + nameLength + extraLength + commentLength;
        if (nextPosition > directoryEndPosition) {
            throw std::runtime_error("ZIP central directory entry 名が壊れています");
        }
        VvmEntryRecord archiveEntry;
        archiveEntry.name.assign(reinterpret_cast<const char *>(directoryBytes.data() + namePosition), nameLength);
        archiveEntry.generalPurposeFlags = readLittleUint16(directoryBytes, position + 8);
        archiveEntry.compressionMethod = readLittleUint16(directoryBytes, position + 10);
        archiveEntry.crc32 = readLittleUint32(directoryBytes, position + 16);
        archiveEntry.compressedSize = readLittleUint32(directoryBytes, position + 20);
        archiveEntry.uncompressedSize = readLittleUint32(directoryBytes, position + 24);
        archiveEntry.localHeaderOffset = readLittleUint32(directoryBytes, position + 42);
        if (archiveEntry.compressedSize == 0xffffffff || archiveEntry.uncompressedSize == 0xffffffff || archiveEntry.localHeaderOffset == 0xffffffff) {
            throw std::runtime_error("ZIP64 VVM entry は未対応です: " + archiveEntry.name);
        }
        archiveEntry.dataOffset = readArchiveEntryDataOffset(inputStream, archivePath, archiveSize, archiveEntry);
        entries.push_back(std::move(archiveEntry));
        position = nextPosition;
    }
    return entries;
}

static const VvmEntryRecord *findArchiveEntry(const std::vector<VvmEntryRecord> &entries, const std::string &entryName) {
    for (const VvmEntryRecord &archiveEntry : entries) {
        if (archiveEntry.name == entryName) {
            return &archiveEntry;
        }
    }
    return nullptr;
}

static std::vector<uint8_t> inflateEntryBytes(const std::vector<uint8_t> &compressedBytes, uint64_t uncompressedSize, const std::string &entryName) {
    if (compressedBytes.size() > std::numeric_limits<uInt>::max() || uncompressedSize > std::numeric_limits<uInt>::max()) {
        throw std::runtime_error("deflate entry が大きすぎます: " + entryName);
    }
    std::vector<uint8_t> expandedBytes(static_cast<size_t>(uncompressedSize));
    z_stream inflateStream{};
    inflateStream.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(compressedBytes.data()));
    inflateStream.avail_in = static_cast<uInt>(compressedBytes.size());
    inflateStream.next_out = reinterpret_cast<Bytef *>(expandedBytes.data());
    inflateStream.avail_out = static_cast<uInt>(expandedBytes.size());
    int initStatus = inflateInit2(&inflateStream, -MAX_WBITS);
    if (initStatus != Z_OK) {
        throw std::runtime_error("deflate 初期化に失敗しました: " + entryName);
    }
    int inflateStatus = inflate(&inflateStream, Z_FINISH);
    inflateEnd(&inflateStream);
    if (inflateStatus != Z_STREAM_END || inflateStream.total_out != uncompressedSize) {
        throw std::runtime_error("deflate 展開に失敗しました: " + entryName);
    }
    return expandedBytes;
}

static std::vector<uint8_t> extractArchiveEntryBytes(const fs::path &archivePath, uint64_t archiveSize, const VvmEntryRecord &archiveEntry) {
    std::ifstream inputStream = openArchive(archivePath);
    uint64_t bodyPosition = archiveEntry.dataOffset == 0
        ? readArchiveEntryDataOffset(inputStream, archivePath, archiveSize, archiveEntry)
        : archiveEntry.dataOffset;
    std::vector<uint8_t> compressedBytes = readArchiveBytesAt(inputStream, archivePath, bodyPosition, archiveEntry.compressedSize);
    if (archiveEntry.compressionMethod == 0) {
        if (archiveEntry.compressedSize != archiveEntry.uncompressedSize) {
            throw std::runtime_error("stored entry サイズが一致しません: " + archiveEntry.name);
        }
        return compressedBytes;
    }
    if (archiveEntry.compressionMethod == 8) {
        return inflateEntryBytes(compressedBytes, archiveEntry.uncompressedSize, archiveEntry.name);
    }
    throw std::runtime_error("未対応の ZIP 圧縮方式です: " + archiveEntry.name);
}

std::vector<uint8_t> extractVvmEntryBytesAt(const fs::path &archivePath, const std::string &entryName, uint64_t dataOffset, uint64_t compressedSize, uint64_t uncompressedSize, uint16_t compressionMethod) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    if (dataOffset + compressedSize > archiveSize) {
        throw std::runtime_error("ZIP entry 本体が壊れています: " + entryName);
    }
    std::vector<uint8_t> compressedBytes = readArchiveBytesAt(archivePath, dataOffset, compressedSize);
    if (compressionMethod == 0) {
        if (compressedSize != uncompressedSize) {
            throw std::runtime_error("stored entry サイズが一致しません: " + entryName);
        }
        return compressedBytes;
    }
    if (compressionMethod == 8) {
        return inflateEntryBytes(compressedBytes, uncompressedSize, entryName);
    }
    throw std::runtime_error("未対応の ZIP 圧縮方式です: " + entryName);
}

std::vector<uint8_t> readVvmEntryPrefixBytesAt(const fs::path &archivePath, const std::string &entryName, uint64_t dataOffset, uint64_t compressedSize, uint64_t uncompressedSize, uint16_t compressionMethod, size_t prefixBytes) {
    if (prefixBytes == 0) {
        return {};
    }
    uint64_t archiveSize = readArchiveSize(archivePath);
    if (dataOffset + compressedSize > archiveSize) {
        throw std::runtime_error("ZIP entry 本体が壊れています: " + entryName);
    }
    if (compressionMethod == 0) {
        if (compressedSize != uncompressedSize) {
            throw std::runtime_error("stored entry サイズが一致しません: " + entryName);
        }
        uint64_t byteCount = std::min<uint64_t>(compressedSize, static_cast<uint64_t>(prefixBytes));
        return readArchiveBytesAt(archivePath, dataOffset, byteCount);
    }
    std::vector<uint8_t> entryBytes = extractVvmEntryBytesAt(archivePath, entryName, dataOffset, compressedSize, uncompressedSize, compressionMethod);
    if (entryBytes.size() > prefixBytes) {
        entryBytes.resize(prefixBytes);
    }
    return entryBytes;
}

static uint32_t calculateCrc32(const std::vector<uint8_t> &entryBytes) {
    uLong checksum = crc32(0L, Z_NULL, 0);
    size_t byteOffset = 0;
    while (byteOffset < entryBytes.size()) {
        size_t remainingBytes = entryBytes.size() - byteOffset;
        size_t chunkBytes = std::min(remainingBytes, static_cast<size_t>(std::numeric_limits<uInt>::max()));
        checksum = crc32(checksum, reinterpret_cast<const Bytef *>(entryBytes.data() + byteOffset), static_cast<uInt>(chunkBytes));
        byteOffset += chunkBytes;
    }
    return static_cast<uint32_t>(checksum);
}

static std::string formatEntryHash(const std::vector<uint8_t> &entryBytes) {
    return createSha256Hex(entryBytes.data(), entryBytes.size());
}

static std::string extractArchiveEntryText(const fs::path &archivePath, uint64_t archiveSize, const std::vector<VvmEntryRecord> &entries, const std::string &entryName) {
    const VvmEntryRecord *archiveEntry = findArchiveEntry(entries, entryName);
    if (!archiveEntry) {
        throw std::runtime_error(entryName + " がありません");
    }
    std::vector<uint8_t> entryBytes = extractArchiveEntryBytes(archivePath, archiveSize, *archiveEntry);
    return std::string(entryBytes.begin(), entryBytes.end());
}

static fs::path createSafeOutputPath(const fs::path &outputDirectory, const std::string &entryName) {
    fs::path entryPath(entryName);
    if (entryPath.is_absolute()) {
        throw std::runtime_error("絶対パス entry は展開できません: " + entryName);
    }
    fs::path safePath = outputDirectory;
    for (const fs::path &pathPart : entryPath) {
        std::string pathText = pathPart.string();
        if (pathText.empty() || pathText == "." || pathText == "..") {
            throw std::runtime_error("不正な entry パスです: " + entryName);
        }
        safePath /= pathPart;
    }
    return safePath;
}

static bool isVvmEntryName(const std::string &entryName) {
    return hasSuffix(entryName, ".vvm") && !entryName.empty() && entryName.back() != '/';
}

static fs::path createExtractedZipVvmDirectory(const fs::path &zipPath) {
    fs::path parentPath = zipPath.parent_path();
    fs::path basePath = parentPath.empty() ? fs::path(".") : parentPath;
    return basePath / ".litevox-model-vvm" / zipPath.stem();
}

static std::vector<fs::path> extractVvmModelsFromZip(const fs::path &zipPath) {
    uint64_t archiveSize = readArchiveSize(zipPath);
    std::vector<VvmEntryRecord> entries = readCentralDirectory(zipPath, archiveSize);
    fs::path outputDirectory = createExtractedZipVvmDirectory(zipPath);
    std::vector<fs::path> modelFiles;
    for (const VvmEntryRecord &archiveEntry : entries) {
        if (!isVvmEntryName(archiveEntry.name)) {
            continue;
        }
        fs::path outputPath = outputDirectory / fs::path(archiveEntry.name).filename();
        if (!fs::exists(outputPath) || fs::file_size(outputPath) != archiveEntry.uncompressedSize) {
            writeBinaryFile(outputPath, extractArchiveEntryBytes(zipPath, archiveSize, archiveEntry));
        }
        modelFiles.push_back(outputPath);
    }
    return modelFiles;
}

static void appendUniqueVvmModelFile(std::vector<fs::path> &modelFiles, const fs::path &modelPath) {
    fs::path canonicalModelPath = fs::weakly_canonical(modelPath);
    if (std::find(modelFiles.begin(), modelFiles.end(), canonicalModelPath) != modelFiles.end()) {
        return;
    }
    modelFiles.push_back(canonicalModelPath);
}

std::vector<fs::path> collectVvmModelFiles(const std::vector<fs::path> &modelRoots) {
    std::vector<fs::path> modelFiles;
    for (const fs::path &modelRoot : modelRoots) {
        ensurePathExists(modelRoot, "モデルパス");
        if (fs::is_regular_file(modelRoot)) {
            if (modelRoot.extension() == ".zip") {
                std::vector<fs::path> extractedModelFiles = extractVvmModelsFromZip(modelRoot);
                for (const fs::path &extractedModelFile : extractedModelFiles) {
                    appendUniqueVvmModelFile(modelFiles, extractedModelFile);
                }
                continue;
            }
            if (modelRoot.extension() != ".vvm") {
                throw std::runtime_error("VVM ではありません: " + modelRoot.string());
            }
            appendUniqueVvmModelFile(modelFiles, modelRoot);
            continue;
        }
        if (!fs::is_directory(modelRoot)) {
            throw std::runtime_error("モデルパスを読めません: " + modelRoot.string());
        }
        for (const fs::directory_entry &directoryEntry : fs::recursive_directory_iterator(modelRoot)) {
            if (!directoryEntry.is_regular_file()) {
                continue;
            }
            if (directoryEntry.path().extension() == ".zip") {
                std::vector<fs::path> extractedModelFiles = extractVvmModelsFromZip(directoryEntry.path());
                for (const fs::path &extractedModelFile : extractedModelFiles) {
                    appendUniqueVvmModelFile(modelFiles, extractedModelFile);
                }
                continue;
            }
            if (directoryEntry.path().extension() == ".vvm") {
                appendUniqueVvmModelFile(modelFiles, directoryEntry.path());
            }
        }
    }
    if (modelFiles.empty()) {
        throw std::runtime_error("VVM が見つかりません");
    }
    return modelFiles;
}

VvmArchiveSummary inspectVvmArchive(const fs::path &archivePath) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    VvmArchiveSummary archiveSummary;
    archiveSummary.archivePath = archivePath;
    archiveSummary.entries = readCentralDirectory(archivePath, archiveSize);
    archiveSummary.manifestJson = extractArchiveEntryText(archivePath, archiveSize, archiveSummary.entries, "manifest.json");
    archiveSummary.metasJson = extractArchiveEntryText(archivePath, archiveSize, archiveSummary.entries, "metas.json");
    for (const VvmEntryRecord &archiveEntry : archiveSummary.entries) {
        if (hasPrefix(archiveEntry.name, "models/") && hasSuffix(archiveEntry.name, ".bin")) {
            archiveSummary.modelBinNames.push_back(archiveEntry.name);
        }
    }
    return archiveSummary;
}

std::vector<uint8_t> extractVvmEntryBytes(const fs::path &archivePath, const std::string &entryName) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    std::vector<VvmEntryRecord> entries = readCentralDirectory(archivePath, archiveSize);
    const VvmEntryRecord *archiveEntry = findArchiveEntry(entries, entryName);
    if (!archiveEntry) {
        throw std::runtime_error("entry がありません: " + entryName);
    }
    return extractArchiveEntryBytes(archivePath, archiveSize, *archiveEntry);
}

std::vector<VvmEntryValidationRecord> validateVvmArchiveEntries(const fs::path &archivePath) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    std::vector<VvmEntryRecord> entries = readCentralDirectory(archivePath, archiveSize);
    std::vector<VvmEntryValidationRecord> validationRecords;
    for (const VvmEntryRecord &archiveEntry : entries) {
        if (!archiveEntry.name.empty() && archiveEntry.name.back() == '/') {
            continue;
        }
        std::vector<uint8_t> entryBytes = extractArchiveEntryBytes(archivePath, archiveSize, archiveEntry);
        uint32_t actualCrc32 = calculateCrc32(entryBytes);
        VvmEntryValidationRecord validationRecord;
        validationRecord.name = archiveEntry.name;
        validationRecord.uncompressedSize = entryBytes.size();
        validationRecord.expectedCrc32 = archiveEntry.crc32;
        validationRecord.actualCrc32 = actualCrc32;
        validationRecord.isValid = entryBytes.size() == archiveEntry.uncompressedSize && actualCrc32 == archiveEntry.crc32;
        validationRecords.push_back(std::move(validationRecord));
    }
    return validationRecords;
}

void extractVvmArchive(const fs::path &archivePath, const fs::path &outputDirectory) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    std::vector<VvmEntryRecord> entries = readCentralDirectory(archivePath, archiveSize);
    for (const VvmEntryRecord &archiveEntry : entries) {
        if (!archiveEntry.name.empty() && archiveEntry.name.back() == '/') {
            continue;
        }
        fs::path outputPath = createSafeOutputPath(outputDirectory, archiveEntry.name);
        writeBinaryFile(outputPath, extractArchiveEntryBytes(archivePath, archiveSize, archiveEntry));
    }
}

ZipDirectoryExtractionSummary extractZipDirectoryEntries(const fs::path &archivePath, const std::string &directoryMarker, const fs::path &outputDirectory) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    std::vector<VvmEntryRecord> entries = readCentralDirectory(archivePath, archiveSize);
    ZipDirectoryExtractionSummary extractionSummary;
    std::vector<std::pair<std::string, std::string>> filemapEntries;
    for (const VvmEntryRecord &archiveEntry : entries) {
        if (!archiveEntry.name.empty() && archiveEntry.name.back() == '/') {
            continue;
        }
        size_t markerPosition = archiveEntry.name.find(directoryMarker);
        if (markerPosition == std::string::npos) {
            continue;
        }
        std::string relativeName = archiveEntry.name.substr(markerPosition + directoryMarker.size());
        if (relativeName.empty()) {
            continue;
        }
        fs::path outputPath = createSafeOutputPath(outputDirectory, relativeName);
        std::vector<uint8_t> entryBytes = extractArchiveEntryBytes(archivePath, archiveSize, archiveEntry);
        writeBinaryFile(outputPath, entryBytes);
        filemapEntries.push_back({relativeName, formatEntryHash(entryBytes)});
        extractionSummary.fileCount++;
        extractionSummary.byteCount += archiveEntry.uncompressedSize;
    }
    if (extractionSummary.fileCount == 0) {
        throw std::runtime_error("展開対象 entry が見つかりません: " + directoryMarker);
    }
    std::ostringstream filemapStream;
    filemapStream << "{";
    for (size_t filemapIndex = 0; filemapIndex < filemapEntries.size(); filemapIndex++) {
        if (filemapIndex > 0) {
            filemapStream << ",";
        }
        filemapStream << quoteJsonString(filemapEntries[filemapIndex].first) << ":" << quoteJsonString(filemapEntries[filemapIndex].second);
    }
    filemapStream << "}";
    writeTextFile(outputDirectory / "filemap.json", filemapStream.str());
    return extractionSummary;
}

ZipDirectoryExtractionSummary extractZipArchivePreservingPaths(const fs::path &archivePath, const fs::path &outputDirectory) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    std::vector<VvmEntryRecord> entries = readCentralDirectory(archivePath, archiveSize);
    ZipDirectoryExtractionSummary extractionSummary;
    for (const VvmEntryRecord &archiveEntry : entries) {
        if (!archiveEntry.name.empty() && archiveEntry.name.back() == '/') {
            continue;
        }
        std::vector<uint8_t> entryBytes = extractArchiveEntryBytes(archivePath, archiveSize, archiveEntry);
        writeBinaryFile(createSafeOutputPath(outputDirectory, archiveEntry.name), entryBytes);
        extractionSummary.fileCount++;
        extractionSummary.byteCount += archiveEntry.uncompressedSize;
    }
    if (extractionSummary.fileCount == 0) {
        throw std::runtime_error("展開対象 entry が見つかりません");
    }
    return extractionSummary;
}

bool extractZipEntryByMarker(const fs::path &archivePath, const std::string &entryMarker, const fs::path &outputPath, uint64_t &byteCount) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    std::vector<VvmEntryRecord> entries = readCentralDirectory(archivePath, archiveSize);
    for (const VvmEntryRecord &archiveEntry : entries) {
        if (!archiveEntry.name.empty() && archiveEntry.name.back() == '/') {
            continue;
        }
        size_t markerPosition = archiveEntry.name.find(entryMarker);
        if (markerPosition == std::string::npos) {
            continue;
        }
        writeBinaryFile(outputPath, extractArchiveEntryBytes(archivePath, archiveSize, archiveEntry));
        byteCount = archiveEntry.uncompressedSize;
        return true;
    }
    byteCount = 0;
    return false;
}

bool zipArchiveHasEntryMarker(const fs::path &archivePath, const std::string &entryMarker) {
    uint64_t archiveSize = readArchiveSize(archivePath);
    std::vector<VvmEntryRecord> entries = readCentralDirectory(archivePath, archiveSize);
    for (const VvmEntryRecord &archiveEntry : entries) {
        if (!archiveEntry.name.empty() && archiveEntry.name.back() == '/') {
            continue;
        }
        if (archiveEntry.name.find(entryMarker) != std::string::npos) {
            return true;
        }
    }
    return false;
}
