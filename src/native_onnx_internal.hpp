#pragma once

#include "dynamic_library.hpp"
#include "model_asset.hpp"
#include "native_audio_query.hpp"
#include "native_onnx.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct OrtStatus;
struct OrtEnv;
struct OrtSessionOptions;
struct OrtSession;
struct OrtAllocator;
struct OrtTypeInfo;
struct OrtTensorTypeAndShapeInfo;
struct OrtMemoryInfo;
struct OrtValue;

struct OrtApiBase {
    const void *(*getApi)(uint32_t version);
    const char *(*getVersionString)();
};

inline constexpr uint32_t ortApiVersion = 17;
inline constexpr int32_t ortLoggingLevelWarning = 2;
inline constexpr int32_t ortGraphOptimizationLevelBasic = 1;
inline constexpr int64_t nativeOnnxPhonemeSize = 45;
inline constexpr size_t nativeOnnxDecoderPaddingFrames = 38;
inline constexpr size_t nativeOnnxDecoderMinimumChunkFrames = 128;
inline constexpr size_t nativeOnnxSamplesPerFrame = 256;
inline constexpr uint32_t nativeOnnxDefaultSamplingRate = 24000;

inline constexpr size_t ortApiIndexGetErrorMessage = 2;
inline constexpr size_t ortApiIndexCreateEnv = 3;
inline constexpr size_t ortApiIndexCreateSession = 7;
inline constexpr size_t ortApiIndexRun = 9;
inline constexpr size_t ortApiIndexCreateSessionFromArray = 8;
inline constexpr size_t ortApiIndexCreateSessionOptions = 10;
inline constexpr size_t ortApiIndexSetOptimizedModelFilePath = 11;
inline constexpr size_t ortApiIndexSetSessionGraphOptimizationLevel = 23;
inline constexpr size_t ortApiIndexSetIntraOpNumThreads = 24;
inline constexpr size_t ortApiIndexSetInterOpNumThreads = 25;
inline constexpr size_t ortApiIndexSessionGetInputCount = 30;
inline constexpr size_t ortApiIndexSessionGetOutputCount = 31;
inline constexpr size_t ortApiIndexSessionGetInputTypeInfo = 33;
inline constexpr size_t ortApiIndexSessionGetOutputTypeInfo = 34;
inline constexpr size_t ortApiIndexCastTypeInfoToTensorInfo = 55;
inline constexpr size_t ortApiIndexGetTensorElementType = 60;
inline constexpr size_t ortApiIndexGetDimensionsCount = 61;
inline constexpr size_t ortApiIndexGetDimensions = 62;
inline constexpr size_t ortApiIndexGetTensorShapeElementCount = 64;
inline constexpr size_t ortApiIndexGetTensorTypeAndShape = 65;
inline constexpr size_t ortApiIndexCreateTensorWithDataAsOrtValue = 49;
inline constexpr size_t ortApiIndexGetTensorMutableData = 51;
inline constexpr size_t ortApiIndexCreateCpuMemoryInfo = 69;
inline constexpr size_t ortApiIndexSessionGetInputName = 36;
inline constexpr size_t ortApiIndexSessionGetOutputName = 37;
inline constexpr size_t ortApiIndexAllocatorFree = 76;
inline constexpr size_t ortApiIndexGetAllocatorWithDefaultOptions = 78;
inline constexpr size_t ortApiIndexReleaseEnv = 92;
inline constexpr size_t ortApiIndexReleaseStatus = 93;
inline constexpr size_t ortApiIndexReleaseMemoryInfo = 94;
inline constexpr size_t ortApiIndexReleaseSession = 95;
inline constexpr size_t ortApiIndexReleaseValue = 96;
inline constexpr size_t ortApiIndexReleaseTypeInfo = 98;
inline constexpr size_t ortApiIndexReleaseTensorTypeAndShapeInfo = 99;
inline constexpr size_t ortApiIndexReleaseSessionOptions = 100;
inline constexpr size_t ortApiIndexGetAvailableProviders = 125;
inline constexpr size_t ortApiIndexReleaseAvailableProviders = 126;
inline constexpr size_t ortApiIndexAddSessionConfigEntry = 130;
inline constexpr size_t ortApiIndexSessionOptionsAppendExecutionProvider = 192;
inline constexpr size_t ortApiIndexGetTrainingApi = 219;
inline constexpr size_t ortTrainingApiIndexSetSeed = 22;
inline constexpr uint32_t nativeOnnxCoreMlFlags = 0x008;

using OrtGetApiBaseFunction = const OrtApiBase *(*)();
using OrtGetErrorMessageFunction = const char *(*)(const OrtStatus *);
using OrtCreateEnvFunction = OrtStatus *(*)(int32_t, const char *, OrtEnv **);
#if defined(_WIN32)
using OrtPathChar = wchar_t;
#else
using OrtPathChar = char;
#endif
using OrtCreateSessionFunction = OrtStatus *(*)(const OrtEnv *, const OrtPathChar *, const OrtSessionOptions *, OrtSession **);
using OrtRunFunction = OrtStatus *(*)(OrtSession *, const void *, const char *const *, const OrtValue *const *, size_t, const char *const *, size_t, OrtValue **);
using OrtCreateSessionFromArrayFunction = OrtStatus *(*)(const OrtEnv *, const void *, size_t, const OrtSessionOptions *, OrtSession **);
using OrtCreateSessionOptionsFunction = OrtStatus *(*)(OrtSessionOptions **);
using OrtSetOptimizedModelFilePathFunction = OrtStatus *(*)(OrtSessionOptions *, const OrtPathChar *);
using OrtSetSessionGraphOptimizationLevelFunction = OrtStatus *(*)(OrtSessionOptions *, int32_t);
using OrtSetThreadCountFunction = OrtStatus *(*)(OrtSessionOptions *, int);
using OrtAddSessionConfigEntryFunction = OrtStatus *(*)(OrtSessionOptions *, const char *, const char *);
using OrtSessionOptionsAppendExecutionProviderFunction = OrtStatus *(*)(OrtSessionOptions *, const char *, const char *const *, const char *const *, size_t);
using OrtSessionOptionsAppendExecutionProviderCoreMLFunction = OrtStatus *(*)(OrtSessionOptions *, uint32_t);
using OrtSessionGetCountFunction = OrtStatus *(*)(const OrtSession *, size_t *);
using OrtSessionGetNameFunction = OrtStatus *(*)(const OrtSession *, size_t, OrtAllocator *, char **);
using OrtSessionGetTypeInfoFunction = OrtStatus *(*)(const OrtSession *, size_t, OrtTypeInfo **);
using OrtCastTypeInfoToTensorInfoFunction = OrtStatus *(*)(const OrtTypeInfo *, const OrtTensorTypeAndShapeInfo **);
using OrtGetTensorElementTypeFunction = OrtStatus *(*)(const OrtTensorTypeAndShapeInfo *, int32_t *);
using OrtGetDimensionsCountFunction = OrtStatus *(*)(const OrtTensorTypeAndShapeInfo *, size_t *);
using OrtGetDimensionsFunction = OrtStatus *(*)(const OrtTensorTypeAndShapeInfo *, int64_t *, size_t);
using OrtGetTensorShapeElementCountFunction = OrtStatus *(*)(const OrtTensorTypeAndShapeInfo *, size_t *);
using OrtGetTensorTypeAndShapeFunction = OrtStatus *(*)(const OrtValue *, OrtTensorTypeAndShapeInfo **);
using OrtCreateTensorWithDataAsOrtValueFunction = OrtStatus *(*)(const OrtMemoryInfo *, void *, size_t, const int64_t *, size_t, int32_t, OrtValue **);
using OrtGetTensorMutableDataFunction = OrtStatus *(*)(OrtValue *, void **);
using OrtCreateCpuMemoryInfoFunction = OrtStatus *(*)(int32_t, int32_t, OrtMemoryInfo **);
using OrtAllocatorFreeFunction = OrtStatus *(*)(OrtAllocator *, void *);
using OrtGetAllocatorWithDefaultOptionsFunction = OrtStatus *(*)(OrtAllocator **);
using OrtGetAvailableProvidersFunction = OrtStatus *(*)(char ***, int *);
using OrtReleaseAvailableProvidersFunction = OrtStatus *(*)(char **, int);
using OrtGetTrainingApiFunction = const void *(*)(uint32_t version);
using OrtTrainingSetSeedFunction = OrtStatus *(*)(int64_t);
using OrtReleaseEnvFunction = void (*)(OrtEnv *);
using OrtReleaseStatusFunction = void (*)(OrtStatus *);
using OrtReleaseMemoryInfoFunction = void (*)(OrtMemoryInfo *);
using OrtReleaseSessionFunction = void (*)(OrtSession *);
using OrtReleaseValueFunction = void (*)(OrtValue *);
using OrtReleaseTypeInfoFunction = void (*)(OrtTypeInfo *);
using OrtReleaseTensorTypeAndShapeInfoFunction = void (*)(OrtTensorTypeAndShapeInfo *);
using OrtReleaseSessionOptionsFunction = void (*)(OrtSessionOptions *);

struct NativeOnnxApi {
    void *libraryHandle = nullptr;
    fs::path libraryPath;
    const OrtApiBase *apiBase = nullptr;
    const void *api = nullptr;
    const void *trainingApi = nullptr;
    OrtGetErrorMessageFunction getErrorMessage = nullptr;
    OrtCreateEnvFunction createEnv = nullptr;
    OrtCreateSessionFunction createSession = nullptr;
    OrtRunFunction run = nullptr;
    OrtCreateSessionFromArrayFunction createSessionFromArray = nullptr;
    OrtCreateSessionOptionsFunction createSessionOptions = nullptr;
    OrtSetOptimizedModelFilePathFunction setOptimizedModelFilePath = nullptr;
    OrtSetSessionGraphOptimizationLevelFunction setSessionGraphOptimizationLevel = nullptr;
    OrtSetThreadCountFunction setIntraOpNumThreads = nullptr;
    OrtSetThreadCountFunction setInterOpNumThreads = nullptr;
    OrtAddSessionConfigEntryFunction addSessionConfigEntry = nullptr;
    OrtSessionOptionsAppendExecutionProviderFunction appendExecutionProvider = nullptr;
    OrtSessionOptionsAppendExecutionProviderCoreMLFunction appendExecutionProviderCoreML = nullptr;
    OrtSessionGetCountFunction sessionGetInputCount = nullptr;
    OrtSessionGetCountFunction sessionGetOutputCount = nullptr;
    OrtSessionGetNameFunction sessionGetInputName = nullptr;
    OrtSessionGetNameFunction sessionGetOutputName = nullptr;
    OrtSessionGetTypeInfoFunction sessionGetInputTypeInfo = nullptr;
    OrtSessionGetTypeInfoFunction sessionGetOutputTypeInfo = nullptr;
    OrtCastTypeInfoToTensorInfoFunction castTypeInfoToTensorInfo = nullptr;
    OrtGetTensorElementTypeFunction getTensorElementType = nullptr;
    OrtGetDimensionsCountFunction getDimensionsCount = nullptr;
    OrtGetDimensionsFunction getDimensions = nullptr;
    OrtGetTensorShapeElementCountFunction getTensorShapeElementCount = nullptr;
    OrtGetTensorTypeAndShapeFunction getTensorTypeAndShape = nullptr;
    OrtCreateTensorWithDataAsOrtValueFunction createTensorWithDataAsOrtValue = nullptr;
    OrtGetTensorMutableDataFunction getTensorMutableData = nullptr;
    OrtCreateCpuMemoryInfoFunction createCpuMemoryInfo = nullptr;
    OrtAllocatorFreeFunction allocatorFree = nullptr;
    OrtGetAllocatorWithDefaultOptionsFunction getAllocatorWithDefaultOptions = nullptr;
    OrtGetAvailableProvidersFunction getAvailableProviders = nullptr;
    OrtReleaseAvailableProvidersFunction releaseAvailableProviders = nullptr;
    OrtTrainingSetSeedFunction setSeed = nullptr;
    OrtReleaseEnvFunction releaseEnv = nullptr;
    OrtReleaseStatusFunction releaseStatus = nullptr;
    OrtReleaseMemoryInfoFunction releaseMemoryInfo = nullptr;
    OrtReleaseSessionFunction releaseSession = nullptr;
    OrtReleaseValueFunction releaseValue = nullptr;
    OrtReleaseTypeInfoFunction releaseTypeInfo = nullptr;
    OrtReleaseTensorTypeAndShapeInfoFunction releaseTensorTypeAndShapeInfo = nullptr;
    OrtReleaseSessionOptionsFunction releaseSessionOptions = nullptr;
    bool hasConfiguredSeed = false;
    int64_t configuredSeed = 0;
};

struct NativeOnnxValueDescriptor {
    std::string name;
    int32_t elementType = 0;
    std::vector<int64_t> dimensions;
};

struct NativeOnnxTraceInput {
    std::string name;
    int32_t elementType = 0;
    std::vector<int64_t> dimensions;
    std::vector<uint8_t> bytes;
};

struct NativeOnnxAudioQuerySettings {
    float speedScale = 1.0f;
    float pitchScale = 0.0f;
    float intonationScale = 1.0f;
    float volumeScale = 1.0f;
    float prePhonemeLength = 0.1f;
    float postPhonemeLength = 0.1f;
    uint32_t outputSamplingRate = 24000;
    bool outputStereo = false;
};

struct soxr;

class NativeOnnxPcm {
public:
    explicit NativeOnnxPcm(const NativeOnnxAudioQuerySettings &Settings);
    ~NativeOnnxPcm();
    NativeOnnxPcm(const NativeOnnxPcm &) = delete;
    NativeOnnxPcm &operator=(const NativeOnnxPcm &) = delete;
    std::vector<uint8_t> Convert(const std::vector<float> &Wave, bool Final);
private:
    NativeOnnxAudioQuerySettings Settings;
    // 遅延と位相をチャンク間で保持し、末尾でだけ残りの音声を排出する。
    soxr *Resampler = nullptr;
};

struct NativeOnnxMora {
    std::string text;
    std::string consonant;
    std::string vowel;
    float consonantLength = 0.0f;
    float vowelLength = 0.0f;
    float pitch = 0.0f;
    bool hasConsonant = false;
    bool hasConsonantLength = false;
    bool hasVowelLength = false;
    bool hasPitch = false;
};

struct NativeOnnxAccentPhrase {
    std::vector<NativeOnnxMora> moras;
    NativeOnnxMora pauseMora;
    bool hasPauseMora = false;
    bool isInterrogative = false;
    int64_t accent = 1;
};

struct NativeOnnxDecoderChunkInputSet {
    std::vector<NativeOnnxTraceInput> tensors;
    size_t frontCropFrames = 0;
    size_t backCropFrames = 0;
};

struct NativeOnnxScoreNote {
    std::string lyric;
    std::string noteId;
    NativeAudioQueryMora mora;
    int64_t key = -1;
    uint64_t frameLength = 0;
    bool hasKey = false;
    bool hasNoteId = false;
    bool hasMora = false;
};

struct NativeOnnxFramePhoneme {
    std::string phoneme;
    std::string noteId;
    uint64_t frameLength = 0;
    bool hasNoteId = false;
};

struct NativeOnnxFrameAudioQuery {
    std::vector<float> f0Values;
    std::vector<float> volumeValues;
    std::vector<NativeOnnxFramePhoneme> phonemes;
    float volumeScale = 1.0f;
    uint32_t outputSamplingRate = 24000;
    bool outputStereo = false;
};

struct NativeOnnxSongPhonemeFeature {
    std::string phoneme;
    std::string noteId;
    int64_t phonemeCode = 0;
    int64_t key = -1;
    bool hasNoteId = false;
};

struct NativeOnnxSongFrameInputs {
    std::vector<int64_t> phonemeValues;
    std::vector<int64_t> keyValues;
};

enum class NativeOnnxSingTeacherMode {
    VvBin,
    Deterministic,
};

struct NativeOnnxCachedSession {
    void *libraryHandle = nullptr;
    OrtEnv *env = nullptr;
    OrtSession *session = nullptr;
    OrtMemoryInfo *memoryInfo = nullptr;
    OrtReleaseEnvFunction releaseEnv = nullptr;
    OrtReleaseMemoryInfoFunction releaseMemoryInfo = nullptr;
    OrtReleaseSessionFunction releaseSession = nullptr;
    std::vector<NativeOnnxValueDescriptor> inputDescriptors;
    std::vector<NativeOnnxValueDescriptor> outputDescriptors;

    ~NativeOnnxCachedSession() {
        if (memoryInfo && releaseMemoryInfo) {
            releaseMemoryInfo(memoryInfo);
        }
        if (session && releaseSession) {
            releaseSession(session);
        }
        if (env && releaseEnv) {
            releaseEnv(env);
        }
        if (libraryHandle) {
            closeDynamicLibrary(libraryHandle);
        }
    }
};

struct NativeOnnxPreparedInputSet {
    std::vector<NativeOnnxTraceInput> tensors;
    size_t traceInputCount = 0;
    size_t syntheticInputCount = 0;
};

struct NativeOnnxCompareBenchmark {
    size_t inputCount = 0;
    size_t outputCount = 0;
    double sessionCreateMilliseconds = 0.0;
    double averageRunMilliseconds = 0.0;
    std::vector<NativeOnnxTraceInput> outputs;
};

extern std::mutex nativeOnnxSessionCacheMutex;
extern std::condition_variable nativeOnnxSessionCacheCondition;
extern std::map<std::string, std::shared_ptr<NativeOnnxCachedSession>> nativeOnnxSessionCache;
extern std::set<std::string> nativeOnnxSessionKeysInProgress;
extern uint64_t nativeOnnxSessionCacheHits;
extern uint64_t nativeOnnxSessionCacheMisses;
extern std::mutex nativeOnnxExportedModelCacheMutex;
extern std::condition_variable nativeOnnxExportedModelCacheCondition;
extern std::map<std::string, fs::path> nativeOnnxExportedModelCache;
extern std::set<std::string> nativeOnnxExportedModelKeysInProgress;
extern uint64_t nativeOnnxExportedModelMemoryHits;
extern uint64_t nativeOnnxExportedModelMemoryMisses;
extern uint64_t nativeOnnxExportedModelFileHits;
extern uint64_t nativeOnnxExportedModelFileMisses;
extern uint64_t nativeOnnxExportedModelFileWrites;
extern uint64_t nativeOnnxExportedModelFileReadErrors;
extern uint64_t nativeOnnxExportedModelFileWriteErrors;
extern std::atomic_size_t nativeOnnxNextTraceId;

std::vector<uint8_t> readNativeOnnxBinaryFile(const fs::path &filePath);
std::string readNativeOnnxTextFile(const fs::path &filePath);
NativeOnnxApi loadNativeOnnxApi(const fs::path &onnxruntimeLibraryPath);
void closeNativeOnnxApi(NativeOnnxApi &nativeOnnxApi);
std::string getNativeOnnxVersion(const NativeOnnxApi &nativeOnnxApi);
std::string formatNativeOnnxElementType(int32_t elementType);
std::string formatNativeOnnxShape(const std::vector<int64_t> &dimensions);
std::string extractNativeOnnxJsonStringField(const std::string &jsonText, const std::string &fieldName);
double extractNativeOnnxJsonNumberField(const std::string &jsonText, const std::string &fieldName, double fallbackNumber);
bool extractNativeOnnxJsonBoolField(const std::string &jsonText, const std::string &fieldName, bool fallbackValue);
bool extractNativeOnnxJsonFloatField(const std::string &jsonText, const std::string &fieldName, float &numberValue);
int64_t parseNativeOnnxJsonInteger(const std::string &numberText);
std::vector<int64_t> extractNativeOnnxJsonShapeField(const std::string &jsonText);
void ensureNativeOnnxCall(NativeOnnxApi &nativeOnnxApi, OrtStatus *callStatus, const std::string &operationName);
std::vector<std::string> collectNativeOnnxAvailableProviders(NativeOnnxApi &nativeOnnxApi);
void applyNativeOnnxSeedIfConfigured(NativeOnnxApi &nativeOnnxApi);
void configureNativeOnnxSessionOptions(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, OrtSessionOptions *sessionOptions, uint16_t cpuThreadCount, bool shouldUseVvBinConfig);
const NativeOnnxTraceInput *findNativeOnnxTraceTensor(const std::vector<NativeOnnxTraceInput> &traceTensors, const std::string &tensorName);
const NativeOnnxTraceInput &requireNativeOnnxTensor(const std::vector<NativeOnnxTraceInput> &traceTensors, const std::string &tensorName);

template <typename TensorValueType>
std::vector<TensorValueType> readNativeOnnxTensorValues(const NativeOnnxTraceInput &tensor, int32_t expectedElementType) {
    if (tensor.elementType != expectedElementType) {
        throw std::runtime_error("tensor dtype が一致しません: " + tensor.name);
    }
    if (tensor.bytes.size() % sizeof(TensorValueType) != 0) {
        throw std::runtime_error("tensor byte size が一致しません: " + tensor.name);
    }
    std::vector<TensorValueType> values(tensor.bytes.size() / sizeof(TensorValueType));
    if (!values.empty()) {
        std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    }
    return values;
}

template <typename TensorValueType>
std::vector<uint8_t> createNativeOnnxTensorBytes(const std::vector<TensorValueType> &values) {
    std::vector<uint8_t> tensorBytes(values.size() * sizeof(TensorValueType));
    if (!tensorBytes.empty()) {
        std::memcpy(tensorBytes.data(), values.data(), tensorBytes.size());
    }
    return tensorBytes;
}

std::string compareNativeOnnxTensors(const NativeOnnxTraceInput &actualTensor, const NativeOnnxTraceInput *expectedTensor);
NativeOnnxAudioQuerySettings parseNativeOnnxAudioQuerySettings(const std::string &audioQueryText);
int64_t parseNativeOnnxPhonemeCode(const std::string &phonemeText);
std::vector<NativeOnnxScoreNote> parseNativeOnnxScore(const std::string &scoreText);
NativeOnnxFrameAudioQuery parseNativeOnnxFrameAudioQuery(const std::string &frameAudioQueryText);
void validateNativeOnnxParsedFrameAudioQuery(const NativeOnnxFrameAudioQuery &frameAudioQuery);
std::vector<NativeOnnxSongPhonemeFeature> createNativeOnnxSongPhonemeFeatures(const std::vector<NativeOnnxScoreNote> &scoreNotes);
std::vector<uint64_t> createNativeOnnxSongPhonemeLengths(const std::vector<NativeOnnxScoreNote> &scoreNotes, const std::vector<int64_t> &consonantLengths);
NativeOnnxSongFrameInputs createNativeOnnxSongFrameInputs(const std::vector<NativeOnnxSongPhonemeFeature> &phonemeFeatures, const std::vector<NativeOnnxFramePhoneme> &framePhonemes);
NativeOnnxAudioQuerySettings createNativeOnnxSettingsFromFrameAudioQuery(const NativeOnnxFrameAudioQuery &frameAudioQuery);
std::string createNativeOnnxFloatArrayJson(const std::vector<float> &numberValues);
std::string createNativeOnnxFrameAudioQueryJson(const NativeOnnxFrameAudioQuery &frameAudioQuery);
int64_t resolveNativeOnnxInnerVoiceId(const ModelAssetRecord &modelAsset, const std::string &domainName, uint32_t styleId);
std::vector<NativeOnnxAccentPhrase> parseNativeOnnxAccentPhrases(const std::string &audioQueryText);
NativeOnnxTraceInput createNativeOnnxInt64Tensor(const std::string &tensorName, const std::vector<int64_t> &dimensions, const std::vector<int64_t> &values);
NativeOnnxTraceInput createNativeOnnxFloatTensor(const std::string &tensorName, const std::vector<int64_t> &dimensions, const std::vector<float> &values);
std::vector<NativeOnnxTraceInput> createNativeOnnxFrontendInputs(const std::string &audioQueryText, int64_t innerVoiceId);
std::vector<int64_t> findNativeOnnxVowelIndexes(const std::vector<int64_t> &phonemeValues, const std::vector<int64_t> &vowelValues);
bool isNativeOnnxUnvoicedVowel(int64_t phonemeValue);
std::vector<NativeOnnxTraceInput> createNativeOnnxDecoderInputs(const std::vector<NativeOnnxTraceInput> &traceInputs, const std::vector<NativeOnnxTraceInput> &durationOutputs, const std::vector<NativeOnnxTraceInput> &intonationOutputs, const NativeOnnxAudioQuerySettings &audioQuerySettings, bool shouldZeroUnvoicedVowels);
const ModelAssetRecord &requireNativeOnnxModelAsset(const std::vector<ModelAssetRecord> &modelAssets, const std::string &entryName);
std::vector<NativeOnnxTraceInput> loadNativeOnnxTraceInputs(const fs::path &inputDirectory);
std::vector<NativeOnnxTraceInput> loadNativeOnnxTraceOutputs(const fs::path &inputDirectory);
fs::path resolveNativeOnnxAudioQueryPath(const fs::path &inputDirectory, const fs::path &audioQueryPath);
void appendNativeOnnxValueInfo(std::ostringstream &inspectStream, NativeOnnxApi &nativeOnnxApi, OrtSession *session, OrtAllocator *allocator, const std::string &prefixText, size_t valueIndex);
int32_t parseNativeOnnxElementType(const std::string &typeText);
NativeOnnxValueDescriptor readNativeOnnxValueDescriptor(NativeOnnxApi &nativeOnnxApi, OrtSession *session, OrtAllocator *allocator, bool isInput, size_t valueIndex);
bool areNativeOnnxDimensionsEqual(const std::vector<int64_t> &leftDimensions, const std::vector<int64_t> &rightDimensions);
size_t calculatePositiveShapeElementCount(const std::vector<int64_t> &dimensions);
size_t getNativeOnnxElementByteCount(int32_t elementType);
bool canCreateNativeOnnxSmokeInput(const NativeOnnxValueDescriptor &inputDescriptor);
std::vector<uint8_t> createNativeOnnxInputBytes(const NativeOnnxValueDescriptor &inputDescriptor, size_t inputIndex, size_t elementCount);
std::vector<int64_t> readNativeOnnxTensorDimensions(NativeOnnxApi &nativeOnnxApi, const OrtTensorTypeAndShapeInfo *tensorShapeInfo);
std::string compareNativeOnnxTraceOutput(const NativeOnnxValueDescriptor &outputDescriptor, const std::vector<int64_t> &outputDimensions, const void *outputPointer, size_t outputByteCount, const std::vector<NativeOnnxTraceInput> &traceOutputs);
void sanitizeNativeOnnxTableCell(std::string &cellText);
void writeNativeOnnxTensorTrace(const ModelAssetRecord &modelAsset, const std::vector<NativeOnnxTraceInput> &inputTensors, const std::vector<NativeOnnxTraceInput> &outputTensors);
void appendNativeOnnxSessionInfoFromBytes(std::ostringstream &inspectStream, NativeOnnxApi &nativeOnnxApi, const std::vector<uint8_t> &modelBytes, const fs::path &inputDirectory, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const ModelAssetRecord *traceModelAsset = nullptr);
void appendNativeOnnxSessionInfo(std::ostringstream &inspectStream, NativeOnnxApi &nativeOnnxApi, const fs::path &modelPath, const fs::path &inputDirectory, uint16_t cpuThreadCount);
std::shared_ptr<NativeOnnxCachedSession> createNativeOnnxCachedSession(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<uint8_t> &modelBytes, uint16_t cpuThreadCount, bool shouldUseVvBinConfig);
std::shared_ptr<NativeOnnxCachedSession> createNativeOnnxCachedSession(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const fs::path &modelPath, uint16_t cpuThreadCount, bool shouldUseVvBinConfig);
std::shared_ptr<NativeOnnxCachedSession> getNativeOnnxCachedSession(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<uint8_t> &modelBytes, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey);
std::shared_ptr<NativeOnnxCachedSession> getNativeOnnxCachedSession(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const fs::path &modelPath, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey);
std::vector<NativeOnnxTraceInput> runNativeOnnxPreparedSession(NativeOnnxApi &nativeOnnxApi, const std::shared_ptr<NativeOnnxCachedSession> &cachedSession, const std::vector<NativeOnnxTraceInput> &inputTensors);
std::vector<NativeOnnxTraceInput> runNativeOnnxModelBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const std::vector<uint8_t> &modelBytes, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey = "");
std::vector<NativeOnnxTraceInput> runNativeOnnxModelPath(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const fs::path &modelPath, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const std::string &sessionCacheKey = "");
std::vector<uint8_t> extractNativeOnnxModelAssetBytes(const ModelAssetRecord &modelAsset);
NativeOnnxSingTeacherMode getNativeOnnxSingTeacherMode();
float getNativeOnnxDeterministicSingTeacherSeed();
std::vector<uint8_t> exportNativeOnnxOptimizedModelBytes(NativeOnnxApi &nativeOnnxApi, const ModelAssetRecord &modelAsset, const std::vector<uint8_t> &modelBytes, uint16_t cpuThreadCount, bool shouldUseVvBinConfig);
std::vector<uint8_t> rewriteNativeOnnxModelRandomSeed(const uint8_t *messageBytes, size_t messageSize, float seedValue, size_t &rewrittenNodeCount);
std::vector<NativeOnnxTraceInput> runNativeOnnxSingTeacherModelAssetBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig);
std::vector<NativeOnnxTraceInput> runNativeOnnxModelAssetBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig);
std::vector<NativeOnnxTraceInput> runNativeOnnxModelAssetBytes(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, const ModelAssetRecord &modelAsset, const std::vector<uint8_t> &modelBytes, const std::vector<NativeOnnxTraceInput> &inputTensors, uint16_t cpuThreadCount, bool shouldUseVvBinConfig);
std::map<std::string, ManifestModelRecord> createNativeOnnxManifestModelMap(const std::vector<VvmArchiveSummary> &archiveSummaries);
const ManifestModelRecord *findNativeOnnxManifestModelRecord(const std::map<std::string, ManifestModelRecord> &manifestModelMap, const ModelAssetRecord &modelAsset);
std::string summarizeNativeOnnxValueDescriptors(const std::vector<NativeOnnxValueDescriptor> &valueDescriptors);
std::map<std::string, size_t> collectNativeOnnxOperatorCounts(const std::vector<uint8_t> &optimizedModelBytes);
std::map<std::string, size_t> collectNativeOnnxRandomOperatorCounts(const std::map<std::string, size_t> &operatorCounts);
NativeOnnxPreparedInputSet createNativeOnnxPreparedInputs(const std::vector<NativeOnnxValueDescriptor> &inputDescriptors, const std::vector<NativeOnnxTraceInput> &traceInputs);
std::string createNativeOnnxInputModeText(const NativeOnnxPreparedInputSet &preparedInputs);
NativeOnnxCompareBenchmark benchmarkNativeOnnxSession(const std::shared_ptr<NativeOnnxCachedSession> &cachedSession, NativeOnnxApi &nativeOnnxApi, const std::vector<NativeOnnxTraceInput> &inputTensors, size_t runCount);
std::vector<NativeOnnxTraceInput> createNativeOnnxDecoderInputsFromAudioQuery(const std::vector<NativeOnnxTraceInput> &frontendInputs, const std::string &audioQueryText, const NativeOnnxAudioQuerySettings &audioQuerySettings);
std::vector<NativeOnnxTraceInput> createNativeOnnxCompareAudioQueryInputs(const std::vector<ModelAssetRecord> &modelAssets, const ModelAssetRecord &modelAsset, const std::string &audioQueryText, uint32_t styleId);
std::string formatNativeOnnxFloat(float value);
std::string formatNativeOnnxSettingFloat(float value);
std::vector<uint8_t> createNativeOnnxPcmBytes(const std::vector<float> &waveValues, const NativeOnnxAudioQuerySettings &audioQuerySettings);
