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


std::vector<uint8_t> readNativeOnnxBinaryFile(const fs::path &filePath) {
    std::ifstream inputStream(filePath, std::ios::binary);
    if (!inputStream) {
        throw std::runtime_error("ONNX model を読めません: " + filePath.string());
    }
    inputStream.seekg(0, std::ios::end);
    std::streamoff fileSize = inputStream.tellg();
    if (fileSize < 0) {
        throw std::runtime_error("ONNX model サイズを読めません: " + filePath.string());
    }
    inputStream.seekg(0, std::ios::beg);
    std::vector<uint8_t> fileBytes(static_cast<size_t>(fileSize));
    if (!fileBytes.empty()) {
        inputStream.read(reinterpret_cast<char *>(fileBytes.data()), static_cast<std::streamsize>(fileBytes.size()));
        if (!inputStream) {
            throw std::runtime_error("ONNX model 読み込みに失敗しました: " + filePath.string());
        }
    }
    return fileBytes;
}

std::string readNativeOnnxTextFile(const fs::path &filePath) {
    std::ifstream inputStream(filePath, std::ios::binary);
    if (!inputStream) {
        throw std::runtime_error("trace json を読めません: " + filePath.string());
    }
    std::ostringstream textStream;
    textStream << inputStream.rdbuf();
    return textStream.str();
}

std::string extractNativeOnnxJsonStringField(const std::string &jsonText, const std::string &fieldName) {
    std::string markerText = "\"" + fieldName + "\"";
    size_t markerPosition = jsonText.find(markerText);
    if (markerPosition == std::string::npos) {
        return "";
    }
    size_t colonPosition = jsonText.find(':', markerPosition + markerText.size());
    if (colonPosition == std::string::npos) {
        return "";
    }
    size_t quotePosition = jsonText.find('"', colonPosition + 1);
    if (quotePosition == std::string::npos) {
        return "";
    }
    std::string valueText;
    for (size_t characterPosition = quotePosition + 1; characterPosition < jsonText.size(); characterPosition++) {
        char character = jsonText[characterPosition];
        if (character == '"') {
            return valueText;
        }
        if (character == '\\' && characterPosition + 1 < jsonText.size()) {
            characterPosition++;
            valueText.push_back(jsonText[characterPosition]);
            continue;
        }
        valueText.push_back(character);
    }
    return "";
}

double extractNativeOnnxJsonNumberField(const std::string &jsonText, const std::string &fieldName, double fallbackNumber) {
    std::string markerText = "\"" + fieldName + "\"";
    size_t markerPosition = jsonText.find(markerText);
    if (markerPosition == std::string::npos) {
        return fallbackNumber;
    }
    size_t colonPosition = jsonText.find(':', markerPosition + markerText.size());
    if (colonPosition == std::string::npos) {
        return fallbackNumber;
    }
    size_t numberStartPosition = jsonText.find_first_of("-0123456789", colonPosition + 1);
    if (numberStartPosition == std::string::npos) {
        return fallbackNumber;
    }
    size_t numberEndPosition = numberStartPosition;
    while (numberEndPosition < jsonText.size()) {
        char character = jsonText[numberEndPosition];
        if (!(std::isdigit(static_cast<unsigned char>(character)) || character == '-' || character == '+' || character == '.' || character == 'e' || character == 'E')) {
            break;
        }
        numberEndPosition++;
    }
    return std::stod(jsonText.substr(numberStartPosition, numberEndPosition - numberStartPosition));
}

bool extractNativeOnnxJsonBoolField(const std::string &jsonText, const std::string &fieldName, bool fallbackValue) {
    std::string markerText = "\"" + fieldName + "\"";
    size_t markerPosition = jsonText.find(markerText);
    if (markerPosition == std::string::npos) {
        return fallbackValue;
    }
    size_t colonPosition = jsonText.find(':', markerPosition + markerText.size());
    if (colonPosition == std::string::npos) {
        return fallbackValue;
    }
    size_t valuePosition = colonPosition + 1;
    while (valuePosition < jsonText.size() && std::isspace(static_cast<unsigned char>(jsonText[valuePosition]))) {
        valuePosition++;
    }
    if (jsonText.compare(valuePosition, 4, "true") == 0) {
        return true;
    }
    if (jsonText.compare(valuePosition, 5, "false") == 0) {
        return false;
    }
    return fallbackValue;
}

bool extractNativeOnnxJsonFloatField(const std::string &jsonText, const std::string &fieldName, float &numberValue) {
    size_t valuePosition = findJsonFieldValuePosition(jsonText, fieldName);
    if (valuePosition == std::string::npos || valuePosition >= jsonText.size()) {
        return false;
    }
    if (jsonText.compare(valuePosition, 4, "null") == 0) {
        return false;
    }
    size_t numberEndPosition = valuePosition;
    while (numberEndPosition < jsonText.size()) {
        char character = jsonText[numberEndPosition];
        if (!(std::isdigit(static_cast<unsigned char>(character)) || character == '-' || character == '+' || character == '.' || character == 'e' || character == 'E')) {
            break;
        }
        numberEndPosition++;
    }
    if (numberEndPosition == valuePosition) {
        return false;
    }
    numberValue = std::stof(jsonText.substr(valuePosition, numberEndPosition - valuePosition));
    return true;
}

int64_t parseNativeOnnxJsonInteger(const std::string &numberText) {
    if (numberText.empty()) {
        throw std::runtime_error("整数が空です");
    }
    size_t consumedLength = 0;
    int64_t numberValue = std::stoll(numberText, &consumedLength);
    if (consumedLength != numberText.size()) {
        throw std::runtime_error("整数が不正です: " + numberText);
    }
    return numberValue;
}

std::vector<int64_t> extractNativeOnnxJsonShapeField(const std::string &jsonText) {
    std::string markerText = "\"shape\"";
    size_t markerPosition = jsonText.find(markerText);
    if (markerPosition == std::string::npos) {
        return {};
    }
    size_t openPosition = jsonText.find('[', markerPosition + markerText.size());
    size_t closePosition = jsonText.find(']', openPosition == std::string::npos ? markerPosition : openPosition);
    if (openPosition == std::string::npos || closePosition == std::string::npos || closePosition < openPosition) {
        throw std::runtime_error("shape が不正です");
    }
    std::vector<int64_t> dimensions;
    std::string currentNumber;
    for (size_t characterPosition = openPosition + 1; characterPosition < closePosition; characterPosition++) {
        char character = jsonText[characterPosition];
        if (std::isdigit(static_cast<unsigned char>(character)) || character == '-') {
            currentNumber.push_back(character);
            continue;
        }
        if (!currentNumber.empty()) {
            dimensions.push_back(parseNativeOnnxJsonInteger(currentNumber));
            currentNumber.clear();
        }
    }
    if (!currentNumber.empty()) {
        dimensions.push_back(parseNativeOnnxJsonInteger(currentNumber));
    }
    return dimensions;
}

template <typename FunctionType>
static FunctionType loadNativeOnnxApiFunction(const void *api, size_t functionIndex) {
    const void *const *apiFunctions = reinterpret_cast<const void *const *>(api);
    void *functionPointer = const_cast<void *>(apiFunctions[functionIndex]);
    return reinterpret_cast<FunctionType>(functionPointer);
}

static bool tryReadNativeOnnxSeedFromEnvironment(int64_t &seedValue) {
    const char *seedText = std::getenv("LITEVOX_ORT_SEED");
    if (!seedText || seedText[0] == '\0') {
        return false;
    }
    errno = 0;
    char *endPointer = nullptr;
    long long parsedSeed = std::strtoll(seedText, &endPointer, 10);
    if (errno != 0 || !endPointer || endPointer == seedText || *endPointer != '\0') {
        throw std::runtime_error("LITEVOX_ORT_SEED が不正です");
    }
    seedValue = static_cast<int64_t>(parsedSeed);
    return true;
}

NativeOnnxApi loadNativeOnnxApi(const fs::path &onnxruntimeLibraryPath) {
    ensurePathExists(onnxruntimeLibraryPath, "onnxruntime library");
    NativeOnnxApi nativeOnnxApi;
    nativeOnnxApi.libraryPath = onnxruntimeLibraryPath;
    nativeOnnxApi.libraryHandle = openDynamicLibrary(onnxruntimeLibraryPath);
    if (!nativeOnnxApi.libraryHandle) {
        throw std::runtime_error(std::string("ONNX Runtime を開けません: ") + getDynamicLibraryErrorText());
    }
    void *getApiBasePointer = loadDynamicLibrarySymbol(nativeOnnxApi.libraryHandle, "OrtGetApiBase");
    if (!getApiBasePointer) {
        throw std::runtime_error("OrtGetApiBase がありません");
    }
    OrtGetApiBaseFunction getApiBase = reinterpret_cast<OrtGetApiBaseFunction>(getApiBasePointer);
    nativeOnnxApi.apiBase = getApiBase();
    if (!nativeOnnxApi.apiBase) {
        throw std::runtime_error("OrtApiBase を取得できません");
    }
    nativeOnnxApi.api = nativeOnnxApi.apiBase->getApi(ortApiVersion);
    if (!nativeOnnxApi.api) {
        throw std::runtime_error("OrtApi v17 を取得できません");
    }
    nativeOnnxApi.getErrorMessage = loadNativeOnnxApiFunction<OrtGetErrorMessageFunction>(nativeOnnxApi.api, ortApiIndexGetErrorMessage);
    nativeOnnxApi.createEnv = loadNativeOnnxApiFunction<OrtCreateEnvFunction>(nativeOnnxApi.api, ortApiIndexCreateEnv);
    nativeOnnxApi.createSession = loadNativeOnnxApiFunction<OrtCreateSessionFunction>(nativeOnnxApi.api, ortApiIndexCreateSession);
    nativeOnnxApi.run = loadNativeOnnxApiFunction<OrtRunFunction>(nativeOnnxApi.api, ortApiIndexRun);
    nativeOnnxApi.createSessionFromArray = loadNativeOnnxApiFunction<OrtCreateSessionFromArrayFunction>(nativeOnnxApi.api, ortApiIndexCreateSessionFromArray);
    nativeOnnxApi.createSessionOptions = loadNativeOnnxApiFunction<OrtCreateSessionOptionsFunction>(nativeOnnxApi.api, ortApiIndexCreateSessionOptions);
    nativeOnnxApi.setOptimizedModelFilePath = loadNativeOnnxApiFunction<OrtSetOptimizedModelFilePathFunction>(nativeOnnxApi.api, ortApiIndexSetOptimizedModelFilePath);
    nativeOnnxApi.setSessionGraphOptimizationLevel = loadNativeOnnxApiFunction<OrtSetSessionGraphOptimizationLevelFunction>(nativeOnnxApi.api, ortApiIndexSetSessionGraphOptimizationLevel);
    nativeOnnxApi.setIntraOpNumThreads = loadNativeOnnxApiFunction<OrtSetThreadCountFunction>(nativeOnnxApi.api, ortApiIndexSetIntraOpNumThreads);
    nativeOnnxApi.setInterOpNumThreads = loadNativeOnnxApiFunction<OrtSetThreadCountFunction>(nativeOnnxApi.api, ortApiIndexSetInterOpNumThreads);
    nativeOnnxApi.addSessionConfigEntry = loadNativeOnnxApiFunction<OrtAddSessionConfigEntryFunction>(nativeOnnxApi.api, ortApiIndexAddSessionConfigEntry);
    nativeOnnxApi.appendExecutionProvider = loadNativeOnnxApiFunction<OrtSessionOptionsAppendExecutionProviderFunction>(nativeOnnxApi.api, ortApiIndexSessionOptionsAppendExecutionProvider);
    nativeOnnxApi.appendExecutionProviderCoreML = reinterpret_cast<OrtSessionOptionsAppendExecutionProviderCoreMLFunction>(loadDynamicLibrarySymbol(nativeOnnxApi.libraryHandle, "OrtSessionOptionsAppendExecutionProvider_CoreML"));
    nativeOnnxApi.CreateCuda = loadNativeOnnxApiFunction<CreateCudaOptions>(nativeOnnxApi.api, CudaCreateIndex);
    nativeOnnxApi.ReleaseCuda = loadNativeOnnxApiFunction<ReleaseCudaOptions>(nativeOnnxApi.api, CudaReleaseIndex);
    nativeOnnxApi.Cuda = loadNativeOnnxApiFunction<AppendCuda>(nativeOnnxApi.api, CudaIndex);
    nativeOnnxApi.ProviderApi = loadNativeOnnxApiFunction<GetProviderApi>(nativeOnnxApi.api, ProviderApiIndex);
    nativeOnnxApi.DisablePattern = loadNativeOnnxApiFunction<ConfigureSession>(nativeOnnxApi.api, PatternIndex);
    nativeOnnxApi.ExecutionMode = loadNativeOnnxApiFunction<SetExecutionMode>(nativeOnnxApi.api, ExecutionModeIndex);
    nativeOnnxApi.sessionGetInputCount = loadNativeOnnxApiFunction<OrtSessionGetCountFunction>(nativeOnnxApi.api, ortApiIndexSessionGetInputCount);
    nativeOnnxApi.sessionGetOutputCount = loadNativeOnnxApiFunction<OrtSessionGetCountFunction>(nativeOnnxApi.api, ortApiIndexSessionGetOutputCount);
    nativeOnnxApi.sessionGetInputName = loadNativeOnnxApiFunction<OrtSessionGetNameFunction>(nativeOnnxApi.api, ortApiIndexSessionGetInputName);
    nativeOnnxApi.sessionGetOutputName = loadNativeOnnxApiFunction<OrtSessionGetNameFunction>(nativeOnnxApi.api, ortApiIndexSessionGetOutputName);
    nativeOnnxApi.sessionGetInputTypeInfo = loadNativeOnnxApiFunction<OrtSessionGetTypeInfoFunction>(nativeOnnxApi.api, ortApiIndexSessionGetInputTypeInfo);
    nativeOnnxApi.sessionGetOutputTypeInfo = loadNativeOnnxApiFunction<OrtSessionGetTypeInfoFunction>(nativeOnnxApi.api, ortApiIndexSessionGetOutputTypeInfo);
    nativeOnnxApi.castTypeInfoToTensorInfo = loadNativeOnnxApiFunction<OrtCastTypeInfoToTensorInfoFunction>(nativeOnnxApi.api, ortApiIndexCastTypeInfoToTensorInfo);
    nativeOnnxApi.getTensorElementType = loadNativeOnnxApiFunction<OrtGetTensorElementTypeFunction>(nativeOnnxApi.api, ortApiIndexGetTensorElementType);
    nativeOnnxApi.getDimensionsCount = loadNativeOnnxApiFunction<OrtGetDimensionsCountFunction>(nativeOnnxApi.api, ortApiIndexGetDimensionsCount);
    nativeOnnxApi.getDimensions = loadNativeOnnxApiFunction<OrtGetDimensionsFunction>(nativeOnnxApi.api, ortApiIndexGetDimensions);
    nativeOnnxApi.getTensorShapeElementCount = loadNativeOnnxApiFunction<OrtGetTensorShapeElementCountFunction>(nativeOnnxApi.api, ortApiIndexGetTensorShapeElementCount);
    nativeOnnxApi.getTensorTypeAndShape = loadNativeOnnxApiFunction<OrtGetTensorTypeAndShapeFunction>(nativeOnnxApi.api, ortApiIndexGetTensorTypeAndShape);
    nativeOnnxApi.createTensorWithDataAsOrtValue = loadNativeOnnxApiFunction<OrtCreateTensorWithDataAsOrtValueFunction>(nativeOnnxApi.api, ortApiIndexCreateTensorWithDataAsOrtValue);
    nativeOnnxApi.getTensorMutableData = loadNativeOnnxApiFunction<OrtGetTensorMutableDataFunction>(nativeOnnxApi.api, ortApiIndexGetTensorMutableData);
    nativeOnnxApi.createCpuMemoryInfo = loadNativeOnnxApiFunction<OrtCreateCpuMemoryInfoFunction>(nativeOnnxApi.api, ortApiIndexCreateCpuMemoryInfo);
    nativeOnnxApi.allocatorFree = loadNativeOnnxApiFunction<OrtAllocatorFreeFunction>(nativeOnnxApi.api, ortApiIndexAllocatorFree);
    nativeOnnxApi.getAllocatorWithDefaultOptions = loadNativeOnnxApiFunction<OrtGetAllocatorWithDefaultOptionsFunction>(nativeOnnxApi.api, ortApiIndexGetAllocatorWithDefaultOptions);
    nativeOnnxApi.getAvailableProviders = loadNativeOnnxApiFunction<OrtGetAvailableProvidersFunction>(nativeOnnxApi.api, ortApiIndexGetAvailableProviders);
    nativeOnnxApi.releaseAvailableProviders = loadNativeOnnxApiFunction<OrtReleaseAvailableProvidersFunction>(nativeOnnxApi.api, ortApiIndexReleaseAvailableProviders);
    OrtGetTrainingApiFunction getTrainingApi = loadNativeOnnxApiFunction<OrtGetTrainingApiFunction>(nativeOnnxApi.api, ortApiIndexGetTrainingApi);
    if (getTrainingApi) {
        nativeOnnxApi.trainingApi = getTrainingApi(ortApiVersion);
        if (nativeOnnxApi.trainingApi) {
            nativeOnnxApi.setSeed = loadNativeOnnxApiFunction<OrtTrainingSetSeedFunction>(nativeOnnxApi.trainingApi, ortTrainingApiIndexSetSeed);
        }
    }
    nativeOnnxApi.releaseEnv = loadNativeOnnxApiFunction<OrtReleaseEnvFunction>(nativeOnnxApi.api, ortApiIndexReleaseEnv);
    nativeOnnxApi.releaseStatus = loadNativeOnnxApiFunction<OrtReleaseStatusFunction>(nativeOnnxApi.api, ortApiIndexReleaseStatus);
    nativeOnnxApi.releaseMemoryInfo = loadNativeOnnxApiFunction<OrtReleaseMemoryInfoFunction>(nativeOnnxApi.api, ortApiIndexReleaseMemoryInfo);
    nativeOnnxApi.releaseSession = loadNativeOnnxApiFunction<OrtReleaseSessionFunction>(nativeOnnxApi.api, ortApiIndexReleaseSession);
    nativeOnnxApi.releaseValue = loadNativeOnnxApiFunction<OrtReleaseValueFunction>(nativeOnnxApi.api, ortApiIndexReleaseValue);
    nativeOnnxApi.releaseTypeInfo = loadNativeOnnxApiFunction<OrtReleaseTypeInfoFunction>(nativeOnnxApi.api, ortApiIndexReleaseTypeInfo);
    nativeOnnxApi.releaseTensorTypeAndShapeInfo = loadNativeOnnxApiFunction<OrtReleaseTensorTypeAndShapeInfoFunction>(nativeOnnxApi.api, ortApiIndexReleaseTensorTypeAndShapeInfo);
    nativeOnnxApi.releaseSessionOptions = loadNativeOnnxApiFunction<OrtReleaseSessionOptionsFunction>(nativeOnnxApi.api, ortApiIndexReleaseSessionOptions);
    nativeOnnxApi.hasConfiguredSeed = tryReadNativeOnnxSeedFromEnvironment(nativeOnnxApi.configuredSeed);
    if (nativeOnnxApi.hasConfiguredSeed && !nativeOnnxApi.setSeed) {
        throw std::runtime_error("LITEVOX_ORT_SEED / --seed はこの ONNX Runtime では未対応です");
    }
    return nativeOnnxApi;
}

void ensureNativeOnnxCall(NativeOnnxApi &nativeOnnxApi, OrtStatus *callStatus, const std::string &operationName);
std::vector<std::string> collectNativeOnnxAvailableProviders(NativeOnnxApi &nativeOnnxApi);
void applyNativeOnnxSeedIfConfigured(NativeOnnxApi &nativeOnnxApi);
void configureNativeOnnxSessionOptions(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, OrtSessionOptions *sessionOptions, uint16_t cpuThreadCount, bool shouldUseVvBinConfig);

void closeNativeOnnxApi(NativeOnnxApi &nativeOnnxApi) {
    if (nativeOnnxApi.libraryHandle) {
        closeDynamicLibrary(nativeOnnxApi.libraryHandle);
        nativeOnnxApi.libraryHandle = nullptr;
    }
}

static bool hasNativeOnnxProviderName(const std::vector<std::string> &providerNames, const std::string &providerName) {
    return std::find(providerNames.begin(), providerNames.end(), providerName) != providerNames.end();
}

static std::string consumeNativeOnnxStatusMessage(NativeOnnxApi &nativeOnnxApi, OrtStatus *callStatus) {
    if (!callStatus) {
        return "";
    }
    std::string errorMessage = nativeOnnxApi.getErrorMessage ? nativeOnnxApi.getErrorMessage(callStatus) : "unknown ONNX Runtime error";
    nativeOnnxApi.releaseStatus(callStatus);
    return errorMessage;
}

bool IsCoremlProfile() {
    const char *Value = std::getenv("LITEVOX_COREML_PROFILE");
    if (!Value || !*Value || std::strcmp(Value, "0") == 0) return false;
    if (std::strcmp(Value, "1") == 0) return true;
    throw std::invalid_argument("LITEVOX_COREML_PROFILE は 0 / 1 が必要です");
}

static void appendNativeOnnxExecutionProvider(NativeOnnxApi &Api, OrtSessionOptions *Options, const std::string &Provider) {
    if (Provider.empty() || Provider == "CPUExecutionProvider") return;
    if (Provider == "CUDAExecutionProvider" && Api.CreateCuda && Api.Cuda && Api.ReleaseCuda) {
        OrtCUDAProviderOptionsV2 *Raw = nullptr;
        OrtStatus *Status = Api.CreateCuda(&Raw);
        std::unique_ptr<OrtCUDAProviderOptionsV2, ReleaseCudaOptions> Cuda(Raw, Api.ReleaseCuda);
        ensureNativeOnnxCall(Api, Status, "CUDA options 作成");
        ensureNativeOnnxCall(Api, Api.Cuda(Options, Cuda.get()), "CUDA provider 設定");
        return;
    }
    if (Provider == "DmlExecutionProvider" && Api.ProviderApi && Api.DisablePattern && Api.ExecutionMode) {
        // DirectML は memory pattern と並列実行モードをサポートしない。
        ensureNativeOnnxCall(Api, Api.DisablePattern(Options), "DML memory pattern 無効化");
        ensureNativeOnnxCall(Api, Api.ExecutionMode(Options, 0), "DML sequential mode 設定");
        const void *Dml = nullptr;
        ensureNativeOnnxCall(Api, Api.ProviderApi("DML", ortApiVersion, &Dml), "DML API 取得");
        if (!Dml) throw std::runtime_error("DML API がありません");
        struct DeviceOptions { int32_t Preference; uint32_t Filter; } Device{1, 1};
        using AppendDevice = OrtStatus *(*)(OrtSessionOptions *, DeviceOptions *);
        auto Append = loadNativeOnnxApiFunction<AppendDevice>(Dml, 5);
        if (!Append) throw std::runtime_error("DML high-performance API がありません");
        ensureNativeOnnxCall(Api, Append(Options, &Device), "DML high-performance GPU 設定");
        return;
    }
    if (Provider == "WebGpuExecutionProvider" && Api.appendExecutionProvider) {
        const char *Keys[] = {"powerPreference"};
        const char *Values[] = {"high-performance"};
        ensureNativeOnnxCall(Api, Api.appendExecutionProvider(Options, "WebGPU", Keys, Values, 1), "WebGPU provider 設定");
        return;
    }
    if (Provider == "CoreMLExecutionProvider") {
        bool IsProfile = IsCoremlProfile();
        if (Api.appendExecutionProvider) {
            const char *Keys[] = {"ModelFormat", "MLComputeUnits", "ProfileComputePlan"};
            const char *Values[] = {"MLProgram", "ALL", "1"};
            OrtStatus *Status = Api.appendExecutionProvider(Options, "CoreML", Keys, Values, IsProfile ? 3 : 2);
            if (!Status) return;
            std::string Error = consumeNativeOnnxStatusMessage(Api, Status);
            if (IsProfile || !Api.appendExecutionProviderCoreML) {
                throw std::runtime_error("CoreML provider 設定に失敗しました: " + Error);
            }
        }
        if (IsProfile) throw std::runtime_error("この ORT は CoreML compute plan を提供しません");
        if (Api.appendExecutionProviderCoreML) {
            ensureNativeOnnxCall(Api, Api.appendExecutionProviderCoreML(Options, nativeOnnxCoreMlFlags), "CoreML MLProgram 設定");
            return;
        }
    }
    throw std::runtime_error("未対応の execution provider です: " + Provider);
}

static bool tryConfigureNativeOnnxExecutionProvider(NativeOnnxApi &nativeOnnxApi, const std::string &providerName, std::string &errorMessage) {
    if (providerName.empty() || providerName == "CPUExecutionProvider") {
        return true;
    }
    OrtSessionOptions *sessionOptions = nullptr;
    OrtEnv *Env = nullptr;
    try {
        // WebGPU の adapter 探索などは ORT の既定 logger を必要とする。
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createEnv(ortLoggingLevelWarning, "litevox-provider", &Env), "provider Env 作成");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionOptions(&sessionOptions), "SessionOptions 作成");
        appendNativeOnnxExecutionProvider(nativeOnnxApi, sessionOptions, providerName);
        nativeOnnxApi.releaseSessionOptions(sessionOptions);
        nativeOnnxApi.releaseEnv(Env);
        return true;
    } catch (const std::exception &exception) {
        if (sessionOptions) {
            nativeOnnxApi.releaseSessionOptions(sessionOptions);
        }
        if (Env) nativeOnnxApi.releaseEnv(Env);
        errorMessage = exception.what();
        return false;
    }
}

std::string SelectProvider(NativeOnnxApi &Api, const std::string &Mode, const std::vector<std::string> &Providers, bool &IsUsable) {
    if (Mode != "cpu" && Mode != "auto" && Mode != "gpu") throw std::invalid_argument("acceleration mode が不正です");
    IsUsable = false;
    std::string Error;
    const char *Choice = std::getenv("LITEVOX_EXECUTION_PROVIDER");
    if (Choice && *Choice) {
        if ((Mode == "cpu" && std::string(Choice) != "CPUExecutionProvider") ||
            (Mode == "gpu" && std::string(Choice) == "CPUExecutionProvider")) {
            throw std::invalid_argument("LITEVOX_EXECUTION_PROVIDER と acceleration mode が矛盾しています");
        }
        if (!hasNativeOnnxProviderName(Providers, Choice) || !tryConfigureNativeOnnxExecutionProvider(Api, Choice, Error)) {
            throw std::runtime_error(std::string("指定した provider を使用できません: ") + Choice + ": " + Error);
        }
        IsUsable = std::string(Choice) != "CPUExecutionProvider";
        return Choice;
    }
    // WebGPU は VOICEVOX の動的形状を扱える。CoreML は一部グラフの中間形状が未解決だと作成に失敗する。
    for (const char *Provider : {"CUDAExecutionProvider", "DmlExecutionProvider", "WebGpuExecutionProvider", "CoreMLExecutionProvider"}) {
        if (!hasNativeOnnxProviderName(Providers, Provider)) continue;
        std::string Candidate;
        if (tryConfigureNativeOnnxExecutionProvider(Api, Provider, Candidate)) {
            IsUsable = true;
            return Mode == "cpu" ? "CPUExecutionProvider" : Provider;
        }
        if (Error.empty()) Error = Candidate;
    }
    if (Mode == "gpu") throw std::runtime_error("GPU provider を有効化できません: " + (Error.empty() ? "対応プロバイダーがありません" : Error));
    return "CPUExecutionProvider";
}

NativeOnnxRuntimeState createNativeOnnxRuntimeState(const fs::path &onnxruntimeLibraryPath, const std::string &requestedAccelerationMode) {
    NativeOnnxApi nativeOnnxApi = loadNativeOnnxApi(onnxruntimeLibraryPath);
    try {
        NativeOnnxRuntimeState runtimeState;
        runtimeState.libraryHandle = nativeOnnxApi.libraryHandle;
        runtimeState.libraryPath = onnxruntimeLibraryPath;
        runtimeState.version = nativeOnnxApi.apiBase && nativeOnnxApi.apiBase->getVersionString ? nativeOnnxApi.apiBase->getVersionString() : "";
        runtimeState.availableProviders = collectNativeOnnxAvailableProviders(nativeOnnxApi);
        runtimeState.requestedAccelerationMode = requestedAccelerationMode.empty() ? "auto" : requestedAccelerationMode;
        runtimeState.selectedExecutionProvider = SelectProvider(nativeOnnxApi, runtimeState.requestedAccelerationMode, runtimeState.availableProviders, runtimeState.hasUsableGpuProvider);
        runtimeState.isGpuExecutionProviderSelected = runtimeState.selectedExecutionProvider != "CPUExecutionProvider";
        runtimeState.apiVersion = ortApiVersion;
        runtimeState.isLoaded = true;
        nativeOnnxApi.libraryHandle = nullptr;
        return runtimeState;
    } catch (...) {
        closeNativeOnnxApi(nativeOnnxApi);
        throw;
    }
}

void destroyNativeOnnxRuntimeState(NativeOnnxRuntimeState &runtimeState) {
    if (runtimeState.libraryHandle) {
        closeDynamicLibrary(runtimeState.libraryHandle);
    }
    runtimeState.libraryHandle = nullptr;
    runtimeState.libraryPath.clear();
    runtimeState.version.clear();
    runtimeState.availableProviders.clear();
    runtimeState.apiVersion = 0;
    runtimeState.isLoaded = false;
}

void ensureNativeOnnxCall(NativeOnnxApi &nativeOnnxApi, OrtStatus *callStatus, const std::string &operationName) {
    if (!callStatus) {
        return;
    }
    std::string errorMessage = nativeOnnxApi.getErrorMessage ? nativeOnnxApi.getErrorMessage(callStatus) : "unknown ONNX Runtime error";
    nativeOnnxApi.releaseStatus(callStatus);
    throw std::runtime_error(operationName + " に失敗しました: " + errorMessage);
}

std::vector<std::string> collectNativeOnnxAvailableProviders(NativeOnnxApi &nativeOnnxApi) {
    std::vector<std::string> providers;
    if (!nativeOnnxApi.getAvailableProviders || !nativeOnnxApi.releaseAvailableProviders) {
        return providers;
    }
    char **providerPointers = nullptr;
    int providerCount = 0;
    ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getAvailableProviders(&providerPointers, &providerCount), "available providers 取得");
    try {
        for (int providerIndex = 0; providerIndex < providerCount; providerIndex++) {
            if (providerPointers[providerIndex]) {
                providers.push_back(providerPointers[providerIndex]);
            }
        }
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.releaseAvailableProviders(providerPointers, providerCount), "available providers 解放");
    } catch (...) {
        nativeOnnxApi.releaseAvailableProviders(providerPointers, providerCount);
        throw;
    }
    return providers;
}

void applyNativeOnnxSeedIfConfigured(NativeOnnxApi &nativeOnnxApi) {
    if (!nativeOnnxApi.hasConfiguredSeed || !nativeOnnxApi.setSeed) {
        return;
    }
    ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setSeed(nativeOnnxApi.configuredSeed), "training random seed 設定");
}

void configureNativeOnnxSessionOptions(NativeOnnxApi &nativeOnnxApi, const NativeOnnxRuntimeState *runtimeState, OrtSessionOptions *sessionOptions, uint16_t cpuThreadCount, bool shouldUseVvBinConfig) {
    ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setSessionGraphOptimizationLevel(sessionOptions, runtimeState ? OptimizationAll : ortGraphOptimizationLevelBasic), "graph optimization 設定");
    if (cpuThreadCount > 0) {
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setIntraOpNumThreads(sessionOptions, cpuThreadCount), "intra op thread 設定");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setInterOpNumThreads(sessionOptions, cpuThreadCount), "inter op thread 設定");
    }
    if (shouldUseVvBinConfig) {
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.addSessionConfigEntry(sessionOptions, "session.use_vv_bin", "1"), "vv_bin session 設定");
    }
    if (runtimeState) {
        appendNativeOnnxExecutionProvider(nativeOnnxApi, sessionOptions, runtimeState->selectedExecutionProvider);
    }
}

static bool hasNativeOnnxProviderName(const NativeOnnxRuntimeState &runtimeState, const std::string &providerName) {
    return hasNativeOnnxProviderName(runtimeState.availableProviders, providerName);
}

bool hasNativeOnnxGpuProvider(const NativeOnnxRuntimeState &runtimeState) {
    return hasNativeOnnxProviderName(runtimeState, "CUDAExecutionProvider")
        || hasNativeOnnxProviderName(runtimeState, "DmlExecutionProvider")
        || hasNativeOnnxProviderName(runtimeState, "CoreMLExecutionProvider")
        || hasNativeOnnxProviderName(runtimeState, "ROCMExecutionProvider")
        || hasNativeOnnxProviderName(runtimeState, "TensorrtExecutionProvider")
        || hasNativeOnnxProviderName(runtimeState, "MIGraphXExecutionProvider")
        || hasNativeOnnxProviderName(runtimeState, "WebGpuExecutionProvider");
}

std::string createNativeOnnxSupportedDevicesJson(const NativeOnnxRuntimeState &runtimeState) {
    bool hasCpuProvider = hasNativeOnnxProviderName(runtimeState, "CPUExecutionProvider");
    bool hasCudaProvider = hasNativeOnnxProviderName(runtimeState, "CUDAExecutionProvider");
    bool hasDmlProvider = hasNativeOnnxProviderName(runtimeState, "DmlExecutionProvider");
    return std::string("{\"cpu\":") + (hasCpuProvider ? "true" : "false")
        + ",\"cuda\":" + (hasCudaProvider ? "true" : "false")
        + ",\"dml\":" + (hasDmlProvider ? "true" : "false") + "}";
}

std::string createNativeOnnxProviderInfoJson(const NativeOnnxRuntimeState &runtimeState) {
    std::ostringstream jsonStream;
    jsonStream << "{\"available\":[";
    for (size_t providerIndex = 0; providerIndex < runtimeState.availableProviders.size(); providerIndex++) {
        if (providerIndex > 0) {
            jsonStream << ",";
        }
        jsonStream << quoteJsonString(runtimeState.availableProviders[providerIndex]);
    }
    jsonStream << "],";
    jsonStream << "\"cpu\":" << (hasNativeOnnxProviderName(runtimeState, "CPUExecutionProvider") ? "true" : "false") << ",";
    jsonStream << "\"cuda\":" << (hasNativeOnnxProviderName(runtimeState, "CUDAExecutionProvider") ? "true" : "false") << ",";
    jsonStream << "\"dml\":" << (hasNativeOnnxProviderName(runtimeState, "DmlExecutionProvider") ? "true" : "false") << ",";
    jsonStream << "\"coreml\":" << (hasNativeOnnxProviderName(runtimeState, "CoreMLExecutionProvider") ? "true" : "false") << ",";
    jsonStream << "\"gpu\":" << (hasNativeOnnxGpuProvider(runtimeState) ? "true" : "false") << ",";
    jsonStream << "\"usable_gpu\":" << (runtimeState.hasUsableGpuProvider ? "true" : "false") << ",";
    jsonStream << "\"selected\":" << quoteJsonString(runtimeState.selectedExecutionProvider) << "}";
    return jsonStream.str();
}

