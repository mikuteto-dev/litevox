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

static bool isNativeOnnxTraceJsonPath(const fs::path &filePath, const std::string &nameMarker) {
    std::string filename = filePath.filename().string();
    return filename.find(nameMarker) != std::string::npos && filePath.extension() == ".json";
}

static std::vector<NativeOnnxTraceInput> loadNativeOnnxTraceTensors(const fs::path &inputDirectory, const std::string &nameMarker, const std::string &directoryLabel) {
    if (inputDirectory.empty()) {
        return {};
    }
    ensurePathExists(inputDirectory, directoryLabel);
    std::vector<fs::path> jsonPaths;
    for (const fs::directory_entry &directoryEntry : fs::directory_iterator(inputDirectory)) {
        if (directoryEntry.is_regular_file() && isNativeOnnxTraceJsonPath(directoryEntry.path(), nameMarker)) {
            jsonPaths.push_back(directoryEntry.path());
        }
    }
    std::sort(jsonPaths.begin(), jsonPaths.end());
    std::vector<NativeOnnxTraceInput> traceInputs;
    for (const fs::path &jsonPath : jsonPaths) {
        std::string jsonText = readNativeOnnxTextFile(jsonPath);
        std::string binaryName = extractNativeOnnxJsonStringField(jsonText, "binary");
        if (binaryName.empty()) {
            throw std::runtime_error("binary がありません: " + jsonPath.string());
        }
        std::string dtypeText = extractNativeOnnxJsonStringField(jsonText, "dtype");
        std::string nameText = extractNativeOnnxJsonStringField(jsonText, "name");
        NativeOnnxTraceInput traceInput;
        traceInput.name = nameText;
        traceInput.elementType = parseNativeOnnxElementType(dtypeText);
        traceInput.dimensions = extractNativeOnnxJsonShapeField(jsonText);
        traceInput.bytes = readNativeOnnxBinaryFile(inputDirectory / binaryName);
        traceInputs.push_back(std::move(traceInput));
    }
    return traceInputs;
}

std::vector<NativeOnnxTraceInput> loadNativeOnnxTraceInputs(const fs::path &inputDirectory) {
    return loadNativeOnnxTraceTensors(inputDirectory, "-input-", "trace input directory");
}

std::vector<NativeOnnxTraceInput> loadNativeOnnxTraceOutputs(const fs::path &inputDirectory) {
    return loadNativeOnnxTraceTensors(inputDirectory, "-output-", "trace output directory");
}

static fs::path getNativeOnnxTensorTraceDirectory() {
    const char *directoryText = std::getenv("LITEVOX_TENSOR_TRACE_DIR");
    if (!directoryText || directoryText[0] == '\0') {
        return {};
    }
    return fs::path(directoryText);
}

static ModelAssetRecord createNativeOnnxPseudoModelAsset(const fs::path &modelPath) {
    ModelAssetRecord modelAsset;
    modelAsset.archivePath = modelPath.parent_path();
    modelAsset.entryName = modelPath.filename().string();
    return modelAsset;
}

static std::string sanitizeNativeOnnxTraceName(const std::string &nameText) {
    std::string sanitizedText;
    sanitizedText.reserve(nameText.size());
    for (char character : nameText) {
        unsigned char unsignedCharacter = static_cast<unsigned char>(character);
        if (std::isalnum(unsignedCharacter) || character == '_' || character == '-') {
            sanitizedText.push_back(character);
        } else {
            sanitizedText.push_back('_');
        }
    }
    if (sanitizedText.empty()) {
        return "tensor";
    }
    return sanitizedText;
}


static std::string createNativeOnnxTraceShapeJson(const std::vector<int64_t> &dimensions) {
    std::ostringstream jsonStream;
    jsonStream << "[";
    for (size_t dimensionIndex = 0; dimensionIndex < dimensions.size(); dimensionIndex++) {
        if (dimensionIndex > 0) {
            jsonStream << ",";
        }
        jsonStream << dimensions[dimensionIndex];
    }
    jsonStream << "]";
    return jsonStream.str();
}

static std::string inferNativeOnnxTraceOperationName(const ModelAssetRecord &modelAsset, const std::vector<NativeOnnxTraceInput> &inputTensors) {
    auto hasInput = [&](const std::string &tensorName) {
        return findNativeOnnxTraceTensor(inputTensors, tensorName) != nullptr;
    };
    if (hasInput("phoneme_list")) {
        return "predict_duration";
    }
    if (hasInput("vowel_phoneme_list")) {
        return "predict_intonation";
    }
    if (hasInput("spec")) {
        return "render_audio_segment";
    }
    if (hasInput("f0") && hasInput("phoneme")) {
        return "decode";
    }
    return sanitizeNativeOnnxTraceName(fs::path(modelAsset.entryName).stem().string());
}

static void writeNativeOnnxTraceTensorFile(const fs::path &traceDirectory, size_t traceId, const std::string &roleText, const NativeOnnxTraceInput &tensor) {
    fs::create_directories(traceDirectory);
    std::ostringstream filePrefixStream;
    filePrefixStream << std::setfill('0') << std::setw(4) << traceId << "-" << roleText << "-" << sanitizeNativeOnnxTraceName(tensor.name);
    fs::path filePrefix = traceDirectory / filePrefixStream.str();
    fs::path binaryPath = filePrefix;
    binaryPath += ".bin";
    fs::path metadataPath = filePrefix;
    metadataPath += ".json";
    writeBinaryFile(binaryPath, tensor.bytes);
    std::ostringstream metadataStream;
    metadataStream << "{";
    metadataStream << "\"trace_id\":" << traceId << ",";
    metadataStream << "\"role\":" << quoteJsonString(roleText) << ",";
    metadataStream << "\"name\":" << quoteJsonString(tensor.name) << ",";
    metadataStream << "\"dtype\":" << quoteJsonString(formatNativeOnnxElementType(tensor.elementType)) << ",";
    metadataStream << "\"shape\":" << createNativeOnnxTraceShapeJson(tensor.dimensions) << ",";
    metadataStream << "\"binary\":" << quoteJsonString(binaryPath.filename().string());
    metadataStream << "}";
    writeTextFile(metadataPath, metadataStream.str());
}

void writeNativeOnnxTensorTrace(const ModelAssetRecord &modelAsset, const std::vector<NativeOnnxTraceInput> &inputTensors, const std::vector<NativeOnnxTraceInput> &outputTensors) {
    fs::path traceDirectory = getNativeOnnxTensorTraceDirectory();
    if (traceDirectory.empty()) {
        return;
    }
    size_t traceId = nativeOnnxNextTraceId.fetch_add(1);
    for (const NativeOnnxTraceInput &inputTensor : inputTensors) {
        writeNativeOnnxTraceTensorFile(traceDirectory, traceId, "input", inputTensor);
    }
    for (const NativeOnnxTraceInput &outputTensor : outputTensors) {
        writeNativeOnnxTraceTensorFile(traceDirectory, traceId, "output", outputTensor);
    }
    std::ostringstream sessionStream;
    sessionStream << "{";
    sessionStream << "\"trace_id\":" << traceId << ",";
    sessionStream << "\"operation\":" << quoteJsonString(inferNativeOnnxTraceOperationName(modelAsset, inputTensors)) << ",";
    sessionStream << "\"asset\":" << quoteJsonString(modelAsset.entryName) << ",";
    sessionStream << "\"outputs\":" << outputTensors.size();
    sessionStream << "}";
    std::ostringstream sessionNameStream;
    sessionNameStream << std::setfill('0') << std::setw(4) << traceId << "-session.json";
    writeTextFile(traceDirectory / sessionNameStream.str(), sessionStream.str());
}

NativeOnnxAudioQuerySettings parseNativeOnnxAudioQuerySettings(const std::string &audioQueryText) {
    NativeOnnxAudioQuerySettings audioQuerySettings;
    audioQuerySettings.speedScale = static_cast<float>(extractNativeOnnxJsonNumberField(audioQueryText, "speedScale", audioQuerySettings.speedScale));
    audioQuerySettings.pitchScale = static_cast<float>(extractNativeOnnxJsonNumberField(audioQueryText, "pitchScale", audioQuerySettings.pitchScale));
    audioQuerySettings.intonationScale = static_cast<float>(extractNativeOnnxJsonNumberField(audioQueryText, "intonationScale", audioQuerySettings.intonationScale));
    audioQuerySettings.volumeScale = static_cast<float>(extractNativeOnnxJsonNumberField(audioQueryText, "volumeScale", audioQuerySettings.volumeScale));
    audioQuerySettings.prePhonemeLength = static_cast<float>(extractNativeOnnxJsonNumberField(audioQueryText, "prePhonemeLength", audioQuerySettings.prePhonemeLength));
    audioQuerySettings.postPhonemeLength = static_cast<float>(extractNativeOnnxJsonNumberField(audioQueryText, "postPhonemeLength", audioQuerySettings.postPhonemeLength));
    double Rate = extractNativeOnnxJsonNumberField(audioQueryText, "outputSamplingRate", audioQuerySettings.outputSamplingRate);
    if (!std::isfinite(Rate) || Rate < 1 || Rate > std::numeric_limits<uint32_t>::max() || std::floor(Rate) != Rate) {
        throw std::invalid_argument("outputSamplingRate が不正です");
    }
    audioQuerySettings.outputSamplingRate = static_cast<uint32_t>(Rate);
    audioQuerySettings.outputStereo = extractNativeOnnxJsonBoolField(audioQueryText, "outputStereo", audioQuerySettings.outputStereo);
    return audioQuerySettings;
}

fs::path resolveNativeOnnxAudioQueryPath(const fs::path &inputDirectory, const fs::path &audioQueryPath) {
    if (!audioQueryPath.empty()) {
        ensurePathExists(audioQueryPath, "audio query");
        return audioQueryPath;
    }
    if (!inputDirectory.empty()) {
        fs::path tracedAudioQueryPath = inputDirectory.parent_path() / "audio_query.json";
        if (fs::exists(tracedAudioQueryPath)) {
            return tracedAudioQueryPath;
        }
    }
    throw std::runtime_error("--audio-query または trace 由来の audio_query.json が必要です");
}

const NativeOnnxTraceInput *findNativeOnnxTraceTensor(const std::vector<NativeOnnxTraceInput> &traceTensors, const std::string &tensorName) {
    for (const NativeOnnxTraceInput &traceTensor : traceTensors) {
        if (traceTensor.name == tensorName) {
            return &traceTensor;
        }
    }
    return nullptr;
}

bool areNativeOnnxDimensionsEqual(const std::vector<int64_t> &leftDimensions, const std::vector<int64_t> &rightDimensions) {
    if (leftDimensions.size() != rightDimensions.size()) {
        return false;
    }
    for (size_t dimensionIndex = 0; dimensionIndex < leftDimensions.size(); dimensionIndex++) {
        if (leftDimensions[dimensionIndex] >= 0 && rightDimensions[dimensionIndex] >= 0 && leftDimensions[dimensionIndex] != rightDimensions[dimensionIndex]) {
            return false;
        }
    }
    return true;
}

size_t calculatePositiveShapeElementCount(const std::vector<int64_t> &dimensions) {
    size_t elementCount = 1;
    for (int64_t dimension : dimensions) {
        if (dimension <= 0) {
            throw std::runtime_error("固定 shape ではありません");
        }
        elementCount *= static_cast<size_t>(dimension);
    }
    return elementCount;
}

size_t getNativeOnnxElementByteCount(int32_t elementType) {
    switch (elementType) {
        case 1:
            return sizeof(float);
        case 6:
            return sizeof(int32_t);
        case 7:
            return sizeof(int64_t);
        case 9:
            return sizeof(uint8_t);
        default:
            return 0;
    }
}

bool canCreateNativeOnnxSmokeInput(const NativeOnnxValueDescriptor &inputDescriptor) {
    if (inputDescriptor.dimensions.empty() || getNativeOnnxElementByteCount(inputDescriptor.elementType) == 0) {
        return false;
    }
    for (int64_t dimension : inputDescriptor.dimensions) {
        if (dimension <= 0) {
            return false;
        }
    }
    return true;
}

static bool canRunNativeOnnxSmokeTest(const std::vector<NativeOnnxValueDescriptor> &inputDescriptors, const std::vector<NativeOnnxValueDescriptor> &outputDescriptors, const std::vector<NativeOnnxTraceInput> &traceInputs) {
    if (inputDescriptors.empty() || outputDescriptors.empty()) {
        return false;
    }
    for (const NativeOnnxValueDescriptor &inputDescriptor : inputDescriptors) {
        const NativeOnnxTraceInput *traceInput = findNativeOnnxTraceTensor(traceInputs, inputDescriptor.name);
        if (traceInput) {
            if (traceInput->elementType != inputDescriptor.elementType || !areNativeOnnxDimensionsEqual(inputDescriptor.dimensions, traceInput->dimensions)) {
                return false;
            }
        } else if (!canCreateNativeOnnxSmokeInput(inputDescriptor)) {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> createNativeOnnxInputBytes(const NativeOnnxValueDescriptor &inputDescriptor, size_t inputIndex, size_t elementCount) {
    size_t elementBytes = getNativeOnnxElementByteCount(inputDescriptor.elementType);
    std::vector<uint8_t> inputBytes(elementCount * elementBytes);
    if (inputDescriptor.elementType == 1) {
        std::vector<float> values(elementCount);
        for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
            values[valueIndex] = static_cast<float>(valueIndex + 1 + inputIndex);
        }
        std::memcpy(inputBytes.data(), values.data(), inputBytes.size());
    } else if (inputDescriptor.elementType == 6) {
        std::vector<int32_t> values(elementCount);
        for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
            values[valueIndex] = static_cast<int32_t>(valueIndex + 1 + inputIndex);
        }
        std::memcpy(inputBytes.data(), values.data(), inputBytes.size());
    } else if (inputDescriptor.elementType == 7) {
        std::vector<int64_t> values(elementCount);
        for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
            values[valueIndex] = static_cast<int64_t>(valueIndex + 1 + inputIndex);
        }
        std::memcpy(inputBytes.data(), values.data(), inputBytes.size());
    } else if (inputDescriptor.elementType == 9) {
        for (size_t valueIndex = 0; valueIndex < inputBytes.size(); valueIndex++) {
            inputBytes[valueIndex] = (valueIndex + inputIndex) % 2 == 0 ? 1 : 0;
        }
    }
    return inputBytes;
}

static std::string readNativeOnnxFirstValueText(const NativeOnnxValueDescriptor &outputDescriptor, void *outputData, size_t outputElementCount) {
    if (!outputData || outputElementCount == 0) {
        return "";
    }
    if (outputDescriptor.elementType == 1) {
        return std::to_string(static_cast<float *>(outputData)[0]);
    }
    if (outputDescriptor.elementType == 6) {
        return std::to_string(static_cast<int32_t *>(outputData)[0]);
    }
    if (outputDescriptor.elementType == 7) {
        return std::to_string(static_cast<int64_t *>(outputData)[0]);
    }
    if (outputDescriptor.elementType == 9) {
        return static_cast<uint8_t *>(outputData)[0] ? "true" : "false";
    }
    return "";
}

std::vector<int64_t> readNativeOnnxTensorDimensions(NativeOnnxApi &nativeOnnxApi, const OrtTensorTypeAndShapeInfo *tensorShapeInfo) {
    size_t dimensionCount = 0;
    ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getDimensionsCount(tensorShapeInfo, &dimensionCount), "output dimension count 取得");
    std::vector<int64_t> dimensions(dimensionCount);
    if (!dimensions.empty()) {
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getDimensions(tensorShapeInfo, dimensions.data(), dimensions.size()), "output dimension 取得");
    }
    return dimensions;
}

std::string compareNativeOnnxTraceOutput(const NativeOnnxValueDescriptor &outputDescriptor, const std::vector<int64_t> &outputDimensions, const void *outputPointer, size_t outputByteCount, const std::vector<NativeOnnxTraceInput> &traceOutputs) {
    const NativeOnnxTraceInput *traceOutput = findNativeOnnxTraceTensor(traceOutputs, outputDescriptor.name);
    if (!traceOutput) {
        return "not_found";
    }
    if (traceOutput->elementType != outputDescriptor.elementType) {
        return "type_mismatch";
    }
    if (!areNativeOnnxDimensionsEqual(outputDimensions, traceOutput->dimensions)) {
        return "shape_mismatch";
    }
    if (outputByteCount != traceOutput->bytes.size()) {
        return "byte_size_mismatch";
    }
    if (!outputPointer && outputByteCount > 0) {
        return "output_null";
    }
    if (outputByteCount == 0 || std::memcmp(outputPointer, traceOutput->bytes.data(), outputByteCount) == 0) {
        return "exact";
    }
    if (outputDescriptor.elementType == 1) {
        const float *actualValues = static_cast<const float *>(outputPointer);
        const float *expectedValues = reinterpret_cast<const float *>(traceOutput->bytes.data());
        size_t valueCount = outputByteCount / sizeof(float);
        float maxDifference = 0.0f;
        for (size_t valueIndex = 0; valueIndex < valueCount; valueIndex++) {
            float difference = actualValues[valueIndex] > expectedValues[valueIndex] ? actualValues[valueIndex] - expectedValues[valueIndex] : expectedValues[valueIndex] - actualValues[valueIndex];
            if (difference > maxDifference) {
                maxDifference = difference;
            }
        }
        return "float32_diff_max=" + std::to_string(maxDifference);
    }
    return "bytes_mismatch";
}

const NativeOnnxTraceInput &requireNativeOnnxTensor(const std::vector<NativeOnnxTraceInput> &traceTensors, const std::string &tensorName) {
    const NativeOnnxTraceInput *traceTensor = findNativeOnnxTraceTensor(traceTensors, tensorName);
    if (!traceTensor) {
        throw std::runtime_error("trace tensor がありません: " + tensorName);
    }
    return *traceTensor;
}


std::string compareNativeOnnxTensors(const NativeOnnxTraceInput &actualTensor, const NativeOnnxTraceInput *expectedTensor) {
    if (!expectedTensor) {
        return "not_found";
    }
    if (actualTensor.elementType != expectedTensor->elementType) {
        return "type_mismatch";
    }
    if (!areNativeOnnxDimensionsEqual(actualTensor.dimensions, expectedTensor->dimensions)) {
        return "shape_mismatch";
    }
    if (actualTensor.bytes.size() != expectedTensor->bytes.size()) {
        return "byte_size_mismatch";
    }
    if (actualTensor.bytes.empty() || std::memcmp(actualTensor.bytes.data(), expectedTensor->bytes.data(), actualTensor.bytes.size()) == 0) {
        return "exact";
    }
    if (actualTensor.elementType == 1) {
        const float *actualValues = reinterpret_cast<const float *>(actualTensor.bytes.data());
        const float *expectedValues = reinterpret_cast<const float *>(expectedTensor->bytes.data());
        size_t valueCount = actualTensor.bytes.size() / sizeof(float);
        float maxDifference = 0.0f;
        for (size_t valueIndex = 0; valueIndex < valueCount; valueIndex++) {
            float difference = actualValues[valueIndex] > expectedValues[valueIndex] ? actualValues[valueIndex] - expectedValues[valueIndex] : expectedValues[valueIndex] - actualValues[valueIndex];
            if (difference > maxDifference) {
                maxDifference = difference;
            }
        }
        return "float32_diff_max=" + std::to_string(maxDifference);
    }
    return "bytes_mismatch";
}


static void appendNativeOnnxSmokeRunInfo(std::ostringstream &inspectStream, NativeOnnxApi &nativeOnnxApi, OrtSession *session, const std::vector<NativeOnnxValueDescriptor> &inputDescriptors, const std::vector<NativeOnnxValueDescriptor> &outputDescriptors, const std::vector<NativeOnnxTraceInput> &traceInputs, const std::vector<NativeOnnxTraceInput> &traceOutputs, const ModelAssetRecord *traceModelAsset) {
    if (!canRunNativeOnnxSmokeTest(inputDescriptors, outputDescriptors, traceInputs)) {
        inspectStream << "run_status\tskipped\n";
        return;
    }
    OrtMemoryInfo *memoryInfo = nullptr;
    std::vector<std::vector<uint8_t>> inputBuffers;
    std::vector<OrtValue *> inputValues;
    std::vector<OrtValue *> outputValues(outputDescriptors.size(), nullptr);
    std::vector<NativeOnnxTraceInput> actualInputTensors;
    std::vector<NativeOnnxTraceInput> actualOutputTensors;
    inputBuffers.reserve(inputDescriptors.size());
    inputValues.reserve(inputDescriptors.size());
    if (traceModelAsset) {
        actualInputTensors.reserve(inputDescriptors.size());
        actualOutputTensors.reserve(outputDescriptors.size());
    }
    try {
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createCpuMemoryInfo(0, 0, &memoryInfo), "CPU memory info 作成");
        std::vector<const char *> inputNames;
        std::vector<const OrtValue *> inputPointers;
        size_t totalInputElements = 0;
        inputNames.reserve(inputDescriptors.size());
        inputPointers.reserve(inputDescriptors.size());
        for (size_t inputIndex = 0; inputIndex < inputDescriptors.size(); inputIndex++) {
            const NativeOnnxValueDescriptor &inputDescriptor = inputDescriptors[inputIndex];
            const NativeOnnxTraceInput *traceInput = findNativeOnnxTraceTensor(traceInputs, inputDescriptor.name);
            std::vector<int64_t> inputDimensions = inputDescriptor.dimensions;
            if (traceInput) {
                inputDimensions = traceInput->dimensions;
                inputBuffers.push_back(traceInput->bytes);
            } else {
                size_t inputElementCount = calculatePositiveShapeElementCount(inputDescriptor.dimensions);
                inputBuffers.push_back(createNativeOnnxInputBytes(inputDescriptor, inputIndex, inputElementCount));
            }
            size_t inputElementCount = calculatePositiveShapeElementCount(inputDimensions);
            totalInputElements += inputElementCount;
            OrtValue *inputValue = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createTensorWithDataAsOrtValue(memoryInfo, inputBuffers.back().data(), inputBuffers.back().size(), inputDimensions.data(), inputDimensions.size(), inputDescriptor.elementType, &inputValue), "input tensor 作成");
            inputValues.push_back(inputValue);
            inputPointers.push_back(inputValue);
            inputNames.push_back(inputDescriptor.name.c_str());
            if (traceModelAsset) {
                NativeOnnxTraceInput actualInput;
                actualInput.name = inputDescriptor.name;
                actualInput.elementType = inputDescriptor.elementType;
                actualInput.dimensions = inputDimensions;
                actualInput.bytes = inputBuffers.back();
                actualInputTensors.push_back(std::move(actualInput));
            }
        }
        std::vector<const char *> outputNames;
        outputNames.reserve(outputDescriptors.size());
        for (const NativeOnnxValueDescriptor &outputDescriptor : outputDescriptors) {
            outputNames.push_back(outputDescriptor.name.c_str());
        }
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.run(session, nullptr, inputNames.data(), inputPointers.data(), inputPointers.size(), outputNames.data(), outputNames.size(), outputValues.data()), "ONNX Run");
        inspectStream << "run_status\tok\n";
        inspectStream << "run_input_count\t" << inputDescriptors.size() << "\n";
        inspectStream << "run_output_count\t" << outputDescriptors.size() << "\n";
        inspectStream << "run_input_elements\t" << totalInputElements << "\n";
        inspectStream << "run_output\tname\ttype\telements\tfirst\ttrace_match\n";
        for (size_t outputIndex = 0; outputIndex < outputValues.size(); outputIndex++) {
            OrtTensorTypeAndShapeInfo *outputShapeInfo = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorTypeAndShape(outputValues[outputIndex], &outputShapeInfo), "output tensor info 取得");
            size_t outputElementCount = 0;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorShapeElementCount(outputShapeInfo, &outputElementCount), "output element count 取得");
            std::vector<int64_t> outputDimensions = readNativeOnnxTensorDimensions(nativeOnnxApi, outputShapeInfo);
            void *outputData = nullptr;
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getTensorMutableData(outputValues[outputIndex], &outputData), "output tensor data 取得");
            size_t outputByteCount = outputElementCount * getNativeOnnxElementByteCount(outputDescriptors[outputIndex].elementType);
            std::string traceMatchText = compareNativeOnnxTraceOutput(outputDescriptors[outputIndex], outputDimensions, outputData, outputByteCount, traceOutputs);
            inspectStream << "output_" << outputIndex << "\t"
                          << outputDescriptors[outputIndex].name << "\t"
                          << formatNativeOnnxElementType(outputDescriptors[outputIndex].elementType) << "\t"
                          << outputElementCount << "\t"
                          << readNativeOnnxFirstValueText(outputDescriptors[outputIndex], outputData, outputElementCount) << "\t"
                          << traceMatchText << "\n";
            if (traceModelAsset) {
                NativeOnnxTraceInput actualOutput;
                actualOutput.name = outputDescriptors[outputIndex].name;
                actualOutput.elementType = outputDescriptors[outputIndex].elementType;
                actualOutput.dimensions = outputDimensions;
                actualOutput.bytes.resize(outputByteCount);
                if (outputByteCount > 0) {
                    std::memcpy(actualOutput.bytes.data(), outputData, outputByteCount);
                }
                actualOutputTensors.push_back(std::move(actualOutput));
            }
            nativeOnnxApi.releaseTensorTypeAndShapeInfo(outputShapeInfo);
        }
        if (traceModelAsset) {
            writeNativeOnnxTensorTrace(*traceModelAsset, actualInputTensors, actualOutputTensors);
        }
        for (OrtValue *outputValue : outputValues) {
            if (outputValue) {
                nativeOnnxApi.releaseValue(outputValue);
            }
        }
        for (OrtValue *inputValue : inputValues) {
            if (inputValue) {
                nativeOnnxApi.releaseValue(inputValue);
            }
        }
        nativeOnnxApi.releaseMemoryInfo(memoryInfo);
    } catch (...) {
        for (OrtValue *outputValue : outputValues) {
            if (outputValue) {
                nativeOnnxApi.releaseValue(outputValue);
            }
        }
        for (OrtValue *inputValue : inputValues) {
            if (inputValue) {
                nativeOnnxApi.releaseValue(inputValue);
            }
        }
        if (memoryInfo) {
            nativeOnnxApi.releaseMemoryInfo(memoryInfo);
        }
        throw;
    }
}

void appendNativeOnnxSessionInfoFromBytes(std::ostringstream &inspectStream, NativeOnnxApi &nativeOnnxApi, const std::vector<uint8_t> &modelBytes, const fs::path &inputDirectory, uint16_t cpuThreadCount, bool shouldUseVvBinConfig, const ModelAssetRecord *traceModelAsset) {
    std::vector<NativeOnnxTraceInput> traceInputs = loadNativeOnnxTraceInputs(inputDirectory);
    std::vector<NativeOnnxTraceInput> traceOutputs = loadNativeOnnxTraceOutputs(inputDirectory);
    OrtEnv *env = nullptr;
    OrtSessionOptions *sessionOptions = nullptr;
    OrtSession *session = nullptr;
    OrtAllocator *allocator = nullptr;
    try {
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createEnv(ortLoggingLevelWarning, "litevox-native", &env), "OrtEnv 作成");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.getAllocatorWithDefaultOptions(&allocator), "default allocator 取得");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionOptions(&sessionOptions), "SessionOptions 作成");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setSessionGraphOptimizationLevel(sessionOptions, ortGraphOptimizationLevelBasic), "graph optimization 設定");
        if (cpuThreadCount > 0) {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setIntraOpNumThreads(sessionOptions, cpuThreadCount), "intra op thread 設定");
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.setInterOpNumThreads(sessionOptions, cpuThreadCount), "inter op thread 設定");
        }
        if (shouldUseVvBinConfig) {
            ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.addSessionConfigEntry(sessionOptions, "session.use_vv_bin", "1"), "vv_bin session 設定");
        }
        applyNativeOnnxSeedIfConfigured(nativeOnnxApi);
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.createSessionFromArray(env, modelBytes.data(), modelBytes.size(), sessionOptions, &session), "ONNX session 作成");
        size_t inputCount = 0;
        size_t outputCount = 0;
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetInputCount(session, &inputCount), "input count 取得");
        ensureNativeOnnxCall(nativeOnnxApi, nativeOnnxApi.sessionGetOutputCount(session, &outputCount), "output count 取得");
        inspectStream << "model_bytes\t" << modelBytes.size() << "\n";
        inspectStream << "trace_input_count\t" << traceInputs.size() << "\n";
        inspectStream << "trace_output_count\t" << traceOutputs.size() << "\n";
        inspectStream << "session_status\tcreated\n";
        inspectStream << "input_count\t" << inputCount << "\n";
        inspectStream << "output_count\t" << outputCount << "\n";
        inspectStream << "value\tname\ttype\tshape\n";
        std::vector<NativeOnnxValueDescriptor> inputDescriptors;
        std::vector<NativeOnnxValueDescriptor> outputDescriptors;
        inputDescriptors.reserve(inputCount);
        outputDescriptors.reserve(outputCount);
        for (size_t inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            appendNativeOnnxValueInfo(inspectStream, nativeOnnxApi, session, allocator, "input", inputIndex);
            inputDescriptors.push_back(readNativeOnnxValueDescriptor(nativeOnnxApi, session, allocator, true, inputIndex));
        }
        for (size_t outputIndex = 0; outputIndex < outputCount; outputIndex++) {
            appendNativeOnnxValueInfo(inspectStream, nativeOnnxApi, session, allocator, "output", outputIndex);
            outputDescriptors.push_back(readNativeOnnxValueDescriptor(nativeOnnxApi, session, allocator, false, outputIndex));
        }
        appendNativeOnnxSmokeRunInfo(inspectStream, nativeOnnxApi, session, inputDescriptors, outputDescriptors, traceInputs, traceOutputs, traceModelAsset);
        nativeOnnxApi.releaseSession(session);
        nativeOnnxApi.releaseSessionOptions(sessionOptions);
        nativeOnnxApi.releaseEnv(env);
    } catch (...) {
        if (session) {
            nativeOnnxApi.releaseSession(session);
        }
        if (sessionOptions) {
            nativeOnnxApi.releaseSessionOptions(sessionOptions);
        }
        if (env) {
            nativeOnnxApi.releaseEnv(env);
        }
        throw;
    }
}

void appendNativeOnnxSessionInfo(std::ostringstream &inspectStream, NativeOnnxApi &nativeOnnxApi, const fs::path &modelPath, const fs::path &inputDirectory, uint16_t cpuThreadCount) {
    std::vector<uint8_t> modelBytes = readNativeOnnxBinaryFile(modelPath);
    inspectStream << "model_path\t" << modelPath.string() << "\n";
    ModelAssetRecord traceModelAsset = createNativeOnnxPseudoModelAsset(modelPath);
    appendNativeOnnxSessionInfoFromBytes(inspectStream, nativeOnnxApi, modelBytes, inputDirectory, cpuThreadCount, false, &traceModelAsset);
}

