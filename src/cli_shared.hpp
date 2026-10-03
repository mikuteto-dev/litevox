#pragma once

#include "runtime.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

enum class CommandMode {
    Help,
    Server,
    Tts,
    Synthesis,
    Stream,
    Query,
    Sing,
    SingQuery,
    SingF0,
    SingVolume,
    FrameSynthesis,
    Models,
    Inspect,
    Inventory,
    Assets,
    ModelDump,
    VvBin,
    OrtInspect,
    NativeOnnx,
    OnnxRandomSeed,
    VvBinOrt,
    VvBinModels,
    VvBinOperators,
    VvBinExportOnnx,
    VvBinCompareOnnx,
    VvBinChain,
    VvBinSynth,
    Cache,
    Extract,
    Validate,
    Speakers,
    Singers,
    Deps,
    Devices,
    Version,
    RuntimeInfo,
    Trace,
    Bench,
    BenchSong,
    BenchHttp,
    BenchHttpSong,
    ApiSession,
    OpenApi
};

struct CliOptions {
    CommandMode commandMode = CommandMode::Server;
    RuntimePaths runtimePaths;
    bool hasExplicitModels = false;
    bool hasExplicitOnnxruntimePath = false;
    bool shouldPreload = false;
    bool shouldScanModels = false;
    bool shouldExtractOnnx = false;
    bool shouldExtractResources = false;
    bool shouldExtractEngineAssets = false;
    bool shouldExtractOnnxruntime = false;
    bool shouldExtractRuntime = false;
    bool shouldFullScan = false;
    bool isKana = false;
    bool httpKeepAlive = false;
    int port = 50021;
    size_t workers = 1;
    uint32_t speaker = 3;
    uint32_t Teacher = 6000;
    std::vector<std::string> SpeakerNames;
    std::string TeacherName;
    std::string ExecutionProvider;
    std::vector<uint32_t> benchSpeakers;
    std::vector<std::string> benchTexts;
    std::vector<std::filesystem::path> benchScorePaths;
    std::vector<std::filesystem::path> benchFrameAudioQueryPaths;
    size_t runs = 1;
    size_t chunkSamples = 1024;
    size_t previewBytes = 64;
    int64_t onnxSeed = 0;
    bool hasOnnxSeed = false;
    std::string nativeSingTeacherMode;
    std::string text;
    std::string httpHost = "127.0.0.1";
    std::string httpPath = "/tts";
    std::string modelAssetName;
    std::vector<std::string> benchHttpPaths;
    AudioStreamFormat audioStreamFormat = AudioStreamFormat::Wav;
    std::filesystem::path inspectPath;
    std::filesystem::path outputPath = "out.wav";
    std::filesystem::path onnxOutputDirectory;
    std::filesystem::path compareOnnxruntimeLibraryPath;
    std::filesystem::path autoDiscoveredOnnxruntimeLibraryPath;
    std::filesystem::path resourceOutputDirectory;
    std::filesystem::path engineAssetOutputDirectory;
    std::filesystem::path onnxruntimeOutputDirectory;
    std::filesystem::path runtimeOutputDirectory;
    std::filesystem::path nativeOnnxInputDirectory;
    std::filesystem::path audioQueryPath;
    std::filesystem::path scorePath;
    std::filesystem::path frameAudioQueryPath;
};

struct RuntimeHolder {
    RuntimeState *runtimeState = nullptr;
    explicit RuntimeHolder(RuntimeState *createdRuntimeState) : runtimeState(createdRuntimeState) {}
    RuntimeHolder(const RuntimeHolder &) = delete;
    RuntimeHolder &operator=(const RuntimeHolder &) = delete;
    ~RuntimeHolder() = default;
};
