#include "cli.hpp"

#include "archive_extract.hpp"
#include "cli_archive.hpp"
#include "cli_bench.hpp"
#include "cli_shared.hpp"
#include "http_server.hpp"
#include "json_utility.hpp"
#include "model_asset.hpp"
#include "model_metadata.hpp"
#include "model_session_cache.hpp"
#include "native_onnx.hpp"
#include "ort_inspect.hpp"
#include "runtime.hpp"
#include "runtime_trace.hpp"
#include "socket_compat.hpp"
#include "streaming_audio.hpp"
#include "utility.hpp"
#include "vvm_archive.hpp"
#include "vv_bin_inspect.hpp"
#include "api_schema.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

#include "cli_parse.hpp"

static void writeCliTextOutput(const fs::path &outputPath, const std::string &outputText) {
    if (outputPath.empty() || outputPath == "-") {
        std::cout << outputText << "\n";
        return;
    }
    writeTextFile(outputPath, outputText);
    std::cout << outputPath.string() << "\n";
}

static std::string createCliOpenApiJson(const CliOptions &cliOptions) {
    fs::path schemaPath = cliOptions.runtimePaths.engineManifestAssetDirectory.parent_path() / "openapi.json";
    if (fs::exists(schemaPath)) {
        return readTextFile(schemaPath);
    }
    return createOpenApiJson();
}

static int runCommand(const CliOptions &cliOptions) {
    if (cliOptions.commandMode == CommandMode::Help) {
        printUsage();
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::OpenApi) {
        std::cout << createCliOpenApiJson(cliOptions);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::ApiSession) {
        return runApiSessionCommand(cliOptions);
    }
    if ((cliOptions.commandMode == CommandMode::Inspect || cliOptions.commandMode == CommandMode::Inventory || cliOptions.commandMode == CommandMode::Assets || cliOptions.commandMode == CommandMode::ModelDump || cliOptions.commandMode == CommandMode::VvBin || cliOptions.commandMode == CommandMode::VvBinOrt || cliOptions.commandMode == CommandMode::VvBinModels || cliOptions.commandMode == CommandMode::VvBinOperators || cliOptions.commandMode == CommandMode::VvBinExportOnnx || cliOptions.commandMode == CommandMode::VvBinCompareOnnx || cliOptions.commandMode == CommandMode::VvBinChain || cliOptions.commandMode == CommandMode::VvBinSynth || cliOptions.commandMode == CommandMode::Extract || cliOptions.commandMode == CommandMode::Validate) && !cliOptions.hasExplicitModels) {
        throw std::runtime_error("MODEL.vvm または --models が必要です");
    }
    if (cliOptions.commandMode == CommandMode::OrtInspect) {
        std::cout << createOrtInspectText(cliOptions.runtimePaths.onnxruntimeLibraryPath);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::NativeOnnx) {
        std::cout << createNativeOnnxInspectText(cliOptions.runtimePaths.onnxruntimeLibraryPath, cliOptions.inspectPath, cliOptions.nativeOnnxInputDirectory, cliOptions.runtimePaths.cpuNumThreads);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::OnnxRandomSeed) {
        if (!cliOptions.hasOnnxSeed) {
            throw std::runtime_error("--seed が必要です");
        }
        std::cout << patchNativeOnnxRandomSeed(cliOptions.inspectPath, cliOptions.outputPath, static_cast<float>(cliOptions.onnxSeed));
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::VvBinOrt) {
        std::cout << createNativeOnnxVvmInspectText(cliOptions.runtimePaths.onnxruntimeLibraryPath, collectArchiveSummaries(cliOptions.runtimePaths.modelPaths), cliOptions.nativeOnnxInputDirectory, cliOptions.runtimePaths.cpuNumThreads);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::VvBinModels) {
        std::cout << createNativeOnnxVvmModelInfoText(cliOptions.runtimePaths.onnxruntimeLibraryPath, collectArchiveSummaries(cliOptions.runtimePaths.modelPaths), cliOptions.runtimePaths.cpuNumThreads);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::VvBinOperators) {
        std::cout << createNativeOnnxVvmOperatorText(cliOptions.runtimePaths.onnxruntimeLibraryPath, collectArchiveSummaries(cliOptions.runtimePaths.modelPaths), cliOptions.runtimePaths.cpuNumThreads);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::VvBinExportOnnx) {
        if (cliOptions.onnxOutputDirectory.empty()) {
            throw std::runtime_error("--extract-onnx DIR が必要です");
        }
        std::cout << exportNativeOnnxVvmOnnxFiles(cliOptions.runtimePaths.onnxruntimeLibraryPath, collectArchiveSummaries(cliOptions.runtimePaths.modelPaths), cliOptions.onnxOutputDirectory, cliOptions.runtimePaths.cpuNumThreads, cliOptions.hasOnnxSeed, static_cast<float>(cliOptions.onnxSeed));
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::VvBinCompareOnnx) {
        fs::path compareOnnxruntimeLibraryPath = cliOptions.compareOnnxruntimeLibraryPath.empty() ? cliOptions.runtimePaths.onnxruntimeLibraryPath : cliOptions.compareOnnxruntimeLibraryPath;
        std::cout << createNativeOnnxVvmCompareText(cliOptions.runtimePaths.onnxruntimeLibraryPath, compareOnnxruntimeLibraryPath, collectArchiveSummaries(cliOptions.runtimePaths.modelPaths), cliOptions.nativeOnnxInputDirectory, cliOptions.audioQueryPath, cliOptions.speaker, cliOptions.modelAssetName, cliOptions.runtimePaths.cpuNumThreads, cliOptions.runs);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::VvBinChain) {
        std::cout << createNativeOnnxVvmChainText(cliOptions.runtimePaths.onnxruntimeLibraryPath, collectArchiveSummaries(cliOptions.runtimePaths.modelPaths), cliOptions.nativeOnnxInputDirectory, cliOptions.audioQueryPath, cliOptions.speaker, cliOptions.runtimePaths.cpuNumThreads);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::VvBinSynth) {
        if (cliOptions.audioQueryPath.empty()) {
            throw std::runtime_error("--audio-query が必要です");
        }
        std::vector<uint8_t> wavBytes = synthesizeNativeOnnxVvmAudioQuery(cliOptions.runtimePaths.onnxruntimeLibraryPath, collectArchiveSummaries(cliOptions.runtimePaths.modelPaths), cliOptions.audioQueryPath, cliOptions.speaker, cliOptions.runtimePaths.cpuNumThreads);
        if (cliOptions.outputPath == "-") {
            std::cout.write(reinterpret_cast<const char *>(wavBytes.data()), static_cast<std::streamsize>(wavBytes.size()));
        } else {
            writeBinaryFile(cliOptions.outputPath, wavBytes);
            std::cout << cliOptions.outputPath.string() << "\n";
        }
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::Inspect) {
        std::cout << createArchiveInspectionText(collectArchiveSummaries(cliOptions.runtimePaths.modelPaths));
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::Inventory) {
        std::cout << createArchiveInventoryText(collectArchiveSummaries(cliOptions.runtimePaths.modelPaths));
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::Assets) {
        std::cout << createModelAssetTable(collectModelAssetsFromModelRoots(cliOptions.runtimePaths.modelPaths));
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::ModelDump) {
        if (cliOptions.shouldExtractResources) {
            std::cout << createResourceExtractionText(cliOptions);
            return 0;
        }
        if (cliOptions.shouldExtractEngineAssets) {
            std::cout << createEngineAssetExtractionText(cliOptions);
            return 0;
        }
        if (cliOptions.shouldExtractRuntime) {
            std::cout << createRuntimeExtractionText(cliOptions);
            return 0;
        }
        if (cliOptions.shouldExtractOnnxruntime) {
            std::cout << createOnnxruntimeExtractionText(cliOptions);
            return 0;
        }
        std::vector<VvmArchiveSummary> archiveSummaries = collectArchiveSummaries(cliOptions.runtimePaths.modelPaths);
        if (cliOptions.shouldExtractOnnx) {
            if (cliOptions.onnxOutputDirectory.empty()) {
                throw std::runtime_error("--extract-onnx DIR が必要です");
            }
            std::cout << exportOnnxModelFiles(archiveSummaries, cliOptions.onnxOutputDirectory);
            return 0;
        }
        std::cout << createModelDumpText(archiveSummaries, cliOptions.previewBytes, cliOptions.shouldScanModels);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::VvBin) {
        std::cout << createVvBinInspectText(collectArchiveSummaries(cliOptions.runtimePaths.modelPaths), cliOptions.previewBytes, cliOptions.shouldFullScan);
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::Cache) {
        return runCacheCommand(cliOptions);
    }
    if (cliOptions.commandMode == CommandMode::Extract) {
        return runExtractCommand(cliOptions);
    }
    if (cliOptions.commandMode == CommandMode::Validate) {
        std::cout << createArchiveValidationText(collectArchiveSummaries(cliOptions.runtimePaths.modelPaths));
        return 0;
    }
    if (cliOptions.commandMode == CommandMode::Bench) {
        return runBenchCommand(cliOptions);
    }
    if (cliOptions.commandMode == CommandMode::BenchSong) {
        return runSongBenchCommand(cliOptions);
    }
    if (cliOptions.commandMode == CommandMode::BenchHttp) {
        return runHttpBenchCommand(cliOptions);
    }
    if (cliOptions.commandMode == CommandMode::BenchHttpSong) {
        return runHttpSongBenchCommand(cliOptions);
    }
    if (cliOptions.commandMode == CommandMode::Server && cliOptions.workers > 1) {
        std::vector<RuntimeWorker> runtimeWorkers;
        std::map<uint32_t, size_t> sharedLoadedStyleCounts;
        std::mutex sharedLoadedStylesMutex;
        std::map<uint32_t, uint64_t> sharedStyleUnloadGenerations;
        std::mutex sharedStyleUnloadMutex;
        std::mutex sharedUserDictMutex;
        std::mutex sharedPresetMutex;
        std::mutex sharedSettingMutex;
        std::mutex sharedLibraryMutex;
        std::vector<std::unique_ptr<RuntimeHolder>> runtimeHolders = createRuntimeHolders(cliOptions, &sharedLoadedStyleCounts, &sharedLoadedStylesMutex, &sharedStyleUnloadGenerations, &sharedStyleUnloadMutex, &sharedUserDictMutex, &sharedPresetMutex, &sharedSettingMutex, &sharedLibraryMutex);
        runtimeWorkers.reserve(cliOptions.workers);
        for (size_t workerIndex = 0; workerIndex < cliOptions.workers; workerIndex++) {
            runtimeWorkers.push_back(RuntimeWorker{runtimeHolders[workerIndex]->runtimeState});
        }
        serveRuntimePool(runtimeWorkers, cliOptions.port);
        return 0;
    }
    RuntimeState *runtimeState = new RuntimeState(createRuntimeState(cliOptions.runtimePaths, cliOptions.shouldPreload));
    runtimeState->workerIndex = 0;
    runtimeState->workerCount = 1;
    try {
        if (cliOptions.commandMode == CommandMode::Server) {
            serve(*runtimeState, cliOptions.port);
        } else if (cliOptions.commandMode == CommandMode::Synthesis) {
            if (cliOptions.audioQueryPath.empty()) {
                throw std::runtime_error("--audio-query が必要です");
            }
            std::vector<uint8_t> wavBytes = synthesizeAudioQuery(*runtimeState, readTextFile(cliOptions.audioQueryPath), cliOptions.speaker);
            AudioStreamPayload audioStreamPayload = createAudioStreamPayload(wavBytes, cliOptions.audioStreamFormat);
            if (cliOptions.outputPath == "-") {
                std::cout.write(reinterpret_cast<const char *>(audioStreamPayload.audioBytes.data()), static_cast<std::streamsize>(audioStreamPayload.audioBytes.size()));
            } else {
                writeBinaryFile(cliOptions.outputPath, audioStreamPayload.audioBytes);
                std::cout << cliOptions.outputPath.string() << "\n";
            }
        } else if (cliOptions.commandMode == CommandMode::Tts || cliOptions.commandMode == CommandMode::Stream) {
            if (cliOptions.text.empty()) {
                throw std::runtime_error("--text が必要です");
            }
            if (cliOptions.outputPath == "-") {
                RuntimeAudioStreamOptions streamOptions;
                streamOptions.audioStreamFormat = cliOptions.audioStreamFormat;
                streamOptions.chunkSamples = cliOptions.chunkSamples;
                auto writeStdoutChunk = [](const uint8_t *audioBytes, size_t byteCount) {
                    std::cout.write(reinterpret_cast<const char *>(audioBytes), static_cast<std::streamsize>(byteCount));
                };
                if (cliOptions.isKana) {
                    streamKana(*runtimeState, cliOptions.text, cliOptions.speaker, streamOptions, writeStdoutChunk);
                } else {
                    streamText(*runtimeState, cliOptions.text, cliOptions.speaker, streamOptions, writeStdoutChunk);
                }
            } else {
                std::vector<uint8_t> wavBytes = cliOptions.isKana
                    ? synthesizeKana(*runtimeState, cliOptions.text, cliOptions.speaker)
                    : synthesizeText(*runtimeState, cliOptions.text, cliOptions.speaker);
                AudioStreamPayload audioStreamPayload = createAudioStreamPayload(wavBytes, cliOptions.audioStreamFormat);
                writeBinaryFile(cliOptions.outputPath, audioStreamPayload.audioBytes);
                std::cout << cliOptions.outputPath.string() << "\n";
            }
        } else if (cliOptions.commandMode == CommandMode::Query) {
            if (cliOptions.text.empty()) {
                throw std::runtime_error("--text が必要です");
            }
            if (cliOptions.isKana) {
                std::cout << createAudioQueryFromKana(*runtimeState, cliOptions.text, cliOptions.speaker) << "\n";
            } else {
                std::cout << createAudioQuery(*runtimeState, cliOptions.text, cliOptions.speaker) << "\n";
            }
        } else if (cliOptions.commandMode == CommandMode::SingQuery) {
            if (cliOptions.scorePath.empty()) {
                throw std::runtime_error("--score が必要です");
            }
            writeCliTextOutput(cliOptions.outputPath, createSingFrameAudioQuery(*runtimeState, readTextFile(cliOptions.scorePath), cliOptions.speaker));
        } else if (cliOptions.commandMode == CommandMode::SingF0) {
            if (cliOptions.scorePath.empty()) {
                throw std::runtime_error("--score が必要です");
            }
            if (cliOptions.frameAudioQueryPath.empty()) {
                throw std::runtime_error("--frame-audio-query が必要です");
            }
            writeCliTextOutput(cliOptions.outputPath, createSingFrameF0(*runtimeState, readTextFile(cliOptions.scorePath), readTextFile(cliOptions.frameAudioQueryPath), cliOptions.speaker));
        } else if (cliOptions.commandMode == CommandMode::SingVolume) {
            if (cliOptions.scorePath.empty()) {
                throw std::runtime_error("--score が必要です");
            }
            if (cliOptions.frameAudioQueryPath.empty()) {
                throw std::runtime_error("--frame-audio-query が必要です");
            }
            writeCliTextOutput(cliOptions.outputPath, createSingFrameVolume(*runtimeState, readTextFile(cliOptions.scorePath), readTextFile(cliOptions.frameAudioQueryPath), cliOptions.speaker));
        } else if (cliOptions.commandMode == CommandMode::FrameSynthesis || cliOptions.commandMode == CommandMode::Sing) {
            std::string Query;
            if (cliOptions.commandMode == CommandMode::Sing) {
                if (cliOptions.scorePath.empty()) {
                    throw std::runtime_error("--score が必要です");
                }
                Query = createSingFrameAudioQuery(*runtimeState, readTextFile(cliOptions.scorePath), cliOptions.Teacher);
            } else {
                fs::path frameAudioQueryPath = cliOptions.frameAudioQueryPath.empty() ? cliOptions.audioQueryPath : cliOptions.frameAudioQueryPath;
                if (frameAudioQueryPath.empty()) {
                    throw std::runtime_error("--frame-audio-query が必要です");
                }
                Query = readTextFile(frameAudioQueryPath);
            }
            std::vector<uint8_t> wavBytes = synthesizeFrameAudioQuery(*runtimeState, Query, cliOptions.speaker);
            AudioStreamPayload audioStreamPayload = createAudioStreamPayload(wavBytes, cliOptions.audioStreamFormat);
            if (cliOptions.outputPath == "-") {
                std::cout.write(reinterpret_cast<const char *>(audioStreamPayload.audioBytes.data()), static_cast<std::streamsize>(audioStreamPayload.audioBytes.size()));
            } else {
                writeBinaryFile(cliOptions.outputPath, audioStreamPayload.audioBytes);
                std::cout << cliOptions.outputPath.string() << "\n";
            }
        } else if (cliOptions.commandMode == CommandMode::Models) {
            std::cout << createModelTable(*runtimeState);
        } else if (cliOptions.commandMode == CommandMode::Inspect) {
            size_t styleCount = 0;
            for (const VoiceModelRecord &voiceModel : runtimeState->voiceModels) {
                styleCount += voiceModel.styles.size();
            }
            std::cout << "models\t" << runtimeState->voiceModels.size() << "\n";
            std::cout << "styles\t" << styleCount << "\n";
            std::cout << createModelTable(*runtimeState);
        } else if (cliOptions.commandMode == CommandMode::Singers) {
            std::cout << createSingersJson(runtimeState->combinedMetasJson, createCharacterSupportedFeaturesJsons(runtimeState->characterResources)) << "\n";
        } else if (cliOptions.commandMode == CommandMode::Speakers) {
            std::cout << createSpeakersJson(runtimeState->combinedMetasJson, getCoreBackendCapabilities(runtimeState->coreBackend).supportsMorphing, createCharacterSupportedFeaturesJsons(runtimeState->characterResources)) << "\n";
        } else if (cliOptions.commandMode == CommandMode::Deps) {
            std::cout << createRuntimeDependencyTable(*runtimeState);
        } else if (cliOptions.commandMode == CommandMode::Devices) {
            std::cout << createSupportedDevicesJson(*runtimeState) << "\n";
        } else if (cliOptions.commandMode == CommandMode::Version) {
            std::cout << getRuntimeCoreVersion(*runtimeState) << "\n";
        } else if (cliOptions.commandMode == CommandMode::RuntimeInfo) {
            std::cout << createRuntimeInfoJson(*runtimeState) << "\n";
        } else if (cliOptions.commandMode == CommandMode::Trace) {
            RuntimeTraceOptions traceOptions;
            traceOptions.outputDirectory = cliOptions.outputPath;
            traceOptions.text = cliOptions.text;
            traceOptions.styleId = cliOptions.speaker;
            traceOptions.isKana = cliOptions.isKana;
            if (traceOptions.outputDirectory.empty() || traceOptions.outputDirectory == "out.wav" || traceOptions.outputDirectory == "-") {
                throw std::runtime_error("trace には --out DIR が必要です");
            }
            std::cout << writeRuntimeTrace(*runtimeState, traceOptions).string() << "\n";
        }
    } catch (...) {
        throw;
    }
    return 0;
}

static void ResolveSpeakers(CliOptions &Options) {
    if (Options.SpeakerNames.empty() && Options.TeacherName.empty()) return;
    std::string Type = "talk";
    switch (Options.commandMode) {
        case CommandMode::Sing: case CommandMode::FrameSynthesis: Type = "frame_decode"; break;
        case CommandMode::SingQuery: case CommandMode::SingF0: case CommandMode::SingVolume:
        case CommandMode::BenchSong: case CommandMode::BenchHttpSong: Type = "sing"; break;
        case CommandMode::ApiSession: case CommandMode::BenchHttp:
            if (Options.httpPath.find("/frame_synthesis") == 0) Type = "frame_decode";
            else if (Options.httpPath.find("/sing_frame_") == 0) Type = "sing";
            break;
        default: break;
    }
    std::vector<StyleRecord> Styles;
    bool HasMetas = false;
    auto Resolve = [&](const std::string &Text, const std::string &StyleType) {
        if (!Text.empty() && (Text[0] == '-' || Text[0] == '+' || (Text[0] >= '0' && Text[0] <= '9'))) {
            return ResolveStyle(Text, {}, StyleType);
        }
        if (!HasMetas) {
            if (Options.commandMode == CommandMode::ApiSession || Options.commandMode == CommandMode::BenchHttp || Options.commandMode == CommandMode::BenchHttpSong) {
                Styles = extractStylesFromMetasJson(FetchSpeakers(Options, StyleType));
            } else {
                for (const auto &Archive : collectArchiveSummaries(Options.runtimePaths.modelPaths)) {
                    auto Entries = extractStylesFromMetasJson(Archive.metasJson);
                    Styles.insert(Styles.end(), Entries.begin(), Entries.end());
                }
            }
            HasMetas = true;
        }
        return ResolveStyle(Text, Styles, StyleType);
    };
    for (const auto &Name : Options.SpeakerNames) Options.benchSpeakers.push_back(Resolve(Name, Type));
    if (!Options.benchSpeakers.empty()) Options.speaker = Options.benchSpeakers.front();
    if (!Options.TeacherName.empty()) Options.Teacher = Resolve(Options.TeacherName, "sing");
}

int runCli(int argc, char **argv) {
    CliOptions cliOptions = parseCliOptions(argc, argv);
    if (!cliOptions.ExecutionProvider.empty()) {
        std::string Backend = lowercaseAscii(cliOptions.runtimePaths.backendMode);
        if (Backend != "native" && Backend != "vvm-native" && Backend != "vvm_native" && Backend != "minimal-ort" && Backend != "minimal_ort") {
            throw std::invalid_argument("--execution-provider は native / minimal-ort バックエンド専用です");
        }
        setEnvironmentVariable("LITEVOX_EXECUTION_PROVIDER", cliOptions.ExecutionProvider);
    }
    ResolveSpeakers(cliOptions);
    resolveAutoDiscoveredOnnxruntimePath(cliOptions);
    bool shouldSetOrtSeedEnvironment = false;
    bool shouldSetNativeSingSeedEnvironment = false;
    if (cliOptions.hasOnnxSeed) {
        if (cliOptions.commandMode == CommandMode::NativeOnnx
            || cliOptions.commandMode == CommandMode::VvBinOrt
            || cliOptions.commandMode == CommandMode::VvBinCompareOnnx) {
            shouldSetOrtSeedEnvironment = true;
        } else if (cliOptions.commandMode != CommandMode::OnnxRandomSeed
                   && cliOptions.commandMode != CommandMode::VvBinExportOnnx) {
            shouldSetNativeSingSeedEnvironment = true;
        }
    }
    if (shouldSetOrtSeedEnvironment) {
        std::string seedText = std::to_string(cliOptions.onnxSeed);
        setEnvironmentVariable("LITEVOX_ORT_SEED", seedText);
    }
    if (shouldSetNativeSingSeedEnvironment) {
        std::string seedText = std::to_string(cliOptions.onnxSeed);
        setEnvironmentVariable("LITEVOX_NATIVE_SING_SEED", seedText);
    }
    if (!cliOptions.nativeSingTeacherMode.empty()) {
        setEnvironmentVariable("LITEVOX_NATIVE_SING_TEACHER_MODE", cliOptions.nativeSingTeacherMode);
    }
    if (cliOptions.runtimePaths.backendMode == "minimal-ort" || cliOptions.runtimePaths.backendMode == "native") {
        const char *exportOnnxruntimeText = std::getenv("LITEVOX_VV_BIN_ONNXRUNTIME");
        if (!exportOnnxruntimeText || exportOnnxruntimeText[0] == '\0') {
            fs::path bundledOnnxruntimePath = getFirstNamedPath(cliOptions.runtimePaths.rootDirectory, getVoicevoxOnnxruntimeLibraryFileNames());
            bool shouldSetExportOnnxruntime = cliOptions.runtimePaths.backendMode == "minimal-ort";
            if (!shouldSetExportOnnxruntime && fs::exists(bundledOnnxruntimePath)) {
                shouldSetExportOnnxruntime = cliOptions.runtimePaths.onnxruntimeLibraryPath != bundledOnnxruntimePath
                    || cliOptions.runtimePaths.accelerationMode == "gpu"
                    || cliOptions.runtimePaths.nativeModelMode == "exported-onnx"
                    || cliOptions.runtimePaths.nativeModelMode == "exported_onnx"
                    || cliOptions.runtimePaths.nativeModelMode == "onnx";
            }
            if (shouldSetExportOnnxruntime && fs::exists(bundledOnnxruntimePath)) {
                std::string bundledOnnxruntimeText = bundledOnnxruntimePath.string();
                setEnvironmentVariable("LITEVOX_VV_BIN_ONNXRUNTIME", bundledOnnxruntimeText);
            }
        }
    }
    return runCommand(cliOptions);
}
