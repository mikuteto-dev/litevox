static void printUsage() {
    std::cout
        << "LiteVox\n"
        << "\n"
        << "usage:\n"
        << "  litevox help\n"
        << "  litevox models [--runtime DIR] [--models PATH]\n"
        << "  litevox styles [--runtime DIR] [--models PATH]\n"
        << "  litevox inspect MODEL.vvm\n"
        << "  litevox inventory MODEL.vvm\n"
        << "  litevox assets MODEL.vvm\n"
        << "  litevox model-dump MODEL.vvm|VOICEVOX.zip\n"
        << "  litevox model-dump MODEL.vvm|VOICEVOX.zip --scan-models\n"
        << "  litevox model-dump MODEL.vvm|VOICEVOX.zip --extract-onnx DIR\n"
        << "  litevox model-dump VOICEVOX.zip --extract-resources DIR\n"
        << "  litevox model-dump VOICEVOX.zip --extract-engine-assets DIR\n"
        << "  litevox model-dump VOICEVOX.zip --extract-runtime DIR\n"
        << "  litevox model-dump ONNXRUNTIME.tgz|onnxruntime.zip --extract-onnxruntime DIR\n"
        << "  litevox vv-bin MODEL.vvm|VOICEVOX.zip\n"
        << "  litevox vv-bin-ort MODEL.vvm|VOICEVOX.zip [--onnxruntime PATH] [--inputs TRACE_TENSORS_DIR]\n"
        << "  litevox vv-bin-models MODEL.vvm|VOICEVOX.zip [--onnxruntime PATH] [--cpu-threads N]\n"
        << "  litevox vv-bin-operators MODEL.vvm|VOICEVOX.zip [--onnxruntime PATH] [--cpu-threads N]\n"
        << "  litevox vv-bin-export-onnx MODEL.vvm|VOICEVOX.zip --extract-onnx DIR [--seed 1] [--onnxruntime PATH] [--cpu-threads N]\n"
        << "  litevox vv-bin-compare-onnx MODEL.vvm|VOICEVOX.zip [--inputs TRACE_TENSORS_DIR] [--asset models/pd.bin] [--runs 5] [--onnxruntime PATH] [--compare-onnxruntime PATH] [--cpu-threads N]\n"
        << "  litevox vv-bin-chain MODEL.vvm|VOICEVOX.zip --audio-query audio_query.json [--inputs TRACE_TENSORS_DIR] [--speaker STYLE_ID] [--onnxruntime PATH]\n"
        << "  litevox vv-bin-synth MODEL.vvm|VOICEVOX.zip --audio-query audio_query.json --out out.wav [--speaker STYLE_ID]\n"
        << "  litevox ort-inspect [--onnxruntime PATH]\n"
        << "  litevox native-onnx [MODEL.onnx] [--onnxruntime PATH] [--inputs TRACE_TENSORS_DIR] [--cpu-threads N]\n"
        << "  litevox onnx-random-seed MODEL.onnx --seed 1 --out patched.onnx\n"
        << "  litevox cache MODEL.vvm\n"
        << "  litevox extract MODEL.vvm --out DIR\n"
        << "  litevox validate MODEL.vvm\n"
        << "  litevox tts --speaker STYLE_ID --text TEXT --out out.wav\n"
        << "  litevox synthesis --speaker STYLE_ID --audio-query audio_query.json --out out.wav\n"
        << "  litevox stream --speaker STYLE_ID --text TEXT [--format wav|pcm] [--chunk-samples 1024] > out.wav\n"
        << "  litevox query --speaker STYLE_ID --text TEXT\n"
        << "  litevox sing --score score.json [--teacher 6000] [--speaker 3000] --out out.wav\n"
        << "  litevox sing-query --score score.json [--speaker 6000] [--out frame_audio_query.json]\n"
        << "  litevox sing-f0 --score score.json --frame-audio-query frame_audio_query.json [--speaker 6000]\n"
        << "  litevox sing-volume --score score.json --frame-audio-query frame_audio_query.json [--speaker 6000]\n"
        << "  litevox frame-synthesis --frame-audio-query frame_audio_query.json [--speaker 3000] --out out.wav\n"
        << "  litevox server [--port 50021] [--workers 4] [--acceleration-mode auto|cpu|gpu] [--cpu-threads N] [--enable_cancellable_synthesis] [--preload]\n"
        << "  litevox speakers\n"
        << "  litevox singers\n"
        << "  litevox deps\n"
        << "  litevox devices\n"
        << "  litevox version\n"
        << "  litevox runtime_info\n"
        << "  litevox trace --speaker STYLE_ID --text TEXT --out DIR\n"
        << "  litevox openapi\n"
        << "  litevox bench --speaker STYLE_ID --text TEXT [--runs 3] [--workers 4] [--cpu-threads N]\n"
        << "  litevox bench --speakers 3,23,54,58 --text TEXT [--runs 8] [--workers 4]\n"
        << "  litevox bench-song --score score.json [--add-score score2.json] [--frame-audio-query frame_audio_query.json] [--add-frame-audio-query frame_audio_query2.json] [--speaker 6000] [--runs 5] [--workers 4] [--native-sing-teacher deterministic|vv-bin]\n"
        << "  litevox bench-http --speaker STYLE_ID --text TEXT [--port 50021] [--runs 8] [--workers 4]\n"
        << "  litevox bench-http --speakers 3,23,54,58 --text TEXT [--port 50021] [--runs 8] [--workers 4]\n"
        << "  litevox bench-http --http-path /tts --add-http-path /tts_stream --text 短文 --add-text 長文 [--runs 8] [--workers 4]\n"
        << "  litevox bench-http-song --score score.json [--add-score score2.json] [--frame-audio-query frame_audio_query.json] [--add-frame-audio-query frame_audio_query2.json] [--speaker 6000] [--port 50021] [--runs 5] [--workers 4] [--keep-alive]\n"
        << "  litevox api-session --speaker STYLE_ID [--host 127.0.0.1] [--port 50021] [--http-path /tts] [--format wav|pcm] [--out DIR] < texts.txt\n"
        << "  litevox api-session --speaker 6000 [--host 127.0.0.1] [--port 50021] --http-path /sing_frame_audio_query --score score.json [--out DIR]\n"
        << "  litevox api-session --speaker 6000 [--host 127.0.0.1] [--port 50021] --http-path /sing_frame_f0 --score score.json --frame-audio-query frame_audio_query.json [--out DIR]\n"
        << "  litevox api-session --speaker 6000 [--host 127.0.0.1] [--port 50021] --http-path /sing_frame_f0 --score score.json --add-score score2.json --frame-audio-query frame_audio_query.json --add-frame-audio-query frame_audio_query2.json [--out DIR]\n"
        << "  litevox api-session --speaker 6000 [--host 127.0.0.1] [--port 50021] --http-path /sing_frame_f0 --score score.json [--out DIR]\n"
        << "  litevox api-session --speaker 6000 [--host 127.0.0.1] [--port 50021] --http-path /frame_synthesis --frame-audio-query frame_audio_query.json [--out DIR]\n"
        << "  litevox api-session --speaker 6000 [--host 127.0.0.1] [--port 50021] --http-path /frame_synthesis --frame-audio-query frame_audio_query.json --add-frame-audio-query frame_audio_query2.json [--out DIR]\n"
        << "  litevox api-session --speaker 6000 [--host 127.0.0.1] [--port 50021] --http-path /frame_synthesis --score score.json [--out DIR]\n"
        << "  printf '@score.json\\t@frame_audio_query.json\\n' | litevox api-session --speaker 6000 --http-path /sing_frame_f0 [--host 127.0.0.1] [--port 50021] [--out DIR]\n"
        << "\n"
        << "common options:\n"
        << "  --runtime DIR|VOICEVOX.zip  runtime root or VOICEVOX product zip\n"
        << "  --core PATH         libvoicevox_core compatible library path\n"
        << "  --onnxruntime PATH  ONNX Runtime library path; omitted の場合は runtime 配下の libonnxruntime*.dylib または onnxruntime.dll を自動検出\n"
        << "  --compare-onnxruntime PATH  alternate ONNX Runtime library for exported ONNX compare\n"
        << "  --dict DIR          OpenJTalk dictionary directory\n"
        << "  --models PATH       VVM file, VOICEVOX zip, or directory; repeatable; replaces default model-vvm\n"
        << "  --add-model PATH    VVM file, VOICEVOX zip, or directory; repeatable; keeps default model-vvm\n"
        << "  --manifest PATH     engine manifest json path\n"
        << "  --state-dir DIR     writable state directory for user_dict.json, presets.json, setting.json, core_libraries\n"
        << "  --user-dict PATH    user dictionary json path\n"
        << "  --setting PATH      setting json path\n"
        << "  --presets PATH      presets json path\n"
        << "  --library-dir DIR   installed voice library directory\n"
        << "  --backend MODE      native, minimal-ort, or voicevox-core; default native\n"
        << "  --core-profile NAME auto, talk-only, or full\n"
        << "  --acceleration-mode auto, cpu, or gpu\n"
        << "  --execution-provider NAME  select ORT provider explicitly (e.g. WebGpuExecutionProvider)\n"
        << "  --gpu               shorthand for --acceleration-mode gpu\n"
        << "  --cpu-threads N     CPU threads for synthesis; 0 uses core default\n"
        << "  --seed N            set ONNX Runtime random seed when supported\n"
        << "  --native-model-mode MODE  auto, vv-bin, or exported-onnx\n"
        << "  --native-sing-teacher MODE  deterministic or vv-bin\n"
        << "  --host HOST         HTTP bench host\n"
        << "  --http-path PATH    HTTP bench path, default /tts\n"
        << "  --add-http-path PATH  add another HTTP bench path\n"
        << "  --keep-alive        reuse one HTTP connection in bench-http\n"
        << "  --speaker ID|NAME   style ID, speaker name or alias (zunda); NAME:STYLE selects a style\n"
        << "  --teacher ID|NAME   query style for sing; default 6000\n"
        << "  --speakers VALUES   comma-separated IDs/names for bench/bench-http\n"
        << "  --text TEXT         Japanese text\n"
        << "  --add-text TEXT     add another text for bench/bench-http\n"
        << "  --kana              treat --text as AquesTalk-style kana\n"
        << "  --format wav|pcm    stream output format\n"
        << "  --chunk-samples N   stream chunk size in PCM frames\n"
        << "  --preview-bytes N   model-dump/vv-bin asset preview byte count\n"
        << "  --full-scan         scan vv-bin assets fully for entropy and ONNX markers\n"
        << "  --scan-models       scan full model blobs for ONNX markers\n"
        << "  --extract-onnx DIR  export manifest models whose type is onnx\n"
        << "  --extract-resources DIR  export VOICEVOX character_info resources\n"
        << "  --extract-engine-assets DIR  export VOICEVOX engine manifest assets\n"
        << "  --extract-runtime DIR  assemble runnable runtime root from VOICEVOX zip, optional standard ONNX Runtime archive, and current litevox dist\n"
        << "  --extract-onnxruntime DIR  export standard ONNX Runtime archive into runtime root\n"
        << "  --inputs DIR        native-onnx/vv-bin-ort input or vv-bin-chain compare tensor directory\n"
        << "  --asset NAME        single model asset name such as models/pd.bin\n"
        << "  --audio-query PATH  audio_query json for synthesis/vv-bin-chain\n"
        << "  --score PATH        score json for song commands\n"
        << "  --add-score PATH    add another score json for song bench commands\n"
        << "  --frame-audio-query PATH  frame audio query json for song commands\n"
        << "  --add-frame-audio-query PATH  add another frame audio query json for song bench commands\n"
        << "  --workers N         server worker count or bench parallel runtime count\n"
        << "  --enable_cancellable_synthesis  enable /cancellable_synthesis endpoint\n"
        << "  --preload           preload models at runtime startup\n"
        << "  --out PATH          output wav path\n";
}

static void replaceModelPaths(CliOptions &cliOptions, const fs::path &modelPath) {
    if (!cliOptions.hasExplicitModels) {
        cliOptions.runtimePaths.modelPaths.clear();
        cliOptions.hasExplicitModels = true;
    }
    cliOptions.runtimePaths.modelPaths.push_back(modelPath);
}

static void applyStateDirectory(CliOptions &cliOptions, const fs::path &stateDirectory) {
    cliOptions.runtimePaths.userDictPath = stateDirectory / "user_dict.json";
    cliOptions.runtimePaths.presetPath = stateDirectory / "presets.json";
    cliOptions.runtimePaths.settingPath = stateDirectory / "setting.json";
    cliOptions.runtimePaths.libraryDirectory = stateDirectory / "core_libraries";
}

static std::vector<std::string> parseSpeakerList(const std::string &speakerListText) {
    std::vector<std::string> speakerIds;
    std::stringstream speakerStream(speakerListText);
    std::string speakerToken;
    while (std::getline(speakerStream, speakerToken, ',')) {
        if (speakerToken.empty()) {
            throw std::runtime_error("--speakers に空の要素があります");
        }
        speakerIds.push_back(speakerToken);
    }
    if (!speakerListText.empty() && speakerListText.back() == ',') throw std::invalid_argument("--speakers に空の要素があります");
    if (speakerIds.empty()) {
        throw std::runtime_error("--speakers が空です");
    }
    return speakerIds;
}

static void setPrimaryBenchText(CliOptions &cliOptions, const std::string &textValue) {
    cliOptions.text = textValue;
    if (cliOptions.commandMode == CommandMode::Bench || cliOptions.commandMode == CommandMode::BenchHttp) {
        if (cliOptions.benchTexts.empty()) {
            cliOptions.benchTexts.push_back(textValue);
        } else {
            cliOptions.benchTexts.front() = textValue;
        }
    }
}

static void addBenchText(CliOptions &cliOptions, const std::string &textValue) {
    if (cliOptions.commandMode != CommandMode::Bench && cliOptions.commandMode != CommandMode::BenchHttp) {
        cliOptions.text = textValue;
        return;
    }
    if (cliOptions.benchTexts.empty()) {
        cliOptions.benchTexts.push_back(textValue);
        if (cliOptions.text.empty()) {
            cliOptions.text = textValue;
        }
        return;
    }
    cliOptions.benchTexts.push_back(textValue);
}

static void setPrimaryBenchHttpPath(CliOptions &cliOptions, const std::string &httpPath) {
    cliOptions.httpPath = httpPath;
    if (cliOptions.commandMode == CommandMode::BenchHttp) {
        if (cliOptions.benchHttpPaths.empty()) {
            cliOptions.benchHttpPaths.push_back(httpPath);
        } else {
            cliOptions.benchHttpPaths.front() = httpPath;
        }
    }
}

static void addBenchHttpPath(CliOptions &cliOptions, const std::string &httpPath) {
    if (cliOptions.commandMode != CommandMode::BenchHttp) {
        cliOptions.httpPath = httpPath;
        return;
    }
    if (cliOptions.benchHttpPaths.empty()) {
        cliOptions.benchHttpPaths.push_back(httpPath);
        if (cliOptions.httpPath.empty()) {
            cliOptions.httpPath = httpPath;
        }
        return;
    }
    cliOptions.benchHttpPaths.push_back(httpPath);
}

static bool isStandardOnnxruntimeLibraryPath(const fs::path &libraryPath) {
    std::string filenameText = lowercaseAscii(libraryPath.filename().string());
    std::string extensionText = lowercaseAscii(libraryPath.extension().string());
    if (extensionText == ".dylib") {
        return filenameText.rfind("libonnxruntime", 0) == 0;
    }
    if (extensionText == ".dll") {
        return filenameText == "onnxruntime.dll";
    }
    return false;
}

static fs::path findStandardOnnxruntimeLibraryPath(const fs::path &rootDirectory) {
    if (rootDirectory.empty() || !fs::exists(rootDirectory)) {
        return {};
    }
    std::vector<fs::path> candidatePaths;
    std::error_code iterationError;
    fs::recursive_directory_iterator iterator(rootDirectory, fs::directory_options::skip_permission_denied, iterationError);
    fs::recursive_directory_iterator endIterator;
    for (; iterator != endIterator; iterator.increment(iterationError)) {
        if (iterationError) {
            iterationError.clear();
            continue;
        }
        const fs::directory_entry &directoryEntry = *iterator;
        std::error_code statusError;
        if (!directoryEntry.is_regular_file(statusError) || statusError) {
            continue;
        }
        if (isStandardOnnxruntimeLibraryPath(directoryEntry.path())) {
            candidatePaths.push_back(directoryEntry.path());
        }
    }
    if (candidatePaths.empty()) {
        return {};
    }
    std::sort(candidatePaths.begin(), candidatePaths.end(), [](const fs::path &leftPath, const fs::path &rightPath) {
        size_t leftLength = leftPath.string().size();
        size_t rightLength = rightPath.string().size();
        if (leftLength != rightLength) {
            return leftLength < rightLength;
        }
        return leftPath.string() < rightPath.string();
    });
    return candidatePaths.front();
}

static bool hasUsableGpuProviderAtOnnxruntimePath(const fs::path &onnxruntimeLibraryPath) {
    if (onnxruntimeLibraryPath.empty()) {
        return false;
    }
    try {
        NativeOnnxRuntimeState runtimeState = createNativeOnnxRuntimeState(onnxruntimeLibraryPath, "auto");
        bool hasUsableGpuProvider = runtimeState.hasUsableGpuProvider;
        destroyNativeOnnxRuntimeState(runtimeState);
        return hasUsableGpuProvider;
    } catch (...) {
        return false;
    }
}

static bool shouldAutoDiscoverStandardOnnxruntime(const CliOptions &cliOptions) {
    std::string backendMode = lowercaseAscii(cliOptions.runtimePaths.backendMode);
    if (backendMode == "minimal-ort" || backendMode == "minimal_ort") {
        return true;
    }
    if (backendMode == "native" || backendMode == "vvm-native" || backendMode == "vvm_native") {
        std::string accelerationMode = lowercaseAscii(cliOptions.runtimePaths.accelerationMode);
        return accelerationMode == "gpu" || accelerationMode == "auto";
    }
    return false;
}

static void resolveAutoDiscoveredOnnxruntimePath(CliOptions &cliOptions) {
    if (cliOptions.hasExplicitOnnxruntimePath) {
        return;
    }
    if (!shouldAutoDiscoverStandardOnnxruntime(cliOptions)) {
        return;
    }
    fs::path discoveredOnnxruntimePath = findStandardOnnxruntimeLibraryPath(cliOptions.runtimePaths.rootDirectory);
    if (discoveredOnnxruntimePath.empty()) {
        return;
    }
    std::string backendMode = lowercaseAscii(cliOptions.runtimePaths.backendMode);
    std::string accelerationMode = lowercaseAscii(cliOptions.runtimePaths.accelerationMode);
    if ((backendMode == "native" || backendMode == "vvm-native" || backendMode == "vvm_native") && accelerationMode == "auto" && !hasUsableGpuProviderAtOnnxruntimePath(discoveredOnnxruntimePath)) {
        return;
    }
    cliOptions.runtimePaths.onnxruntimeLibraryPath = discoveredOnnxruntimePath;
    cliOptions.autoDiscoveredOnnxruntimeLibraryPath = discoveredOnnxruntimePath;
}

static CliOptions parseCliOptions(int argc, char **argv) {
    CliOptions cliOptions;
    cliOptions.runtimePaths = makeDefaultRuntimePaths(getExecutableDirectory());
    int argumentIndex = 1;
    if (argumentIndex < argc) {
        std::string commandText = argv[argumentIndex];
        if (commandText == "help") {
            cliOptions.commandMode = CommandMode::Help;
            argumentIndex++;
        } else if (commandText == "server") {
            cliOptions.commandMode = CommandMode::Server;
            argumentIndex++;
        } else if (commandText == "tts") {
            cliOptions.commandMode = CommandMode::Tts;
            argumentIndex++;
        } else if (commandText == "synthesis" || commandText == "synth") {
            cliOptions.commandMode = CommandMode::Synthesis;
            argumentIndex++;
        } else if (commandText == "stream") {
            cliOptions.commandMode = CommandMode::Stream;
            cliOptions.outputPath = "-";
            argumentIndex++;
        } else if (commandText == "query") {
            cliOptions.commandMode = CommandMode::Query;
            argumentIndex++;
        } else if (commandText == "sing") {
            cliOptions.commandMode = CommandMode::Sing;
            cliOptions.speaker = 3000;
            argumentIndex++;
        } else if (commandText == "sing-query") {
            cliOptions.commandMode = CommandMode::SingQuery;
            cliOptions.outputPath = "-";
            cliOptions.speaker = 6000;
            argumentIndex++;
        } else if (commandText == "sing-f0") {
            cliOptions.commandMode = CommandMode::SingF0;
            cliOptions.outputPath = "-";
            cliOptions.speaker = 6000;
            argumentIndex++;
        } else if (commandText == "sing-volume") {
            cliOptions.commandMode = CommandMode::SingVolume;
            cliOptions.outputPath = "-";
            cliOptions.speaker = 6000;
            argumentIndex++;
        } else if (commandText == "frame-synthesis") {
            cliOptions.commandMode = CommandMode::FrameSynthesis;
            cliOptions.speaker = 3000;
            argumentIndex++;
        } else if (commandText == "models" || commandText == "styles") {
            cliOptions.commandMode = CommandMode::Models;
            argumentIndex++;
        } else if (commandText == "inspect") {
            cliOptions.commandMode = CommandMode::Inspect;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string inspectArgument = argv[argumentIndex];
                if (!inspectArgument.empty() && inspectArgument[0] != '-') {
                    cliOptions.inspectPath = inspectArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "inventory") {
            cliOptions.commandMode = CommandMode::Inventory;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string inventoryArgument = argv[argumentIndex];
                if (!inventoryArgument.empty() && inventoryArgument[0] != '-') {
                    cliOptions.inspectPath = inventoryArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "assets") {
            cliOptions.commandMode = CommandMode::Assets;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string assetsArgument = argv[argumentIndex];
                if (!assetsArgument.empty() && assetsArgument[0] != '-') {
                    cliOptions.inspectPath = assetsArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "model-dump") {
            cliOptions.commandMode = CommandMode::ModelDump;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string modelDumpArgument = argv[argumentIndex];
                if (!modelDumpArgument.empty() && modelDumpArgument[0] != '-') {
                    cliOptions.inspectPath = modelDumpArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "vv-bin" || commandText == "vvbin") {
            cliOptions.commandMode = CommandMode::VvBin;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string vvBinArgument = argv[argumentIndex];
                if (!vvBinArgument.empty() && vvBinArgument[0] != '-') {
                    cliOptions.inspectPath = vvBinArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "ort-inspect" || commandText == "ort") {
            cliOptions.commandMode = CommandMode::OrtInspect;
            argumentIndex++;
        } else if (commandText == "vv-bin-ort" || commandText == "vvbin-ort") {
            cliOptions.commandMode = CommandMode::VvBinOrt;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string vvBinOrtArgument = argv[argumentIndex];
                if (!vvBinOrtArgument.empty() && vvBinOrtArgument[0] != '-') {
                    cliOptions.inspectPath = vvBinOrtArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "vv-bin-models" || commandText == "vvbin-models") {
            cliOptions.commandMode = CommandMode::VvBinModels;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string vvBinModelsArgument = argv[argumentIndex];
                if (!vvBinModelsArgument.empty() && vvBinModelsArgument[0] != '-') {
                    cliOptions.inspectPath = vvBinModelsArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "vv-bin-operators" || commandText == "vvbin-operators") {
            cliOptions.commandMode = CommandMode::VvBinOperators;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string vvBinOperatorsArgument = argv[argumentIndex];
                if (!vvBinOperatorsArgument.empty() && vvBinOperatorsArgument[0] != '-') {
                    cliOptions.inspectPath = vvBinOperatorsArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "vv-bin-export-onnx" || commandText == "vvbin-export-onnx") {
            cliOptions.commandMode = CommandMode::VvBinExportOnnx;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string vvBinExportOnnxArgument = argv[argumentIndex];
                if (!vvBinExportOnnxArgument.empty() && vvBinExportOnnxArgument[0] != '-') {
                    cliOptions.inspectPath = vvBinExportOnnxArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "vv-bin-compare-onnx" || commandText == "vvbin-compare-onnx") {
            cliOptions.commandMode = CommandMode::VvBinCompareOnnx;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string vvBinCompareOnnxArgument = argv[argumentIndex];
                if (!vvBinCompareOnnxArgument.empty() && vvBinCompareOnnxArgument[0] != '-') {
                    cliOptions.inspectPath = vvBinCompareOnnxArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "vv-bin-chain" || commandText == "vvbin-chain" || commandText == "native-chain") {
            cliOptions.commandMode = CommandMode::VvBinChain;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string vvBinChainArgument = argv[argumentIndex];
                if (!vvBinChainArgument.empty() && vvBinChainArgument[0] != '-') {
                    cliOptions.inspectPath = vvBinChainArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "vv-bin-synth" || commandText == "vvbin-synth" || commandText == "native-synth") {
            cliOptions.commandMode = CommandMode::VvBinSynth;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string vvBinSynthArgument = argv[argumentIndex];
                if (!vvBinSynthArgument.empty() && vvBinSynthArgument[0] != '-') {
                    cliOptions.inspectPath = vvBinSynthArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "native-onnx" || commandText == "onnx-session") {
            cliOptions.commandMode = CommandMode::NativeOnnx;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string nativeOnnxArgument = argv[argumentIndex];
                if (!nativeOnnxArgument.empty() && nativeOnnxArgument[0] != '-') {
                    cliOptions.inspectPath = nativeOnnxArgument;
                    argumentIndex++;
                }
            }
        } else if (commandText == "onnx-random-seed") {
            cliOptions.commandMode = CommandMode::OnnxRandomSeed;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string onnxRandomSeedArgument = argv[argumentIndex];
                if (!onnxRandomSeedArgument.empty() && onnxRandomSeedArgument[0] != '-') {
                    cliOptions.inspectPath = onnxRandomSeedArgument;
                    argumentIndex++;
                }
            }
        } else if (commandText == "cache") {
            cliOptions.commandMode = CommandMode::Cache;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string cacheArgument = argv[argumentIndex];
                if (!cacheArgument.empty() && cacheArgument[0] != '-') {
                    cliOptions.inspectPath = cacheArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "extract") {
            cliOptions.commandMode = CommandMode::Extract;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string extractArgument = argv[argumentIndex];
                if (!extractArgument.empty() && extractArgument[0] != '-') {
                    cliOptions.inspectPath = extractArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "validate") {
            cliOptions.commandMode = CommandMode::Validate;
            argumentIndex++;
            if (argumentIndex < argc) {
                std::string validateArgument = argv[argumentIndex];
                if (!validateArgument.empty() && validateArgument[0] != '-') {
                    cliOptions.inspectPath = validateArgument;
                    replaceModelPaths(cliOptions, cliOptions.inspectPath);
                    argumentIndex++;
                }
            }
        } else if (commandText == "speakers" || commandText == "singers") {
            cliOptions.commandMode = commandText == "singers" ? CommandMode::Singers : CommandMode::Speakers;
            argumentIndex++;
        } else if (commandText == "deps" || commandText == "dependencies") {
            cliOptions.commandMode = CommandMode::Deps;
            argumentIndex++;
        } else if (commandText == "devices") {
            cliOptions.commandMode = CommandMode::Devices;
            argumentIndex++;
        } else if (commandText == "version") {
            cliOptions.commandMode = CommandMode::Version;
            argumentIndex++;
        } else if (commandText == "runtime_info") {
            cliOptions.commandMode = CommandMode::RuntimeInfo;
            argumentIndex++;
        } else if (commandText == "trace") {
            cliOptions.commandMode = CommandMode::Trace;
            argumentIndex++;
        } else if (commandText == "bench") {
            cliOptions.commandMode = CommandMode::Bench;
            argumentIndex++;
        } else if (commandText == "bench-song" || commandText == "song-bench") {
            cliOptions.commandMode = CommandMode::BenchSong;
            cliOptions.runs = 5;
            cliOptions.speaker = 6000;
            argumentIndex++;
        } else if (commandText == "bench-http" || commandText == "http-bench") {
            cliOptions.commandMode = CommandMode::BenchHttp;
            cliOptions.runs = 8;
            cliOptions.workers = 4;
            argumentIndex++;
        } else if (commandText == "bench-http-song" || commandText == "http-song-bench") {
            cliOptions.commandMode = CommandMode::BenchHttpSong;
            cliOptions.runs = 5;
            cliOptions.speaker = 6000;
            argumentIndex++;
        } else if (commandText == "api-session" || commandText == "http-session") {
            cliOptions.commandMode = CommandMode::ApiSession;
            argumentIndex++;
        } else if (commandText == "openapi") {
            cliOptions.commandMode = CommandMode::OpenApi;
            argumentIndex++;
        } else if (commandText == "--help" || commandText == "-h") {
            printUsage();
            std::exit(0);
        }
    }
    while (argumentIndex < argc) {
        std::string argumentText = argv[argumentIndex++];
        auto requireValue = [&](const std::string &optionName) -> std::string {
            if (argumentIndex >= argc) {
                throw std::runtime_error(optionName + " の値がありません");
            }
            return argv[argumentIndex++];
        };
        if (argumentText == "--runtime") {
            std::vector<fs::path> explicitModelPaths = cliOptions.runtimePaths.modelPaths;
            cliOptions.runtimePaths = makeDefaultRuntimePaths(resolveRuntimeRootPath(requireValue(argumentText)));
            cliOptions.hasExplicitOnnxruntimePath = false;
            cliOptions.autoDiscoveredOnnxruntimeLibraryPath.clear();
            if (cliOptions.hasExplicitModels) {
                cliOptions.runtimePaths.modelPaths = explicitModelPaths;
            }
        } else if (argumentText == "--core") {
            cliOptions.runtimePaths.coreLibraryPath = requireValue(argumentText);
        } else if (argumentText == "--onnxruntime") {
            cliOptions.runtimePaths.onnxruntimeLibraryPath = requireValue(argumentText);
            cliOptions.hasExplicitOnnxruntimePath = true;
            cliOptions.autoDiscoveredOnnxruntimeLibraryPath.clear();
        } else if (argumentText == "--compare-onnxruntime") {
            cliOptions.compareOnnxruntimeLibraryPath = requireValue(argumentText);
        } else if (argumentText == "--dict") {
            cliOptions.runtimePaths.dictionaryDirectory = requireValue(argumentText);
        } else if (argumentText == "--models" || argumentText == "--model") {
            replaceModelPaths(cliOptions, requireValue(argumentText));
        } else if (argumentText == "--add-model") {
            cliOptions.runtimePaths.modelPaths.push_back(requireValue(argumentText));
        } else if (argumentText == "--manifest") {
            cliOptions.runtimePaths.manifestPath = requireValue(argumentText);
            cliOptions.runtimePaths.hasManifestOverride = true;
        } else if (argumentText == "--state-dir") {
            applyStateDirectory(cliOptions, requireValue(argumentText));
        } else if (argumentText == "--user-dict") {
            cliOptions.runtimePaths.userDictPath = requireValue(argumentText);
        } else if (argumentText == "--setting") {
            cliOptions.runtimePaths.settingPath = requireValue(argumentText);
        } else if (argumentText == "--presets") {
            cliOptions.runtimePaths.presetPath = requireValue(argumentText);
        } else if (argumentText == "--library-dir") {
            cliOptions.runtimePaths.libraryDirectory = requireValue(argumentText);
        } else if (argumentText == "--backend") {
            cliOptions.runtimePaths.backendMode = requireValue(argumentText);
        } else if (argumentText == "--core-profile") {
            cliOptions.runtimePaths.coreProfile = requireValue(argumentText);
        } else if (argumentText == "--acceleration-mode") {
            cliOptions.runtimePaths.accelerationMode = requireValue(argumentText);
        } else if (argumentText == "--execution-provider") {
            cliOptions.ExecutionProvider = requireValue(argumentText);
            if (cliOptions.ExecutionProvider.empty()) throw std::invalid_argument("--execution-provider が空です");
        } else if (argumentText == "--gpu") {
            cliOptions.runtimePaths.accelerationMode = "gpu";
        } else if (argumentText == "--cpu-threads") {
            unsigned long cpuThreadCount = std::stoul(requireValue(argumentText));
            if (cpuThreadCount > std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("--cpu-threads が大きすぎます");
            }
            cliOptions.runtimePaths.cpuNumThreads = static_cast<uint16_t>(cpuThreadCount);
        } else if (argumentText == "--seed") {
            cliOptions.onnxSeed = std::stoll(requireValue(argumentText));
            cliOptions.hasOnnxSeed = true;
        } else if (argumentText == "--native-model-mode") {
            cliOptions.runtimePaths.nativeModelMode = requireValue(argumentText);
        } else if (argumentText == "--native-sing-teacher") {
            cliOptions.nativeSingTeacherMode = requireValue(argumentText);
        } else if (argumentText == "--host") {
            cliOptions.httpHost = requireValue(argumentText);
        } else if (argumentText == "--http-path") {
            setPrimaryBenchHttpPath(cliOptions, requireValue(argumentText));
        } else if (argumentText == "--add-http-path") {
            addBenchHttpPath(cliOptions, requireValue(argumentText));
        } else if (argumentText == "--keep-alive") {
            cliOptions.httpKeepAlive = true;
        } else if (argumentText == "--port") {
            cliOptions.port = std::stoi(requireValue(argumentText));
        } else if (argumentText == "--workers") {
            cliOptions.workers = static_cast<size_t>(std::stoul(requireValue(argumentText)));
            if (cliOptions.workers == 0) {
                throw std::runtime_error("--workers は 1 以上が必要です");
            }
        } else if (argumentText == "--enable_cancellable_synthesis" || argumentText == "--enable-cancellable-synthesis") {
            cliOptions.runtimePaths.enableCancellableSynthesis = true;
        } else if (argumentText == "--teacher") {
            if (cliOptions.commandMode != CommandMode::Sing) {
                throw std::runtime_error("--teacher は sing 専用です");
            }
            cliOptions.TeacherName = requireValue(argumentText);
            if (cliOptions.TeacherName.empty()) throw std::invalid_argument("--teacher が空です");
        } else if (argumentText == "--speaker") {
            cliOptions.SpeakerNames = {requireValue(argumentText)};
            if (cliOptions.SpeakerNames.front().empty()) throw std::invalid_argument("--speaker が空です");
        } else if (argumentText == "--speakers") {
            cliOptions.SpeakerNames = parseSpeakerList(requireValue(argumentText));
        } else if (argumentText == "--runs") {
            cliOptions.runs = static_cast<size_t>(std::stoul(requireValue(argumentText)));
            if (cliOptions.runs == 0) {
                throw std::runtime_error("--runs は 1 以上が必要です");
            }
        } else if (argumentText == "--text") {
            setPrimaryBenchText(cliOptions, requireValue(argumentText));
        } else if (argumentText == "--add-text") {
            addBenchText(cliOptions, requireValue(argumentText));
        } else if (argumentText == "--kana") {
            cliOptions.isKana = true;
        } else if (argumentText == "--format") {
            cliOptions.audioStreamFormat = parseAudioStreamFormat(requireValue(argumentText));
        } else if (argumentText == "--chunk-samples") {
            cliOptions.chunkSamples = static_cast<size_t>(std::stoul(requireValue(argumentText)));
            if (cliOptions.chunkSamples == 0) {
                throw std::runtime_error("--chunk-samples は 1 以上が必要です");
            }
        } else if (argumentText == "--preview-bytes") {
            cliOptions.previewBytes = static_cast<size_t>(std::stoul(requireValue(argumentText)));
        } else if (argumentText == "--scan-models") {
            cliOptions.shouldScanModels = true;
        } else if (argumentText == "--full-scan") {
            cliOptions.shouldFullScan = true;
        } else if (argumentText == "--extract-onnx") {
            cliOptions.shouldExtractOnnx = true;
            cliOptions.onnxOutputDirectory = requireValue(argumentText);
        } else if (argumentText == "--extract-resources") {
            cliOptions.shouldExtractResources = true;
            cliOptions.resourceOutputDirectory = requireValue(argumentText);
        } else if (argumentText == "--extract-engine-assets") {
            cliOptions.shouldExtractEngineAssets = true;
            cliOptions.engineAssetOutputDirectory = requireValue(argumentText);
        } else if (argumentText == "--extract-runtime") {
            cliOptions.shouldExtractRuntime = true;
            cliOptions.runtimeOutputDirectory = requireValue(argumentText);
        } else if (argumentText == "--extract-onnxruntime") {
            cliOptions.shouldExtractOnnxruntime = true;
            cliOptions.onnxruntimeOutputDirectory = requireValue(argumentText);
        } else if (argumentText == "--inputs") {
            cliOptions.nativeOnnxInputDirectory = requireValue(argumentText);
        } else if (argumentText == "--asset") {
            cliOptions.modelAssetName = requireValue(argumentText);
        } else if (argumentText == "--audio-query") {
            cliOptions.audioQueryPath = requireValue(argumentText);
        } else if (argumentText == "--score") {
            cliOptions.scorePath = requireValue(argumentText);
        } else if (argumentText == "--add-score") {
            cliOptions.benchScorePaths.push_back(requireValue(argumentText));
        } else if (argumentText == "--frame-audio-query") {
            cliOptions.frameAudioQueryPath = requireValue(argumentText);
        } else if (argumentText == "--add-frame-audio-query") {
            cliOptions.benchFrameAudioQueryPaths.push_back(requireValue(argumentText));
        } else if (argumentText == "--out") {
            cliOptions.outputPath = requireValue(argumentText);
        } else if (argumentText == "--preload") {
            cliOptions.shouldPreload = true;
        } else if (argumentText == "--help" || argumentText == "-h") {
            printUsage();
            std::exit(0);
        } else {
            throw std::runtime_error("不明な引数です: " + argumentText);
        }
    }
    return cliOptions;
}

