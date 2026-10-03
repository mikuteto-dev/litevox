#include "json_utility.hpp"
#include "model_metadata.hpp"
#include "native_onnx_internal.hpp"

#include <cassert>
#include <atomic>
#include <thread>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <set>

static void Reject(const std::function<void()> &Run) {
    bool Failed = false;
    try { Run(); } catch (const std::invalid_argument &) { Failed = true; }
    assert(Failed);
}

int main(int Argc, char **Argv) {
    if (Argc > 1 && Argv[1][0] != '\0') {
        std::vector<StyleRecord> All;
        for (const auto &Path : collectVvmModelFiles({Argv[1]})) {
            auto Entries = extractStylesFromMetasJson(inspectVvmArchive(Path).metasJson);
            All.insert(All.end(), Entries.begin(), Entries.end());
        }
        assert(!All.empty());
        std::set<std::string> Speakers, Aliased;
        std::map<std::string, size_t> Counts;
        for (const auto &Entry : All) {
            Speakers.insert(Entry.speakerName);
            ++Counts[Entry.styleType];
            assert(ResolveStyle(Entry.speakerName + ":" + Entry.styleName, All, Entry.styleType) == Entry.styleId);
            assert(ResolveStyle(std::to_string(Entry.styleId), All, Entry.styleType) == Entry.styleId);
        }
        for (const auto &Alias : GetAliases()) {
            bool HasSpeaker = false;
            for (const auto &Entry : All) {
                if (Entry.speakerName != Alias.second) continue;
                HasSpeaker = true;
                Aliased.insert(Entry.speakerName);
                assert(ResolveStyle(Alias.first + ":" + Entry.styleName, All, Entry.styleType) == Entry.styleId);
                std::string Type = Entry.styleType == "sing" ? "frame_decode" : Entry.styleType;
                assert(ResolveStyle(Alias.first + ":" + std::to_string(Entry.styleId), All, Type) == Entry.styleId);
            }
            assert(HasSpeaker);
        }
        assert(Aliased == Speakers);
        std::cout << Speakers.size() << " standard speakers / " << All.size() << " styles / all aliases checked";
        for (const auto &Count : Counts) std::cout << " / " << Count.first << ": " << Count.second;
        std::cout << "\n";
    }
    const std::vector<StyleRecord> Styles{
        {1, "z", "ずんだもん", "あまあま", "talk"}, {3, "z", "ずんだもん", "ノーマル", "talk"},
        {3003, "z", "ずんだもん", "ノーマル", "frame_decode"},
        {6000, "r", "波音リツ", "ノーマル", "sing"}, {12, "k", "白上虎太郎", "ふつう", "talk"}
    };
    for (const char *Name : {"3", "zunda", "ZUNDAMON", "ずんだもん", "zunda:normal"}) assert(ResolveStyle(Name, Styles) == 3);
    assert(ResolveStyle("zunda:あまあま", Styles) == 1);
    assert(ResolveStyle("zunda", Styles, "frame_decode") == 3003);
    assert(ResolveStyle("ritsu", Styles, "sing") == 6000);
    assert(ResolveStyle("ritsu:6000", Styles, "frame_decode") == 6000);
    assert(ResolveStyle("zunda:7", {{7, "z", "ずんだもん", "ツンツン", "talk"}}) == 7);
    Reject([&] { ResolveStyle("zunda:6000", Styles, "frame_decode"); });
    Reject([&] { ResolveStyle("ritsu:6000", Styles, "talk"); });
    assert(ResolveStyle("kotaro", Styles) == 12);
    assert(ResolveStyle("kotaro:normal", Styles) == 12);
    assert(ResolveStyle("ENGLISH:normal", {{9, "e", "English", "Normal", "talk"}}) == 9);
    assert(ResolveStyle("WHITECUL", {{23, "w", "WhiteCUL", "ノーマル", "talk"}}) == 23);
    assert(ResolveStyle("no7", {{29, "n", "No.7", "ノーマル", "talk"}}) == 29);
    assert(ResolveStyle("4294967295", {}) == 4294967295u);
    for (const char *Name : {"", "-1", "+3", "3x", "4294967296", "missing", "zunda:", ":normal", "zunda:nope"}) Reject([&] { ResolveStyle(Name, Styles); });
    Reject([&] { ResolveStyle("zunda", Styles, "sing"); });
    Reject([&] { ResolveStyle("ずんだもん", {{1, "z", "ずんだもん", "A", "talk"}, {2, "z", "ずんだもん", "B", "talk"}}); });
    std::atomic_size_t Creates{0};
    std::vector<std::thread> Threads;
    std::vector<std::shared_ptr<NativeOnnxCachedSession>> Sessions(8);
    for (size_t Index = 0; Index < Sessions.size(); ++Index) Threads.emplace_back([&, Index] {
        Sessions[Index] = CacheSession("lazy-test", [&] { ++Creates; return std::make_shared<NativeOnnxCachedSession>(); });
    });
    for (auto &Thread : Threads) Thread.join();
    assert(Creates == 1);
    for (auto &Session : Sessions) assert(Session == Sessions.front());
    Reject([] { CacheSession("retry-test", []() -> std::shared_ptr<NativeOnnxCachedSession> { throw std::invalid_argument("load failed"); }); });
    assert(CacheSession("retry-test", [] { return std::make_shared<NativeOnnxCachedSession>(); }));
    clearNativeOnnxCaches();
    std::vector<float> F0(500), Phonemes(500 * nativeOnnxPhonemeSize);
    for (size_t Index = 0; Index < F0.size(); ++Index) F0[Index] = static_cast<float>(Index);
    for (size_t Index = 0; Index < Phonemes.size(); ++Index) Phonemes[Index] = static_cast<float>(Index);
    std::vector<NativeOnnxTraceInput> Inputs{
        createNativeOnnxFloatTensor("f0", {500, 1}, F0),
        createNativeOnnxFloatTensor("phoneme", {500, nativeOnnxPhonemeSize}, Phonemes),
        createNativeOnnxInt64Tensor("speaker_id", {1}, {0})
    };
    auto Chunk = createNativeOnnxDecoderChunkInputs(Inputs, 256, 424, 128);
    assert(Chunk.frontCropFrames == 166 && Chunk.backCropFrames == 38);
    assert(Chunk.tensors[0].dimensions == std::vector<int64_t>({372, 1}));
    assert(readNativeOnnxTensorValues<float>(Chunk.tensors[0], 1) == std::vector<float>(F0.begin() + 128, F0.end()));
    assert(readNativeOnnxTensorValues<float>(Chunk.tensors[1], 1) == std::vector<float>(Phonemes.begin() + 128 * nativeOnnxPhonemeSize, Phonemes.end()));
    assert(Chunk.tensors[2].bytes == Inputs[2].bytes);
    assert(createNativeOnnxDecoderChunkInputs(Inputs, 0, 128, std::numeric_limits<size_t>::max()).tensors[0].bytes == Inputs[0].bytes);
    Reject([&] { createNativeOnnxDecoderChunkInputs(Inputs, 256, 425, 128); });
    Inputs[0].dimensions[0] = 499;
    Reject([&] { createNativeOnnxDecoderChunkInputs(Inputs, 0, 128, 128); });
    Inputs[0].dimensions[0] = 500;
    Inputs[1].bytes.pop_back();
    Reject([&] { createNativeOnnxDecoderChunkInputs(Inputs, 0, 128, 128); });
    assert(decodeJsonString(R"("\u30c9\u30ec\u30df")", 0) == "ドレミ");
    assert(decodeJsonString(R"("\ud83c\udfb5")", 0) == "🎵");
    assert(decodeJsonString(R"("\"\\\/\b\f\n\r\t")", 0) == "\"\\/\b\f\n\r\t");
    for (const std::string &Text : {R"("\u30")", R"("\uZZZZ")", R"("\ud800")", R"("\udc00")", R"("\ud800\u0041")", R"("\q")", R"("unterminated)"}) {
        Reject([&] { decodeJsonString(Text, 0); });
    }

    const std::string Score = R"({"notes":[{"key":null,"frame_length":15,"lyric":""},{"id":"\u97f3\ud83c\udfb5","key":60,"frame_length":45,"lyric":"\u3069"}]})";
    auto Notes = parseNativeOnnxScore(Score);
    assert(Notes.size() == 2 && Notes[1].lyric == "ど" && Notes[1].noteId == "音🎵");
    auto Named = parseNativeOnnxScore(R"({"notes":[{"key":null,"frame_length":15,"lyric":""},{"id":"frame_length","key":60,"frame_length":45,"lyric":"ド"}]})");
    assert(Named[1].noteId == "frame_length" && Named[1].frameLength == 45);
    auto Features = createNativeOnnxSongPhonemeFeatures(Notes);
    auto Lengths = createNativeOnnxSongPhonemeLengths(Notes, {0, 5});
    assert(Lengths == std::vector<uint64_t>({10, 5, 45}));
    assert(Features.size() == 3 && Features[1].phoneme == "d" && Features[2].phoneme == "o");
    assert(Features[1].noteId == Notes[1].noteId && Features[2].noteId == Notes[1].noteId);

    for (const std::string &Length : {"-1", "1.5", "18446744073709551616"}) {
        Reject([&] { parseNativeOnnxScore("{\"notes\":[{\"key\":null,\"frame_length\":" + Length + ",\"lyric\":\"\"}]}"); });
    }
    Reject([] { parseNativeOnnxScore(R"({"notes":[false,{"key":null,"frame_length":15,"lyric":""}]})"); });
    Reject([] { parseNativeOnnxScore(R"({"notes":[{"key":60,"frame_length":45,"lyric":"ド"}]})"); });

    NativeOnnxFrameAudioQuery Query;
    Query.phonemes = {{"pau", "", 1, false}};
    Query.f0Values = {0};
    Query.volumeValues = {0};
    validateNativeOnnxParsedFrameAudioQuery(parseNativeOnnxFrameAudioQuery(createNativeOnnxFrameAudioQueryJson(Query)));
    Query.volumeValues.clear();
    Reject([&] { validateNativeOnnxParsedFrameAudioQuery(Query); });
    Query.phonemes = {{"pau", "", static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), false}, {"pau", "", 1, false}};
    Reject([&] { validateNativeOnnxParsedFrameAudioQuery(Query); });
    Reject([] { parseNativeOnnxFrameAudioQuery(R"({"f0":[0],"volume":[0],"phonemes":[{"phoneme":"pau","frame_length":1}],"outputSamplingRate":-1})"); });
    const std::vector<float> Wave(2400, 0.25f);
    for (uint32_t Rate : {12000u, 22050u, 24000u, 44100u, 48000u}) {
        NativeOnnxAudioQuerySettings Settings;
        Settings.outputSamplingRate = Rate;
        Settings.outputStereo = true;
        auto Whole = createNativeOnnxPcmBytes(Wave, Settings);
        assert(Whole.size() == static_cast<size_t>(Rate / 10) * 4);
        NativeOnnxPcm Pcm(Settings);
        std::vector<uint8_t> Stream;
        for (size_t Start = 0; Start < Wave.size(); Start += 37) {
            size_t End = std::min(Start + 37, Wave.size());
            auto Chunk = Pcm.Convert(std::vector<float>(Wave.begin() + Start, Wave.begin() + End), End == Wave.size());
            Stream.insert(Stream.end(), Chunk.begin(), Chunk.end());
        }
        assert(Stream == Whole);
        for (size_t Position = 0; Position < Whole.size(); Position += 4) {
            assert(Whole[Position] == Whole[Position + 2] && Whole[Position + 1] == Whole[Position + 3]);
        }
    }
    for (const std::vector<uint8_t> &Bytes : std::vector<std::vector<uint8_t>>{{0x3a, 0x80}, {0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02}}) {
        bool IsInvalid = false;
        size_t Count = 0;
        try { rewriteNativeOnnxModelRandomSeed(Bytes.data(), Bytes.size(), 0, Count); } catch (const std::runtime_error &) { IsInvalid = true; }
        assert(IsInvalid);
    }
    NativeOnnxApi Api;
    Api.createEnv = [](int32_t, const char *, OrtEnv **Env) -> OrtStatus * { *Env = nullptr; return nullptr; };
    Api.releaseEnv = [](OrtEnv *) {};
    Api.createSessionOptions = [](OrtSessionOptions **Options) -> OrtStatus * { *Options = nullptr; return nullptr; };
    Api.releaseSessionOptions = [](OrtSessionOptions *) {};
    Api.CreateCuda = [](OrtCUDAProviderOptionsV2 **Options) -> OrtStatus * {
        *Options = reinterpret_cast<OrtCUDAProviderOptionsV2 *>(1); return nullptr;
    };
    static bool IsReleased = false;
    Api.ReleaseCuda = [](OrtCUDAProviderOptionsV2 *) { IsReleased = true; };
    Api.Cuda = [](OrtSessionOptions *, const OrtCUDAProviderOptionsV2 *Options) -> OrtStatus * {
        assert(Options); return nullptr;
    };
    bool IsUsable = false;
    const std::vector<std::string> Providers{"CPUExecutionProvider", "CUDAExecutionProvider"};
    assert(SelectProvider(Api, "auto", Providers, IsUsable) == "CUDAExecutionProvider");
    assert(IsUsable && IsReleased);
    assert(SelectProvider(Api, "cpu", Providers, IsUsable) == "CPUExecutionProvider");
    assert(SelectProvider(Api, "auto", {"CPUExecutionProvider"}, IsUsable) == "CPUExecutionProvider" && !IsUsable);
    bool IsRejected = false;
    try { SelectProvider(Api, "gpu", {"CPUExecutionProvider"}, IsUsable); }
    catch (const std::runtime_error &) { IsRejected = true; }
    assert(IsRejected);
    Api.appendExecutionProvider = [](OrtSessionOptions *, const char *Name, const char *const *Keys, const char *const *Values, size_t Count) -> OrtStatus * {
        if (std::string(Name) == "WebGPU") {
            assert(Count == 1 && std::string(Keys[0]) == "powerPreference" && std::string(Values[0]) == "high-performance");
        } else {
            assert(std::string(Name) == "CoreML" && Count >= 2);
            assert(std::string(Keys[0]) == "ModelFormat" && std::string(Values[0]) == "MLProgram");
            assert(std::string(Keys[1]) == "MLComputeUnits" && std::string(Values[1]) == "ALL");
        }
        return nullptr;
    };
    assert(SelectProvider(Api, "gpu", {"CoreMLExecutionProvider"}, IsUsable) == "CoreMLExecutionProvider" && IsUsable);
    assert(SelectProvider(Api, "auto", {"CoreMLExecutionProvider", "WebGpuExecutionProvider"}, IsUsable) == "WebGpuExecutionProvider" && IsUsable);
    Reject([&] { SelectProvider(Api, "invalid", Providers, IsUsable); });
    NativeOnnxRuntimeState Gpu, Cpu;
    Gpu.selectedExecutionProvider = "WebGpuExecutionProvider";
    Gpu.isGpuExecutionProviderSelected = true;
    ModelAssetRecord Asset;
    Asset.entryName = "models/pi.bin";
    assert(SelectModel(&Gpu, Asset, Cpu)->selectedExecutionProvider == "CPUExecutionProvider");
    assert(!Cpu.isGpuExecutionProviderSelected && Gpu.isGpuExecutionProviderSelected);
    for (const char *Name : {"models/d.bin", "models/sd.bin"}) {
        Asset.entryName = Name;
        assert(SelectModel(&Gpu, Asset, Cpu) == &Gpu);
    }
    assert(SelectModel(nullptr, Asset, Cpu) == nullptr);
    std::cout << "JSON / song / resampling / GPU selection compatibility checks passed\n";
}
