#include "json_utility.hpp"
#include "native_onnx_internal.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

static void Reject(const std::function<void()> &Run) {
    bool Failed = false;
    try { Run(); } catch (const std::invalid_argument &) { Failed = true; }
    assert(Failed);
}

int main() {
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
