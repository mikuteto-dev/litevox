#include "runtime.hpp"

#include "runtime_internal.hpp"
#include "native_onnx.hpp"
#include "utility.hpp"

#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static constexpr size_t minimumStreamingTextSegmentBytes = 48;

static uint64_t getStreamingWavePcmByteCount() {
    return static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 44;
}

static bool startsWithAt(const std::string &text, size_t byteOffset, const std::string &prefixText) {
    return byteOffset + prefixText.size() <= text.size() && text.compare(byteOffset, prefixText.size(), prefixText) == 0;
}

static size_t getUtf8CharacterBytes(unsigned char leadingByte) {
    if (leadingByte < 0x80) {
        return 1;
    }
    if ((leadingByte & 0xe0) == 0xc0) {
        return 2;
    }
    if ((leadingByte & 0xf0) == 0xe0) {
        return 3;
    }
    if ((leadingByte & 0xf8) == 0xf0) {
        return 4;
    }
    return 1;
}

static bool findStreamingTextDelimiter(const std::string &text, size_t byteOffset, std::string &matchedDelimiter) {
    static const std::vector<std::string> delimiterTexts = {"。", "、", "\n", "？", "?", "！", "!", "．", "."};
    for (const std::string &delimiterText : delimiterTexts) {
        if (startsWithAt(text, byteOffset, delimiterText)) {
            matchedDelimiter = delimiterText;
            return true;
        }
    }
    return false;
}

static void appendStreamingTextSegment(std::vector<std::string> &segments, const std::string &segmentText) {
    std::string trimmedSegmentText = trimAscii(segmentText);
    if (!trimmedSegmentText.empty()) {
        segments.push_back(trimmedSegmentText);
    }
}

static std::vector<std::string> splitTextForStreaming(const std::string &text) {
    std::vector<std::string> segments;
    size_t segmentStart = 0;
    size_t byteOffset = 0;
    while (byteOffset < text.size()) {
        std::string delimiterText;
        if (findStreamingTextDelimiter(text, byteOffset, delimiterText)) {
            size_t segmentEnd = byteOffset + delimiterText.size();
            bool isHardBreak = delimiterText == "\n";
            if (isHardBreak || segments.empty() || segmentEnd - segmentStart >= minimumStreamingTextSegmentBytes) {
                appendStreamingTextSegment(segments, text.substr(segmentStart, segmentEnd - segmentStart));
                segmentStart = segmentEnd;
            }
            byteOffset = segmentEnd;
            continue;
        }
        byteOffset += getUtf8CharacterBytes(static_cast<unsigned char>(text[byteOffset]));
    }
    std::string tailText = trimAscii(text.substr(segmentStart));
    if (!tailText.empty()) {
        if (!segments.empty() && tailText.size() < minimumStreamingTextSegmentBytes) {
            segments.back() += tailText;
        } else {
            segments.push_back(tailText);
        }
    }
    if (segments.empty() && !trimAscii(text).empty()) {
        segments.push_back(trimAscii(text));
    }
    return segments;
}

static void streamWavBytes(const std::vector<uint8_t> &wavBytes, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    AudioStreamPayload audioStreamPayload = createAudioStreamPayload(wavBytes, streamOptions.audioStreamFormat);
    size_t chunkBytes = calculateAudioStreamChunkBytes(audioStreamPayload, streamOptions.chunkSamples, streamOptions.fallbackChunkBytes);
    writeAudioStreamChunks(audioStreamPayload, chunkBytes, writeChunk);
}

static void streamPcmPayloadSegment(const AudioStreamPayload &pcmPayload, const RuntimeAudioStreamOptions &streamOptions, bool &hasWrittenWaveHeader, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    if (streamOptions.audioStreamFormat == AudioStreamFormat::Wav && !hasWrittenWaveHeader) {
        std::vector<uint8_t> wavHeader = createPcmWaveHeader(pcmPayload.sampleRate, pcmPayload.channels, pcmPayload.bitsPerSample, getStreamingWavePcmByteCount());
        writeChunk(wavHeader.data(), wavHeader.size());
        hasWrittenWaveHeader = true;
    }
    size_t chunkBytes = calculateAudioStreamChunkBytes(pcmPayload, streamOptions.chunkSamples, streamOptions.fallbackChunkBytes);
    writeAudioStreamChunks(pcmPayload, chunkBytes, writeChunk);
}

static AudioStreamPayload synthesizeSegmentedTextPcmPayload(RuntimeState &runtimeState, const std::string &segmentText, uint32_t styleId) {
    return createAudioStreamPayload(synthesizeText(runtimeState, segmentText, styleId), AudioStreamFormat::Pcm);
}

static void streamNativeBackendPcmAudio(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, bool shouldUseStreamingWaveHeader, bool &hasWrittenWaveHeader, const std::function<void(const uint8_t *, size_t)> &writeChunk);

static void streamSegmentedAudioQuery(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, bool &hasWrittenWaveHeader, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    streamNativeBackendPcmAudio(runtimeState, audioQueryJson, styleId, streamOptions, true, hasWrittenWaveHeader, writeChunk);
}

struct SegmentedTextPrefetchState {
    std::mutex mutex;
    std::condition_variable condition;
    AudioStreamPayload pcmPayload;
    std::exception_ptr exception;
    size_t readySegmentIndex = 0;
    bool hasReadyPayload = false;
    bool isFinished = false;
    bool shouldStop = false;
};

static void streamSegmentedTextByPrefetch(RuntimeState &runtimeState, const std::vector<std::string> &segments, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    SegmentedTextPrefetchState prefetchState;
    std::thread prefetchThread([&runtimeState, &segments, styleId, &prefetchState]() {
        try {
            for (size_t segmentIndex = 0; segmentIndex < segments.size(); segmentIndex++) {
                AudioStreamPayload pcmPayload = synthesizeSegmentedTextPcmPayload(runtimeState, segments[segmentIndex], styleId);
                std::unique_lock<std::mutex> prefetchLock(prefetchState.mutex);
                prefetchState.condition.wait(prefetchLock, [&prefetchState]() {
                    return prefetchState.shouldStop || !prefetchState.hasReadyPayload;
                });
                if (prefetchState.shouldStop) {
                    return;
                }
                prefetchState.pcmPayload = std::move(pcmPayload);
                prefetchState.readySegmentIndex = segmentIndex;
                prefetchState.hasReadyPayload = true;
                prefetchLock.unlock();
                prefetchState.condition.notify_all();
            }
            std::lock_guard<std::mutex> prefetchLock(prefetchState.mutex);
            prefetchState.isFinished = true;
            prefetchState.condition.notify_all();
        } catch (...) {
            std::lock_guard<std::mutex> prefetchLock(prefetchState.mutex);
            prefetchState.exception = std::current_exception();
            prefetchState.isFinished = true;
            prefetchState.condition.notify_all();
        }
    });
    auto stopPrefetchThread = [&prefetchThread, &prefetchState]() {
        {
            std::lock_guard<std::mutex> prefetchLock(prefetchState.mutex);
            prefetchState.shouldStop = true;
        }
        prefetchState.condition.notify_all();
        if (prefetchThread.joinable()) {
            prefetchThread.join();
        }
    };
    bool hasWrittenWaveHeader = false;
    try {
        for (size_t segmentIndex = 0; segmentIndex < segments.size(); segmentIndex++) {
            AudioStreamPayload pcmPayload;
            {
                std::unique_lock<std::mutex> prefetchLock(prefetchState.mutex);
                prefetchState.condition.wait(prefetchLock, [&prefetchState]() {
                    return prefetchState.hasReadyPayload || prefetchState.isFinished || prefetchState.exception != nullptr;
                });
                if (prefetchState.exception) {
                    std::rethrow_exception(prefetchState.exception);
                }
                if (!prefetchState.hasReadyPayload || prefetchState.readySegmentIndex != segmentIndex) {
                    throw std::runtime_error("stream prefetch が途中で終了しました");
                }
                pcmPayload = std::move(prefetchState.pcmPayload);
                prefetchState.hasReadyPayload = false;
            }
            prefetchState.condition.notify_all();
            streamPcmPayloadSegment(pcmPayload, streamOptions, hasWrittenWaveHeader, writeChunk);
        }
    } catch (...) {
        stopPrefetchThread();
        throw;
    }
    stopPrefetchThread();
}

static RuntimeState &getSegmentedTextPrefetchRuntime(RuntimeState &runtimeState) {
    if (runtimeState.segmentedTextPrefetchRuntime && runtimeState.segmentedTextPrefetchRuntime->libraryStateSignature == runtimeState.libraryStateSignature) {
        return *runtimeState.segmentedTextPrefetchRuntime;
    }
    runtimeState.segmentedTextPrefetchRuntime.reset();
    std::unique_ptr<RuntimeState> prefetchRuntime = std::make_unique<RuntimeState>(createRuntimeState(runtimeState.runtimePaths, false));
    prefetchRuntime->workerIndex = runtimeState.workerIndex;
    prefetchRuntime->workerCount = runtimeState.workerCount;
    prefetchRuntime->sharedLoadedStyleCounts = runtimeState.sharedLoadedStyleCounts;
    prefetchRuntime->sharedLoadedStylesMutex = runtimeState.sharedLoadedStylesMutex;
    prefetchRuntime->sharedStyleUnloadGenerations = runtimeState.sharedStyleUnloadGenerations;
    prefetchRuntime->sharedStyleUnloadMutex = runtimeState.sharedStyleUnloadMutex;
    prefetchRuntime->sharedUserDictMutex = runtimeState.sharedUserDictMutex;
    prefetchRuntime->sharedPresetMutex = runtimeState.sharedPresetMutex;
    prefetchRuntime->sharedSettingMutex = runtimeState.sharedSettingMutex;
    prefetchRuntime->sharedLibraryMutex = runtimeState.sharedLibraryMutex;
    runtimeState.segmentedTextPrefetchRuntime = std::move(prefetchRuntime);
    return *runtimeState.segmentedTextPrefetchRuntime;
}

static void streamSegmentedTextByStreamingFirstSegment(RuntimeState &runtimeState, RuntimeState &prefetchRuntimeState, const std::vector<std::string> &segments, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    SegmentedTextPrefetchState prefetchState;
    std::thread prefetchThread([&prefetchRuntimeState, &segments, styleId, &prefetchState]() {
        try {
            for (size_t segmentIndex = 1; segmentIndex < segments.size(); segmentIndex++) {
                AudioStreamPayload pcmPayload = synthesizeSegmentedTextPcmPayload(prefetchRuntimeState, segments[segmentIndex], styleId);
                std::unique_lock<std::mutex> prefetchLock(prefetchState.mutex);
                prefetchState.condition.wait(prefetchLock, [&prefetchState]() {
                    return prefetchState.shouldStop || !prefetchState.hasReadyPayload;
                });
                if (prefetchState.shouldStop) {
                    return;
                }
                prefetchState.pcmPayload = std::move(pcmPayload);
                prefetchState.readySegmentIndex = segmentIndex;
                prefetchState.hasReadyPayload = true;
                prefetchLock.unlock();
                prefetchState.condition.notify_all();
            }
            std::lock_guard<std::mutex> prefetchLock(prefetchState.mutex);
            prefetchState.isFinished = true;
            prefetchState.condition.notify_all();
        } catch (...) {
            std::lock_guard<std::mutex> prefetchLock(prefetchState.mutex);
            prefetchState.exception = std::current_exception();
            prefetchState.isFinished = true;
            prefetchState.condition.notify_all();
        }
    });
    auto stopPrefetchThread = [&prefetchThread, &prefetchState]() {
        {
            std::lock_guard<std::mutex> prefetchLock(prefetchState.mutex);
            prefetchState.shouldStop = true;
        }
        prefetchState.condition.notify_all();
        if (prefetchThread.joinable()) {
            prefetchThread.join();
        }
    };
    bool hasWrittenWaveHeader = false;
    try {
        streamSegmentedAudioQuery(runtimeState, createAudioQuery(runtimeState, segments.front(), styleId), styleId, streamOptions, hasWrittenWaveHeader, writeChunk);
        for (size_t segmentIndex = 1; segmentIndex < segments.size(); segmentIndex++) {
            AudioStreamPayload pcmPayload;
            {
                std::unique_lock<std::mutex> prefetchLock(prefetchState.mutex);
                prefetchState.condition.wait(prefetchLock, [&prefetchState]() {
                    return prefetchState.hasReadyPayload || prefetchState.isFinished || prefetchState.exception != nullptr;
                });
                if (prefetchState.exception) {
                    std::rethrow_exception(prefetchState.exception);
                }
                if (!prefetchState.hasReadyPayload || prefetchState.readySegmentIndex != segmentIndex) {
                    throw std::runtime_error("stream prefetch が途中で終了しました");
                }
                pcmPayload = std::move(prefetchState.pcmPayload);
                prefetchState.hasReadyPayload = false;
            }
            prefetchState.condition.notify_all();
            streamPcmPayloadSegment(pcmPayload, streamOptions, hasWrittenWaveHeader, writeChunk);
        }
    } catch (...) {
        stopPrefetchThread();
        throw;
    }
    stopPrefetchThread();
}

static size_t convertChunkSamplesToDecoderFrames(size_t chunkSamples) {
    return std::max<size_t>(1, (std::max<size_t>(1, chunkSamples) + 255) / 256);
}

static void streamNativeBackendPcmAudio(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, bool shouldUseStreamingWaveHeader, bool &hasWrittenWaveHeader, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    size_t chunkFrames = convertChunkSamplesToDecoderFrames(streamOptions.chunkSamples);
    const VoiceModelRecord &modelRecord = getRuntimeVoiceModelForStyle(runtimeState, styleId);
    auto startStream = [&streamOptions, shouldUseStreamingWaveHeader, &hasWrittenWaveHeader, &writeChunk](const NativeOnnxPcmStreamInfo &streamInfo) {
        if (streamOptions.audioStreamFormat == AudioStreamFormat::Wav) {
            if (shouldUseStreamingWaveHeader && hasWrittenWaveHeader) {
                return;
            }
            uint64_t pcmByteCount = shouldUseStreamingWaveHeader ? getStreamingWavePcmByteCount() : streamInfo.pcmBytes;
            std::vector<uint8_t> wavHeader = createPcmWaveHeader(streamInfo.sampleRate, streamInfo.channels, streamInfo.bitsPerSample, pcmByteCount);
            writeChunk(wavHeader.data(), wavHeader.size());
            hasWrittenWaveHeader = true;
        }
    };
    streamNativeOnnxModelAssetsAudioQueryPcm(runtimeState.coreBackend.nativeOnnxRuntime, modelRecord.modelAssets, audioQueryJson, styleId, getCoreBackendCpuNumThreads(runtimeState.coreBackend), chunkFrames, startStream, writeChunk, shouldUseNativeRuntimeVvBinConfig(runtimeState));
}

static void streamNativeBackendPcmAudio(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    bool hasWrittenWaveHeader = false;
    streamNativeBackendPcmAudio(runtimeState, audioQueryJson, styleId, streamOptions, false, hasWrittenWaveHeader, writeChunk);
}

void streamAudioQuery(RuntimeState &runtimeState, const std::string &audioQueryJson, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    if (getCoreBackendCapabilities(runtimeState.coreBackend).supportsTrueStreaming) {
        ensureStyleLoaded(runtimeState, styleId);
        if (isNativeRuntimeBackend(runtimeState)) {
            streamNativeBackendPcmAudio(runtimeState, audioQueryJson, styleId, streamOptions, writeChunk);
            return;
        }
    }
    streamWavBytes(synthesizeAudioQuery(runtimeState, audioQueryJson, styleId), streamOptions, writeChunk);
}

void streamText(RuntimeState &runtimeState, const std::string &text, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    std::vector<std::string> segments = splitTextForStreaming(text);
    if (segments.size() <= 1) {
        if (getCoreBackendCapabilities(runtimeState.coreBackend).supportsTrueStreaming) {
            streamAudioQuery(runtimeState, createAudioQuery(runtimeState, text, styleId), styleId, streamOptions, writeChunk);
            return;
        }
        streamWavBytes(synthesizeText(runtimeState, text, styleId), streamOptions, writeChunk);
        return;
    }
    if (getCoreBackendCapabilities(runtimeState.coreBackend).supportsTrueStreaming) {
        streamSegmentedTextByStreamingFirstSegment(runtimeState, getSegmentedTextPrefetchRuntime(runtimeState), segments, styleId, streamOptions, writeChunk);
        return;
    }
    streamSegmentedTextByPrefetch(runtimeState, segments, styleId, streamOptions, writeChunk);
}

void streamKana(RuntimeState &runtimeState, const std::string &kana, uint32_t styleId, const RuntimeAudioStreamOptions &streamOptions, const std::function<void(const uint8_t *, size_t)> &writeChunk) {
    if (getCoreBackendCapabilities(runtimeState.coreBackend).supportsTrueStreaming) {
        streamAudioQuery(runtimeState, createAudioQueryFromKana(runtimeState, kana, styleId), styleId, streamOptions, writeChunk);
        return;
    }
    streamWavBytes(synthesizeKana(runtimeState, kana, styleId), streamOptions, writeChunk);
}
