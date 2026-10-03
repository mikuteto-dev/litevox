#include "cli_bench.hpp"

#include "json_utility.hpp"
#include "socket_compat.hpp"
#include "utility.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
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

static double getElapsedMilliseconds(std::chrono::steady_clock::time_point startTime, std::chrono::steady_clock::time_point endTime) {
    return std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

static std::vector<uint32_t> getBenchSpeakerIds(const CliOptions &cliOptions) {
    if (!cliOptions.benchSpeakers.empty()) {
        return cliOptions.benchSpeakers;
    }
    return {cliOptions.speaker};
}

static std::vector<std::string> getBenchTexts(const CliOptions &cliOptions) {
    if (!cliOptions.benchTexts.empty()) {
        return cliOptions.benchTexts;
    }
    return {cliOptions.text};
}

static std::vector<std::string> getBenchHttpPaths(const CliOptions &cliOptions) {
    if (!cliOptions.benchHttpPaths.empty()) {
        return cliOptions.benchHttpPaths;
    }
    return {cliOptions.httpPath};
}

static std::vector<uint8_t> synthesizeBenchAudio(RuntimeState &runtimeState, const std::string &textValue, bool isKana, uint32_t speakerId) {
    if (isKana) {
        return synthesizeKana(runtimeState, textValue, speakerId);
    }
    return synthesizeText(runtimeState, textValue, speakerId);
}

static std::string joinSizeValues(const std::vector<size_t> &values) {
    std::ostringstream valuesStream;
    for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
        if (valueIndex > 0) {
            valuesStream << ",";
        }
        valuesStream << values[valueIndex];
    }
    return valuesStream.str();
}

static std::string joinUint64Values(const std::vector<uint64_t> &values) {
    std::ostringstream valuesStream;
    for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
        if (valueIndex > 0) {
            valuesStream << ",";
        }
        valuesStream << values[valueIndex];
    }
    return valuesStream.str();
}

static std::string joinDoubleValues(const std::vector<double> &values) {
    std::ostringstream valuesStream;
    valuesStream << std::fixed << std::setprecision(3);
    for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
        if (valueIndex > 0) {
            valuesStream << ",";
        }
        valuesStream << values[valueIndex];
    }
    return valuesStream.str();
}

static std::string joinSpeakerIds(const std::vector<uint32_t> &speakerIds) {
    std::ostringstream speakerStream;
    for (size_t speakerIndex = 0; speakerIndex < speakerIds.size(); speakerIndex++) {
        if (speakerIndex > 0) {
            speakerStream << ",";
        }
        speakerStream << speakerIds[speakerIndex];
    }
    return speakerStream.str();
}

static std::string joinSpeakerCounts(const std::vector<uint32_t> &speakerIds, const std::vector<size_t> &speakerCounts) {
    std::ostringstream speakerStream;
    for (size_t speakerIndex = 0; speakerIndex < speakerIds.size(); speakerIndex++) {
        if (speakerIndex > 0) {
            speakerStream << ",";
        }
        speakerStream << speakerIds[speakerIndex] << ":" << speakerCounts[speakerIndex];
    }
    return speakerStream.str();
}

static std::string joinStringValues(const std::vector<std::string> &values) {
    std::ostringstream valueStream;
    for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
        if (valueIndex > 0) {
            valueStream << ",";
        }
        valueStream << values[valueIndex];
    }
    return valueStream.str();
}

static std::string joinStringCounts(const std::vector<std::string> &values, const std::vector<size_t> &counts) {
    std::ostringstream valueStream;
    for (size_t valueIndex = 0; valueIndex < values.size(); valueIndex++) {
        if (valueIndex > 0) {
            valueStream << ",";
        }
        valueStream << values[valueIndex] << ":" << counts[valueIndex];
    }
    return valueStream.str();
}

static std::string joinTextByteLengths(const std::vector<std::string> &texts) {
    std::vector<size_t> textByteLengths;
    textByteLengths.reserve(texts.size());
    for (const std::string &textValue : texts) {
        textByteLengths.push_back(textValue.size());
    }
    return joinSizeValues(textByteLengths);
}

static std::string getRoundRobinModeText(size_t valueCount) {
    if (valueCount > 1) {
        return "round_robin";
    }
    return "single";
}

static std::vector<std::string> getBenchScoreTexts(const CliOptions &cliOptions) {
    if (cliOptions.scorePath.empty()) {
        throw std::runtime_error("--score が必要です");
    }
    std::vector<std::string> scoreTexts;
    scoreTexts.push_back(readTextFile(cliOptions.scorePath));
    for (const fs::path &scorePath : cliOptions.benchScorePaths) {
        scoreTexts.push_back(readTextFile(scorePath));
    }
    return scoreTexts;
}

static std::vector<std::string> getBenchFrameAudioQueryTexts(const CliOptions &cliOptions, size_t scoreCount) {
    std::vector<std::string> frameAudioQueryTexts;
    if (cliOptions.frameAudioQueryPath.empty()) {
        if (!cliOptions.benchFrameAudioQueryPaths.empty()) {
            throw std::runtime_error("--add-frame-audio-query の前に --frame-audio-query が必要です");
        }
        return frameAudioQueryTexts;
    }
    frameAudioQueryTexts.push_back(readTextFile(cliOptions.frameAudioQueryPath));
    for (const fs::path &frameAudioQueryPath : cliOptions.benchFrameAudioQueryPaths) {
        frameAudioQueryTexts.push_back(readTextFile(frameAudioQueryPath));
    }
    if (scoreCount > 1 && frameAudioQueryTexts.size() != scoreCount) {
        throw std::runtime_error("複数 score を使う場合、--frame-audio-query と --add-frame-audio-query の総数は score 数と一致する必要があります");
    }
    return frameAudioQueryTexts;
}

static std::vector<size_t> createRoundRobinCounts(size_t runCount, size_t valueCount) {
    std::vector<size_t> counts(valueCount, 0);
    for (size_t runIndex = 0; runIndex < runCount; runIndex++) {
        counts[runIndex % valueCount]++;
    }
    return counts;
}

static std::string getCartesianModeText(size_t combinationCycle) {
    if (combinationCycle > 1) {
        return "cartesian";
    }
    return "single";
}

struct BenchCaseSelection {
    size_t speakerIndex = 0;
    size_t textIndex = 0;
    size_t pathIndex = 0;
    size_t combinationIndex = 0;
};

static size_t getCombinationCycleSize(size_t speakerCount, size_t textCount, size_t pathCount) {
    return speakerCount * textCount * pathCount;
}

static BenchCaseSelection getBenchCaseSelection(size_t runIndex, size_t speakerCount, size_t textCount, size_t pathCount) {
    BenchCaseSelection selection;
    selection.speakerIndex = runIndex % speakerCount;
    size_t remainingIndex = runIndex / speakerCount;
    selection.textIndex = remainingIndex % textCount;
    remainingIndex /= textCount;
    selection.pathIndex = remainingIndex % pathCount;
    selection.combinationIndex = selection.pathIndex * textCount * speakerCount
        + selection.textIndex * speakerCount
        + selection.speakerIndex;
    return selection;
}

static std::string joinCombinationCounts(const std::vector<uint32_t> &speakerIds, size_t textCount, size_t pathCount, const std::vector<size_t> &combinationCounts) {
    std::ostringstream valueStream;
    size_t combinationIndex = 0;
    for (size_t pathIndex = 0; pathIndex < pathCount; pathIndex++) {
        for (size_t textIndex = 0; textIndex < textCount; textIndex++) {
            for (size_t speakerIndex = 0; speakerIndex < speakerIds.size(); speakerIndex++) {
                if (combinationIndex > 0) {
                    valueStream << ",";
                }
                valueStream << "s" << speakerIds[speakerIndex] << "-t" << textIndex;
                if (pathCount > 1) {
                    valueStream << "-p" << pathIndex;
                }
                valueStream << ":" << combinationCounts[combinationIndex];
                combinationIndex++;
            }
        }
    }
    return valueStream.str();
}

static size_t countActiveBenchWorkers(const std::vector<size_t> &completedRunsByWorker) {
    size_t activeWorkerCount = 0;
    for (size_t completedWorkerRuns : completedRunsByWorker) {
        if (completedWorkerRuns > 0) {
            activeWorkerCount++;
        }
    }
    return activeWorkerCount;
}

static void printBenchMetrics(std::chrono::steady_clock::time_point runtimeStartTime, std::chrono::steady_clock::time_point runtimeReadyTime, std::chrono::steady_clock::time_point synthesisStartTime, std::chrono::steady_clock::time_point synthesisEndTime, size_t runCount, size_t workerCount, const std::vector<uint32_t> &speakerIds, const std::vector<std::string> &benchTexts, size_t completedRunCount, size_t wavBytesSize, uint64_t totalWavBytesSize, const std::vector<size_t> &completedRunsByWorker, const std::vector<uint64_t> &totalWavBytesByWorker, const std::vector<size_t> &completedRunsBySpeaker, const std::vector<size_t> &completedRunsByText, const std::vector<size_t> &completedRunsByCombination) {
    double synthesisMilliseconds = getElapsedMilliseconds(synthesisStartTime, synthesisEndTime);
    double throughputPerSecond = synthesisMilliseconds > 0.0 ? static_cast<double>(completedRunCount) * 1000.0 / synthesisMilliseconds : 0.0;
    size_t combinationCycle = getCombinationCycleSize(speakerIds.size(), benchTexts.size(), 1);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "init_ms\t" << getElapsedMilliseconds(runtimeStartTime, runtimeReadyTime) << "\n";
    std::cout << "synthesis_ms\t" << synthesisMilliseconds << "\n";
    std::cout << "runs\t" << runCount << "\n";
    std::cout << "workers\t" << workerCount << "\n";
    std::cout << "workload_mode\t" << getCartesianModeText(combinationCycle) << "\n";
    std::cout << "combination_cycle\t" << combinationCycle << "\n";
    std::cout << "speaker_mode\t" << getRoundRobinModeText(speakerIds.size()) << "\n";
    std::cout << "speakers\t" << joinSpeakerIds(speakerIds) << "\n";
    std::cout << "text_mode\t" << getRoundRobinModeText(benchTexts.size()) << "\n";
    std::cout << "text_count\t" << benchTexts.size() << "\n";
    std::cout << "text_utf8_bytes\t" << joinTextByteLengths(benchTexts) << "\n";
    std::cout << "completed_runs\t" << completedRunCount << "\n";
    std::cout << "speaker_completed_runs\t" << joinSpeakerCounts(speakerIds, completedRunsBySpeaker) << "\n";
    std::cout << "text_completed_runs\t" << joinSizeValues(completedRunsByText) << "\n";
    std::cout << "combination_completed_runs\t" << joinCombinationCounts(speakerIds, benchTexts.size(), 1, completedRunsByCombination) << "\n";
    std::cout << "wav_bytes\t" << wavBytesSize << "\n";
    std::cout << "total_wav_bytes\t" << totalWavBytesSize << "\n";
    std::cout << "throughput_rps\t" << throughputPerSecond << "\n";
    std::cout << "active_workers\t" << countActiveBenchWorkers(completedRunsByWorker) << "\n";
    std::cout << "worker_completed_runs\t" << joinSizeValues(completedRunsByWorker) << "\n";
    std::cout << "worker_total_wav_bytes\t" << joinUint64Values(totalWavBytesByWorker) << "\n";
    std::cout << "max_rss_bytes\t" << getPeakResidentBytes() << "\n";
}

std::vector<std::unique_ptr<RuntimeHolder>> createRuntimeHolders(const CliOptions &cliOptions, std::map<uint32_t, size_t> *sharedLoadedStyleCounts, std::mutex *sharedLoadedStylesMutex, std::map<uint32_t, uint64_t> *sharedStyleUnloadGenerations, std::mutex *sharedStyleUnloadMutex, std::mutex *sharedUserDictMutex, std::mutex *sharedPresetMutex, std::mutex *sharedSettingMutex, std::mutex *sharedLibraryMutex) {
    std::vector<std::unique_ptr<RuntimeHolder>> runtimeHolders(cliOptions.workers);
    std::atomic<bool> hasRuntimeError{false};
    std::string firstErrorMessage;
    std::mutex firstErrorMessageMutex;
    std::vector<std::thread> initializerThreads;
    initializerThreads.reserve(cliOptions.workers);
    for (size_t workerIndex = 0; workerIndex < cliOptions.workers; workerIndex++) {
        initializerThreads.emplace_back([&, workerIndex]() {
            try {
                RuntimeState *createdRuntimeState = new RuntimeState(createRuntimeState(cliOptions.runtimePaths, false));
                createdRuntimeState->workerIndex = workerIndex;
                createdRuntimeState->workerCount = cliOptions.workers;
                createdRuntimeState->sharedLoadedStyleCounts = sharedLoadedStyleCounts;
                createdRuntimeState->sharedLoadedStylesMutex = sharedLoadedStylesMutex;
                createdRuntimeState->sharedStyleUnloadGenerations = sharedStyleUnloadGenerations;
                createdRuntimeState->sharedStyleUnloadMutex = sharedStyleUnloadMutex;
                createdRuntimeState->sharedUserDictMutex = sharedUserDictMutex;
                createdRuntimeState->sharedPresetMutex = sharedPresetMutex;
                createdRuntimeState->sharedSettingMutex = sharedSettingMutex;
                createdRuntimeState->sharedLibraryMutex = sharedLibraryMutex;
                if (cliOptions.shouldPreload) {
                    loadAllVoiceModels(*createdRuntimeState);
                }
                runtimeHolders[workerIndex] = std::make_unique<RuntimeHolder>(createdRuntimeState);
            } catch (const std::exception &caughtException) {
                if (!hasRuntimeError.exchange(true)) {
                    std::lock_guard<std::mutex> firstErrorLock(firstErrorMessageMutex);
                    firstErrorMessage = caughtException.what();
                }
            } catch (...) {
                if (!hasRuntimeError.exchange(true)) {
                    std::lock_guard<std::mutex> firstErrorLock(firstErrorMessageMutex);
                    firstErrorMessage = "runtime worker 初期化中に不明なエラーが発生しました";
                }
            }
        });
    }
    for (std::thread &initializerThread : initializerThreads) {
        initializerThread.join();
    }
    if (hasRuntimeError.load()) {
        throw std::runtime_error(firstErrorMessage);
    }
    return runtimeHolders;
}

int runBenchCommand(const CliOptions &cliOptions) {
    std::vector<std::string> benchTexts = getBenchTexts(cliOptions);
    if (benchTexts.empty() || benchTexts.front().empty()) {
        throw std::runtime_error("--text が必要です");
    }
    std::vector<uint32_t> benchSpeakerIds = getBenchSpeakerIds(cliOptions);
    auto runtimeStartTime = std::chrono::steady_clock::now();
    if (cliOptions.workers > 1) {
        std::map<uint32_t, size_t> sharedLoadedStyleCounts;
        std::mutex sharedLoadedStylesMutex;
        std::map<uint32_t, uint64_t> sharedStyleUnloadGenerations;
        std::mutex sharedStyleUnloadMutex;
        std::mutex sharedUserDictMutex;
        std::mutex sharedPresetMutex;
        std::mutex sharedSettingMutex;
        std::mutex sharedLibraryMutex;
        std::vector<std::unique_ptr<RuntimeHolder>> runtimeHolders = createRuntimeHolders(cliOptions, &sharedLoadedStyleCounts, &sharedLoadedStylesMutex, &sharedStyleUnloadGenerations, &sharedStyleUnloadMutex, &sharedUserDictMutex, &sharedPresetMutex, &sharedSettingMutex, &sharedLibraryMutex);
        auto runtimeReadyTime = std::chrono::steady_clock::now();
        std::atomic<size_t> nextRunIndex{0};
        std::atomic<size_t> completedRunCount{0};
        std::atomic<size_t> wavBytesSize{0};
        std::atomic<uint64_t> totalWavBytesSize{0};
        std::vector<size_t> completedRunsByWorker(cliOptions.workers, 0);
        std::vector<uint64_t> totalWavBytesByWorker(cliOptions.workers, 0);
        std::vector<size_t> completedRunsBySpeaker(benchSpeakerIds.size(), 0);
        std::vector<size_t> completedRunsByText(benchTexts.size(), 0);
        std::vector<size_t> completedRunsByCombination(getCombinationCycleSize(benchSpeakerIds.size(), benchTexts.size(), 1), 0);
        std::mutex completedRunsBySpeakerMutex;
        std::mutex completedRunsByTextMutex;
        std::mutex completedRunsByCombinationMutex;
        std::atomic<bool> hasBenchError{false};
        std::string firstErrorMessage;
        std::mutex firstErrorMessageMutex;
        std::vector<std::thread> benchThreads;
        benchThreads.reserve(cliOptions.workers);
        auto synthesisStartTime = std::chrono::steady_clock::now();
        for (size_t workerIndex = 0; workerIndex < cliOptions.workers; workerIndex++) {
            benchThreads.emplace_back([&, workerIndex]() {
                RuntimeState &workerRuntimeState = *runtimeHolders[workerIndex]->runtimeState;
                size_t workerCompletedRuns = 0;
                uint64_t workerTotalWavBytes = 0;
                std::vector<size_t> workerCompletedRunsBySpeaker(benchSpeakerIds.size(), 0);
                std::vector<size_t> workerCompletedRunsByText(benchTexts.size(), 0);
                std::vector<size_t> workerCompletedRunsByCombination(completedRunsByCombination.size(), 0);
                while (!hasBenchError.load()) {
                    size_t runIndex = nextRunIndex.fetch_add(1);
                    if (runIndex >= cliOptions.runs) {
                        break;
                    }
                    try {
                        BenchCaseSelection selection = getBenchCaseSelection(runIndex, benchSpeakerIds.size(), benchTexts.size(), 1);
                        std::vector<uint8_t> wavBytes = synthesizeBenchAudio(workerRuntimeState, benchTexts[selection.textIndex], cliOptions.isKana, benchSpeakerIds[selection.speakerIndex]);
                        wavBytesSize.store(wavBytes.size());
                        totalWavBytesSize.fetch_add(static_cast<uint64_t>(wavBytes.size()));
                        completedRunCount.fetch_add(1);
                        workerCompletedRuns++;
                        workerTotalWavBytes += static_cast<uint64_t>(wavBytes.size());
                        workerCompletedRunsBySpeaker[selection.speakerIndex]++;
                        workerCompletedRunsByText[selection.textIndex]++;
                        workerCompletedRunsByCombination[selection.combinationIndex]++;
                    } catch (const std::exception &caughtException) {
                        if (!hasBenchError.exchange(true)) {
                            std::lock_guard<std::mutex> firstErrorLock(firstErrorMessageMutex);
                            firstErrorMessage = caughtException.what();
                        }
                    } catch (...) {
                        if (!hasBenchError.exchange(true)) {
                            std::lock_guard<std::mutex> firstErrorLock(firstErrorMessageMutex);
                            firstErrorMessage = "bench 実行中に不明なエラーが発生しました";
                        }
                    }
                }
                completedRunsByWorker[workerIndex] = workerCompletedRuns;
                totalWavBytesByWorker[workerIndex] = workerTotalWavBytes;
                std::lock_guard<std::mutex> completedRunsBySpeakerLock(completedRunsBySpeakerMutex);
                for (size_t speakerIndex = 0; speakerIndex < benchSpeakerIds.size(); speakerIndex++) {
                    completedRunsBySpeaker[speakerIndex] += workerCompletedRunsBySpeaker[speakerIndex];
                }
                std::lock_guard<std::mutex> completedRunsByTextLock(completedRunsByTextMutex);
                for (size_t textIndex = 0; textIndex < benchTexts.size(); textIndex++) {
                    completedRunsByText[textIndex] += workerCompletedRunsByText[textIndex];
                }
                std::lock_guard<std::mutex> completedRunsByCombinationLock(completedRunsByCombinationMutex);
                for (size_t combinationIndex = 0; combinationIndex < completedRunsByCombination.size(); combinationIndex++) {
                    completedRunsByCombination[combinationIndex] += workerCompletedRunsByCombination[combinationIndex];
                }
            });
        }
        for (std::thread &benchThread : benchThreads) {
            benchThread.join();
        }
        if (hasBenchError.load()) {
            throw std::runtime_error(firstErrorMessage);
        }
        auto synthesisEndTime = std::chrono::steady_clock::now();
        printBenchMetrics(runtimeStartTime, runtimeReadyTime, synthesisStartTime, synthesisEndTime, cliOptions.runs, cliOptions.workers, benchSpeakerIds, benchTexts, completedRunCount.load(), wavBytesSize.load(), totalWavBytesSize.load(), completedRunsByWorker, totalWavBytesByWorker, completedRunsBySpeaker, completedRunsByText, completedRunsByCombination);
        return 0;
    }
    RuntimeState *runtimeState = new RuntimeState(createRuntimeState(cliOptions.runtimePaths, cliOptions.shouldPreload));
    auto runtimeReadyTime = std::chrono::steady_clock::now();
    size_t wavBytesSize = 0;
    uint64_t totalWavBytesSize = 0;
    std::vector<size_t> completedRunsBySpeaker(benchSpeakerIds.size(), 0);
    std::vector<size_t> completedRunsByText(benchTexts.size(), 0);
    std::vector<size_t> completedRunsByCombination(getCombinationCycleSize(benchSpeakerIds.size(), benchTexts.size(), 1), 0);
    auto synthesisStartTime = std::chrono::steady_clock::now();
    for (size_t runIndex = 0; runIndex < cliOptions.runs; runIndex++) {
        BenchCaseSelection selection = getBenchCaseSelection(runIndex, benchSpeakerIds.size(), benchTexts.size(), 1);
        std::vector<uint8_t> wavBytes = synthesizeBenchAudio(*runtimeState, benchTexts[selection.textIndex], cliOptions.isKana, benchSpeakerIds[selection.speakerIndex]);
        wavBytesSize = wavBytes.size();
        totalWavBytesSize += static_cast<uint64_t>(wavBytesSize);
        completedRunsBySpeaker[selection.speakerIndex]++;
        completedRunsByText[selection.textIndex]++;
        completedRunsByCombination[selection.combinationIndex]++;
    }
    auto synthesisEndTime = std::chrono::steady_clock::now();
    printBenchMetrics(runtimeStartTime, runtimeReadyTime, synthesisStartTime, synthesisEndTime, cliOptions.runs, cliOptions.workers, benchSpeakerIds, benchTexts, cliOptions.runs, wavBytesSize, totalWavBytesSize, std::vector<size_t>{cliOptions.runs}, std::vector<uint64_t>{totalWavBytesSize}, completedRunsBySpeaker, completedRunsByText, completedRunsByCombination);
    return 0;
}

struct SongBenchResult {
    std::string endpoint;
    double totalElapsedMilliseconds = 0.0;
    double firstMilliseconds = 0.0;
    double averageMilliseconds = 0.0;
    double averageWarmMilliseconds = 0.0;
    double minimumMilliseconds = 0.0;
    double maximumMilliseconds = 0.0;
    double averageFirstResponseMilliseconds = 0.0;
    double averageFirstBodyMilliseconds = 0.0;
    double throughputPerSecond = 0.0;
    size_t activeWorkers = 0;
    std::vector<size_t> completedRequestsByWorker;
    std::vector<double> averageMillisecondsByWorker;
    std::vector<double> averageFirstResponseMillisecondsByWorker;
    std::vector<double> averageFirstBodyMillisecondsByWorker;
    size_t bytes = 0;
    std::string repeatStatus;
    size_t uniqueShaCount = 0;
    std::string firstSha256;
    std::string lastSha256;
};

static std::string getNativeSingTeacherModeLabel(const CliOptions &cliOptions) {
    if (cliOptions.nativeSingTeacherMode.empty()
        || cliOptions.nativeSingTeacherMode == "vv-bin"
        || cliOptions.nativeSingTeacherMode == "vv_bin"
        || cliOptions.nativeSingTeacherMode == "stochastic") {
        return "vv-bin";
    }
    if (cliOptions.nativeSingTeacherMode == "deterministic"
        || cliOptions.nativeSingTeacherMode == "seeded_exported_onnx") {
        return "deterministic";
    }
    return cliOptions.nativeSingTeacherMode;
}

static SongBenchResult benchmarkSongOperation(const std::string &endpoint, size_t runCount, size_t scoreCount, const std::function<std::vector<uint8_t>(size_t)> &runOperation) {
    if (runCount == 0) {
        throw std::runtime_error("runCount は 1 以上が必要です");
    }
    SongBenchResult result;
    result.endpoint = endpoint;
    std::vector<double> elapsedValues;
    elapsedValues.reserve(runCount);
    std::set<std::string> sha256Values;
    std::vector<std::set<std::string>> sha256ValuesByScore(scoreCount);
    auto benchmarkStartTime = std::chrono::steady_clock::now();
    for (size_t runIndex = 0; runIndex < runCount; runIndex++) {
        auto startTime = std::chrono::steady_clock::now();
        std::vector<uint8_t> outputBytes = runOperation(runIndex);
        auto endTime = std::chrono::steady_clock::now();
        elapsedValues.push_back(getElapsedMilliseconds(startTime, endTime));
        std::string sha256Value = createSha256Hex(outputBytes.data(), outputBytes.size());
        sha256Values.insert(sha256Value);
        sha256ValuesByScore[runIndex % scoreCount].insert(sha256Value);
        if (runIndex == 0) {
            result.bytes = outputBytes.size();
            result.firstSha256 = sha256Value;
        }
        result.lastSha256 = sha256Value;
    }
    auto benchmarkEndTime = std::chrono::steady_clock::now();
    result.totalElapsedMilliseconds = getElapsedMilliseconds(benchmarkStartTime, benchmarkEndTime);
    result.firstMilliseconds = elapsedValues.front();
    result.minimumMilliseconds = *std::min_element(elapsedValues.begin(), elapsedValues.end());
    result.maximumMilliseconds = *std::max_element(elapsedValues.begin(), elapsedValues.end());
    result.averageMilliseconds = std::accumulate(elapsedValues.begin(), elapsedValues.end(), 0.0) / static_cast<double>(elapsedValues.size());
    result.throughputPerSecond = result.totalElapsedMilliseconds > 0.0 ? static_cast<double>(runCount) * 1000.0 / result.totalElapsedMilliseconds : 0.0;
    result.activeWorkers = 1;
    result.completedRequestsByWorker = {runCount};
    result.averageMillisecondsByWorker = {result.averageMilliseconds};
    if (elapsedValues.size() > 1) {
        result.averageWarmMilliseconds = std::accumulate(elapsedValues.begin() + 1, elapsedValues.end(), 0.0) / static_cast<double>(elapsedValues.size() - 1);
    } else {
        result.averageWarmMilliseconds = result.averageMilliseconds;
    }
    result.uniqueShaCount = sha256Values.size();
    result.repeatStatus = "exact";
    for (const std::set<std::string> &scoreShaValues : sha256ValuesByScore) {
        if (scoreShaValues.size() > 1) {
            result.repeatStatus = "different";
            break;
        }
    }
    return result;
}

static SongBenchResult benchmarkSongOperationParallel(const std::string &endpoint, size_t runCount, size_t scoreCount, std::vector<std::unique_ptr<RuntimeHolder>> &runtimeHolders, const std::function<std::vector<uint8_t>(RuntimeState &, size_t)> &runOperation) {
    if (runCount == 0) {
        throw std::runtime_error("runCount は 1 以上が必要です");
    }
    if (runtimeHolders.empty()) {
        throw std::runtime_error("runtime worker がありません");
    }
    SongBenchResult result;
    result.endpoint = endpoint;
    const size_t workerCount = runtimeHolders.size();
    std::vector<double> elapsedValues(runCount, 0.0);
    std::vector<size_t> outputByteSizes(runCount, 0);
    std::vector<std::string> sha256ByRun(runCount);
    result.completedRequestsByWorker.assign(workerCount, 0);
    result.averageMillisecondsByWorker.assign(workerCount, 0.0);
    std::atomic<size_t> nextRunIndex{0};
    std::atomic<bool> hasBenchError{false};
    std::string firstErrorMessage;
    std::mutex firstErrorMessageMutex;
    std::vector<std::thread> benchmarkThreads;
    benchmarkThreads.reserve(workerCount);
    auto benchmarkStartTime = std::chrono::steady_clock::now();
    for (size_t workerIndex = 0; workerIndex < workerCount; workerIndex++) {
        benchmarkThreads.emplace_back([&, workerIndex]() {
            RuntimeState &workerRuntimeState = *runtimeHolders[workerIndex]->runtimeState;
            size_t workerCompletedRequests = 0;
            double workerTotalElapsedMilliseconds = 0.0;
            while (!hasBenchError.load()) {
                size_t runIndex = nextRunIndex.fetch_add(1);
                if (runIndex >= runCount) {
                    break;
                }
                try {
                    auto startTime = std::chrono::steady_clock::now();
                    std::vector<uint8_t> outputBytes = runOperation(workerRuntimeState, runIndex);
                    auto endTime = std::chrono::steady_clock::now();
                    elapsedValues[runIndex] = getElapsedMilliseconds(startTime, endTime);
                    outputByteSizes[runIndex] = outputBytes.size();
                    sha256ByRun[runIndex] = createSha256Hex(outputBytes.data(), outputBytes.size());
                    workerCompletedRequests++;
                    workerTotalElapsedMilliseconds += elapsedValues[runIndex];
                } catch (const std::exception &caughtException) {
                    if (!hasBenchError.exchange(true)) {
                        std::lock_guard<std::mutex> firstErrorLock(firstErrorMessageMutex);
                        firstErrorMessage = caughtException.what();
                    }
                } catch (...) {
                    if (!hasBenchError.exchange(true)) {
                        std::lock_guard<std::mutex> firstErrorLock(firstErrorMessageMutex);
                        firstErrorMessage = endpoint + " の bench 実行中に不明なエラーが発生しました";
                    }
                }
            }
            result.completedRequestsByWorker[workerIndex] = workerCompletedRequests;
            if (workerCompletedRequests > 0) {
                result.averageMillisecondsByWorker[workerIndex] = workerTotalElapsedMilliseconds / static_cast<double>(workerCompletedRequests);
            }
        });
    }
    for (std::thread &benchmarkThread : benchmarkThreads) {
        benchmarkThread.join();
    }
    if (hasBenchError.load()) {
        throw std::runtime_error(firstErrorMessage);
    }
    auto benchmarkEndTime = std::chrono::steady_clock::now();
    std::set<std::string> sha256Values(sha256ByRun.begin(), sha256ByRun.end());
    std::vector<std::set<std::string>> sha256ValuesByScore(scoreCount);
    for (size_t runIndex = 0; runIndex < runCount; runIndex++) {
        sha256ValuesByScore[runIndex % scoreCount].insert(sha256ByRun[runIndex]);
    }
    result.totalElapsedMilliseconds = getElapsedMilliseconds(benchmarkStartTime, benchmarkEndTime);
    result.firstMilliseconds = elapsedValues.front();
    result.minimumMilliseconds = *std::min_element(elapsedValues.begin(), elapsedValues.end());
    result.maximumMilliseconds = *std::max_element(elapsedValues.begin(), elapsedValues.end());
    result.averageMilliseconds = std::accumulate(elapsedValues.begin(), elapsedValues.end(), 0.0) / static_cast<double>(elapsedValues.size());
    if (elapsedValues.size() > 1) {
        result.averageWarmMilliseconds = std::accumulate(elapsedValues.begin() + 1, elapsedValues.end(), 0.0) / static_cast<double>(elapsedValues.size() - 1);
    } else {
        result.averageWarmMilliseconds = result.averageMilliseconds;
    }
    result.throughputPerSecond = result.totalElapsedMilliseconds > 0.0 ? static_cast<double>(runCount) * 1000.0 / result.totalElapsedMilliseconds : 0.0;
    result.activeWorkers = countActiveBenchWorkers(result.completedRequestsByWorker);
    result.bytes = outputByteSizes.front();
    result.firstSha256 = sha256ByRun.front();
    result.lastSha256 = sha256ByRun.back();
    result.uniqueShaCount = sha256Values.size();
    result.repeatStatus = "exact";
    for (const std::set<std::string> &scoreShaValues : sha256ValuesByScore) {
        if (scoreShaValues.size() > 1) {
            result.repeatStatus = "different";
            break;
        }
    }
    return result;
}

int runSongBenchCommand(const CliOptions &cliOptions) {
    std::vector<std::string> scoreTexts = getBenchScoreTexts(cliOptions);
    std::vector<std::string> frameAudioQueryTexts = getBenchFrameAudioQueryTexts(cliOptions, scoreTexts.size());
    std::vector<size_t> scoreCompletedRuns = createRoundRobinCounts(cliOptions.runs, scoreTexts.size());
    std::vector<SongBenchResult> results;
    if (cliOptions.workers == 1) {
        RuntimeState *runtimeState = new RuntimeState(createRuntimeState(cliOptions.runtimePaths, cliOptions.shouldPreload));
        if (cliOptions.frameAudioQueryPath.empty()) {
            frameAudioQueryTexts.reserve(scoreTexts.size());
            for (const std::string &scoreText : scoreTexts) {
                frameAudioQueryTexts.push_back(createSingFrameAudioQuery(*runtimeState, scoreText, cliOptions.speaker));
            }
        }
        results.push_back(benchmarkSongOperation("sing_frame_audio_query", cliOptions.runs, scoreTexts.size(), [runtimeState, &scoreTexts, &cliOptions](size_t runIndex) {
            return makeBodyBytes(createSingFrameAudioQuery(*runtimeState, scoreTexts[runIndex % scoreTexts.size()], cliOptions.speaker));
        }));
        results.push_back(benchmarkSongOperation("sing_frame_f0", cliOptions.runs, scoreTexts.size(), [runtimeState, &scoreTexts, &frameAudioQueryTexts, &cliOptions](size_t runIndex) {
            size_t scoreIndex = runIndex % scoreTexts.size();
            return makeBodyBytes(createSingFrameF0(*runtimeState, scoreTexts[scoreIndex], frameAudioQueryTexts[scoreIndex], cliOptions.speaker));
        }));
        results.push_back(benchmarkSongOperation("sing_frame_volume", cliOptions.runs, scoreTexts.size(), [runtimeState, &scoreTexts, &frameAudioQueryTexts, &cliOptions](size_t runIndex) {
            size_t scoreIndex = runIndex % scoreTexts.size();
            return makeBodyBytes(createSingFrameVolume(*runtimeState, scoreTexts[scoreIndex], frameAudioQueryTexts[scoreIndex], cliOptions.speaker));
        }));
        results.push_back(benchmarkSongOperation("frame_synthesis", cliOptions.runs, scoreTexts.size(), [runtimeState, &scoreTexts, &frameAudioQueryTexts, &cliOptions](size_t runIndex) {
            return synthesizeFrameAudioQuery(*runtimeState, frameAudioQueryTexts[runIndex % scoreTexts.size()], cliOptions.speaker);
        }));
    } else {
        std::map<uint32_t, size_t> sharedLoadedStyleCounts;
        std::mutex sharedLoadedStylesMutex;
        std::map<uint32_t, uint64_t> sharedStyleUnloadGenerations;
        std::mutex sharedStyleUnloadMutex;
        std::mutex sharedUserDictMutex;
        std::mutex sharedPresetMutex;
        std::mutex sharedSettingMutex;
        std::mutex sharedLibraryMutex;
        std::vector<std::unique_ptr<RuntimeHolder>> runtimeHolders = createRuntimeHolders(cliOptions, &sharedLoadedStyleCounts, &sharedLoadedStylesMutex, &sharedStyleUnloadGenerations, &sharedStyleUnloadMutex, &sharedUserDictMutex, &sharedPresetMutex, &sharedSettingMutex, &sharedLibraryMutex);
        if (cliOptions.frameAudioQueryPath.empty()) {
            frameAudioQueryTexts.reserve(scoreTexts.size());
            for (const std::string &scoreText : scoreTexts) {
                frameAudioQueryTexts.push_back(createSingFrameAudioQuery(*runtimeHolders.front()->runtimeState, scoreText, cliOptions.speaker));
            }
        }
        results.push_back(benchmarkSongOperationParallel("sing_frame_audio_query", cliOptions.runs, scoreTexts.size(), runtimeHolders, [&scoreTexts, &cliOptions](RuntimeState &runtimeState, size_t runIndex) {
            return makeBodyBytes(createSingFrameAudioQuery(runtimeState, scoreTexts[runIndex % scoreTexts.size()], cliOptions.speaker));
        }));
        results.push_back(benchmarkSongOperationParallel("sing_frame_f0", cliOptions.runs, scoreTexts.size(), runtimeHolders, [&scoreTexts, &frameAudioQueryTexts, &cliOptions](RuntimeState &runtimeState, size_t runIndex) {
            size_t scoreIndex = runIndex % scoreTexts.size();
            return makeBodyBytes(createSingFrameF0(runtimeState, scoreTexts[scoreIndex], frameAudioQueryTexts[scoreIndex], cliOptions.speaker));
        }));
        results.push_back(benchmarkSongOperationParallel("sing_frame_volume", cliOptions.runs, scoreTexts.size(), runtimeHolders, [&scoreTexts, &frameAudioQueryTexts, &cliOptions](RuntimeState &runtimeState, size_t runIndex) {
            size_t scoreIndex = runIndex % scoreTexts.size();
            return makeBodyBytes(createSingFrameVolume(runtimeState, scoreTexts[scoreIndex], frameAudioQueryTexts[scoreIndex], cliOptions.speaker));
        }));
        results.push_back(benchmarkSongOperationParallel("frame_synthesis", cliOptions.runs, scoreTexts.size(), runtimeHolders, [&scoreTexts, &frameAudioQueryTexts, &cliOptions](RuntimeState &runtimeState, size_t runIndex) {
            return synthesizeFrameAudioQuery(runtimeState, frameAudioQueryTexts[runIndex % scoreTexts.size()], cliOptions.speaker);
        }));
    }
    std::cout << "mode\tendpoint\truns\tworkers\tscore_mode\tscore_count\tscore_utf8_bytes\tscore_completed_runs\telapsed_ms\tthroughput_rps\tfirst_ms\tavg_ms\tavg_warm_ms\tmin_ms\tmax_ms\tactive_workers\tworker_completed_runs\tworker_avg_ms\tbytes\trepeat_status\tunique_sha_count\tsha256_first\tsha256_last\n";
    std::cout << std::fixed << std::setprecision(3);
    std::string modeText = getNativeSingTeacherModeLabel(cliOptions);
    for (const SongBenchResult &result : results) {
        std::cout << modeText << "\t"
                  << result.endpoint << "\t"
                  << cliOptions.runs << "\t"
                  << cliOptions.workers << "\t"
                  << getRoundRobinModeText(scoreTexts.size()) << "\t"
                  << scoreTexts.size() << "\t"
                  << joinTextByteLengths(scoreTexts) << "\t"
                  << joinSizeValues(scoreCompletedRuns) << "\t"
                  << result.totalElapsedMilliseconds << "\t"
                  << result.throughputPerSecond << "\t"
                  << result.firstMilliseconds << "\t"
                  << result.averageMilliseconds << "\t"
                  << result.averageWarmMilliseconds << "\t"
                  << result.minimumMilliseconds << "\t"
                  << result.maximumMilliseconds << "\t"
                  << result.activeWorkers << "\t"
                  << joinSizeValues(result.completedRequestsByWorker) << "\t"
                  << joinDoubleValues(result.averageMillisecondsByWorker) << "\t"
                  << result.bytes << "\t"
                  << result.repeatStatus << "\t"
                  << result.uniqueShaCount << "\t"
                  << result.firstSha256 << "\t"
                  << result.lastSha256 << "\n";
    }
    return 0;
}

struct HttpBenchResponse {
    int statusCode = 0;
    size_t responseBytes = 0;
    size_t bodyBytes = 0;
    double elapsedMilliseconds = 0.0;
    double firstResponseMilliseconds = 0.0;
    double firstBodyMilliseconds = 0.0;
    std::vector<uint8_t> bodyBytesData;
    std::string contentType;
};

struct ApiSessionLineSpec {
    std::string httpPath;
    std::vector<std::string> inputFields;
};

#include "cli_bench_http.hpp"
#include "cli_bench_api_session.hpp"

std::string FetchSpeakers(const CliOptions &Options, const std::string &Type) {
    const char *Path = Type == "talk" ? "/speakers" : "/singers";
    auto Response = requestHttpBenchTarget(Options, Path);
    if (Response.statusCode != 200) throw std::runtime_error(std::string("話者一覧を取得できません: ") + Path + " HTTP " + std::to_string(Response.statusCode));
    return std::string(Response.bodyBytesData.begin(), Response.bodyBytesData.end());
}
