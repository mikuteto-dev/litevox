#pragma once

#include "cli_shared.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

std::vector<std::unique_ptr<RuntimeHolder>> createRuntimeHolders(const CliOptions &cliOptions, std::map<uint32_t, size_t> *sharedLoadedStyleCounts, std::mutex *sharedLoadedStylesMutex, std::map<uint32_t, uint64_t> *sharedStyleUnloadGenerations, std::mutex *sharedStyleUnloadMutex, std::mutex *sharedUserDictMutex, std::mutex *sharedPresetMutex, std::mutex *sharedSettingMutex, std::mutex *sharedLibraryMutex);
int runBenchCommand(const CliOptions &cliOptions);
int runSongBenchCommand(const CliOptions &cliOptions);
int runHttpSongBenchCommand(const CliOptions &cliOptions);
int runHttpBenchCommand(const CliOptions &cliOptions);
int runApiSessionCommand(const CliOptions &cliOptions);
std::string FetchSpeakers(const CliOptions &Options, const std::string &Type);
