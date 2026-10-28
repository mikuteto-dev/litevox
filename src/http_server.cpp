#include "http_server.hpp"

#include "api_schema.hpp"
#include "json_utility.hpp"
#include "socket_compat.hpp"
#include "setting_store.hpp"
#include "streaming_audio.hpp"
#include "utility.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <zlib.h>

namespace fs = std::filesystem;

struct HttpRequest {
    std::string method;
    std::string httpVersion;
    std::string path;
    std::map<std::string, std::string> queryParameters;
    std::map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int statusCode = 200;
    std::string contentType = "application/json";
    std::map<std::string, std::string> headers;
    std::vector<uint8_t> bodyBytes;
    bool isChunked = false;
    size_t chunkBytes = 32768;
    std::function<void(const std::function<void(const uint8_t *, size_t)> &)> writeStream;
};

class HttpStatusError : public std::runtime_error {
public:
    HttpStatusError(int createdStatusCode, const std::string &messageText) : std::runtime_error(messageText), statusCode(createdStatusCode) {}
    HttpStatusError(int createdStatusCode, const std::string &messageText, const std::string &createdResponseBodyText) : std::runtime_error(messageText), statusCode(createdStatusCode), responseBodyText(createdResponseBodyText) {}
    int statusCode;
    std::string responseBodyText;
};

class HttpClientDisconnected : public std::runtime_error {
public:
    HttpClientDisconnected() : std::runtime_error("client disconnected") {}
};

enum class JsonBodyRoot {
    Object,
    Array
};

struct QueuedClientSocket {
    LitevoxSocket clientSocket = static_cast<LitevoxSocket>(-1);
    std::chrono::steady_clock::time_point queuedTime;
};

struct ClientSocketQueue {
    std::mutex mutex;
    std::condition_variable condition;
    std::queue<QueuedClientSocket> sockets;
    RuntimeRequestQueueMetrics requestQueueMetrics;
};

struct ZipFileEntry {
    std::string name;
    std::vector<uint8_t> bodyBytes;
};

static bool startsWithText(const std::string &text, const std::string &prefixText);
static bool isRuntimeCorsRequestAllowed(const HttpRequest &request);

static uint64_t getElapsedMillisecondsCount(std::chrono::steady_clock::time_point startTime, std::chrono::steady_clock::time_point endTime) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
}

#include "http_server_request.hpp"
#include "http_server_response.hpp"

static std::string createHttpOpenApiJson(RuntimeState &runtimeState) {
    fs::path schemaPath = runtimeState.engineManifestAssetDirectory.parent_path() / "openapi.json";
    if (fs::exists(schemaPath)) {
        return readTextFile(schemaPath);
    }
    return createOpenApiJson();
}

static std::string formatMultiSynthesisFileName(size_t audioQueryIndex) {
    std::ostringstream fileNameStream;
    fileNameStream << std::setw(3) << std::setfill('0') << (audioQueryIndex + 1) << ".wav";
    return fileNameStream.str();
}

static HttpResponse makeMultiSynthesisResponse(RuntimeState &runtimeState, const HttpRequest &request) {
    std::vector<std::string> audioQueries = splitJsonObjects(request.body);
    if (audioQueries.empty()) {
        throw HttpStatusError(400, "audio query array が必要です");
    }
    uint32_t styleId = parseSpeakerParameter(request);
    std::vector<ZipFileEntry> zipFileEntries;
    zipFileEntries.reserve(audioQueries.size());
    for (size_t audioQueryIndex = 0; audioQueryIndex < audioQueries.size(); audioQueryIndex++) {
        validateRequestAudioQuery(runtimeState, audioQueries[audioQueryIndex]);
        zipFileEntries.push_back(ZipFileEntry{formatMultiSynthesisFileName(audioQueryIndex), synthesizeAudioQuery(runtimeState, audioQueries[audioQueryIndex], styleId)});
    }
    HttpResponse response;
    response.statusCode = 200;
    response.contentType = "application/zip";
    response.bodyBytes = createStoredZipBytes(zipFileEntries);
    return response;
}

static HttpResponse routeRequest(RuntimeState &runtimeState, const HttpRequest &request, const RuntimeRequestQueueMetrics *requestQueueMetrics) {
    if (request.method == "OPTIONS") {
        return makeNoContentResponse();
    }
    syncInstalledVoiceLibraries(runtimeState);
    if (request.path == "/") {
        requireMethod(request, {"GET"});
        return makePortalPageResponse();
    }
    if (request.path == "/docs") {
        requireMethod(request, {"GET"});
        return makeDocsPageResponse();
    }
    if (request.path == "/redoc") {
        requireMethod(request, {"GET"});
        return makeRedocPageResponse();
    }
    if (request.path == "/docs/oauth2-redirect") {
        requireMethod(request, {"GET"});
        return makeOauth2RedirectResponse();
    }
    if (request.path == "/health") {
        requireMethod(request, {"GET"});
        return makeJsonResponse("{\"status\":\"ok\"}");
    }
    if (request.path == "/request_queue") {
        requireMethod(request, {"GET"});
        if (!requestQueueMetrics) {
            RuntimeRequestQueueMetrics emptyRequestQueueMetrics;
            return makeJsonResponse(createRuntimeRequestQueueMetricsJson(emptyRequestQueueMetrics));
        }
        return makeJsonResponse(createRuntimeRequestQueueMetricsJson(*requestQueueMetrics));
    }
    if (request.path == "/version") {
        requireMethod(request, {"GET"});
        return makeJsonResponse("\"0.25.2\"");
    }
    if (request.path == "/core_versions") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(std::string("[\"") + getRuntimeCoreVersion(runtimeState) + "\"]");
    }
    if (request.path == "/runtime_info") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(createRuntimeInfoJson(runtimeState));
    }
    if (request.path == "/model_assets") {
        requireMethod(request, {"GET"});
        return makeTextResponse(200, "text/tab-separated-values; charset=utf-8", createRuntimeModelAssetTable(runtimeState));
    }
    if (request.path == "/model_cache") {
        requireMethod(request, {"GET", "POST"});
        if (request.method == "POST") {
            loadRuntimeModelSessionCache(runtimeState);
        }
        return makeTextResponse(200, "text/tab-separated-values; charset=utf-8", createRuntimeModelSessionCacheSummary(runtimeState));
    }
    if (request.path == "/speakers") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(createSpeakersJson(runtimeState.combinedMetasJson, getCoreBackendCapabilities(runtimeState.coreBackend).supportsMorphing, createCharacterSupportedFeaturesJsons(getCharacterResources(runtimeState))));
    }
    if (request.path == "/singers") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(createSingersJson(runtimeState.combinedMetasJson, createCharacterSupportedFeaturesJsons(getCharacterResources(runtimeState))));
    }
    if (request.path == "/speaker_info") {
        requireMethod(request, {"GET"});
        std::string speakerUuid = getRequiredQueryParameter(request, "speaker_uuid");
        validateOptionalResourceFormatParameter(request);
        try {
            return makeJsonResponse(createRuntimeSpeakerInfoJson(runtimeState, speakerUuid, "talk", getOptionalQueryParameter(request, "resource_format", "base64"), createResourceBaseUrl(request)));
        } catch (const std::runtime_error &speakerInfoError) {
            if (startsWithText(speakerInfoError.what(), "resource_format")) {
                throw HttpStatusError(400, speakerInfoError.what());
            }
            throw HttpStatusError(404, speakerInfoError.what());
        }
    }
    if (request.path == "/singer_info") {
        requireMethod(request, {"GET"});
        std::string speakerUuid = getRequiredQueryParameter(request, "speaker_uuid");
        validateOptionalResourceFormatParameter(request);
        try {
            return makeJsonResponse(createRuntimeSpeakerInfoJson(runtimeState, speakerUuid, "sing", getOptionalQueryParameter(request, "resource_format", "base64"), createResourceBaseUrl(request)));
        } catch (const std::runtime_error &speakerInfoError) {
            if (startsWithText(speakerInfoError.what(), "resource_format")) {
                throw HttpStatusError(400, speakerInfoError.what());
            }
            throw HttpStatusError(404, speakerInfoError.what());
        }
    }
    if (startsWithText(request.path, "/_resources/")) {
        requireMethod(request, {"GET"});
        std::string resourceHash = request.path.substr(std::string("/_resources/").size());
        try {
            HttpResponse response;
            response.statusCode = 200;
            response.contentType = getRuntimeCharacterResourceContentType(runtimeState, resourceHash);
            response.bodyBytes = readRuntimeCharacterResource(runtimeState, resourceHash);
            response.headers["Cache-Control"] = "max-age=2592000";
            return response;
        } catch (const std::runtime_error &resourceError) {
            throw HttpStatusError(404, resourceError.what());
        }
    }
    if (request.path == "/models" || request.path == "/styles") {
        requireMethod(request, {"GET"});
        return makeTextResponse(200, "text/tab-separated-values; charset=utf-8", createModelTable(runtimeState));
    }
    if (request.path == "/engine_manifest") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(getRuntimeManifestJson(runtimeState));
    }
    if (request.path == "/supported_devices") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(createSupportedDevicesJson(runtimeState));
    }
    if (request.path == "/setting") {
        requireMethod(request, {"GET", "POST"});
        if (request.method == "GET") {
            return makeTextResponse(200, "text/html; charset=utf-8", createSettingPageHtml(runtimeState));
        }
        std::map<std::string, std::string> formParameters = parseFormParameters(request.body);
        requireSettingFormFields(formParameters);
        auto corsPolicyIterator = formParameters.find("cors_policy_mode");
        if (corsPolicyIterator->second != "all" && corsPolicyIterator->second != "localapps") {
            throw createValidationStatusError("cors_policy_mode が不正です", {createBodyEnumValidationDetail("cors_policy_mode", corsPolicyIterator->second, "'all' or 'localapps'")});
        }
        std::string allowOrigin = "";
        auto allowOriginIterator = formParameters.find("allow_origin");
        if (allowOriginIterator != formParameters.end()) {
            allowOrigin = allowOriginIterator->second;
        }
        try {
            updateSetting(runtimeState, corsPolicyIterator->second, allowOrigin);
        } catch (const std::runtime_error &settingError) {
            throw HttpStatusError(400, settingError.what());
        }
        return makeNoContentResponse();
    }
    if (request.path == "/downloadable_libraries") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(createDownloadableLibrariesJson(runtimeState));
    }
    if (request.path == "/installed_libraries") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(createInstalledLibrariesJson(runtimeState));
    }
    if (startsWithText(request.path, "/install_library/")) {
        requireMethod(request, {"POST"});
        std::string libraryUuid = request.path.substr(std::string("/install_library/").size());
        try {
            installVoiceLibrary(runtimeState, libraryUuid, std::vector<uint8_t>(request.body.begin(), request.body.end()));
        } catch (const std::runtime_error &libraryError) {
            throw HttpStatusError(422, libraryError.what());
        }
        return makeNoContentResponse();
    }
    if (startsWithText(request.path, "/uninstall_library/")) {
        requireMethod(request, {"POST"});
        std::string libraryUuid = request.path.substr(std::string("/uninstall_library/").size());
        try {
            uninstallVoiceLibrary(runtimeState, libraryUuid);
        } catch (const std::runtime_error &libraryError) {
            if (startsWithText(libraryError.what(), "標準ライブラリ")) {
                throw HttpStatusError(403, libraryError.what());
            }
            throw HttpStatusError(404, libraryError.what());
        }
        return makeNoContentResponse();
    }
    if (request.path == "/presets") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(createPresetsJson(runtimeState));
    }
    if (request.path == "/add_preset") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Object);
        requirePresetBodyFields(request);
        try {
            return makeJsonResponse(std::to_string(addPreset(runtimeState, request.body)));
        } catch (const std::runtime_error &presetError) {
            throw HttpStatusError(400, presetError.what());
        }
    }
    if (request.path == "/update_preset") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Object);
        requirePresetBodyFields(request);
        try {
            return makeJsonResponse(std::to_string(updatePreset(runtimeState, request.body)));
        } catch (const std::runtime_error &presetError) {
            std::string presetErrorText = presetError.what();
            if (presetErrorText.find("プリセットが見つかりません") == 0) {
                throw createDetailStatusError(422, "更新先のプリセットが存在しません");
            }
            throw HttpStatusError(400, presetErrorText);
        }
    }
    if (request.path == "/delete_preset") {
        requireMethod(request, {"POST"});
        int32_t presetId = parseIntegerParameter(request, "id");
        try {
            deletePreset(runtimeState, presetId);
        } catch (const std::runtime_error &presetError) {
            std::string presetErrorText = presetError.what();
            if (presetErrorText.find("プリセットが見つかりません") == 0) {
                throw createDetailStatusError(422, "削除対象のプリセットが存在しません");
            }
            throw HttpStatusError(400, presetErrorText);
        }
        return makeNoContentResponse();
    }
    if (request.path == "/audio_query_from_preset") {
        requireMethod(request, {"POST"});
        reloadUserDict(runtimeState);
        requireQueryParameters(request, {"text", "preset_id"});
        try {
            return makeJsonResponse(createAudioQueryFromPreset(runtimeState, getTextParameter(request), parseIntegerParameter(request, "preset_id")));
        } catch (const std::runtime_error &presetError) {
            std::string presetErrorText = presetError.what();
            if (presetErrorText.find("プリセットが見つかりません") == 0) {
                throw createDetailStatusError(422, "該当するプリセットIDが見つかりません");
            }
            throw HttpStatusError(400, presetErrorText);
        }
    }
    if (request.path == "/validate_kana") {
        requireMethod(request, {"POST"});
        std::string text = getTextParameter(request);
        try {
            validateKana(runtimeState, text);
        } catch (const std::runtime_error &kanaError) {
            throw createParseKanaBadRequestErrorFromMessage(kanaError.what());
        }
        return makeJsonResponse("true");
    }
    if (request.path == "/connect_waves") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Array);
        return makeConnectedWavesResponse(request);
    }
    if (request.path == "/multi_synthesis") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Array);
        return makeMultiSynthesisResponse(runtimeState, request);
    }
    if (request.path == "/sing_frame_audio_query") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Object);
        requireScoreBodyFields(request);
        if (!canCoreBackendSing(runtimeState.coreBackend)) {
            return makeUnsupportedFeatureResponse(request.path);
        }
        return makeJsonResponse(createSingFrameAudioQuery(runtimeState, request.body, parseSpeakerParameter(request)));
    }
    if (request.path == "/sing_frame_f0") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Object);
        requireSingFrameRequestBodyFields(request);
        if (!canCoreBackendSing(runtimeState.coreBackend)) {
            return makeUnsupportedFeatureResponse(request.path);
        }
        std::string scoreJson = getJsonObjectBodyField(request, "score");
        std::string frameAudioQueryJson = getJsonObjectBodyField(request, "frame_audio_query");
        return makeJsonResponse(createSingFrameF0(runtimeState, scoreJson, frameAudioQueryJson, parseSpeakerParameter(request)));
    }
    if (request.path == "/sing_frame_volume") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Object);
        requireSingFrameRequestBodyFields(request);
        if (!canCoreBackendSing(runtimeState.coreBackend)) {
            return makeUnsupportedFeatureResponse(request.path);
        }
        std::string scoreJson = getJsonObjectBodyField(request, "score");
        std::string frameAudioQueryJson = getJsonObjectBodyField(request, "frame_audio_query");
        return makeJsonResponse(createSingFrameVolume(runtimeState, scoreJson, frameAudioQueryJson, parseSpeakerParameter(request)));
    }
    if (request.path == "/openapi.json") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(createHttpOpenApiJson(runtimeState));
    }
    if (request.path == "/user_dict") {
        requireMethod(request, {"GET"});
        reloadUserDict(runtimeState);
        return makeJsonResponse(createUserDictJson(runtimeState));
    }
    if (request.path == "/import_user_dict") {
        requireMethod(request, {"POST"});
        requireImportUserDictRequestFields(request);
        bool shouldOverride = parseBooleanParameter(request, "override");
        try {
            reloadUserDict(runtimeState);
            importUserDictJson(runtimeState, request.body, shouldOverride);
        } catch (const std::runtime_error &userDictError) {
            throw HttpStatusError(400, userDictError.what());
        }
        return makeNoContentResponse();
    }
    if (request.path == "/user_dict_word") {
        requireMethod(request, {"POST"});
        reloadUserDict(runtimeState);
        if (request.method == "POST") {
            requireQueryParameters(request, {"surface", "pronunciation", "accent_type"});
            validateOptionalUserDictWordTypeParameter(request);
            std::string surface = getRequiredQueryParameter(request, "surface");
            std::string pronunciation = getRequiredQueryParameter(request, "pronunciation");
            uintptr_t accentType = static_cast<uintptr_t>(parseIntegerParameter(request, "accent_type"));
            VoicevoxUserDictWordType wordType = getUserDictWordTypeParameter(request);
            uint32_t priority = getUserDictPriorityParameter(request);
            try {
                std::string wordUuid = addUserDictWord(
                    runtimeState,
                    surface,
                    pronunciation,
                    accentType,
                    wordType,
                    priority);
                return makeJsonResponse(jsonQuote(wordUuid));
            } catch (const std::runtime_error &userDictError) {
                throw HttpStatusError(400, userDictError.what());
            }
        }
    }
    if (startsWithText(request.path, "/user_dict_word/")) {
        requireMethod(request, {"PUT", "DELETE"});
        reloadUserDict(runtimeState);
        std::string wordUuid = request.path.substr(std::string("/user_dict_word/").size());
        if (request.method == "PUT") {
            requireQueryParameters(request, {"surface", "pronunciation", "accent_type"});
            validateOptionalUserDictWordTypeParameter(request);
            std::string surface = getRequiredQueryParameter(request, "surface");
            std::string pronunciation = getRequiredQueryParameter(request, "pronunciation");
            uintptr_t accentType = static_cast<uintptr_t>(parseIntegerParameter(request, "accent_type"));
            VoicevoxUserDictWordType wordType = getUserDictWordTypeParameter(request);
            uint32_t priority = getUserDictPriorityParameter(request);
            try {
                updateUserDictWord(
                    runtimeState,
                    wordUuid,
                    surface,
                    pronunciation,
                    accentType,
                    wordType,
                    priority);
            } catch (const std::runtime_error &userDictError) {
                throw HttpStatusError(400, userDictError.what());
            }
            return makeNoContentResponse();
        }
        if (request.method == "DELETE") {
            try {
                removeUserDictWord(runtimeState, wordUuid);
            } catch (const std::runtime_error &userDictError) {
                throw createDetailStatusError(422, "IDに該当するワードが見つかりませんでした");
            }
            return makeNoContentResponse();
        }
    }
    if (request.path == "/is_initialized_speaker") {
        requireMethod(request, {"GET"});
        return makeJsonResponse(isStyleLoaded(runtimeState, parseSpeakerParameter(request)) ? "true" : "false");
    }
    if (request.path == "/loaded_models") {
        requireMethod(request, {"GET"});
        return makeTextResponse(200, "text/tab-separated-values; charset=utf-8", createLoadedModelTable(runtimeState));
    }
    if (request.path == "/initialize_speaker") {
        requireMethod(request, {"POST"});
        ensureStyleLoaded(runtimeState, parseSpeakerParameter(request));
        return makeNoContentResponse();
    }
    if (request.path == "/unload_speaker") {
        requireMethod(request, {"POST"});
        unloadStyleModel(runtimeState, parseSpeakerParameter(request));
        return makeNoContentResponse();
    }
    if (request.path == "/open_jtalk/analyze") {
        requireMethod(request, {"POST"});
        return makeJsonResponse(analyzeText(runtimeState, getTextParameter(request)));
    }
    if (request.path == "/audio_query_from_accent_phrases") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Array);
        return makeJsonResponse(createAudioQueryFromAccentPhrases(runtimeState, request.body));
    }
    if (request.path == "/morphable_targets") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Array);
        if (!getCoreBackendCapabilities(runtimeState.coreBackend).supportsMorphing) {
            return makeUnsupportedFeatureResponse(request.path);
        }
        return makeJsonResponse(createMorphableTargetsJson(runtimeState, request.body));
    }
    if (request.path == "/synthesis_morphing") {
        requireMethod(request, {"POST"});
        requireSynthesisMorphingRequestFields(request);
        validateRequestAudioQuery(runtimeState, request.body);
        if (!getCoreBackendCapabilities(runtimeState.coreBackend).supportsMorphing) {
            return makeUnsupportedFeatureResponse(request.path);
        }
        uint32_t baseStyleId = static_cast<uint32_t>(parseIntegerParameter(request, "base_speaker"));
        uint32_t targetStyleId = static_cast<uint32_t>(parseIntegerParameter(request, "target_speaker"));
        double morphRate = parseDoubleParameter(request, "morph_rate");
        if (morphRate < 0.0 || morphRate > 1.0) {
            throw HttpStatusError(400, "morph_rate は 0.0 以上 1.0 以下が必要です");
        }
        if (!isRuntimeTalkStyle(runtimeState, baseStyleId) || !isRuntimeTalkStyle(runtimeState, targetStyleId)) {
            return makeUnsupportedFeatureResponse(request.path);
        }
        return makeAudioResponse(synthesizeMorphingAudioQuery(runtimeState, request.body, baseStyleId, targetStyleId, morphRate), getAudioStreamFormatParameter(request));
    }
    if (request.path == "/audio_query") {
        requireMethod(request, {"POST"});
        reloadUserDict(runtimeState);
        requireQueryParameters(request, {"text", "speaker"});
        uint32_t styleId = parseSpeakerParameter(request);
        return makeJsonResponse(createAudioQuery(runtimeState, getTextParameter(request), styleId));
    }
    if (request.path == "/accent_phrases") {
        requireMethod(request, {"POST"});
        reloadUserDict(runtimeState);
        requireQueryParameters(request, {"text", "speaker"});
        uint32_t styleId = parseSpeakerParameter(request);
        if (isTruthyParameter(request, "is_kana")) {
            try {
                return makeJsonResponse(createAccentPhrasesFromKana(runtimeState, getTextParameter(request), styleId));
            } catch (const std::runtime_error &kanaError) {
                throw createParseKanaBadRequestErrorFromMessage(kanaError.what());
            }
        }
        return makeJsonResponse(createAccentPhrases(runtimeState, getTextParameter(request), styleId));
    }
    if (request.path == "/mora_data") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Array);
        return makeJsonResponse(replaceMoraData(runtimeState, request.body, parseSpeakerParameter(request)));
    }
    if (request.path == "/mora_length") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Array);
        return makeJsonResponse(replacePhonemeLength(runtimeState, request.body, parseSpeakerParameter(request)));
    }
    if (request.path == "/mora_pitch") {
        requireMethod(request, {"POST"});
        requireJsonBody(request, JsonBodyRoot::Array);
        return makeJsonResponse(replaceMoraPitch(runtimeState, request.body, parseSpeakerParameter(request)));
    }
    if (request.path == "/synthesis") {
        requireMethod(request, {"POST"});
        requireSpeakerParameterAndAudioQueryBodyFields(request);
        validateRequestAudioQuery(runtimeState, request.body);
        return makeAudioResponse(synthesizeAudioQuery(runtimeState, request.body, parseSpeakerParameter(request)), getAudioStreamFormatParameter(request));
    }
    if (request.path == "/frame_synthesis") {
        requireMethod(request, {"POST"});
        requireSpeakerParameterAndFrameAudioQueryBodyFields(request);
        validateRequestFrameAudioQuery(runtimeState, request.body);
        if (!canCoreBackendFrameSynthesis(runtimeState.coreBackend)) {
            return makeUnsupportedFeatureResponse(request.path);
        }
        return makeAudioResponse(synthesizeFrameAudioQuery(runtimeState, request.body, parseSpeakerParameter(request)), getAudioStreamFormatParameter(request));
    }
    if (request.path == "/cancellable_synthesis") {
        requireMethod(request, {"POST"});
        requireSpeakerParameterAndAudioQueryBodyFields(request);
        validateRequestAudioQuery(runtimeState, request.body);
        if (!runtimeState.enableCancellableSynthesis) {
            return makeExperimentalFeatureDisabledResponse();
        }
        return makeCancellableSynthesisResponse(runtimeState, request);
    }
    if (request.path == "/synthesis_stream") {
        requireMethod(request, {"POST"});
        requireSpeakerParameterAndAudioQueryBodyFields(request);
        validateRequestAudioQuery(runtimeState, request.body);
        uint32_t styleId = parseSpeakerParameter(request);
        std::string audioQueryJson = request.body;
        RuntimeAudioStreamOptions streamOptions;
        streamOptions.audioStreamFormat = getAudioStreamFormatParameter(request);
        streamOptions.chunkSamples = getChunkSamplesParameter(request);
        return makeChunkedRuntimeStreamResponse(streamOptions.audioStreamFormat, [&runtimeState, audioQueryJson, styleId, streamOptions](const std::function<void(const uint8_t *, size_t)> &writeChunk) {
            streamAudioQuery(runtimeState, audioQueryJson, styleId, streamOptions, writeChunk);
        });
    }
    if (request.path == "/tts") {
        requireMethod(request, {"GET", "POST"});
        reloadUserDict(runtimeState);
        uint32_t styleId = parseSpeakerParameter(request);
        if (isTruthyParameter(request, "is_kana")) {
            return makeAudioResponse(synthesizeKana(runtimeState, getTextParameter(request), styleId), getAudioStreamFormatParameter(request));
        }
        return makeAudioResponse(synthesizeText(runtimeState, getTextParameter(request), styleId), getAudioStreamFormatParameter(request));
    }
    if (request.path == "/tts_stream") {
        requireMethod(request, {"GET", "POST"});
        reloadUserDict(runtimeState);
        uint32_t styleId = parseSpeakerParameter(request);
        std::string text = getTextParameter(request);
        RuntimeAudioStreamOptions streamOptions;
        streamOptions.audioStreamFormat = getAudioStreamFormatParameter(request);
        streamOptions.chunkSamples = getChunkSamplesParameter(request);
        if (isTruthyParameter(request, "is_kana")) {
            return makeChunkedRuntimeStreamResponse(streamOptions.audioStreamFormat, [&runtimeState, text, styleId, streamOptions](const std::function<void(const uint8_t *, size_t)> &writeChunk) {
                streamKana(runtimeState, text, styleId, streamOptions, writeChunk);
            });
        }
        return makeChunkedRuntimeStreamResponse(streamOptions.audioStreamFormat, [&runtimeState, text, styleId, streamOptions](const std::function<void(const uint8_t *, size_t)> &writeChunk) {
            streamText(runtimeState, text, styleId, streamOptions, writeChunk);
        });
    }
    if (isUnsupportedFeaturePath(request.path)) {
        return makeUnsupportedFeatureResponse(request.path);
    }
    return makeErrorResponse(404, "未実装のエンドポイントです: " + request.path);
}

static bool sendRawBytes(LitevoxSocket clientSocket, const uint8_t *bytePointer, size_t byteCount) {
    const uint8_t *bodyPointer = bytePointer;
    size_t bodyRemaining = byteCount;
    while (bodyRemaining > 0) {
        int sentLength = static_cast<int>(send(clientSocket, bodyPointer, static_cast<int>(bodyRemaining), 0));
        if (sentLength <= 0) {
            return false;
        }
        bodyPointer += sentLength;
        bodyRemaining -= static_cast<size_t>(sentLength);
    }
    return true;
}

static bool sendAllBytes(LitevoxSocket clientSocket, const std::string &headerText, const std::vector<uint8_t> &bodyBytes) {
    const char *headerPointer = headerText.data();
    size_t headerRemaining = headerText.size();
    while (headerRemaining > 0) {
        int sentLength = static_cast<int>(send(clientSocket, headerPointer, static_cast<int>(headerRemaining), 0));
        if (sentLength <= 0) {
            return false;
        }
        headerPointer += sentLength;
        headerRemaining -= static_cast<size_t>(sentLength);
    }
    if (!bodyBytes.empty()) {
        return sendRawBytes(clientSocket, bodyBytes.data(), bodyBytes.size());
    }
    return true;
}

static void sendHttpResponse(LitevoxSocket clientSocket, const HttpResponse &response, bool shouldKeepAlive) {
    std::ostringstream headerStream;
    headerStream << "HTTP/1.1 " << response.statusCode << " " << getStatusText(response.statusCode) << "\r\n";
    headerStream << "Content-Type: " << response.contentType << "\r\n";
    headerStream << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n";
    headerStream << "Access-Control-Allow-Headers: Content-Type\r\n";
    for (const auto &responseHeader : response.headers) {
        if (responseHeader.first == "Access-Control-Allow-Origin" && responseHeader.second.empty()) {
            continue;
        }
        headerStream << responseHeader.first << ": " << responseHeader.second << "\r\n";
    }
    if (response.isChunked) {
        headerStream << "Transfer-Encoding: chunked\r\n";
        headerStream << "Connection: " << (shouldKeepAlive ? "keep-alive" : "close") << "\r\n\r\n";
        std::vector<uint8_t> emptyBody;
        if (!sendAllBytes(clientSocket, headerStream.str(), emptyBody)) {
            return;
        }
        auto writeHttpChunk = [clientSocket](const uint8_t *audioBytes, size_t byteCount) {
            std::ostringstream chunkHeaderStream;
            chunkHeaderStream << std::hex << byteCount << "\r\n";
            if (!sendAllBytes(clientSocket, chunkHeaderStream.str(), std::vector<uint8_t>())) {
                throw HttpClientDisconnected();
            }
            if (!sendRawBytes(clientSocket, audioBytes, byteCount)) {
                throw HttpClientDisconnected();
            }
            const uint8_t lineBreakBytes[] = {'\r', '\n'};
            if (!sendRawBytes(clientSocket, lineBreakBytes, sizeof(lineBreakBytes))) {
                throw HttpClientDisconnected();
            }
        };
        try {
            if (response.writeStream) {
                response.writeStream(writeHttpChunk);
            } else {
                AudioStreamPayload audioStreamPayload;
                audioStreamPayload.contentType = response.contentType;
                audioStreamPayload.audioBytes = response.bodyBytes;
                writeAudioStreamChunks(audioStreamPayload, response.chunkBytes, writeHttpChunk);
            }
        } catch (const HttpClientDisconnected &) {
            return;
        }
        std::vector<uint8_t> endBytes = {'0', '\r', '\n', '\r', '\n'};
        sendAllBytes(clientSocket, "", endBytes);
        return;
    }
    headerStream << "Content-Length: " << response.bodyBytes.size() << "\r\n";
    headerStream << "Connection: " << (shouldKeepAlive ? "keep-alive" : "close") << "\r\n\r\n";
    sendAllBytes(clientSocket, headerStream.str(), response.bodyBytes);
}

static void sendRuntimeHttpResponse(LitevoxSocket clientSocket, RuntimeState &runtimeState, const HttpResponse &response, bool shouldKeepAlive) {
    HttpResponse runtimeResponse = response;
    runtimeResponse.headers["X-LiteVox-Worker-Index"] = std::to_string(runtimeState.workerIndex);
    runtimeResponse.headers["X-LiteVox-Worker-Count"] = std::to_string(runtimeState.workerCount);
    sendHttpResponse(clientSocket, runtimeResponse, shouldKeepAlive);
}

static bool isRuntimeCorsRequestAllowed(const HttpRequest &request) {
    auto originIterator = request.headers.find("origin");
    if (originIterator == request.headers.end() || originIterator->second.empty()) {
        return true;
    }
    return isLiteVoxLocalAppOrigin(originIterator->second);
}

static bool shouldKeepHttpConnectionAlive(const HttpRequest &request) {
    auto connectionIterator = request.headers.find("connection");
    if (connectionIterator != request.headers.end()) {
        std::string connectionValue = lowercaseAscii(connectionIterator->second);
        if (connectionValue.find("close") != std::string::npos) {
            return false;
        }
        if (connectionValue.find("keep-alive") != std::string::npos) {
            return true;
        }
    }
    return request.httpVersion != "HTTP/1.0";
}

static LitevoxSocket openServerSocket(int port) {
    initializeSocketRuntime();
    LitevoxSocket serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (!isValidSocket(serverSocket)) {
        throw std::runtime_error("socket を作成できません");
    }
    int reuseEnabled = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&reuseEnabled), sizeof(reuseEnabled));

    sockaddr_in serverAddress{};
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    serverAddress.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(serverSocket, reinterpret_cast<sockaddr *>(&serverAddress), sizeof(serverAddress)) < 0) {
        closeSocket(serverSocket);
        throw std::runtime_error("bind に失敗しました: " + getSocketErrorText());
    }
    if (listen(serverSocket, 64) < 0) {
        closeSocket(serverSocket);
        throw std::runtime_error("listen に失敗しました: " + getSocketErrorText());
    }
    return serverSocket;
}

static void handleClient(RuntimeState &runtimeState, LitevoxSocket clientSocket, RuntimeRequestQueueMetrics *requestQueueMetrics, std::chrono::steady_clock::time_point queuedTime) {
    std::string pendingRequestText;
    std::chrono::steady_clock::time_point currentQueuedTime = queuedTime;
    while (true) {
        HttpRequest request;
        try {
            if (!readHttpRequest(clientSocket, pendingRequestText, &request)) {
                break;
            }
        } catch (const std::exception &exception) {
            sendHttpResponse(clientSocket, makeErrorResponse(400, exception.what()), false);
            break;
        }
        bool shouldKeepAlive = shouldKeepHttpConnectionAlive(request);
        auto serviceStartTime = std::chrono::steady_clock::now();
        if (requestQueueMetrics) {
            markRuntimeRequestAccepted(*requestQueueMetrics);
            uint64_t waitMilliseconds = getElapsedMillisecondsCount(currentQueuedTime, serviceStartTime);
            markRuntimeRequestStarted(*requestQueueMetrics, waitMilliseconds);
        }
        try {
            if (!isRuntimeCorsRequestAllowed(request)) {
                throw createDetailStatusError(403, "Origin not allowed");
            }
            sendRuntimeHttpResponse(clientSocket, runtimeState, routeRequest(runtimeState, request, requestQueueMetrics), shouldKeepAlive);
        } catch (const HttpStatusError &statusError) {
            sendHttpResponse(clientSocket, makeStatusErrorResponse(statusError), shouldKeepAlive);
        } catch (const std::invalid_argument &invalidArgument) {
            sendHttpResponse(clientSocket, makeErrorResponse(400, invalidArgument.what()), shouldKeepAlive);
        } catch (const std::out_of_range &outOfRange) {
            sendHttpResponse(clientSocket, makeErrorResponse(400, outOfRange.what()), shouldKeepAlive);
        } catch (const std::exception &exception) {
            sendHttpResponse(clientSocket, makeErrorResponse(500, exception.what()), false);
            shouldKeepAlive = false;
        }
        if (requestQueueMetrics) {
            uint64_t serviceMilliseconds = getElapsedMillisecondsCount(serviceStartTime, std::chrono::steady_clock::now());
            markRuntimeRequestCompleted(*requestQueueMetrics, serviceMilliseconds);
        }
        if (!shouldKeepAlive) {
            break;
        }
        currentQueuedTime = std::chrono::steady_clock::now();
    }
    closeSocket(clientSocket);
}

void serve(RuntimeState &runtimeState, int port) {
    LitevoxSocket serverSocket = openServerSocket(port);
    RuntimeRequestQueueMetrics requestQueueMetrics;
    std::cout << "litevox listening on http://127.0.0.1:" << port << "\n";
    while (true) {
        LitevoxSocket clientSocket = accept(serverSocket, nullptr, nullptr);
        if (!isValidSocket(clientSocket)) {
            continue;
        }
        auto queuedTime = std::chrono::steady_clock::now();
        markRuntimeConnectionAccepted(requestQueueMetrics);
        handleClient(runtimeState, clientSocket, &requestQueueMetrics, queuedTime);
    }
}

static void pushClientSocket(ClientSocketQueue &clientSocketQueue, LitevoxSocket clientSocket) {
    {
        std::lock_guard<std::mutex> queueLock(clientSocketQueue.mutex);
        clientSocketQueue.sockets.push(QueuedClientSocket{clientSocket, std::chrono::steady_clock::now()});
        markRuntimeConnectionAccepted(clientSocketQueue.requestQueueMetrics);
        markRuntimeConnectionQueued(clientSocketQueue.requestQueueMetrics);
    }
    clientSocketQueue.condition.notify_one();
}

static QueuedClientSocket waitClientSocket(ClientSocketQueue &clientSocketQueue) {
    std::unique_lock<std::mutex> queueLock(clientSocketQueue.mutex);
    clientSocketQueue.condition.wait(queueLock, [&clientSocketQueue]() {
        return !clientSocketQueue.sockets.empty();
    });
    QueuedClientSocket queuedClientSocket = clientSocketQueue.sockets.front();
    clientSocketQueue.sockets.pop();
    markRuntimeConnectionDequeued(clientSocketQueue.requestQueueMetrics);
    return queuedClientSocket;
}

static void runRuntimeWorkerLoop(RuntimeWorker runtimeWorker, ClientSocketQueue &clientSocketQueue) {
    while (true) {
        QueuedClientSocket queuedClientSocket = waitClientSocket(clientSocketQueue);
        handleClient(*runtimeWorker.runtimeState, queuedClientSocket.clientSocket, &clientSocketQueue.requestQueueMetrics, queuedClientSocket.queuedTime);
    }
}

static std::vector<std::thread> startRuntimeWorkerThreads(const std::vector<RuntimeWorker> &runtimeWorkers, ClientSocketQueue &clientSocketQueue) {
    std::vector<std::thread> workerThreads;
    workerThreads.reserve(runtimeWorkers.size());
    for (RuntimeWorker runtimeWorker : runtimeWorkers) {
        workerThreads.emplace_back([runtimeWorker, &clientSocketQueue]() {
            runRuntimeWorkerLoop(runtimeWorker, clientSocketQueue);
        });
    }
    return workerThreads;
}

static void acceptClientSockets(LitevoxSocket serverSocket, ClientSocketQueue &clientSocketQueue) {
    while (true) {
        LitevoxSocket clientSocket = accept(serverSocket, nullptr, nullptr);
        if (!isValidSocket(clientSocket)) {
            continue;
        }
        pushClientSocket(clientSocketQueue, clientSocket);
    }
}

void serveRuntimePool(std::vector<RuntimeWorker> runtimeWorkers, int port) {
    if (runtimeWorkers.empty()) {
        throw std::runtime_error("worker がありません");
    }
    LitevoxSocket serverSocket = openServerSocket(port);
    ClientSocketQueue clientSocketQueue;
    std::vector<std::thread> workerThreads = startRuntimeWorkerThreads(runtimeWorkers, clientSocketQueue);
    std::cout << "litevox listening on http://127.0.0.1:" << port << " workers=" << runtimeWorkers.size() << "\n";
    acceptClientSockets(serverSocket, clientSocketQueue);
}
