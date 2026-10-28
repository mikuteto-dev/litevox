static HttpResponse makeTextResponse(int statusCode, const std::string &contentType, const std::string &bodyText) {
    HttpResponse response;
    response.statusCode = statusCode;
    response.contentType = contentType;
    response.bodyBytes = makeBodyBytes(bodyText);
    return response;
}

static HttpResponse makeJsonResponse(const std::string &bodyText) {
    return makeTextResponse(200, "application/json", bodyText);
}

static HttpResponse makeNoContentResponse() {
    return makeTextResponse(204, "application/json", "");
}

static HttpResponse makeErrorResponse(int statusCode, const std::string &messageText) {
    std::string escapedText;
    for (char character : messageText) {
        if (character == '"' || character == '\\') {
            escapedText.push_back('\\');
        }
        escapedText.push_back(character);
    }
    return makeTextResponse(statusCode, "application/json", "{\"error\":\"" + escapedText + "\"}");
}

static HttpResponse makeStatusErrorResponse(const HttpStatusError &statusError) {
    if (!statusError.responseBodyText.empty()) {
        return makeTextResponse(statusError.statusCode, "application/json", statusError.responseBodyText);
    }
    return makeErrorResponse(statusError.statusCode, statusError.what());
}

static HttpResponse makeUnsupportedFeatureResponse(const std::string &featureName) {
    return makeErrorResponse(501, featureName + " は現在の LiteVox backend では未対応です");
}

static HttpResponse makeExperimentalFeatureDisabledResponse() {
    return makeTextResponse(404, "application/json", "{\"detail\":\"実験的機能はデフォルトで無効になっています。使用するには引数を指定してください。\"}");
}

static bool isUnsupportedFeaturePath(const std::string &pathText) {
    return pathText == "/sing_frame_audio_query"
        || pathText == "/sing_frame_f0"
        || pathText == "/sing_frame_volume"
        || pathText == "/frame_synthesis";
}

static std::vector<uint32_t> parseJsonIntegerArrayText(const std::string &jsonText) {
    std::vector<uint32_t> numbers;
    uint64_t numberValue = 0;
    bool hasNumber = false;
    for (char character : jsonText) {
        if (character >= '0' && character <= '9') {
            hasNumber = true;
            numberValue = numberValue * 10 + static_cast<uint64_t>(character - '0');
        } else if (hasNumber) {
            numbers.push_back(static_cast<uint32_t>(numberValue));
            numberValue = 0;
            hasNumber = false;
        }
    }
    if (hasNumber) {
        numbers.push_back(static_cast<uint32_t>(numberValue));
    }
    return numbers;
}

static bool isRuntimeTalkStyle(const RuntimeState &runtimeState, uint32_t styleId) {
    for (const VoiceModelRecord &voiceModel : runtimeState.voiceModels) {
        for (const StyleRecord &styleRecord : voiceModel.styles) {
            if (styleRecord.styleId == styleId) {
                return styleRecord.styleType == "talk";
            }
        }
    }
    return false;
}

static bool isTalkStyleRecord(const StyleRecord &styleRecord) {
    return styleRecord.styleType.empty() || styleRecord.styleType == "talk";
}

static std::map<std::string, std::string> createRuntimeSpeakerMorphingPermissions(const RuntimeState &runtimeState, const std::vector<StyleRecord> &styleRecords) {
    std::map<std::string, bool> speakerHasTalkStyles;
    for (const StyleRecord &styleRecord : styleRecords) {
        if (isTalkStyleRecord(styleRecord)) {
            speakerHasTalkStyles[styleRecord.speakerUuid] = true;
        }
    }
    std::map<std::string, std::string> speakerPermissions;
    std::map<std::string, std::string> speakerSupportedFeaturesJsons = createCharacterSupportedFeaturesJsons(getCharacterResources(runtimeState));
    bool supportsMorphing = getCoreBackendCapabilities(runtimeState.coreBackend).supportsMorphing;
    for (const StyleRecord &styleRecord : styleRecords) {
        if (speakerPermissions.find(styleRecord.speakerUuid) != speakerPermissions.end()) {
            continue;
        }
        auto featureEntry = speakerSupportedFeaturesJsons.find(styleRecord.speakerUuid);
        if (featureEntry != speakerSupportedFeaturesJsons.end()) {
            speakerPermissions[styleRecord.speakerUuid] = extractJsonStringField(featureEntry->second, "permitted_synthesis_morphing");
        }
        if (speakerPermissions[styleRecord.speakerUuid].empty()) {
            speakerPermissions[styleRecord.speakerUuid] = supportsMorphing && speakerHasTalkStyles[styleRecord.speakerUuid] ? "ALL" : "NOTHING";
        }
    }
    return speakerPermissions;
}

static bool isMorphableStylePair(const StyleRecord &baseStyleRecord, const StyleRecord &targetStyleRecord, const std::map<std::string, std::string> &speakerPermissions) {
    if (!isTalkStyleRecord(baseStyleRecord) || !isTalkStyleRecord(targetStyleRecord)) {
        return false;
    }
    auto basePermissionEntry = speakerPermissions.find(baseStyleRecord.speakerUuid);
    auto targetPermissionEntry = speakerPermissions.find(targetStyleRecord.speakerUuid);
    std::string basePermission = basePermissionEntry == speakerPermissions.end() ? "NOTHING" : basePermissionEntry->second;
    std::string targetPermission = targetPermissionEntry == speakerPermissions.end() ? "NOTHING" : targetPermissionEntry->second;
    if (basePermission == "NOTHING" || targetPermission == "NOTHING") {
        return false;
    }
    if (baseStyleRecord.speakerUuid == targetStyleRecord.speakerUuid) {
        return true;
    }
    return basePermission == "ALL" && targetPermission == "ALL";
}

static std::string createMorphableTargetsJson(const RuntimeState &runtimeState, const std::string &baseStyleIdsJson) {
    std::vector<uint32_t> baseStyleIds = parseJsonIntegerArrayText(baseStyleIdsJson);
    std::vector<StyleRecord> targetStyleRecords = extractOrderedStylesFromMetasJson(runtimeState.combinedMetasJson);
    std::map<uint32_t, StyleRecord> styleRecordsById;
    for (const StyleRecord &styleRecord : targetStyleRecords) {
        styleRecordsById[styleRecord.styleId] = styleRecord;
    }
    std::map<std::string, std::string> speakerPermissions = createRuntimeSpeakerMorphingPermissions(runtimeState, targetStyleRecords);
    std::ostringstream morphableTargetsStream;
    morphableTargetsStream << "[";
    for (size_t baseIndex = 0; baseIndex < baseStyleIds.size(); baseIndex++) {
        if (baseIndex > 0) {
            morphableTargetsStream << ",";
        }
        morphableTargetsStream << "{";
        auto baseStyleRecordEntry = styleRecordsById.find(baseStyleIds[baseIndex]);
        for (size_t targetIndex = 0; targetIndex < targetStyleRecords.size(); targetIndex++) {
            if (targetIndex > 0) {
                morphableTargetsStream << ",";
            }
            bool isMorphable = baseStyleRecordEntry != styleRecordsById.end() && isMorphableStylePair(baseStyleRecordEntry->second, targetStyleRecords[targetIndex], speakerPermissions);
            morphableTargetsStream << "\"" << targetStyleRecords[targetIndex].styleId << "\":{\"is_morphable\":" << (isMorphable ? "true" : "false") << "}";
        }
        morphableTargetsStream << "}";
    }
    morphableTargetsStream << "]";
    return morphableTargetsStream.str();
}

static AudioStreamFormat getAudioStreamFormatParameter(const HttpRequest &request) {
    return parseAudioStreamFormat(getOptionalQueryParameter(request, "format", "wav"));
}

static size_t getChunkSamplesParameter(const HttpRequest &request) {
    std::string chunkSamplesText = getOptionalQueryParameter(request, "chunk_samples", "1024");
    size_t chunkSamples = 0;
    try {
        chunkSamples = static_cast<size_t>(std::stoul(chunkSamplesText));
    } catch (...) {
        throw HttpStatusError(400, "chunk_samples が不正です: " + chunkSamplesText);
    }
    if (chunkSamples == 0) {
        throw HttpStatusError(400, "chunk_samples は 1 以上が必要です");
    }
    return chunkSamples;
}

static HttpResponse makeAudioResponse(const std::vector<uint8_t> &wavBytes, AudioStreamFormat audioStreamFormat) {
    AudioStreamPayload audioStreamPayload = createAudioStreamPayload(wavBytes, audioStreamFormat);
    HttpResponse response;
    response.statusCode = 200;
    response.contentType = audioStreamPayload.contentType;
    response.bodyBytes = std::move(audioStreamPayload.audioBytes);
    return response;
}

static std::string createHttpAudioStreamContentType(AudioStreamFormat audioStreamFormat) {
    if (audioStreamFormat == AudioStreamFormat::Pcm) {
        return "audio/L16; rate=24000; channels=1";
    }
    return "audio/wav";
}

static HttpResponse makeChunkedRuntimeStreamResponse(AudioStreamFormat audioStreamFormat, const std::function<void(const std::function<void(const uint8_t *, size_t)> &)> &writeStream) {
    HttpResponse response;
    response.statusCode = 200;
    response.contentType = createHttpAudioStreamContentType(audioStreamFormat);
    response.isChunked = true;
    response.writeStream = writeStream;
    return response;
}

static HttpResponse makeCancellableSynthesisResponse(RuntimeState &runtimeState, const HttpRequest &request) {
    uint32_t styleId = parseSpeakerParameter(request);
    AudioStreamFormat audioStreamFormat = getAudioStreamFormatParameter(request);
    CoreBackendCapabilities backendCapabilities = getCoreBackendCapabilities(runtimeState.coreBackend);
    if (backendCapabilities.supportsTrueStreaming) {
        std::string audioQueryJson = request.body;
        RuntimeAudioStreamOptions streamOptions;
        streamOptions.audioStreamFormat = audioStreamFormat;
        streamOptions.chunkSamples = getChunkSamplesParameter(request);
        HttpResponse response = makeChunkedRuntimeStreamResponse(audioStreamFormat, [&runtimeState, audioQueryJson, styleId, streamOptions](const std::function<void(const uint8_t *, size_t)> &writeChunk) {
            streamAudioQuery(runtimeState, audioQueryJson, styleId, streamOptions, writeChunk);
        });
        response.headers["X-LiteVox-Cancellable"] = "true";
        response.headers["X-LiteVox-Backend"] = getCoreBackendMode(runtimeState.coreBackend);
        return response;
    }
    HttpResponse response = makeAudioResponse(synthesizeAudioQuery(runtimeState, request.body, styleId), audioStreamFormat);
    response.headers["X-LiteVox-Cancellable"] = "false";
    response.headers["X-LiteVox-Backend"] = getCoreBackendMode(runtimeState.coreBackend);
    return response;
}

static void validateRequestAudioQuery(RuntimeState &runtimeState, const std::string &audioQueryJson) {
    try {
        validateAudioQuery(runtimeState, audioQueryJson);
    } catch (const std::runtime_error &validationError) {
        throw HttpStatusError(400, validationError.what());
    }
}

static void validateRequestFrameAudioQuery(RuntimeState &runtimeState, const std::string &frameAudioQueryJson) {
    try {
        validateFrameAudioQuery(runtimeState, frameAudioQueryJson);
    } catch (const std::runtime_error &validationError) {
        throw HttpStatusError(400, validationError.what());
    }
}

static std::string getJsonObjectBodyField(const HttpRequest &request, const std::string &fieldName) {
    std::string objectText = extractJsonObjectField(request.body, fieldName);
    if (objectText.empty()) {
        throw HttpStatusError(400, fieldName + " がありません");
    }
    return objectText;
}

static HttpResponse makePortalPageResponse() {
    return makeTextResponse(200, "text/html; charset=utf-8", "\n        <html>\n            <head>\n                <title>VOICEVOX Engine</title>\n            </head>\n            <body>\n                <h1>VOICEVOX Engine</h1>\n                VOICEVOX Engine へようこそ！\n                <ul>\n                    <li><a href='/setting'>設定</a></li>\n                    <li><a href='/docs'>API ドキュメント</a></li>\n        </ul></body></html>\n        ");
}

static HttpResponse makeDocsPageResponse() {
    return makeTextResponse(200, "text/html; charset=utf-8", R"litevox_html(
    <!DOCTYPE html>
    <html>
    <head>
    <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <title>VOICEVOX Engine - Swagger UI</title>
    </head>
    <body>
    <div id="swagger-ui">
    </div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <!-- `SwaggerUIBundle` is now available on the page -->
    <script>
    const ui = SwaggerUIBundle({
        url: '/openapi.json',
    "dom_id": "#swagger-ui",
"layout": "BaseLayout",
"deepLinking": true,
"showExtensions": true,
"showCommonExtensions": true,
oauth2RedirectUrl: window.location.origin + '/docs/oauth2-redirect',
    presets: [
        SwaggerUIBundle.presets.apis,
        SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
    })
    </script>
    </body>
    </html>
    )litevox_html");
}

static HttpResponse makeRedocPageResponse() {
    return makeTextResponse(200, "text/html; charset=utf-8", R"litevox_html(
    <!DOCTYPE html>
    <html>
    <head>
    <title>VOICEVOX Engine - ReDoc</title>
    <!-- needed for adaptive design -->
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    
    <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <!--
    ReDoc doesn't change outer page styles
    -->
    <style>
      body {
        margin: 0;
        padding: 0;
      }
    </style>
    </head>
    <body>
    <noscript>
        ReDoc requires Javascript to function. Please enable it to browse the documentation.
    </noscript>
    <redoc spec-url="/openapi.json"></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"> </script>
    </body>
    </html>
    )litevox_html");
}

static HttpResponse makeOauth2RedirectResponse() {
    return makeTextResponse(200, "text/html; charset=utf-8", R"litevox_html(
    <!doctype html>
    <html lang="en-US">
    <head>
        <title>Swagger UI: OAuth2 Redirect</title>
    </head>
    <body>
    <script>
        'use strict';
        function run () {
            var oauth2 = window.opener.swaggerUIRedirectOauth2;
            var sentState = oauth2.state;
            var redirectUrl = oauth2.redirectUrl;
            var isValid, qp, arr;

            if (/code|token|error/.test(window.location.hash)) {
                qp = window.location.hash.substring(1).replace('?', '&');
            } else {
                qp = location.search.substring(1);
            }

            arr = qp.split("&");
            arr.forEach(function (v,i,_arr) { _arr[i] = '"' + v.replace('=', '":"') + '"';});
            qp = qp ? JSON.parse('{' + arr.join() + '}',
                    function (key, value) {
                        return key === "" ? value : decodeURIComponent(value);
                    }
            ) : {};

            isValid = qp.state === sentState;

            if ((
              oauth2.auth.schema.get("flow") === "accessCode" ||
              oauth2.auth.schema.get("flow") === "authorizationCode" ||
              oauth2.auth.schema.get("flow") === "authorization_code"
            ) && !oauth2.auth.code) {
                if (!isValid) {
                    oauth2.errCb({
                        authId: oauth2.auth.name,
                        source: "auth",
                        level: "warning",
                        message: "Authorization may be unsafe, passed state was changed in server. The passed state wasn't returned from auth server."
                    });
                }

                if (qp.code) {
                    delete oauth2.state;
                    oauth2.auth.code = qp.code;
                    oauth2.callback({auth: oauth2.auth, redirectUrl: redirectUrl});
                } else {
                    let oauthErrorMsg;
                    if (qp.error) {
                        oauthErrorMsg = "["+qp.error+"]: " +
                            (qp.error_description ? qp.error_description+ ". " : "no accessCode received from the server. ") +
                            (qp.error_uri ? "More info: "+qp.error_uri : "");
                    }

                    oauth2.errCb({
                        authId: oauth2.auth.name,
                        source: "auth",
                        level: "error",
                        message: oauthErrorMsg || "[Authorization failed]: no accessCode received from the server."
                    });
                }
            } else {
                oauth2.callback({auth: oauth2.auth, token: qp, isValid: isValid, redirectUrl: redirectUrl});
            }
            window.close();
        }

        if (document.readyState !== 'loading') {
            run();
        } else {
            document.addEventListener('DOMContentLoaded', function () {
                run();
            });
        }
    </script>
    </body>
    </html>
        )litevox_html");
}

static HttpResponse makeConnectedWavesResponse(const HttpRequest &request) {
    std::vector<std::string> encodedWaves = parseJsonStringArray(request.body);
    std::vector<std::vector<uint8_t>> wavFiles;
    wavFiles.reserve(encodedWaves.size());
    for (const std::string &encodedWave : encodedWaves) {
        wavFiles.push_back(decodeBase64Text(encodedWave));
    }
    HttpResponse response;
    response.statusCode = 200;
    response.contentType = "audio/wav";
    response.bodyBytes = connectPcmWaveBytes(wavFiles);
    return response;
}

