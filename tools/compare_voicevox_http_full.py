import base64
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
import zipfile
from io import BytesIO


OFFICIAL_BASE = sys.argv[1]
LITEVOX_BASE = sys.argv[2]
TEXT = "ずんだもんなのだ"
KANA = "ズンダモンナノダ'"
SPEAKER_UUID = "7ffcb7ce-00ec-4bdc-82cd-45a8889e43ff"
FIXED_IMPORTED_WORD_UUID = "11111111-1111-1111-1111-111111111111"
DEFAULT_SONG_SCORE = {
    "notes": [
        {"key": None, "frame_length": 15, "lyric": ""},
        {"id": "frame_length", "key": 60, "frame_length": 45, "lyric": "ド"},
        {"id": "音符-レ", "key": 62, "frame_length": 45, "lyric": "レ"},
        {"id": "音符-ミ", "key": 64, "frame_length": 45, "lyric": "ミ"},
        {"key": None, "frame_length": 15, "lyric": ""},
    ]
}


def request(base_url, method, path, body=None, headers=None):
    request_headers = headers or {}
    request_body = body
    if isinstance(request_body, str):
        request_body = request_body.encode()
    http_request = urllib.request.Request(base_url + path, data=request_body, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(http_request, timeout=120) as response:
            return {
                "status": response.status,
                "content_type": response.headers.get("Content-Type"),
                "headers": dict(response.headers.items()),
                "body": response.read(),
            }
    except urllib.error.HTTPError as error:
        return {
            "status": error.code,
            "content_type": error.headers.get("Content-Type"),
            "headers": dict(error.headers.items()),
            "body": error.read(),
        }


def quote(text):
    return urllib.parse.quote(text)


def post(path, body=b"", headers=None):
    return "POST", path, body, headers or {}


def get(path, headers=None):
    return "GET", path, None, headers or {}


def put(path, body=b"", headers=None):
    return "PUT", path, body, headers or {}


def delete(path, headers=None):
    return "DELETE", path, None, headers or {}


def load_json(body):
    return json.loads(body.decode())


def call(base_url, step):
    method, path, body, headers = step
    return request(base_url, method, path, body, headers)


def call_both(step):
    return call(OFFICIAL_BASE, step), call(LITEVOX_BASE, step)


def call_split(official_step, litevox_step):
    return call(OFFICIAL_BASE, official_step), call(LITEVOX_BASE, litevox_step)


def compare_bytes(label, official_response, litevox_response):
    return {
        "label": label,
        "status_equal": official_response["status"] == litevox_response["status"],
        "content_type_equal": official_response["content_type"] == litevox_response["content_type"],
        "body_equal": official_response["body"] == litevox_response["body"],
        "official_status": official_response["status"],
        "litevox_status": litevox_response["status"],
        "official_content_type": official_response["content_type"],
        "litevox_content_type": litevox_response["content_type"],
        "official_size": len(official_response["body"]),
        "litevox_size": len(litevox_response["body"]),
        "official_preview": official_response["body"][:160].decode("utf-8", "replace"),
        "litevox_preview": litevox_response["body"][:160].decode("utf-8", "replace"),
    }


def compare_json_semantic(label, official_response, litevox_response):
    result = compare_bytes(label, official_response, litevox_response)
    try:
        result["json_equal"] = load_json(official_response["body"]) == load_json(litevox_response["body"])
    except Exception:
        result["json_equal"] = False
    return result


def compare_header_value(label, official_response, litevox_response, header_name):
    result = compare_bytes(label, official_response, litevox_response)
    official_header_value = official_response["headers"].get(header_name)
    litevox_header_value = litevox_response["headers"].get(header_name)
    result["header_equal"] = official_header_value == litevox_header_value
    result["official_header"] = official_header_value
    result["litevox_header"] = litevox_header_value
    return result


def get_wav_info(body):
    with wave.open(BytesIO(body), "rb") as wav_file:
        return {
            "channels": wav_file.getnchannels(),
            "sample_width": wav_file.getsampwidth(),
            "frame_rate": wav_file.getframerate(),
            "frames": wav_file.getnframes(),
        }


def compare_wav_shape(label, official_response, litevox_response):
    result = compare_bytes(label, official_response, litevox_response)
    result["raw_body_equal"] = result["body_equal"]
    result["body_equal"] = True
    result["official_wav"] = get_wav_info(official_response["body"]) if official_response["status"] == 200 else None
    result["litevox_wav"] = get_wav_info(litevox_response["body"]) if litevox_response["status"] == 200 else None
    result["wav_shape_equal"] = result["official_wav"] == result["litevox_wav"]
    return result


def compare_zip_wav_shape(label, official_response, litevox_response):
    result = compare_bytes(label, official_response, litevox_response)
    result["raw_body_equal"] = result["body_equal"]
    result["body_equal"] = True
    if official_response["status"] != 200 or litevox_response["status"] != 200:
        result["zip_wav_shape_equal"] = official_response["body"] == litevox_response["body"]
        return result
    with zipfile.ZipFile(BytesIO(official_response["body"])) as official_zip:
        official_names = official_zip.namelist()
        official_wavs = {name: get_wav_info(official_zip.read(name)) for name in official_names}
    with zipfile.ZipFile(BytesIO(litevox_response["body"])) as litevox_zip:
        litevox_names = litevox_zip.namelist()
        litevox_wavs = {name: get_wav_info(litevox_zip.read(name)) for name in litevox_names}
    result["official_zip_names"] = official_names
    result["litevox_zip_names"] = litevox_names
    result["zip_wav_shape_equal"] = official_wavs == litevox_wavs
    return result


def are_finite_numbers(values, allow_negative):
    if not isinstance(values, list):
        return False
    for number in values:
        if not isinstance(number, (int, float)) or not math.isfinite(number):
            return False
        if not allow_negative and number < 0:
            return False
    return True


def get_frame_count(frame_query):
    if not isinstance(frame_query.get("phonemes"), list):
        return -1
    frame_count = 0
    for phoneme in frame_query["phonemes"]:
        if not isinstance(phoneme, dict) or not isinstance(phoneme.get("frame_length"), int):
            return -1
        frame_count += phoneme["frame_length"]
    return frame_count


def compare_frame_audio_query_shape(label, official_response, litevox_response):
    result = compare_bytes(label, official_response, litevox_response)
    result["raw_body_equal"] = result["body_equal"]
    result["body_equal"] = True
    try:
        official_query = load_json(official_response["body"])
        litevox_query = load_json(litevox_response["body"])
        official_frame_count = get_frame_count(official_query)
        litevox_frame_count = get_frame_count(litevox_query)
        result["phonemes_equal"] = official_query.get("phonemes") == litevox_query.get("phonemes")
        result["f0_shape_equal"] = len(official_query.get("f0", [])) == len(litevox_query.get("f0", [])) == official_frame_count == litevox_frame_count
        result["volume_shape_equal"] = len(official_query.get("volume", [])) == len(litevox_query.get("volume", [])) == official_frame_count == litevox_frame_count
        result["f0_values_valid"] = are_finite_numbers(official_query.get("f0"), False) and are_finite_numbers(litevox_query.get("f0"), False)
        result["volume_values_valid"] = are_finite_numbers(official_query.get("volume"), True) and are_finite_numbers(litevox_query.get("volume"), True)
    except Exception:
        result["phonemes_equal"] = False
        result["f0_shape_equal"] = False
        result["volume_shape_equal"] = False
        result["f0_values_valid"] = False
        result["volume_values_valid"] = False
    return result


def compare_float_array_shape(label, official_response, litevox_response):
    result = compare_bytes(label, official_response, litevox_response)
    result["raw_body_equal"] = result["body_equal"]
    result["body_equal"] = True
    try:
        official_values = load_json(official_response["body"])
        litevox_values = load_json(litevox_response["body"])
        result["array_length_equal"] = isinstance(official_values, list) and isinstance(litevox_values, list) and len(official_values) == len(litevox_values)
        result["array_values_valid"] = are_finite_numbers(official_values, True) and are_finite_numbers(litevox_values, True)
    except Exception:
        result["array_length_equal"] = False
        result["array_values_valid"] = False
    return result


def normalize_resource_value(value):
    if isinstance(value, dict):
        return {key: normalize_resource_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [normalize_resource_value(item) for item in value]
    if isinstance(value, str) and (value.startswith("http://") or value.startswith("https://")):
        parsed_url = urllib.parse.urlparse(value)
        return parsed_url.path
    return value


def compare_resource_json(label, official_response, litevox_response):
    result = compare_bytes(label, official_response, litevox_response)
    result["raw_body_equal"] = result["body_equal"]
    result["body_equal"] = True
    try:
        official_json = normalize_resource_value(load_json(official_response["body"]))
        litevox_json = normalize_resource_value(load_json(litevox_response["body"]))
        result["json_equal"] = official_json == litevox_json
    except Exception:
        result["json_equal"] = False
    return result


def is_valid_uuid_text(text):
    try:
        uuid.UUID(text)
        return True
    except Exception:
        return False


def compare_uuid_string(label, official_response, litevox_response):
    result = compare_bytes(label, official_response, litevox_response)
    result["raw_body_equal"] = result["body_equal"]
    result["body_equal"] = True
    try:
        official_uuid = load_json(official_response["body"])
        litevox_uuid = load_json(litevox_response["body"])
        result["uuid_valid"] = isinstance(official_uuid, str) and isinstance(litevox_uuid, str) and is_valid_uuid_text(official_uuid) and is_valid_uuid_text(litevox_uuid)
        result["official_uuid"] = official_uuid
        result["litevox_uuid"] = litevox_uuid
    except Exception:
        result["uuid_valid"] = False
        result["official_uuid"] = None
        result["litevox_uuid"] = None
    return result


def normalize_user_dict_words(body):
    user_dict = load_json(body)
    if not isinstance(user_dict, dict):
        raise ValueError("user dict is not object")
    normalized_words = []
    for word in user_dict.values():
        if not isinstance(word, dict):
            raise ValueError("user dict value is not object")
        normalized_words.append({key: word[key] for key in sorted(word.keys())})
    normalized_words.sort(key=lambda word: json.dumps(word, ensure_ascii=False, sort_keys=True))
    return normalized_words


def compare_user_dict_semantic(label, official_response, litevox_response):
    result = compare_bytes(label, official_response, litevox_response)
    result["raw_body_equal"] = result["body_equal"]
    result["body_equal"] = True
    try:
        result["json_equal"] = normalize_user_dict_words(official_response["body"]) == normalize_user_dict_words(litevox_response["body"])
    except Exception:
        result["json_equal"] = False
    return result


def add_preset(preset_id):
    preset = {
        "id": preset_id,
        "name": "LiteVox Verify",
        "speaker_uuid": "dummy",
        "style_id": 3,
        "speedScale": 1,
        "pitchScale": 0,
        "intonationScale": 1,
        "volumeScale": 1,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
        "pauseLength": 0.4,
        "pauseLengthScale": 1,
    }
    body = json.dumps(preset, ensure_ascii=False, separators=(",", ":")).encode()
    return post("/add_preset", body, {"Content-Type": "application/json"})


def delete_preset(preset_id):
    return post(f"/delete_preset?id={preset_id}")


def make_audio_query():
    return call_both(post(f"/audio_query?speaker=3&text={quote(TEXT)}"))


def make_frame_query():
    body = json.dumps(DEFAULT_SONG_SCORE, ensure_ascii=True, separators=(",", ":")).encode()
    return body, call_both(post("/sing_frame_audio_query?speaker=6000", body, {"Content-Type": "application/json"}))


def make_form_body(parameters):
    return urllib.parse.urlencode(parameters).encode()


def make_query_path(path, parameters):
    return path + "?" + urllib.parse.urlencode(parameters)


def record_result(results, result):
    results.append(result)
    failed_keys = [
        key
        for key in (
            "status_equal",
            "content_type_equal",
            "body_equal",
            "json_equal",
            "wav_shape_equal",
            "zip_wav_shape_equal",
            "phonemes_equal",
            "f0_shape_equal",
            "volume_shape_equal",
            "f0_values_valid",
            "volume_values_valid",
            "array_length_equal",
            "array_values_valid",
            "uuid_valid",
            "header_equal",
        )
        if key in result and not result[key]
    ]
    print(json.dumps({"label": result["label"], "failed": failed_keys, "official_status": result["official_status"], "litevox_status": result["litevox_status"], "official_size": result["official_size"], "litevox_size": result["litevox_size"]}, ensure_ascii=False, separators=(",", ":")))


def find_user_dict_word(body, pronunciation):
    user_dict = load_json(body)
    if not isinstance(user_dict, dict):
        raise ValueError("user dict expected")
    for word in user_dict.values():
        if isinstance(word, dict) and word.get("pronunciation") == pronunciation:
            return dict(word)
    raise ValueError("word not found")


def create_import_body(word_object, word_uuid):
    return json.dumps({word_uuid: word_object}, ensure_ascii=False, separators=(",", ":")).encode()


def run():
    preset_id = 480000
    for cleanup_id in range(preset_id, preset_id + 4):
        call_both(delete_preset(cleanup_id))
    reset_setting_body = make_form_body({"cors_policy_mode": "localapps", "allow_origin": ""})
    call_both(post("/setting", reset_setting_body, {"Content-Type": "application/x-www-form-urlencoded"}))
    call_both(post("/import_user_dict?override=true", b"{}", {"Content-Type": "application/json"}))
    official_audio_query, litevox_audio_query = make_audio_query()
    audio_query_body = official_audio_query["body"]
    official_accent, litevox_accent = call_both(post(f"/accent_phrases?speaker=3&text={quote(TEXT)}"))
    accent_body = official_accent["body"]
    score_body, frame_query_pair = make_frame_query()
    frame_query_body = frame_query_pair[0]["body"]
    synthesized_wav_response = request(OFFICIAL_BASE, "POST", "/synthesis?speaker=3", audio_query_body, {"Content-Type": "application/json"})
    wav_body = json.dumps([base64.b64encode(synthesized_wav_response["body"]).decode()], ensure_ascii=False, separators=(",", ":")).encode()
    cases = [
        ("root", get("/"), compare_bytes),
        ("version", get("/version"), compare_bytes),
        ("core_versions", get("/core_versions"), compare_bytes),
        ("supported_devices", get("/supported_devices"), compare_bytes),
        ("speakers", get("/speakers"), compare_bytes),
        ("singers", get("/singers"), compare_bytes),
        ("speaker_info_base64", get(f"/speaker_info?speaker_uuid={SPEAKER_UUID}"), compare_bytes),
        ("speaker_info_url", get(f"/speaker_info?speaker_uuid={SPEAKER_UUID}&resource_format=url"), compare_resource_json),
        ("singer_info_base64", get(f"/singer_info?speaker_uuid={SPEAKER_UUID}"), compare_json_semantic),
        ("singer_info_url", get(f"/singer_info?speaker_uuid={SPEAKER_UUID}&resource_format=url"), compare_resource_json),
        ("presets_initial", get("/presets"), compare_bytes),
        ("engine_manifest", get("/engine_manifest"), compare_bytes),
        ("setting", get("/setting"), compare_bytes),
        ("openapi", get("/openapi.json"), compare_bytes),
        ("docs", get("/docs"), compare_bytes),
        ("redoc", get("/redoc"), compare_bytes),
        ("docs_oauth2_redirect", get("/docs/oauth2-redirect"), compare_bytes),
        ("validate_kana_ok", post(f"/validate_kana?text={quote(KANA)}"), compare_bytes),
        ("audio_query", post(f"/audio_query?speaker=3&text={quote(TEXT)}"), compare_bytes),
        ("audio_query_kana", post(f"/audio_query?speaker=3&text={quote(KANA)}&is_kana=true"), compare_bytes),
        ("accent_phrases", post(f"/accent_phrases?speaker=3&text={quote(TEXT)}"), compare_bytes),
        ("accent_phrases_kana", post(f"/accent_phrases?speaker=3&text={quote(KANA)}&is_kana=true"), compare_bytes),
        ("mora_data", post("/mora_data?speaker=3", accent_body, {"Content-Type": "application/json"}), compare_bytes),
        ("mora_length", post("/mora_length?speaker=3", accent_body, {"Content-Type": "application/json"}), compare_bytes),
        ("mora_pitch", post("/mora_pitch?speaker=3", accent_body, {"Content-Type": "application/json"}), compare_bytes),
        ("synthesis", post("/synthesis?speaker=3", audio_query_body, {"Content-Type": "application/json"}), compare_wav_shape),
        ("cancellable_synthesis", post("/cancellable_synthesis?speaker=3", audio_query_body, {"Content-Type": "application/json"}), compare_wav_shape),
        ("multi_synthesis", post("/multi_synthesis?speaker=3", b"[" + audio_query_body + b"]", {"Content-Type": "application/json"}), compare_zip_wav_shape),
        ("connect_waves", post("/connect_waves", wav_body, {"Content-Type": "application/json"}), compare_wav_shape),
        ("morphable_targets", post("/morphable_targets", b"[3,8,89,2]", {"Content-Type": "application/json"}), compare_bytes),
        ("synthesis_morphing", post("/synthesis_morphing?base_speaker=3&target_speaker=1&morph_rate=0.5", audio_query_body, {"Content-Type": "application/json"}), compare_wav_shape),
        ("sing_frame_audio_query", post("/sing_frame_audio_query?speaker=6000", score_body, {"Content-Type": "application/json"}), compare_frame_audio_query_shape),
        ("sing_frame_f0", post("/sing_frame_f0?speaker=6000", json.dumps({"score": load_json(score_body), "frame_audio_query": load_json(frame_query_body)}, ensure_ascii=False, separators=(",", ":")).encode(), {"Content-Type": "application/json"}), compare_float_array_shape),
        ("sing_frame_volume", post("/sing_frame_volume?speaker=6000", json.dumps({"score": load_json(score_body), "frame_audio_query": load_json(frame_query_body)}, ensure_ascii=False, separators=(",", ":")).encode(), {"Content-Type": "application/json"}), compare_float_array_shape),
        ("frame_synthesis", post("/frame_synthesis?speaker=3000", frame_query_body, {"Content-Type": "application/json"}), compare_wav_shape),
        ("user_dict_initial", get("/user_dict"), compare_bytes),
        ("user_dict_word_get_not_allowed", get("/user_dict_word"), compare_bytes),
        ("is_initialized_before", get("/is_initialized_speaker?speaker=3"), compare_bytes),
        ("initialize_speaker", post("/initialize_speaker?speaker=3"), compare_bytes),
        ("is_initialized_after", get("/is_initialized_speaker?speaker=3"), compare_bytes),
        ("audio_query_from_preset_missing", post(f"/audio_query_from_preset?preset_id={preset_id}&text={quote(TEXT)}"), compare_bytes),
        ("delete_preset_missing", delete_preset(preset_id), compare_bytes),
        ("update_preset_missing", post("/update_preset", json.dumps({"id": preset_id, "name": "LiteVox Verify", "speaker_uuid": "dummy", "style_id": 3, "speedScale": 1, "pitchScale": 0, "intonationScale": 1, "volumeScale": 1, "prePhonemeLength": 0.1, "postPhonemeLength": 0.1}, separators=(",", ":")).encode(), {"Content-Type": "application/json"}), compare_bytes),
        ("add_preset", add_preset(preset_id), compare_bytes),
        ("add_preset_duplicate", add_preset(preset_id), compare_bytes),
        ("audio_query_from_preset", post(f"/audio_query_from_preset?preset_id={preset_id}&text={quote(TEXT)}"), compare_bytes),
        ("update_preset", post("/update_preset", json.dumps({"id": preset_id, "name": "LiteVox Verify", "speaker_uuid": "dummy", "style_id": 3, "speedScale": 1, "pitchScale": 0, "intonationScale": 1, "volumeScale": 1, "prePhonemeLength": 0.1, "postPhonemeLength": 0.1}, separators=(",", ":")).encode(), {"Content-Type": "application/json"}), compare_bytes),
        ("delete_preset", delete_preset(preset_id), compare_bytes),
    ]
    results = []
    for label, step, comparator in cases:
        official_response, litevox_response = call_both(step)
        record_result(results, comparator(label, official_response, litevox_response))

    add_word_query = make_query_path("/user_dict_word", {
        "surface": "LiteVox",
        "pronunciation": "ライトボックス",
        "accent_type": 4,
        "word_type": "PROPER_NOUN",
        "priority": 5,
    })
    official_add_word_response, litevox_add_word_response = call_both(post(add_word_query))
    add_word_result = compare_uuid_string("user_dict_word_add", official_add_word_response, litevox_add_word_response)
    record_result(results, add_word_result)
    official_word_uuid = add_word_result["official_uuid"]
    litevox_word_uuid = add_word_result["litevox_uuid"]

    record_result(results, compare_user_dict_semantic("user_dict_after_add", *call_both(get("/user_dict"))))

    update_word_query = "?" + urllib.parse.urlencode({
        "surface": "LiteVox2",
        "pronunciation": "ライトボックスツー",
        "accent_type": 5,
        "word_type": "PROPER_NOUN",
        "priority": 7,
    })
    record_result(results, compare_bytes("user_dict_word_update", *call_split(put(f"/user_dict_word/{official_word_uuid}{update_word_query}"), put(f"/user_dict_word/{litevox_word_uuid}{update_word_query}"))))

    official_after_update_response, litevox_after_update_response = call_both(get("/user_dict"))
    record_result(results, compare_user_dict_semantic("user_dict_after_update", official_after_update_response, litevox_after_update_response))

    imported_word_object = find_user_dict_word(official_after_update_response["body"], "ライトボックスツー")
    imported_word_object["surface"] = "LiteVoxImport"
    imported_word_object["pronunciation"] = "ライトボックスインポート"
    imported_word_object["yomi"] = "ライトボックスインポート"
    imported_word_object["accent_type"] = 6
    imported_word_object["priority"] = 6
    import_body = create_import_body(imported_word_object, FIXED_IMPORTED_WORD_UUID)
    record_result(results, compare_bytes("import_user_dict_merge", *call_both(post("/import_user_dict?override=false", import_body, {"Content-Type": "application/json"}))))
    record_result(results, compare_user_dict_semantic("user_dict_after_import_merge", *call_both(get("/user_dict"))))

    record_result(results, compare_bytes("import_user_dict_override", *call_both(post("/import_user_dict?override=true", import_body, {"Content-Type": "application/json"}))))
    record_result(results, compare_user_dict_semantic("user_dict_after_import_override", *call_both(get("/user_dict"))))
    record_result(results, compare_bytes("user_dict_word_delete", *call_both(delete(f"/user_dict_word/{FIXED_IMPORTED_WORD_UUID}"))))
    record_result(results, compare_user_dict_semantic("user_dict_after_delete", *call_both(get("/user_dict"))))

    setting_all_body = make_form_body({"cors_policy_mode": "all", "allow_origin": "https://example.com"})
    record_result(results, compare_bytes("setting_post_all", *call_both(post("/setting", setting_all_body, {"Content-Type": "application/x-www-form-urlencoded"}))))
    record_result(results, compare_bytes("setting_after_all", *call_both(get("/setting"))))
    record_result(results, compare_header_value("cors_header_all", *call_both(get("/version", {"Origin": "https://example.com"})), "Access-Control-Allow-Origin"))

    setting_localapps_body = make_form_body({"cors_policy_mode": "localapps", "allow_origin": ""})
    record_result(results, compare_bytes("setting_post_localapps", *call_both(post("/setting", setting_localapps_body, {"Content-Type": "application/x-www-form-urlencoded"}))))
    record_result(results, compare_bytes("setting_after_localapps", *call_both(get("/setting"))))
    record_result(results, compare_header_value("cors_header_localapps_allowed", *call_both(get("/version", {"Origin": "http://localhost:3000"})), "Access-Control-Allow-Origin"))
    record_result(results, compare_header_value("cors_header_localapps_blocked", *call_both(get("/version", {"Origin": "https://example.com"})), "Access-Control-Allow-Origin"))

    failed_results = [
        result
        for result in results
        if any(
            key in result and not result[key]
            for key in (
                "status_equal",
                "content_type_equal",
                "body_equal",
                "json_equal",
                "wav_shape_equal",
                "zip_wav_shape_equal",
                "phonemes_equal",
                "f0_shape_equal",
                "volume_shape_equal",
                "f0_values_valid",
                "volume_values_valid",
                "array_length_equal",
                "array_values_valid",
                "uuid_valid",
                "header_equal",
            )
        )
    ]
    print(json.dumps({"total": len(results), "failed": len(failed_results), "failed_labels": [result["label"] for result in failed_results]}, ensure_ascii=False, separators=(",", ":")))
    if failed_results:
        print(json.dumps(failed_results, ensure_ascii=False, indent=2))
        sys.exit(1)
if __name__ == "__main__":
    run()
