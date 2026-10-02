#!/bin/sh
set -eu

if [ "${1:-}" = "" ]; then
  echo "usage: verify-cli-smoke.sh RUNTIME_ROOT [RESULT_PREFIX]" >&2
  exit 1
fi

runtime_root=$1
result_prefix=${2:-audio-compare/cli-smoke}
bin_path="$runtime_root/litevox"
tmp_dir=$(mktemp -d /tmp/litevox-cli-smoke.XXXXXX)
port=$((50880 + ($$ % 1000)))
server_pid=""

cleanup() {
  if [ "$server_pid" != "" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp_dir"
}

trap cleanup EXIT INT TERM

cat > "$tmp_dir/score.json" <<'EOF'
{"notes":[{"key":null,"frame_length":15,"lyric":""},{"id":"note-d","key":60,"frame_length":45,"lyric":"\u30c9"},{"id":"note-r","key":62,"frame_length":45,"lyric":"レ"},{"id":"note-m","key":64,"frame_length":45,"lyric":"ミ"},{"key":null,"frame_length":15,"lyric":""}]}
EOF

"$bin_path" help > "$result_prefix-help.txt"
"$bin_path" version --runtime "$runtime_root" > "$result_prefix-version.txt"
"$bin_path" deps --runtime "$runtime_root" --backend native > "$result_prefix-deps.txt"
"$bin_path" runtime_info --runtime "$runtime_root" > "$result_prefix-runtime_info.json"
"$bin_path" models --runtime "$runtime_root" > "$result_prefix-models.txt"
"$bin_path" styles --runtime "$runtime_root" > "$result_prefix-styles.txt"
"$bin_path" singers --runtime "$runtime_root" > "$result_prefix-singers.json"
"$bin_path" query --runtime "$runtime_root" --speaker 3 --text "ずんだもんなのだ" > "$result_prefix-query.json"
"$bin_path" tts --runtime "$runtime_root" --speaker 3 --text "ずんだもんなのだ" --out "$result_prefix-tts.wav"
"$bin_path" stream --runtime "$runtime_root" --speaker 3 --text "ずんだもんなのだ" --format pcm --chunk-samples 4096 > "$result_prefix-stream.pcm"
"$bin_path" sing-query --runtime "$runtime_root" --backend native --score "$tmp_dir/score.json" --speaker 6000 --out "$result_prefix-frame_audio_query.json"
"$bin_path" sing-f0 --runtime "$runtime_root" --backend native --score "$tmp_dir/score.json" --frame-audio-query "$result_prefix-frame_audio_query.json" --speaker 6000 > "$result_prefix-sing-f0.json"
"$bin_path" sing-volume --runtime "$runtime_root" --backend native --score "$tmp_dir/score.json" --frame-audio-query "$result_prefix-frame_audio_query.json" --speaker 6000 > "$result_prefix-sing-volume.json"
"$bin_path" frame-synthesis --runtime "$runtime_root" --backend native --frame-audio-query "$result_prefix-frame_audio_query.json" --speaker 3000 --out "$result_prefix-frame.wav"
"$bin_path" sing --runtime "$runtime_root" --backend native --score "$tmp_dir/score.json" --teacher 6000 --speaker 3000 --out "$result_prefix-sing.wav"
python3 - "$result_prefix" <<'PY'
import json
import math
import sys
import wave
from pathlib import Path

Prefix = sys.argv[1]
Query = json.loads(Path(Prefix + "-frame_audio_query.json").read_text())
Frames = sum(Phoneme["frame_length"] for Phoneme in Query["phonemes"])
assert Frames == 165
assert {Phoneme["note_id"] for Phoneme in Query["phonemes"] if Phoneme["note_id"]} == {"note-d", "note-r", "note-m"}
for Field in ("f0", "volume"):
    assert len(Query[Field]) == Frames
    assert all(math.isfinite(Value) for Value in Query[Field])
for Name in ("sing-f0", "sing-volume"):
    Values = json.loads(Path(Prefix + "-" + Name + ".json").read_text())
    assert len(Values) == Frames and all(math.isfinite(Value) for Value in Values)
for Name in ("frame", "sing"):
    with wave.open(Prefix + "-" + Name + ".wav", "rb") as Audio:
        assert (Audio.getframerate(), Audio.getnchannels(), Audio.getsampwidth(), Audio.getnframes()) == (24000, 1, 2, Frames * 256)
        assert any(Audio.readframes(Audio.getnframes()))
Singers = json.loads(Path(Prefix + "-singers.json").read_text())
assert any(Style["id"] == 3000 for Singer in Singers for Style in Singer["styles"])
PY

"$bin_path" bench --runtime "$runtime_root" --speaker 3 --text "ずんだもんなのだ" --runs 1 --workers 1 --cpu-threads 1 > "$result_prefix-bench.tsv"
"$bin_path" bench-song --runtime "$runtime_root" --backend native --score "$tmp_dir/score.json" --speaker 6000 --runs 1 --workers 1 > "$result_prefix-bench-song.tsv"

"$bin_path" server --runtime "$runtime_root" --port "$port" > "$result_prefix-server.log" 2>&1 &
server_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 60 ]; do
  if curl -fsS "http://127.0.0.1:$port/version" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
  attempt=$((attempt + 1))
done

if [ "$ready" -ne 1 ]; then
  echo "server did not become ready" >&2
  exit 1
fi

python3 - "$port" "$result_prefix" <<'PY'
import io
import json
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

Port, Prefix = sys.argv[1:]
Base = "http://127.0.0.1:" + Port

def Post(Pathname, Query):
    Request = urllib.request.Request(Base + Pathname, data=json.dumps(Query).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(Request, timeout=120) as Response:
        return Response.read()

Query = json.loads(Path(Prefix + "-frame_audio_query.json").read_text())
Query["outputSamplingRate"], Query["outputStereo"] = 44100, True
Audio = Post("/frame_synthesis?speaker=3000", Query)
Path(Prefix + "-song-44100.wav").write_bytes(Audio)
with wave.open(io.BytesIO(Audio)) as Wav:
    assert (Wav.getframerate(), Wav.getnchannels(), Wav.getnframes()) == (44100, 2, 77616)

Query = json.loads(Path(Prefix + "-query.json").read_text())
Query["outputSamplingRate"], Query["outputStereo"] = 44100, True
Lengths = []
for Pathname in ("/synthesis", "/synthesis_stream"):
    Audio = Post(Pathname + "?speaker=3", Query)
    with wave.open(io.BytesIO(Audio)) as Wav:
        assert Wav.getframerate() == 44100 and Wav.getnchannels() == 2
        Frames = Wav.getnframes()
        assert len(Wav.readframes(Frames)) == Frames * 4
        Lengths.append(Frames)
assert Lengths[0] == Lengths[1]

Invalid = {"notes": [{"key": None, "frame_length": 15, "lyric": "bad"}]}
try:
    Post("/sing_frame_audio_query?speaker=6000", Invalid)
    raise AssertionError("invalid score was accepted")
except urllib.error.HTTPError as Error:
    assert Error.code == 400
PY

printf 'ずんだもんなのだ\n' | "$bin_path" api-session --host 127.0.0.1 --port "$port" --speaker 3 --http-path /tts --out "$tmp_dir/api-session-tts" >/dev/null
"$bin_path" api-session --host 127.0.0.1 --port "$port" --speaker 6000 --http-path /sing_frame_audio_query --score "$tmp_dir/score.json" --out "$tmp_dir/api-session-song" >/dev/null

wc -c "$result_prefix-tts.wav" | awk '{print "tts_wav_bytes\t" $1}' > "$result_prefix-summary.tsv"
wc -c "$result_prefix-stream.pcm" | awk '{print "stream_pcm_bytes\t" $1}' >> "$result_prefix-summary.tsv"
wc -c "$result_prefix-frame.wav" | awk '{print "frame_wav_bytes\t" $1}' >> "$result_prefix-summary.tsv"
wc -c "$result_prefix-query.json" | awk '{print "query_json_bytes\t" $1}' >> "$result_prefix-summary.tsv"
wc -c "$result_prefix-frame_audio_query.json" | awk '{print "frame_audio_query_bytes\t" $1}' >> "$result_prefix-summary.tsv"
wc -c "$result_prefix-sing-f0.json" | awk '{print "sing_f0_json_bytes\t" $1}' >> "$result_prefix-summary.tsv"
wc -c "$result_prefix-sing-volume.json" | awk '{print "sing_volume_json_bytes\t" $1}' >> "$result_prefix-summary.tsv"
wc -l "$result_prefix-models.txt" | awk '{print "models_lines\t" $1}' >> "$result_prefix-summary.tsv"
wc -l "$result_prefix-styles.txt" | awk '{print "styles_lines\t" $1}' >> "$result_prefix-summary.tsv"
wc -c "$tmp_dir/api-session-tts/0001.wav" | awk '{print "api_session_tts_wav_bytes\t" $1}' >> "$result_prefix-summary.tsv"
wc -c "$tmp_dir/api-session-song/0001.json" | awk '{print "api_session_song_json_bytes\t" $1}' >> "$result_prefix-summary.tsv"
