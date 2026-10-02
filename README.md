# LiteVox

LiteVox は、VOICEVOX Core を前提にせずに動かせる CLI / HTTP ランタイムです。  
既定 backend は `native` です。

モデルデータ、VOICEVOX 製品 zip、標準 ONNX Runtime archive は同梱しません。

以下のコマンドは `litevox/` ディレクトリで実行する前提です。

## できること

- `VOICEVOX.zip` をそのまま runtime として使う
- `.vvm` を読む
- CLI で TTS / query / stream / song / frame synthesis を実行する
- HTTP API を `server` で提供する
- `vv_bin` から ONNX を取り出す
- VOICEVOX 製品 zip から runtime root を組み立てる


## ビルド

C++17 のコンパイラー、make、CMake、Git、zlib、libsoxr、pkg-config が必要です。
Rust / Cargo は不要です。macOS では `brew install cmake libsoxr pkg-config`、
Debian / Ubuntu では `libsoxr-dev` と `pkg-config` を用意してください。
libsoxr は、最新版 VOICEVOX と同じ再サンプリングを行うために使います。
動的リンク版で配布する場合は実行先にも libsoxr が必要です。
別の配置を使う場合は `SOXR_CFLAGS` と `SOXR_LIBS` で指定できます。
初回は公式 Open JTalk 1.11 の C/C++ ソースを取得して静的ライブラリをビルドします。
既存の Open JTalk を使う場合は `OPEN_JTALK_INCLUDE_DIR` と `OPEN_JTALK_LIB` を指定できます。

```sh
make dist
make check
```

生成物:

- `dist/litevox`
- `dist/resources/`

## まず使う

### 1. VOICEVOX 製品 zip をそのまま使う

`--runtime` には mac 版 zip と Windows 版 zip のどちらも渡せます。  
Windows は `voicevox-windows-directml-*.zip` をそのまま使えます。

```sh
dist/litevox runtime_info \
  --runtime /path/to/voicevox-<platform>.zip

dist/litevox tts \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text 'ずんだもんなのだ' \
  --out zundamon.wav
```

初回だけ zip の横に `.litevox-runtime-cache/` を作り、必要な runtime を自動展開します。  
2 回目以降はその cache を再利用します。

### 2. HTTP サーバーを立てる

```sh
dist/litevox server \
  --runtime /path/to/voicevox-<platform>.zip \
  --port 50021
```

### 3. モデル一覧を見る

```sh
dist/litevox models \
  --runtime /path/to/voicevox-<platform>.zip
```

## GPU を使う場合

- mac 版 VOICEVOX 製品 zip 同梱 ORT は CPU provider のみです
- Windows DirectML 版 VOICEVOX 製品 zip 同梱 ORT は Windows 上で GPU を使えます
- mac で GPU を使う場合は、GPU provider を含む標準 ORT を別途指定します

```sh
dist/litevox runtime_info \
  --runtime /path/to/voicevox-macos-cpu-arm64-0.25.2.zip \
  --onnxruntime /path/to/<onnxruntime-library> \
  --acceleration-mode gpu
```

`--acceleration-mode auto` でも、指定した ORT に GPU provider があれば自動でそちらを使います。

## runtime root を明示的に作る

### VOICEVOX 製品 zip から作る

```sh
dist/litevox model-dump \
  /path/to/voicevox-<platform>.zip \
  --extract-runtime runtime-root
```

### VOICEVOX 製品 zip と標準 ORT archive を 1 回で入れる

```sh
dist/litevox model-dump \
  /path/to/voicevox-<platform>.zip \
  --add-model /path/to/onnxruntime-<platform>.archive \
  --extract-runtime runtime-root
```

### 標準 ORT archive だけ後から足す

```sh
dist/litevox model-dump \
  /path/to/onnxruntime-<platform>.archive \
  --extract-onnxruntime runtime-root
```

作った runtime root はこう使います。

```sh
runtime-root/litevox runtime_info --runtime runtime-root
runtime-root/litevox server --runtime runtime-root --port 50021
```

## よく使うコマンド

### TTS

```sh
dist/litevox tts \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text 'ずんだもんなのだ。' \
  --out out.wav
```

### AudioQuery

```sh
dist/litevox query \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text 'ずんだもんなのだ。'
```

### ストリーミング

```sh
dist/litevox stream \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text 'ずんだもんなのだ。ずんだもんなのだ。' \
  --format wav > out.wav
```

### 歌唱

[公式サイト](https://voicevox.hiroshiba.jp/)で配布されている VOICEVOX 0.25.2 の
トーク・歌唱モデルを、C++ の `native` backend で利用できます。
モデルと辞書の利用規約は公式配布物・キャラクターごとの規約に従ってください。

歌唱スタイルの一覧と、楽譜から WAV を直接生成するコマンド:

```sh
dist/litevox singers --runtime /path/to/voicevox-macos-cpu-arm64-0.25.2.zip

dist/litevox sing \
  --runtime /path/to/voicevox-macos-cpu-arm64-0.25.2.zip \
  --score score.json \
  --teacher 6000 \
  --speaker 3000 \
  --out song.wav
```

`--teacher` は歌唱クエリを予測するスタイル（既定 6000）、`--speaker` は
音声を出す歌唱・ハミングスタイル（既定 3000）です。トークのスタイル ID とは異なります。
`--format pcm` や `--out -` も使えます。

楽譜は公式 `Score` 形式です。先頭に休符を置き、歌詞は音符ごとにひらがな・カタカナの
1 モーラを指定します。`frame_length` は 93.75 fps（24,000 Hz / 256）の長さです。
休符は `key: null`、`lyric: ""` とします。音符 ID は任意で、クエリの `note_id` に引き継がれます。
Unicode エスケープされた歌詞も読み込めます。

```json
{"notes":[
  {"key":null,"frame_length":15,"lyric":""},
  {"id":"note-1","key":60,"frame_length":45,"lyric":"ド"},
  {"id":"note-2","key":62,"frame_length":45,"lyric":"レ"},
  {"id":"note-3","key":64,"frame_length":45,"lyric":"ミ"},
  {"key":null,"frame_length":15,"lyric":""}
]}
```

クエリ・音高・音量を個別に扱う場合は既存コマンドを使います。
HTTP でも `/singers`、`/sing_frame_audio_query`、`/sing_frame_f0`、
`/sing_frame_volume`、`/frame_synthesis` を提供します。

```sh
dist/litevox sing-query \
  --runtime runtime-root \
  --score score.json \
  --speaker 6000 \
  --out frame_audio_query.json

dist/litevox sing-f0 \
  --runtime runtime-root \
  --score score.json \
  --frame-audio-query frame_audio_query.json \
  --speaker 6000

dist/litevox sing-volume \
  --runtime runtime-root \
  --score score.json \
  --frame-audio-query frame_audio_query.json \
  --speaker 6000

dist/litevox frame-synthesis \
  --runtime runtime-root \
  --frame-audio-query frame_audio_query.json \
  --speaker 3000 \
  --out frame.wav
```

## ONNX を取り出す

### `.vvm` または `VOICEVOX.zip` から exported ONNX を出す

```sh
dist/litevox vv-bin-export-onnx \
  /path/to/model.vvm \
  --runtime runtime-root \
  --extract-onnx exported-onnx
```

### runtime root から ONNX を出す

```sh
dist/litevox vv-bin-export-onnx \
  runtime-root/model-vvm/0.vvm \
  --runtime runtime-root \
  --extract-onnx exported-onnx
```

## 配布時の考え方

このリポジトリは、次の形で配布する前提です。

- GitHub repo: `mikumiku-jp/litevox`
- リポジトリ本体: ソースコードと tool
- release asset: 必要なら `bootstrap-bundle.tar.gz`

`bootstrap-bundle` は model なしです。  
ユーザーが VOICEVOX 製品 zip と標準 ORT archive を指定して runtime を組み立てます。

## 検証

`make check` は JSON の Unicode エスケープ、歌詞・音符 ID、入力エラー、
再サンプリングの長さ・ステレオ出力・チャンク間の連続性を確認します。
CLI smoke は実モデルでの読み上げ・歌唱・音高／音量再計算、44.1 kHz 歌唱出力と
HTTP ストリーミング、無効な楽譜に対する HTTP 400 を確認します。

### runtime 組み立てから HTTP 比較まで一括実行

```sh
tools/verify-runtime-from-archives.sh \
  /path/to/voicevox-<platform>.zip \
  /path/to/onnxruntime-<platform>.archive \
  runtime-root \
  verify-output
```

### Makefile から呼ぶ

```sh
make verify-runtime-from-archives \
  VOICEVOX_ZIP=/path/to/voicevox-<platform>.zip \
  ONNXRUNTIME_ARCHIVE=/path/to/onnxruntime-<platform>.archive \
  RUNTIME_ROOT=runtime-root \
  RESULT_PREFIX=verify-output
```

### CLI smoke

```sh
make verify-cli-smoke \
  RUNTIME_ROOT=runtime-root \
  RESULT_PREFIX=cli-smoke
```

## 重要な注意

- `--runtime VOICEVOX.zip` はそのまま使えます
- mac 版 zip 同梱 ORT は CPU のみです
- Windows DirectML 版 zip 同梱 ORT は Windows 上で GPU を使えます
- mac で GPU を使うなら、GPU provider を持つ標準 ORT を別途指定してください
- `model-dump --extract-runtime` は runtime root を固定で持ちたい場合に使います
- `--state-dir DIR` を使うと、`user_dict.json`、`presets.json`、`setting.json`、`core_libraries/` を runtime root から分離できます

## 主要ファイル

- `src/` 実装
- `resources/openapi.json` OpenAPI 定義
- `tools/verify-runtime-from-archives.sh` HTTP 比較付き検証
- `tools/verify-cli-smoke.sh` CLI smoke
- `README.bundle.md` bootstrap bundle 向け説明
