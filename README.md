# LiteVox

LiteVox は、VOICEVOX 互換の音声合成・歌唱音声合成を提供する C++ 製のスタンドアロン CLI / HTTP ランタイムです。

公式の VOICEVOX 配布パッケージ（製品 zip）や ONNX Runtime と組み合わせることで、Python 実行環境や VOICEVOX Core に依存することなく、高速・軽量に音声合成を実行できます。

## 主な特徴

- **公式 zip の直接利用**: 公式配布の `voicevox-*.zip` を指定するだけで、自動的に展開・キャッシュして即座に起動可能
- **高速な C++ ネイティブ推論**: 依存関係を最小限に抑えた C++17 実装による高速な音声合成
- **GPU アクセラレーション**: WebGPU（Metal / DirectX / Vulkan）、DirectML、CUDA、CoreML に対応
- **完全な機能サポート**: テキスト音声合成（TTS）、AudioQuery 生成、ストリーミング出力に加え、楽譜（Score）からの歌唱音声合成に対応
- **VOICEVOX 互換 HTTP サーバー**: 既存の VOICEVOX クライアントやエコシステムとそのまま連携可能
- **状態の分離管理**: ユーザー辞書やプリセットなどの設定を、モデルデータから分離して永続化可能

---

## パフォーマンス（公式 VOICEVOX との比較）

LiteVox は、計算負荷の大半を占める波形合成デコーダを GPU（Metal / DirectML / CUDA / WebGPU）にオフロードし、軽量な音長・音高予測モデルを CPU で効率的に実行するハイブリッド推論パイプラインを採用しています。

これにより、公式の CPU 実行と比較して大幅な高速化（レイテンシ短縮・RTF 改善）を実現します。

### 測定例（Apple M2）

同一モデル（VOICEVOX 0.25.2）、同一クエリ、同一出力フォーマット（24 kHz / 16-bit PCM）での波形合成レイテンシの比較例です。

| 処理内容 | 公式 VOICEVOX (CPU, 1スレッド) | LiteVox (GPU / Metal) | 高速化倍率 | LiteVox RTF※ |
|---|---:|---:|---:|---:|
| **音声合成（読み上げ）波形生成** | 1,432 ms | **263 ms** | **約 5.4 倍** | **0.25** |
| **歌唱音声（ソング）波形生成** | 884 ms | **211 ms** | **約 4.2 倍** | **0.12** |

※ **RTF（Real-Time Factor）**: 音声の長さに対する生成時間の比率（値が小さいほど高速、1.0 未満で実時間より高速）。  
※ CPU 同士の同一スレッド・同一 ONNX Runtime 比較では公式と同等水準の処理速度を維持します。また、CPU 実行時もスレッド数を指定（`--cpu-threads 4` など）することで高速化が可能です。

同一条件でのレイテンシ比較には、付属のスクリプト（`tools/Benchmark.py`）を利用できます。

---

## 動作要件・ビルド

### 必要な環境・ライブラリ

- C++17 対応コンパイラ（Clang / GCC）
- CMake, GNU Make, Git, pkg-config
- zlib
- libsoxr（音声リサンプリング用）

#### ライブラリのインストール例

- **macOS (Homebrew)**:
  ```sh
  brew install cmake libsoxr pkg-config
  ```
- **Ubuntu / Debian**:
  ```sh
  sudo apt-get install cmake make g++ git pkg-config zlib1g-dev libsoxr-dev
  ```

※ Open JTalk（1.11）のソースコードは、初回ビルド時に自動で取得・静的ビルドされます。既存の Open JTalk を使用する場合は、`OPEN_JTALK_INCLUDE_DIR` と `OPEN_JTALK_LIB` を指定してください。

### ビルド手順

```sh
# ビルド
make dist

# テスト実行
make check
```

ビルドが完了すると、`dist/` ディレクトリに以下のファイルが生成されます：
- `dist/litevox`（実行可能バイナリ）
- `dist/resources/`（OpenAPI 定義など）

---

## クイックスタート

### 1. 公式 zip を指定して音声を合成する (TTS)

`--runtime` オプションにダウンロードした VOICEVOX の配布 zip（macOS 版、Windows DirectML 版など）を直接指定して実行できます。

初回実行時に zip と同じディレクトリに `.litevox-runtime-cache/` が自動作成され、必要なランタイムファイルが展開されます。2回目以降はこのキャッシュが再利用されるため、高速に起動します。

```sh
dist/litevox tts \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text 'こんにちは、音声合成のテストです。' \
  --out output.wav
```

### 2. HTTP サーバーを起動する

VOICEVOX 互換の HTTP サーバーを起動します（デフォルトポート: 50021）。

```sh
dist/litevox server \
  --runtime /path/to/voicevox-<platform>.zip \
  --port 50021
```

起動後は、通常の VOICEVOX エディタや API クライアントから接続して利用できます。

### 3. 利用可能なモデル一覧を確認する

```sh
dist/litevox models \
  --runtime /path/to/voicevox-<platform>.zip
```

---

## 主な機能とコマンド例

### 音声合成 (TTS)

```sh
dist/litevox tts \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text 'ずんだもんなのだ。' \
  --out out.wav
```

### 音声合成クエリの生成 (AudioQuery)

```sh
dist/litevox query \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text 'ずんだもんなのだ。' \
  --out query.json
```

### 音声ストリーミング (Stream)

長文を文単位で逐次生成しながら標準出力やパイプへ出力します。

```sh
dist/litevox stream \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text '第一文です。第二文です。' \
  --format wav > out.wav
```

### 歌唱音声合成 (Song)

公式の楽譜フォーマット（Score JSON）から歌声を合成できます。

```sh
# 歌唱対応話者（シンガー）一覧の表示
dist/litevox singers \
  --runtime /path/to/voicevox-<platform>.zip

# 楽譜から直接 WAV を合成
dist/litevox sing \
  --runtime /path/to/voicevox-<platform>.zip \
  --score score.json \
  --teacher 6000 \
  --speaker 3000 \
  --out song.wav
```

- `--teacher`: 歌唱クエリ（音高・音量予測）用スタイル ID（デフォルト: 6000）
- `--speaker`: 波形合成用歌唱スタイル ID（デフォルト: 3000）

#### 楽譜ファイル形式 (Score JSON)

楽譜は VOICEVOX 公式の `Score` 形式に準拠します。歌詞にはモーラ（ひらがな・カタカナ）、`frame_length` にはフレーム数（93.75 fps: 24,000 Hz / 256 サンプル単位）を指定します。休符は `key: null`, `lyric: ""` とします。

```json
{
  "notes": [
    {"key": null, "frame_length": 15, "lyric": ""},
    {"id": "note-1", "key": 60, "frame_length": 45, "lyric": "ド"},
    {"id": "note-2", "key": 62, "frame_length": 45, "lyric": "レ"},
    {"id": "note-3", "key": 64, "frame_length": 45, "lyric": "ミ"},
    {"key": null, "frame_length": 15, "lyric": ""}
  ]
}
```

#### フレーム単位の詳細な歌唱処理

HTTP API および CLI では、歌唱クエリの生成、ピッチ（F0）や音量の個別調整、波形生成を段階的に実行できます。

```sh
# 1. 歌唱クエリの生成
dist/litevox sing-query \
  --runtime runtime-root \
  --score score.json \
  --speaker 6000 \
  --out frame_audio_query.json

# 2. ピッチ (F0) の予測・編集
dist/litevox sing-f0 \
  --runtime runtime-root \
  --score score.json \
  --frame-audio-query frame_audio_query.json \
  --speaker 6000

# 3. 音量の予測・編集
dist/litevox sing-volume \
  --runtime runtime-root \
  --score score.json \
  --frame-audio-query frame_audio_query.json \
  --speaker 6000

# 4. 波形合成
dist/litevox frame-synthesis \
  --runtime runtime-root \
  --frame-audio-query frame_audio_query.json \
  --speaker 3000 \
  --out frame.wav
```

---

## GPU アクセラレーション

LiteVox は各プラットフォームに応じた GPU アクセラレーションに対応しています。

### 対応 Execution Provider

- **macOS**: WebGPU (Metal), CoreML
- **Windows**: DirectML, CUDA, WebGPU (Direct3D 12)
- **Linux**: CUDA, WebGPU (Vulkan)

推論パイプラインでは、転送オーバーヘッドの小さい音長・音高予測モデルを CPU で処理し、計算負荷の大半を占める波形合成デコーダを GPU に配置するハイブリッド実行を行うことで、高いスループットを実現しています。

### GPU の利用方法

`--acceleration-mode` オプションで動作モードを指定できます：
- `auto`: 利用可能な GPU Execution Provider を自動検出し、利用できない場合は CPU で動作します（デフォルト）
- `gpu`: GPU による実行を要求します（対応環境が見つからない場合はエラー）
- `cpu`: 常に CPU で実行します

```sh
# GPU モードで実行
dist/litevox tts \
  --runtime /path/to/voicevox-<platform>.zip \
  --acceleration-mode gpu \
  --speaker 3 \
  --text 'GPU で合成しています。' \
  --out gpu_out.wav
```

> **macOS での利用時の注意**  
> macOS 版の VOICEVOX 公式 zip に同梱されている ONNX Runtime は CPU 専用ビルドです。macOS 上で GPU（Metal / WebGPU）を利用する場合は、GPU プロバイダを有効化してビルドした ONNX Runtime ライブラリを `--onnxruntime` オプションで指定してください。
> ```sh
> LITEVOX_VV_BIN_ONNXRUNTIME=/path/to/runtime-root/libvoicevox_onnxruntime.dylib \
> dist/litevox tts \
>   --runtime runtime-root \
>   --onnxruntime /path/to/libonnxruntime.dylib \
>   --acceleration-mode gpu \
>   --speaker 3 \
>   --text 'Metal で合成しています。'
> ```

---

## ランタイム環境の構築・管理

### ランタイムディレクトリ (runtime root) の明示的な展開

配布 zip を毎回読み込むのではなく、あらかじめ展開されたディレクトリ（runtime root）を作成して運用することができます。

```sh
# 公式 zip から runtime root を構築
dist/litevox model-dump \
  /path/to/voicevox-<platform>.zip \
  --extract-runtime runtime-root

# 標準 ONNX Runtime も同時に組み込む場合
dist/litevox model-dump \
  /path/to/voicevox-<platform>.zip \
  --add-model /path/to/onnxruntime-<platform>.archive \
  --extract-runtime runtime-root

# 展開した runtime root を指定して起動
runtime-root/litevox server \
  --runtime runtime-root \
  --port 50021
```

### ユーザーデータ・設定の分離 (`--state-dir`)

`--state-dir` を指定することで、ユーザー辞書（`user_dict.json`）、プリセット（`presets.json`）、設定ファイル（`setting.json`）をランタイム本体とは別のディレクトリに保存・永続化できます。

```sh
dist/litevox server \
  --runtime runtime-root \
  --state-dir /path/to/user-data \
  --port 50021
```

### ONNX モデルの抽出

`.vvm` ファイルや製品 zip から ONNX モデルを抽出するユーティリティも備えています。

```sh
dist/litevox vv-bin-export-onnx \
  /path/to/model.vvm \
  --runtime runtime-root \
  --extract-onnx exported-onnx
```

---

## テストと検証

```sh
# 単体テストおよび互換性チェック
make check

# 公式 zip と ONNX Runtime アーカイブからの組み立て検証
tools/verify-runtime-from-archives.sh \
  /path/to/voicevox-<platform>.zip \
  /path/to/onnxruntime-<platform>.archive \
  runtime-root \
  verify-output

# CLI の動作検証 (smoke test)
tools/verify-cli-smoke.sh runtime-root cli-smoke
```

---

## 主な構成

- `src/`: C++ 実装ソースコード
- `resources/openapi.json`: VOICEVOX 互換 OpenAPI 定義
- `tools/`: 検証・比較用ユーティリティスクリプト
- `README.bundle.md`: 配布用 Bootstrap Bundle 向けドキュメント

---

## ライセンス・利用規約

- LiteVox 自体のソースコードは本リポジトリのライセンスに従います。
- 合成に使用する音声モデル、キャラクターボイス、辞書データ等の著作権および利用条件は、VOICEVOX 公式および各キャラクターの利用規約に準拠します。商用利用やクレジット表記等の条件については各モデル・キャラクターの規約をご確認ください。
