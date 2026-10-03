# LiteVox Bootstrap Bundle

LiteVox Bootstrap Bundle は、モデルデータを含まない軽量なポータブル配布パッケージです。  
お手元の VOICEVOX 公式配布パッケージ（zip）や ONNX Runtime と組み合わせてランタイム環境を構築して利用します。

> **依存ライブラリについて**  
> 動的リンク版バイナリを使用する場合は `libsoxr` が必要です。  
> - **macOS**: `brew install libsoxr`  
> - **Debian / Ubuntu**: `sudo apt-get install libsoxr0`  
> ※ 静的リンク版の場合は追加インストール不要です。

---

## パッケージ内容

### 含まれるファイル

- `litevox`（実行可能バイナリ）
- `bundle-manifest.json`（マニフェスト）
- `SHA256SUMS`（チェックサム）
- `resources/openapi.json`（OpenAPI 定義）
- `tools/`（動作検証・比較用スクリプト群）

### 別途必要なもの

- VOICEVOX 製品 zip（各プラットフォーム向け公式配布物）
- （GPU 利用時）対応する Execution Provider を含む ONNX Runtime アーカイブ

---

## ファイルの検証

配布アーカイブの整合性を確認します。

```sh
cd /path/to/bootstrap-bundle
shasum -a 256 -c SHA256SUMS
```

---

## 使い方

### 方法 1: 公式 zip を直接指定して実行する

VOICEVOX 製品 zip をそのまま `--runtime` に指定して実行できます。初回実行時に zip と同じディレクトリに自動展開キャッシュ（`.litevox-runtime-cache/`）が作成され、次回以降は高速に起動します。

```sh
# ランタイム情報の確認
./litevox runtime_info \
  --runtime /path/to/voicevox-<platform>.zip

# 音声合成 (TTS)
./litevox tts \
  --runtime /path/to/voicevox-<platform>.zip \
  --speaker 3 \
  --text 'こんにちは。' \
  --out output.wav

# HTTP サーバーの起動
./litevox server \
  --runtime /path/to/voicevox-<platform>.zip \
  --port 50021
```

### 方法 2: ランタイムディレクトリ (runtime root) を作成して実行する

あらかじめ展開済みのランタイムディレクトリ（runtime root）を構築して運用することも可能です。

#### CPU 実行用ランタイムの作成

```sh
./litevox model-dump \
  /path/to/voicevox-<platform>.zip \
  --extract-runtime runtime-root
```

#### GPU 対応 ONNX Runtime を組み込んだランタイムの作成

```sh
./litevox model-dump \
  /path/to/voicevox-<platform>.zip \
  --add-model /path/to/onnxruntime-<platform>.archive \
  --extract-runtime runtime-root
```

#### サーバーの起動

```sh
runtime-root/litevox server \
  --runtime runtime-root \
  --port 50021
```

---

## 動作検証

同梱の検証スクリプトを使用して、環境のセットアップと動作を確認できます。

```sh
# ランタイム構築から HTTP 互換性テストまでの検証
./tools/verify-runtime-from-archives.sh \
  /path/to/voicevox-<platform>.zip \
  /path/to/onnxruntime-<platform>.archive \
  runtime-root \
  verify-output

# CLI 基本動作の検証 (smoke test)
./tools/verify-cli-smoke.sh \
  runtime-root \
  cli-smoke
```

---

## 注意事項

- **macOS での GPU 利用**: macOS 版の公式製品 zip に同梱されている ONNX Runtime は CPU 専用です。macOS 上で GPU（Metal / WebGPU）を利用する場合は、GPU プロバイダに対応した標準 ONNX Runtime を別途指定してください。
- **Windows での GPU 利用**: `voicevox-windows-directml-*.zip` に同梱されている ONNX Runtime は DirectML に対応しており、そのまま GPU を利用可能です。
- **モデルの利用条件**: 本パッケージには音声モデルは含まれていません。合成に使用する音声モデルの利用規約は、VOICEVOX 公式および各キャラクターの利用規約をご確認ください。
