# LiteVox Bootstrap Bundle

この bundle にはモデルデータを含めません。  
実行に必要なモデルや ORT は、ユーザーが手元の配布物から与える前提です。
動的リンク版は libsoxr も必要です。macOS では `brew install libsoxr`、
Debian / Ubuntu では `libsoxr0` を用意してください。静的リンク版では追加不要です。

## 含まれるもの

- `litevox`
- `bundle-manifest.json`
- `SHA256SUMS`
- `resources/openapi.json`
- `tools/verify-runtime-from-archives.sh`
- `tools/verify-cli-smoke.sh`
- `tools/compare_voicevox_http.py`
- `tools/compare_voicevox_http_full.py`
- `tools/write_bundle_manifest.py`

## 含まれないもの

- VOICEVOX 製品 zip
- 標準 ONNX Runtime archive
- `.vvm`
- `.onnx`

## まず確認する

```sh
cd /path/to/bootstrap-bundle
shasum -a 256 -c SHA256SUMS
```

## runtime root を作る

### CPU だけで動かす

```sh
./litevox model-dump \
  /path/to/voicevox-<platform>.zip \
  --extract-runtime runtime-root
```

### GPU 用 ORT も一緒に入れる

```sh
./litevox model-dump \
  /path/to/voicevox-<platform>.zip \
  --add-model /path/to/onnxruntime-<platform>.archive \
  --extract-runtime runtime-root
```

## そのまま VOICEVOX zip を runtime に使う

```sh
./litevox runtime_info \
  --runtime /path/to/voicevox-<platform>.zip
```

初回だけ zip の横に `.litevox-runtime-cache/` を作ります。  
2 回目以降はその cache を再利用します。

Windows は `voicevox-windows-directml-*.zip` をそのまま指定できます。

## 起動

```sh
runtime-root/litevox server \
  --runtime runtime-root \
  --port 50021
```

## 検証

### runtime 組み立てから HTTP 比較まで

```sh
./tools/verify-runtime-from-archives.sh \
  /path/to/voicevox-<platform>.zip \
  /path/to/onnxruntime-<platform>.archive \
  runtime-root \
  verify-output
```

### CLI smoke

```sh
./tools/verify-cli-smoke.sh \
  runtime-root \
  cli-smoke
```

## 注意

- mac 版 VOICEVOX 製品 zip 同梱 ORT は CPU provider のみです
- Windows DirectML 版 VOICEVOX 製品 zip 同梱 ORT は Windows 上で GPU を使えます
- mac で GPU を使う場合は、GPU provider を持つ標準 ORT を別途与えてください
- bundle 自体は model なしです
