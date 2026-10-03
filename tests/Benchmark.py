"""Run with python3 tests/Benchmark.py; no inference runtime is required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

Root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as Dir:
    Score = Path(Dir) / "Score.json"
    Score.write_text(json.dumps(json.loads((Root / "tests/fixtures/sample_replay.json").read_text())["score"]))
    Result = subprocess.run(
        [sys.executable, str(Root / "tools/Benchmark.py"), "--engine", "test=http://127.0.0.1:1",
         "--score", str(Score), "--replay", str(Root / "tests/fixtures/sample_replay.json"), "--case", "talk-long",
         "--out", str(Path(Dir) / "Result.json")], capture_output=True, text=True)
    assert Result.returncode == 2, Result.stderr
    assert "replay missing requested cases: talk-long" in Result.stderr, Result.stderr
    assert not (Path(Dir) / "Result.json").exists()
# Both first-response differences and later nondeterminism must fail --exact.
import importlib.util
import io
from unittest.mock import patch
import wave

Spec = importlib.util.spec_from_file_location("Benchmark", Root / "tools/Benchmark.py")
Bench = importlib.util.module_from_spec(Spec)
Spec.loader.exec_module(Bench)

def Audio(Value):
    Buffer = io.BytesIO()
    with wave.open(Buffer, "wb") as Output:
        Output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        Output.writeframes(Value.to_bytes(2, "little", signed=True) * 2)
    return Buffer.getvalue()

for IsFirst in (True, False):
    Counts = {}
    def Request(Url, Body=None):
        if "/synthesis?" not in Url:
            return b"{}"
        Counts[Url] = Counts.get(Url, 0) + 1
        Changed = ("packed" in Url) if IsFirst else Counts[Url] == 4
        return Audio(2 if Changed else 1)
    with tempfile.TemporaryDirectory() as Dir:
        Score = Path(Dir) / "Score.json"
        Score.write_text(json.dumps(json.loads((Root / "tests/fixtures/sample_replay.json").read_text())["score"]))
        Argv = ["Benchmark", "--engine", "control=http://control", "--engine", "packed=http://packed",
                "--score", str(Score), "--replay", str(Root / "tests/fixtures/sample_replay.json"),
                "--case", "talk-short", "--runs", "2", "--exact", "--out", str(Path(Dir) / "Result.json")]
        with patch.object(sys, "argv", Argv), patch.object(Bench, "Request", Request):
            try:
                Bench.Main()
            except ValueError as Error:
                assert ("engines returned different WAV bytes" if IsFirst else
                        "measured WAV bytes changed") in str(Error), Error
            else:
                raise AssertionError("--exact accepted differing audio")
print("Benchmark replay and exact-audio validation passed")
