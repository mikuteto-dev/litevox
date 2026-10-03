#!/usr/bin/env python3
"""Compare HTTP engines with identical official queries and existing song scores."""
import argparse
import array
import concurrent.futures
import hashlib
import io
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path


def Request(Url, Body=None):
    Call = urllib.request.Request(Url, data=Body, method="POST" if Body is not None else "GET",
                                  headers={"Content-Type": "application/json", "Connection": "close"})
    try:
        with urllib.request.urlopen(Call, timeout=600) as Response:
            return Response.read()
    except urllib.error.HTTPError as Error:
        raise RuntimeError(f"{Url}: HTTP {Error.code}: {Error.read(2048).decode(errors='replace')}") from Error


def Encode(Value):
    return json.dumps(Value, ensure_ascii=True, separators=(",", ":")).encode()


def Measure(Url, Body):
    Start = time.perf_counter()
    Data = Request(Url, Body)
    return (time.perf_counter() - Start) * 1000, Data


def WaveInfo(Data):
    with wave.open(io.BytesIO(Data)) as Audio:
        Shape = [Audio.getframerate(), Audio.getnchannels(), Audio.getsampwidth(), Audio.getnframes()]
        if Shape[0] != 24000 or Shape[1:3] != [1, 2] or Shape[3] <= 0:
            raise ValueError(f"unexpected WAV shape: {Shape}")
        Samples = array.array("h", Audio.readframes(Shape[3]))
    if sys.byteorder != "little":
        Samples.byteswap()
    if len(Samples) != Shape[3]:
        raise ValueError("truncated WAV")
    Rms = math.sqrt(sum(Value * Value for Value in Samples) / len(Samples)) / 32768
    if Rms == 0:
        raise ValueError("silent WAV")
    return {"shape": Shape, "seconds": Shape[3] / Shape[0], "rms": Rms,
            "peak": max(abs(Value) for Value in Samples) / 32768,
            "sha256": hashlib.sha256(Data).hexdigest()}


def Main():
    Parser = argparse.ArgumentParser(description=__doc__)
    Parser.add_argument("--engine", action="append", required=True, metavar="NAME=HTTP_URL",
                        help="First engine supplies reference queries; normally official VOICEVOX")
    Parser.add_argument("--score", type=Path, required=True)
    Parser.add_argument("--runs", type=int, default=10)
    Parser.add_argument("--workers", type=int, default=1)
    Parser.add_argument("--out", type=Path, required=True)
    Parser.add_argument("--case", action="append", choices=["talk-short", "talk-long", "song"], dest="Cases")
    Parser.add_argument("--replay", type=Path, help="Reuse exact saved request payloads instead of generating new reference queries")
    Parser.add_argument("--exact", action="store_true", help="Require byte-identical first and measured WAV responses across engines")
    Args = Parser.parse_args()
    if Args.runs < 2 or not 1 <= Args.workers <= Args.runs:
        Parser.error("runs >= 2 and 1 <= workers <= runs are required")
    Engines = {}
    for Entry in Args.engine:
        if "=" not in Entry:
            Parser.error("Use NAME=http://HOST:PORT")
        Name, Url = Entry.split("=", 1)
        Parsed = urllib.parse.urlsplit(Url)
        if not Name or Name in Engines or Parsed.scheme != "http" or not Parsed.hostname or Parsed.username or Parsed.password or Parsed.path not in ("", "/") or Parsed.query or Parsed.fragment:
            Parser.error("Use unique NAME=http://HOST:PORT entries without credentials or URL paths")
        Engines[Name] = Url.rstrip("/")
    Score = json.loads(Args.score.read_text())
    Selected = Args.Cases or ["talk-short", "talk-long", "song"]
    if Args.replay:
        Saved = json.loads(Args.replay.read_text())
        if Saved["score"] != Score:
            Parser.error("replay score differs from --score")
        Missing = set(Args.Cases or []) - {Name.rsplit("-", 1)[0] for Name in Saved["cases"]}
        if Missing:
            Parser.error("replay missing requested cases: " + ", ".join(sorted(Missing)))
        Cases = []
        for Name, Case in Saved["cases"].items():
            if Name.rsplit("-", 1)[0] not in Selected:
                continue
            Target = Case["path"]
            Endpoint = urllib.parse.urlsplit(Target).path
            if not Target.startswith("/") or Endpoint not in ["/audio_query", "/synthesis", "/sing_frame_audio_query", "/frame_synthesis"]:
                Parser.error("replay contains a non-benchmark endpoint")
            Body = Encode(Case["request"]) if Case["request"] is not None else b""
            if hashlib.sha256(Body).hexdigest() != Case["body_sha256"]:
                Parser.error("replay payload hash differs")
            Cases.append((Name, Target, Body, Endpoint in ["/synthesis", "/frame_synthesis"]))
        if not Cases:
            Parser.error("replay contains no selected cases")
    else:
        Reference = next(iter(Engines.values()))
        Cases = []
        for Name, Text in [("talk-short", "こんにちは。"), ("talk-long", "こんにちは。音声合成の速度を確認しています。自然な日本語の読み上げと歌声を同じ条件で比較します。")]:
            if Name not in Selected:
                continue
            Target = "/audio_query?" + urllib.parse.urlencode({"speaker": 3, "text": Text})
            Query = json.loads(Request(Reference + Target, b""))
            Query.update(outputSamplingRate=24000, outputStereo=False)
            Cases.extend([(Name + "-query", Target, b"", False),
                          (Name + "-synthesis", "/synthesis?speaker=3", Encode(Query), True)])
        if "song" in Selected:
            Frame = json.loads(Request(Reference + "/sing_frame_audio_query?speaker=6000", Encode(Score)))
            Frame.update(outputSamplingRate=24000, outputStereo=False)
            Cases.extend([("song-query", "/sing_frame_audio_query?speaker=6000", Encode(Score), False),
                          ("song-synthesis", "/frame_synthesis?speaker=3000", Encode(Frame), True)])
    Result = {"versions": {Name: json.loads(Request(Url + "/version")) for Name, Url in Engines.items()},
              "runs": Args.runs, "workers": Args.workers, "score": Score,
              "method": "HTTP POST, new connection; reference query bytes shared; first request and two warmups excluded; engines alternate by trial",
              "cases": {}}
    for Name, Target, Body, IsWave in Cases:
        Records = {Engine: {"samples_ms": []} for Engine in Engines}
        Shapes = []
        for Engine, Url in Engines.items():
            First, Data = Measure(Url + Target, Body)
            Records[Engine]["first_request_ms"] = First
            if IsWave:
                Info = WaveInfo(Data)
                Shapes.append(Info["shape"])
                Records[Engine]["wave"] = Info
            else:
                json.loads(Data)
            for _ in range(2):
                Request(Url + Target, Body)
        if IsWave and any(Shape != Shapes[0] for Shape in Shapes):
            raise ValueError(f"{Name}: engines returned different WAV shapes: {Shapes}")
        if IsWave and Args.exact and len({Record["wave"]["sha256"] for Record in Records.values()}) != 1:
            raise ValueError(f"{Name}: engines returned different WAV bytes")
        # Alternating engine order reduces order/thermal bias; do not run engines simultaneously.
        with concurrent.futures.ThreadPoolExecutor(max_workers=Args.workers) as Pool:
            for Trial in range(Args.runs):
                Order = list(Engines.items())
                if Trial % 2:
                    Order.reverse()
                for Engine, Url in Order:
                    Calls = [Pool.submit(Measure, Url + Target, Body) for _ in range(Args.workers)]
                    for Call in Calls:
                        Elapsed, Data = Call.result()
                        if IsWave:
                            if Args.exact and hashlib.sha256(Data).hexdigest() != Records[Engine]["wave"]["sha256"]:
                                raise ValueError(f"{Name}/{Engine}: measured WAV bytes changed")
                            with wave.open(io.BytesIO(Data)) as Audio:
                                Shape = [Audio.getframerate(), Audio.getnchannels(), Audio.getsampwidth(), Audio.getnframes()]
                                if Shape != Shapes[0] or len(Audio.readframes(Shape[3])) != Shape[3] * Shape[1] * Shape[2]:
                                    raise ValueError(f"{Name}/{Engine}: inconsistent or truncated WAV: {Shape}")
                        else:
                            json.loads(Data)
                        Records[Engine]["samples_ms"].append(Elapsed)
        for Record in Records.values():
            Samples = Record["samples_ms"]
            Record.update(mean_ms=statistics.mean(Samples), median_ms=statistics.median(Samples),
                          min_ms=min(Samples), max_ms=max(Samples))
            if IsWave:
                Record["rtf"] = Record["median_ms"] / 1000 / Record["wave"]["seconds"]
        Result["cases"][Name] = {"path": Target, "request": json.loads(Body) if Body else None,
                                  "body_sha256": hashlib.sha256(Body).hexdigest(), "engines": Records}
        Args.out.parent.mkdir(parents=True, exist_ok=True)
        Args.out.write_text(json.dumps(Result, ensure_ascii=False, indent=2) + "\n")
        print(Name, " / ".join(f"{Engine}: {Record['median_ms']:.3f} ms" for Engine, Record in Records.items()), flush=True)


if __name__ == "__main__":
    Main()
