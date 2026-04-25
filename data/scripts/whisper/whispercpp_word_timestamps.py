#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Jean-Baptiste Mardelle <jb@kdenlive.org>
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile


def run_cmd(command):
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(shlex.quote(part) for part in command)
            + "\n\nstdout:\n"
            + process.stdout
            + "\n\nstderr:\n"
            + process.stderr
        )


def align_to_fps(seconds, fps):
    return round(round(seconds * fps) / fps, 6)


def normalize_audio(input_file, output_wav, ffmpeg_path, sample_rate):
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_file,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        "highpass=f=80,lowpass=f=8000,afftdn=nf=-25",
        output_wav,
    ]
    run_cmd(command)
    return command


def whisper_output_json_path(prefix):
    return prefix + ".json"


def run_whisper_cpp(
    normalized_wav,
    model_path,
    output_prefix,
    whisper_cli_path,
    use_gpu,
    gpu_layers,
):
    output_file_flags = [["--output-file", output_prefix], ["-of", output_prefix]]
    json_flags = [["--output-json-full"], ["-ojf"], ["--output-json"], ["-oj"]]
    extra_gpu_flags = []
    if use_gpu:
        extra_gpu_flags = ["--gpu-layers", str(gpu_layers)]

    last_error = None
    for output_flag in output_file_flags:
        for json_flag in json_flags:
            command = [
                whisper_cli_path,
                "-m",
                model_path,
                "-f",
                normalized_wav,
                *output_flag,
                *json_flag,
                "--no-prints",
                *extra_gpu_flags,
            ]
            try:
                run_cmd(command)
                json_path = whisper_output_json_path(output_prefix)
                if os.path.exists(json_path):
                    return command, json_path
                last_error = RuntimeError(f"Whisper.cpp finished but no JSON was generated at: {json_path}")
            except RuntimeError as exc:
                last_error = exc
                continue

    if last_error is None:
        raise RuntimeError("Whisper.cpp invocation failed with unknown error.")
    raise last_error


def _extract_token_text(token):
    for key in ("text", "token", "word"):
        value = token.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_token_times(token):
    for start_key, end_key in (("start", "end"), ("from", "to"), ("t0", "t1")):
        if start_key in token and end_key in token:
            start = float(token[start_key])
            end = float(token[end_key])
            if start_key == "t0" or end_key == "t1":
                # Some whisper.cpp JSON variants expose token times in 10 ms ticks
                # (for example t0=123 means 1.23 s), so convert those to seconds.
                if start > 100:
                    start *= 0.01
                if end > 100:
                    end *= 0.01
            return start, end
    return None


def _extract_words_from_segment(segment):
    words = []
    segment_words = segment.get("words", [])
    if isinstance(segment_words, list) and segment_words:
        for item in segment_words:
            text = _extract_token_text(item)
            times = _extract_token_times(item)
            if text and times is not None:
                words.append((text, times[0], times[1]))
        return words

    tokens = segment.get("tokens", [])
    if isinstance(tokens, list) and tokens:
        for token in tokens:
            text = _extract_token_text(token)
            times = _extract_token_times(token)
            if text and times is not None:
                words.append((text, times[0], times[1]))
        return words

    text = str(segment.get("text", "")).strip()
    start = segment.get("start")
    end = segment.get("end")
    if text and start is not None and end is not None:
        words.append((text, float(start), float(end)))
    return words


def parse_whisper_json_to_words(json_path, fps):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    segments = data.get("segments", [])
    words = []
    for segment in segments:
        for text, start, end in _extract_words_from_segment(segment):
            start_aligned = align_to_fps(start, fps)
            end_aligned = align_to_fps(end, fps)
            if end_aligned < start_aligned:
                end_aligned = start_aligned
            words.append(
                {
                    "word": text,
                    "start_time": start_aligned,
                    "end_time": end_aligned,
                }
            )
    return words


def main():
    parser = argparse.ArgumentParser(
        description="Normalize voiceover audio and extract Whisper.cpp word timestamps as JSON."
    )
    parser.add_argument("--input", required=True, help="Input audio/video file path.")
    parser.add_argument("--model", required=True, help="Path to Whisper.cpp model file (ggml*.bin).")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target timeline FPS (e.g. 24 or 30).")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Normalized audio sample rate.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg binary path.")
    parser.add_argument("--whisper-cli", default="whisper-cli", help="Whisper.cpp CLI binary path.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable Whisper.cpp GPU path.")
    parser.add_argument("--gpu-layers", type=int, default=35, help="Whisper.cpp GPU layers when --use-gpu is set.")
    parser.add_argument(
        "--tmp-wav",
        default="",
        help="Optional normalized wav path (kept if provided). Default uses a temporary file.",
    )
    parser.add_argument("--print-commands", action="store_true", help="Print executed FFmpeg/Whisper commands.")
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    model_path = os.path.abspath(args.model)
    output_file = os.path.abspath(args.output)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if args.fps <= 0:
        raise ValueError("FPS must be greater than 0.")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kdenlive-whispercpp-") as tmp_dir:
        normalized_wav = args.tmp_wav if args.tmp_wav else os.path.join(tmp_dir, "normalized.wav")
        whisper_prefix = os.path.join(tmp_dir, "whisper_output")

        ffmpeg_command = normalize_audio(input_file, normalized_wav, args.ffmpeg, args.sample_rate)
        whisper_command, whisper_json = run_whisper_cpp(
            normalized_wav=normalized_wav,
            model_path=model_path,
            output_prefix=whisper_prefix,
            whisper_cli_path=args.whisper_cli,
            use_gpu=args.use_gpu,
            gpu_layers=args.gpu_layers,
        )

        words = parse_whisper_json_to_words(whisper_json, args.fps)
        payload = {
            "input_file": input_file,
            "normalized_audio": os.path.abspath(normalized_wav),
            "fps": args.fps,
            "sample_rate": args.sample_rate,
            "words": words,
        }
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        if args.print_commands:
            print("FFmpeg command:", " ".join(shlex.quote(part) for part in ffmpeg_command), flush=True)
            print("Whisper.cpp command:", " ".join(shlex.quote(part) for part in whisper_command), flush=True)
        print(f"Wrote {len(words)} words to: {output_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
