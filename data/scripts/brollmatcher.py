#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kdenlive Contributors
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

SCENE_CHANGE_THRESHOLD = 0.45
YOLO_CONFIDENCE_THRESHOLD = 0.5
MAX_SCENE_BONUS = 6
SCENE_BONUS_DIVISOR = 10.0
KEYWORD_MATCH_WEIGHT = 1.5
NO_KEYWORD_MATCH_PENALTY = 0.55


def _tokenize(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2}


def _safe_json_dump(payload: object, output: Optional[str]) -> None:
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        print(json.dumps(payload, indent=2), flush=True)


def _load_json_file(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class TranscriptEntry:
    timestamp: float
    text: str
    keywords: Set[str]


@dataclass
class ClipInfo:
    path: str
    objects: Set[str]
    scenes: int
    tags: Set[str]
    quality: float
    notes: List[str]


def _parse_transcript(data: object) -> List[TranscriptEntry]:
    if not isinstance(data, list):
        raise ValueError("Transcript JSON must be a list of entries")

    entries: List[TranscriptEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ts = float(item.get("timestamp", 0.0))
        text = str(item.get("text", ""))
        kw = item.get("keywords")
        if isinstance(kw, list):
            keywords = {str(k).strip().lower() for k in kw if str(k).strip()}
        else:
            keywords = _tokenize(text)
        entries.append(TranscriptEntry(timestamp=ts, text=text, keywords=keywords))
    return entries


def _parse_broll_list(data: object) -> List[str]:
    clips: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                clips.append(item)
            elif isinstance(item, dict) and "path" in item:
                clips.append(str(item["path"]))
    elif isinstance(data, dict) and isinstance(data.get("clips"), list):
        for item in data["clips"]:
            if isinstance(item, str):
                clips.append(item)
            elif isinstance(item, dict) and "path" in item:
                clips.append(str(item["path"]))
    return clips


def _filename_tags(path: str) -> Set[str]:
    base = os.path.splitext(os.path.basename(path))[0]
    return _tokenize(base.replace("_", " ").replace("-", " "))


def _try_load_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def _detect_scenes_basic(cv2, capture, sample_seconds: float, max_frames: int, notes: List[str]) -> int:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0
        notes.append("FPS unavailable, using fallback=25")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        notes.append("Could not read frame count")
        return 1

    step = max(1, int(fps * sample_seconds))
    last_hist = None
    scene_changes = 0
    read_frames = 0
    for frame_idx in range(0, total, step):
        if read_frames >= max_frames:
            notes.append("Scene scan truncated by max_frames")
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if last_hist is not None:
            diff = cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > SCENE_CHANGE_THRESHOLD:
                scene_changes += 1
        last_hist = hist
        read_frames += 1

    return max(1, scene_changes + 1)


def _load_yolo_detector(cv2, yolo_cfg: Optional[str], yolo_weights: Optional[str], yolo_names: Optional[str], notes: List[str]):
    if not yolo_cfg or not yolo_weights:
        notes.append("YOLO disabled (no cfg/weights provided)")
        return None, []
    if not os.path.isfile(yolo_cfg) or not os.path.isfile(yolo_weights):
        notes.append("YOLO disabled (cfg/weights path missing)")
        return None, []
    try:
        net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
        classes: List[str] = []
        if yolo_names and os.path.isfile(yolo_names):
            with open(yolo_names, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
        return net, classes
    except Exception as e:
        notes.append(f"YOLO init failed: {e}")
        return None, []


def _detect_objects_yolo(cv2, capture, net, classes: Sequence[str], sample_seconds: float, max_frames: int, notes: List[str]) -> Set[str]:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        return set()

    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    step = max(1, int(fps * sample_seconds))
    labels: Set[str] = set()
    scanned = 0

    for frame_idx in range(0, total, step):
        if scanned >= max_frames:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(out_layers)
        for out in outputs:
            for det in out:
                scores = det[5:]
                cls_id = int(scores.argmax())
                confidence = float(scores[cls_id])
                if confidence >= YOLO_CONFIDENCE_THRESHOLD:
                    if classes and 0 <= cls_id < len(classes):
                        labels.add(classes[cls_id].lower())
                    else:
                        labels.add(f"class_{cls_id}")
        scanned += 1

    if not labels:
        notes.append("YOLO ran but found no confident objects")
    return labels


def _analyze_clip(
    path: str,
    sample_seconds: float,
    max_frames: int,
    yolo_cfg: Optional[str],
    yolo_weights: Optional[str],
    yolo_names: Optional[str],
) -> ClipInfo:
    notes: List[str] = []
    tags = _filename_tags(path)
    objects: Set[str] = set()
    scenes = 1
    quality = 0.25

    if not os.path.isfile(path):
        notes.append("Clip file missing")
        return ClipInfo(path=path, objects=objects, scenes=scenes, tags=tags, quality=0.0, notes=notes)

    cv2 = _try_load_cv2()
    if cv2 is None:
        notes.append("OpenCV unavailable, using filename tags only")
        return ClipInfo(path=path, objects=objects, scenes=scenes, tags=tags, quality=0.3, notes=notes)

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        notes.append("OpenCV could not open clip")
        return ClipInfo(path=path, objects=objects, scenes=scenes, tags=tags, quality=0.2, notes=notes)

    try:
        scenes = _detect_scenes_basic(cv2, capture, sample_seconds, max_frames, notes)
        yolo_net, yolo_classes = _load_yolo_detector(cv2, yolo_cfg, yolo_weights, yolo_names, notes)
        if yolo_net is not None:
            objects = _detect_objects_yolo(cv2, capture, yolo_net, yolo_classes, sample_seconds, max_frames, notes)
        else:
            notes.append("Object detection fallback: tags only")
    finally:
        capture.release()

    tags = tags | objects
    quality = 0.4
    if objects:
        quality += 0.35
    if scenes > 1:
        quality += 0.15
    quality = min(1.0, quality)
    return ClipInfo(path=path, objects=objects, scenes=scenes, tags=tags, quality=quality, notes=notes)


def _rank_clips_for_entry(entry: TranscriptEntry, clips: Sequence[ClipInfo], top_k: int) -> List[Dict[str, object]]:
    ranked = []
    for clip in clips:
        overlap = sorted(entry.keywords & clip.tags)
        lexical = len(overlap)
        scene_bonus = min(clip.scenes, MAX_SCENE_BONUS) / SCENE_BONUS_DIVISOR
        score = (lexical * KEYWORD_MATCH_WEIGHT) + scene_bonus + clip.quality
        if lexical == 0:
            score *= NO_KEYWORD_MATCH_PENALTY
        ranked.append(
            {
                "path": clip.path,
                "score": round(score, 4),
                "matched_keywords": overlap,
                "objects": sorted(clip.objects),
                "scenes": clip.scenes,
                "analysis_notes": clip.notes,
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline B-roll contextual matcher")
    parser.add_argument("--transcript", required=True, help="JSON list with timestamp/text/keywords")
    parser.add_argument("--broll", required=True, help="JSON list with clip paths or {path: ...}")
    parser.add_argument("--output", default="", help="Output JSON path, default stdout")
    parser.add_argument("--top-k", type=int, default=3, help="Best clips returned per transcript timestamp")
    parser.add_argument("--sample-seconds", type=float, default=1.5, help="Frame sampling interval")
    parser.add_argument("--max-frames", type=int, default=120, help="Max sampled frames per clip")
    parser.add_argument("--yolo-cfg", default="", help="Optional local YOLO cfg file")
    parser.add_argument("--yolo-weights", default="", help="Optional local YOLO weights file")
    parser.add_argument("--yolo-names", default="", help="Optional local class names file")
    args = parser.parse_args()

    transcript_data = _load_json_file(args.transcript)
    broll_data = _load_json_file(args.broll)
    transcript = _parse_transcript(transcript_data)
    broll_paths = _parse_broll_list(broll_data)

    if not transcript:
        print("No transcript entries found", file=sys.stderr)
        return 1
    if not broll_paths:
        print("No B-roll clips found", file=sys.stderr)
        return 1

    clips = [
        _analyze_clip(
            path=path,
            sample_seconds=max(0.1, args.sample_seconds),
            max_frames=max(1, args.max_frames),
            yolo_cfg=args.yolo_cfg or None,
            yolo_weights=args.yolo_weights or None,
            yolo_names=args.yolo_names or None,
        )
        for path in broll_paths
    ]

    results = []
    for entry in transcript:
        results.append(
            {
                "timestamp": entry.timestamp,
                "text": entry.text,
                "keywords": sorted(entry.keywords),
                "matches": _rank_clips_for_entry(entry, clips, max(1, args.top_k)),
            }
        )

    payload = {
        "mode": "offline",
        "notes": [
            "OpenCV is used when available for scene analysis.",
            "YOLO is optional and only used with local cfg/weights.",
            "If dependencies are missing, ranking falls back to filename/keyword overlap.",
        ],
        "results": results,
    }
    _safe_json_dump(payload, args.output or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
