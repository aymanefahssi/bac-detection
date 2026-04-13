"""
Video analysis pipeline -- batch detection, no cloud/infra.

1. Sample frames at 1.5s intervals
2. System 1 (bins) + System 2 (emptying) run in parallel
3. Multi-method aggregation + smart decision engine
4. Save JSON result locally
"""

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import cv2
from PIL import Image

from config import cfg
from detection.bins import SimpleTrack, aggregate_bin_results
from detection.emptying import EmptyingStateMachine
from detection.models import get_bin_models, get_emptying_models


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(video_path: str, interval: float, label: str = "") -> List[Dict]:
    """Sample frames from *video_path* at *interval* seconds."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    interval_frames = max(1, int(fps * interval))
    indices = list(range(0, total_frames, interval_frames))

    tag = f" ({label})" if label else ""
    print(f"Video: {total_frames} frames @ {fps:.1f} fps ({total_frames / fps:.1f}s)")
    print(f"Sampling{tag}: {len(indices)} frames (every {interval}s)")

    sampled = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        sampled.append({
            "frame_idx": idx,
            "timestamp": idx / fps,
            "frame": frame,
        })

    cap.release()
    return sampled


# ---------------------------------------------------------------------------
# System 1: Bin Detection
# ---------------------------------------------------------------------------

def _run_system1(frames: List[Dict]) -> Dict:
    print("=" * 70)
    print("SYSTEM 1: BIN DETECTION")
    print("=" * 70)

    bin_model, material_model, size_model = get_bin_models()
    device = cfg.DEVICE
    half = cfg.HALF_PRECISION
    conf = cfg.YOLO_CONFIDENCE

    frame_results = []
    all_crops = []
    crop_map = []

    # Detect bins in all frames
    for idx, fd in enumerate(frames):
        pil = Image.fromarray(cv2.cvtColor(fd["frame"], cv2.COLOR_BGR2RGB))
        results = bin_model.predict(
            pil, imgsz=640, conf=conf, device=device, half=half, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()

        frame_results.append({
            "frame_idx": fd["frame_idx"],
            "timestamp": fd["timestamp"],
            "detections": [],
            "bin_count": len(boxes),
        })

        for box in boxes:
            crop = pil.crop((box[0], box[1], box[2], box[3]))
            all_crops.append(crop)
            crop_map.append({"list_idx": idx, "box": box, "timestamp": fd["timestamp"]})

        if (idx + 1) % 5 == 0:
            print(f"  [S1] Detected bins in {idx + 1}/{len(frames)} frames")

    # Batch classify material + size in parallel
    print(f"  [S1] Classifying {len(all_crops)} crops...")

    def _classify_mat(crops):
        return [material_model(c, device=device, verbose=False)[0].names[
            material_model(c, device=device, verbose=False)[0].probs.top1
        ] for c in crops] if False else []  # placeholder replaced below

    # Actually run classification properly
    mat_results = []
    sz_results = []

    def _do_material(crops):
        out = []
        for c in crops:
            pred = material_model(c, device=device, verbose=False)[0]
            out.append(pred.names[pred.probs.top1])
        return out

    def _do_size(crops):
        out = []
        for c in crops:
            pred = size_model(c, device=device, verbose=False)[0]
            out.append(pred.names[pred.probs.top1])
        return out

    if all_crops:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_mat = pool.submit(_do_material, all_crops)
            f_sz = pool.submit(_do_size, all_crops)
            mat_results = f_mat.result()
            sz_results = f_sz.result()

    print("  [S1] Classification complete")

    # Build detections + tracking
    tracks: List[SimpleTrack] = []
    next_id = 0

    for ci, info in enumerate(crop_map):
        li = info["list_idx"]
        box = info["box"]
        material = mat_results[ci]
        size_label = sz_results[ci]

        frame_results[li]["detections"].append({
            "box": box, "material": material, "size": size_label})

        matched = False
        for track in tracks:
            if track.matches(box):
                track.update(box, material, size_label, info["timestamp"])
                matched = True
                break
        if not matched:
            tracks.append(SimpleTrack(next_id, box, material, size_label, info["timestamp"]))
            next_id += 1

    print(f"  [S1] Tracking: {len(tracks)} tracks")

    aggregated = aggregate_bin_results(frame_results, tracks)
    print("=" * 70)
    return aggregated


# ---------------------------------------------------------------------------
# System 2: Emptying Detection
# ---------------------------------------------------------------------------

def _run_system2(video_path: str) -> Dict:
    """Detect emptying events with dense sampling (0.5s) and low confidence."""
    print("=" * 70)
    print("SYSTEM 2: EMPTYING DETECTION")
    print("=" * 70)

    emptying_model, fullness_model = get_emptying_models()
    device = cfg.DEVICE
    half = cfg.HALF_PRECISION
    conf = cfg.EMPTYING_CONFIDENCE

    # Sample frames at denser interval than System 1
    frames = _sample_frames(video_path, cfg.EMPTYING_FRAME_INTERVAL, "S2 emptying")

    sm = EmptyingStateMachine()

    print(f"[S2] Processing {len(frames)} frames (conf={conf})...")

    for idx, fd in enumerate(frames):
        ts = fd["timestamp"]
        pil = Image.fromarray(cv2.cvtColor(fd["frame"], cv2.COLOR_BGR2RGB))

        results = emptying_model.predict(
            pil, imgsz=640, conf=conf, device=device, half=half, verbose=False)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            detected_class = results[0].names[int(boxes.cls[best_idx].item())]
            bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
            crop = pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            state = "normal" if detected_class == "emptying" else "emptying"

            if idx < 5 or idx % 10 == 0:
                c = boxes.conf[best_idx].item()
                print(f"  [S2] t={ts:.1f}s: '{detected_class}' (conf={c:.2f}) -> {state}")

            sm.process_frame(state, ts, crop, bbox, num_bins=-1,
                             is_model_detection=True)
        else:
            # No detection = nothing happening = treat as normal
            if idx < 5 or idx % 10 == 0:
                print(f"  [S2] t={ts:.1f}s: No detection -> normal (implicit)")
            sm.process_frame("normal", ts, crop=None, bbox=None,
                             num_bins=-1, is_model_detection=False)

    # Classify fullness
    events = []
    for event in sm.completed_events:
        crops = [f["crop"] for f in event.get("frames", []) if f.get("crop")]

        if crops:
            votes = []
            for crop in crops:
                try:
                    pred = fullness_model(crop, device=device, verbose=False)[0]
                    votes.append(pred.names[pred.probs.top1])
                except Exception:
                    votes.append("unknown")
            fullness = "full" if "full" in votes else "empty"
        else:
            # No crop images available — still count the event
            fullness = "unknown"

        start = event.get("start_time") or 0
        end = event.get("end_time") or start + 5
        events.append({
            "event_id": event["event_id"],
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
            "fullness": fullness,
        })

    empty_n = sum(1 for e in events if e["fullness"] == "empty")
    full_n = sum(1 for e in events if e["fullness"] == "full")

    print(f"\n[S2] Found {len(events)} emptying events ({empty_n} empty, {full_n} full)")
    print("=" * 70)

    return {
        "emptying_events": events,
        "summary": {
            "total_emptying_events": len(events),
            "empty_count": empty_n,
            "full_count": full_n,
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(video_path: str) -> Dict:
    overall_start = time.time()

    print("\n" + "=" * 70)
    print("VIDEO MODE (Batch pipeline)")
    print("=" * 70)
    print(f"Config:")
    print(f"   S1 frame interval : {cfg.FRAME_INTERVAL}s  (conf={cfg.YOLO_CONFIDENCE})")
    print(f"   S2 frame interval : {cfg.EMPTYING_FRAME_INTERVAL}s  (conf={cfg.EMPTYING_CONFIDENCE})")
    print("=" * 70)

    # Pre-load models
    print("\nLoading models...")
    get_bin_models()
    get_emptying_models()
    print("All models loaded\n")

    # Sample frames for System 1 (System 2 samples its own)
    s1_frames = _sample_frames(video_path, cfg.FRAME_INTERVAL, "S1 bins")
    if not s1_frames:
        print("No frames read from video")
        return {}

    # Run System 1 + System 2 in parallel
    print("\n" + "=" * 70)
    print("RUNNING SYSTEMS 1 & 2 IN PARALLEL")
    print("=" * 70)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_run_system1, s1_frames)
        f2 = pool.submit(_run_system2, video_path)
        s1 = f1.result()
        s2 = f2.result()
    parallel_time = time.time() - t0

    # Build final result
    s1_sum = s1["summary"]
    s2_sum = s2["summary"]
    result = {
        "total_bacs": s1_sum["total_bacs"],
        "small_bacs": s1_sum["small_bacs"],
        "large_bacs": s1_sum["large_bacs"],
        "plastique_bacs": s1_sum["plastique_bacs"],
        "metal_bacs": s1_sum["metal_bacs"],
        "empty_bacs": s2_sum["empty_count"],
        "full_bacs": s2_sum["full_count"],
        "emptying_events": s2_sum["total_emptying_events"],
        "event_details": s2["emptying_events"],
        "aggregation": s1["aggregation_methods"],
    }

    total_time = time.time() - overall_start

    # Print results
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"   Total bins      : {result['total_bacs']}")
    print(f"   Material        : {result['plastique_bacs']}P + {result['metal_bacs']}M")
    print(f"   Size            : {result['small_bacs']}S + {result['large_bacs']}L")
    print(f"   Emptying events : {result['emptying_events']}")
    print(f"   Fullness        : {result['empty_bacs']}E + {result['full_bacs']}F")
    print(f"\n   Parallel time   : {parallel_time:.1f}s")
    print(f"   Total time      : {total_time:.1f}s")
    print(f"{'=' * 70}")

    # Save locally
    os.makedirs("results", exist_ok=True)
    out_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join("results", f"{out_name}_result.json")

    # Remove non-serializable frame data from aggregation
    save_result = {k: v for k, v in result.items() if k != "aggregation"}
    save_result["aggregation_note"] = result["aggregation"].get("decision_note", "")
    save_result["mode"] = result["aggregation"].get("mode", {})
    save_result["tracking"] = {
        "total_tracks": result["aggregation"].get("tracking", {}).get("total_tracks", 0),
        "top_tracks": result["aggregation"].get("tracking", {}).get("track_details", []),
    }

    with open(out_path, "w") as f:
        json.dump(save_result, f, indent=2)
    print(f"\nResult saved: {out_path}")

    return result
