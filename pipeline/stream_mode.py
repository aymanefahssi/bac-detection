"""
Stream mode -- process a video with cycle-based detection.

State machine:
  IDLE   → scan frames, wait for consecutive bin detections
  ACTIVE → per-frame detection + tracking + emptying state machine
  END    → aggregate cycle results (batch S1 + S2), save JSON, return to IDLE

Works with video files (treats them like a stream) or RTSP sources.
Each cycle produces one result JSON in results/.
"""

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import cv2
from PIL import Image

from config import cfg
from detection.bins import SimpleTrack, aggregate_bin_results
from detection.emptying import EmptyingStateMachine
from detection.models import get_bin_models, get_emptying_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_cycle_result(result: dict, session_id: str) -> str:
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"{session_id}.json")
    save = {k: v for k, v in result.items() if k != "aggregation"}
    save["aggregation_note"] = result.get("aggregation", {}).get("decision_note", "")
    save["mode"] = result.get("aggregation", {}).get("mode", {})
    save["tracking"] = {
        "total_tracks": result.get("aggregation", {}).get("tracking", {}).get("total_tracks", 0),
        "top_tracks": result.get("aggregation", {}).get("tracking", {}).get("track_details", []),
    }
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    return path


def _classify_fullness(events, fullness_model, device):
    """Classify fullness for completed emptying events."""
    results = []
    for event in events:
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
            fullness = "unknown"

        start = event.get("start_time") or 0
        end = event.get("end_time") or start + 5
        results.append({
            "event_id": event["event_id"],
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
            "fullness": fullness,
        })
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(video_path: str) -> Dict:
    """
    Process *video_path* with cycle-based stream detection.

    Returns a dict summarising all detected cycles.
    """
    overall_start = time.time()

    # Config
    FRAME_INTERVAL = cfg.STREAM_FRAME_INTERVAL
    BIN_CONF = cfg.STREAM_CONFIDENCE
    EMP_CONF = cfg.STREAM_EMPTYING_CONFIDENCE
    MIN_CONSECUTIVE = cfg.STREAM_MIN_CONSECUTIVE
    NO_DET_TIMEOUT = cfg.STREAM_NO_DETECTION_TIMEOUT
    MIN_CYCLE_FRAMES = cfg.STREAM_MIN_CYCLE_FRAMES

    print("\n" + "=" * 70)
    print("STREAM MODE (IDLE -> ACTIVE -> END -> repeat)")
    print("=" * 70)
    print(f"Config:")
    print(f"   Frame interval       : {FRAME_INTERVAL}s")
    print(f"   Bin confidence       : {BIN_CONF}")
    print(f"   Emptying confidence  : {EMP_CONF}")
    print(f"   Consecutive to start : {MIN_CONSECUTIVE}")
    print(f"   No-detection timeout : {NO_DET_TIMEOUT}s")
    print(f"   Min frames / cycle   : {MIN_CYCLE_FRAMES}")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    bin_model, material_model, size_model = get_bin_models()
    emptying_model, fullness_model = get_emptying_models()
    print("All models loaded")

    device = cfg.DEVICE
    half = cfg.HALF_PRECISION

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_duration = total_frames / fps
    frame_skip = max(1, int(fps * FRAME_INTERVAL))

    print(f"\nVideo: {total_frames} frames @ {fps:.1f} fps ({video_duration:.1f}s)")
    print(f"Sampling every {frame_skip} frames ({FRAME_INTERVAL}s)")

    # State
    all_cycles = []
    cycle_count = 0
    frame_idx = 0

    # ==================================================================
    # Main loop
    # ==================================================================
    while frame_idx < total_frames:

        # ==============================================================
        # IDLE — scan for bins
        # ==============================================================
        print(f"\n{'=' * 60}")
        print(f"IDLE - Scanning for bins (need {MIN_CONSECUTIVE} consecutive)...")
        print(f"   Frame {frame_idx}/{total_frames} "
              f"({frame_idx / total_frames * 100:.1f}%)")
        print(f"{'=' * 60}")

        consecutive = 0
        idle_start_idx = frame_idx

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx += frame_skip
                continue

            timestamp = frame_idx / fps
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            results = bin_model.predict(
                pil, imgsz=640, conf=BIN_CONF,
                device=device, half=half, verbose=False)
            num_bins = len(results[0].boxes)

            if num_bins > 0:
                consecutive += 1
                if consecutive >= MIN_CONSECUTIVE:
                    print(f"Bins confirmed ({num_bins})! t={timestamp:.1f}s")
                    break
                else:
                    print(f"   Detection {consecutive}/{MIN_CONSECUTIVE} "
                          f"at t={timestamp:.1f}s ({num_bins} bins)")
            else:
                if consecutive > 0:
                    print(f"   Reset at t={timestamp:.1f}s "
                          f"(was {consecutive}/{MIN_CONSECUTIVE})")
                consecutive = 0

            frame_idx += frame_skip

        if frame_idx >= total_frames:
            print("\nVideo ended during IDLE")
            break

        # ==============================================================
        # ACTIVE — track bins + detect emptying
        # ==============================================================
        cycle_count += 1
        session_id = f"stream_{int(time.time())}_{cycle_count}"
        cycle_start_frame = frame_idx
        cycle_start_time = frame_idx / fps
        last_detection_frame = frame_idx

        print(f"\n{'=' * 60}")
        print(f"CYCLE {cycle_count} STARTED | t={cycle_start_time:.1f}s "
              f"| {session_id}")
        print(f"{'=' * 60}")

        tracks: List[SimpleTrack] = []
        next_track_id = 0
        frame_results = []
        frames_with_bins = 0
        state_machine = EmptyingStateMachine()
        status_count = 0

        # Collect raw frames for batch S2 reprocessing at cycle end
        cycle_raw_frames = []

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx += frame_skip
                continue

            timestamp = frame_idx / fps
            elapsed = timestamp - cycle_start_time
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Store raw frame for later S2 batch reprocessing
            cycle_raw_frames.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "frame": frame,
            })

            # -- Bin Detection -----------------------------------------
            bin_results = bin_model.predict(
                pil, imgsz=640, conf=BIN_CONF,
                device=device, half=half, verbose=False)
            boxes = bin_results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
            num_bins = len(boxes)

            frame_results.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "elapsed": elapsed,
                "bin_count": num_bins,
                "detections": [],
            })

            if num_bins > 0:
                frames_with_bins += 1
                last_detection_frame = frame_idx

            # Classify each bin
            for box in boxes:
                x1, y1, x2, y2 = box
                crop = pil.crop((x1, y1, x2, y2))

                mat_pred = material_model(crop, device=device, verbose=False)[0]
                material = mat_pred.names[mat_pred.probs.top1]

                size_pred = size_model(crop, device=device, verbose=False)[0]
                size_label = size_pred.names[size_pred.probs.top1]

                frame_results[-1]["detections"].append({
                    "box": box, "material": material, "size": size_label})

                # Track
                matched = False
                for track in tracks:
                    if track.matches(box):
                        track.update(box, material, size_label, elapsed)
                        matched = True
                        break
                if not matched:
                    tracks.append(
                        SimpleTrack(next_track_id, box, material, size_label, elapsed))
                    next_track_id += 1

            # -- Emptying Detection (inline, low conf) -----------------
            emp_results = emptying_model.predict(
                pil, imgsz=640, conf=EMP_CONF,
                device=device, half=half, verbose=False)

            if len(emp_results[0].boxes) > 0:
                emp_boxes = emp_results[0].boxes
                best_idx = emp_boxes.conf.argmax().item()
                detected_class = emp_results[0].names[
                    int(emp_boxes.cls[best_idx].item())]
                bbox = emp_boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
                crop = pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                state = "normal" if detected_class == "emptying" else "emptying"
                state_machine.process_frame(
                    state, elapsed, crop, bbox,
                    num_bins=num_bins, is_model_detection=True)
            else:
                state_machine.process_frame(
                    "normal", elapsed, crop=None, bbox=None,
                    num_bins=num_bins, is_model_detection=False)

            # -- Timeout -----------------------------------------------
            time_since_last = (frame_idx - last_detection_frame) / fps

            status_count += 1
            if status_count % 4 == 0:
                print(f"   t={timestamp:.1f}s | Bins: {num_bins} "
                      f"| Tracks: {len(tracks)} "
                      f"| Events: {len(state_machine.completed_events)} "
                      f"| Silent: {time_since_last:.1f}s")

            if time_since_last >= NO_DET_TIMEOUT:
                print(f"\nNo bins for {NO_DET_TIMEOUT}s — ending cycle")
                state_machine.force_end_if_active(elapsed)
                break

            frame_idx += frame_skip

        # ==============================================================
        # END CYCLE — aggregate + save
        # ==============================================================
        cycle_end_time = frame_idx / fps
        cycle_duration = cycle_end_time - cycle_start_time

        print(f"\n{'=' * 60}")
        print(f"CYCLE {cycle_count} COMPLETE | Duration: {cycle_duration:.1f}s")
        print(f"{'=' * 60}")

        if frames_with_bins < MIN_CYCLE_FRAMES:
            print(f"Invalid cycle: only {frames_with_bins} frames with bins "
                  f"(min: {MIN_CYCLE_FRAMES}). Discarding.")
            continue

        # -- S1 aggregation (4-method + smart decision) ----------------
        print(f"\n[S1] Aggregating {len(frame_results)} frames, "
              f"{len(tracks)} tracks...")
        s1 = aggregate_bin_results(frame_results, tracks)
        s1_sum = s1["summary"]

        # -- S2 emptying: classify fullness ----------------------------
        emptying_events = _classify_fullness(
            state_machine.completed_events, fullness_model, device)

        empty_n = sum(1 for e in emptying_events if e["fullness"] == "empty")
        full_n = sum(1 for e in emptying_events if e["fullness"] == "full")

        # -- Build result ----------------------------------------------
        result = {
            "cycle": cycle_count,
            "session_id": session_id,
            "total_bacs": s1_sum["total_bacs"],
            "small_bacs": s1_sum["small_bacs"],
            "large_bacs": s1_sum["large_bacs"],
            "plastique_bacs": s1_sum["plastique_bacs"],
            "metal_bacs": s1_sum["metal_bacs"],
            "empty_bacs": empty_n,
            "full_bacs": full_n,
            "emptying_events": len(emptying_events),
            "event_details": emptying_events,
            "aggregation": s1.get("aggregation_methods", {}),
            "start_time": round(cycle_start_time, 2),
            "end_time": round(cycle_end_time, 2),
            "duration": round(cycle_duration, 2),
        }

        # Print
        print(f"\nCYCLE {cycle_count} RESULTS:")
        print(f"   Bins     : {s1_sum['total_bacs']} "
              f"({s1_sum['plastique_bacs']}P + {s1_sum['metal_bacs']}M)")
        print(f"   Size     : {s1_sum['small_bacs']}S + {s1_sum['large_bacs']}L")
        print(f"   Emptying : {len(emptying_events)} events "
              f"({empty_n}E + {full_n}F)")
        print(f"   Time     : {cycle_start_time:.1f}s — {cycle_end_time:.1f}s "
              f"({cycle_duration:.1f}s)")

        # Save
        path = _save_cycle_result(result, session_id)
        print(f"   Saved    : {path}")

        all_cycles.append(result)

        print(f"\n{'-' * 60}")
        print("Returning to IDLE...")
        print(f"{'-' * 60}")

    # ==================================================================
    # Video complete
    # ==================================================================
    cap.release()
    total_time = time.time() - overall_start

    total_bins = sum(c["total_bacs"] for c in all_cycles)
    total_emptying = sum(c["emptying_events"] for c in all_cycles)

    print(f"\n{'=' * 70}")
    print("STREAM ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"   Total cycles        : {cycle_count}")
    print(f"   Total bins          : {total_bins}")
    print(f"   Total emptying      : {total_emptying}")
    print(f"   Processing time     : {total_time:.1f}s")
    print(f"{'=' * 70}")

    return {
        "total_cycles": cycle_count,
        "total_bacs": total_bins,
        "total_emptying_events": total_emptying,
        "cycles": all_cycles,
        "processing_time": round(total_time, 2),
    }
