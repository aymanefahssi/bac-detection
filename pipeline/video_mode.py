"""
Video analysis pipeline -- batch detection, no cloud/infra.

1. Sample frames at 1.5s intervals
2. System 1 (bins) + System 2 (emptying) run in parallel
3. Multi-method aggregation + smart decision engine
4. Cycle Analysis (State-Driven Bin Regroupments)
5. Save JSON result locally
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

        print(f"  [S1 Raw] t={fd['timestamp']:.1f}s: Saw {len(boxes)} bins")

    print(f"\n  [S1] Classifying {len(all_crops)} crops...")

    mat_results = []
    sz_results = []

    def _do_material(crops):
        display_map = {"plastique": "plastic", "metal": "metal"}
        out = []
        for c in crops:
            pred = material_model(c, device=device, verbose=False)[0]
            raw_name = pred.names[pred.probs.top1]
            out.append(display_map.get(raw_name, raw_name))
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
    
    # Export materials safely so the cycle engine can read them
    aggregated["frame_counts"] = [{
        "timestamp": f["timestamp"], 
        "count": len(f["detections"]),
        "materials": [d.get("material", "unknown") for d in f["detections"]]
    } for f in frame_results]
    
    print("=" * 70)
    return aggregated


# ---------------------------------------------------------------------------
# System 2: Emptying Detection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# System 2: Emptying Detection
# ---------------------------------------------------------------------------

def _run_system2(video_path: str) -> Dict:
    print("=" * 70)
    print("SYSTEM 2: EMPTYING DETECTION (WITH PRE-EVENT LOOKBACK)")
    print("=" * 70)

    emptying_model, fullness_model = get_emptying_models()
    device = cfg.DEVICE
    half = cfg.HALF_PRECISION
    conf = cfg.EMPTYING_CONFIDENCE

    frames = _sample_frames(video_path, cfg.EMPTYING_FRAME_INTERVAL, "S2 emptying")
    sm = EmptyingStateMachine()

    print(f"[S2] Processing {len(frames)} frames (conf={conf})...")

    # 🛑 THE NEW LOOKBACK BUFFER
    rolling_crops = []
    in_emptying = False
    pre_event_snapshots = []

    for idx, fd in enumerate(frames):
        ts = fd["timestamp"]
        pil = Image.fromarray(cv2.cvtColor(fd["frame"], cv2.COLOR_BGR2RGB))

        results = emptying_model.predict(
            pil, imgsz=640, conf=conf, device=device, half=half, verbose=False)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            detected_class = results[0].names[int(boxes.cls[best_idx].item())]
            
            state = "emptying" if detected_class.lower() == "emptying" else "normal"

            if state == "emptying":
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                w = float(x2 - x1)
                h = float(y2 - y1)
                h = h if h > 0 else 1.0 
                
                if y2 < (pil.height * 0.50):
                    state = "normal"
                elif w < (h * 0.95):
                    state = "normal"
                else:
                    box_area = w * h
                    screen_area = pil.width * pil.height
                    if (box_area / screen_area) < 0.04:
                        state = "normal"

            bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
            crop = pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            
            # 🛑 FREEZE PRE-EMPTING CROPS
            if state == "emptying":
                if not in_emptying:
                    # State just switched! Snapshot the recent normal bins from the past 4 seconds
                    pre_event_snapshots.append((ts, [rc["crop"] for rc in rolling_crops]))
                    in_emptying = True
            else:
                in_emptying = False
                rolling_crops.append({"ts": ts, "crop": crop})
                # Keep only the last 4 seconds of frames in the rolling buffer
                rolling_crops = [rc for rc in rolling_crops if ts - rc["ts"] <= 4.0]

            sm.process_frame(state, ts, crop, bbox, num_bins=-1, is_model_detection=True)
        else:
            in_emptying = False
            sm.process_frame("normal", ts, crop=None, bbox=None, num_bins=-1, is_model_detection=False)

    events = []
    
    # 🛑 DETAILED S2 LOGGER
    print("\n" + "-" * 70)
    print("🗑️ DETAILED EMPTYING EVENT LOG (PRE-EVENT FULLNESS) 🗑️")
    print("-" * 70)

    for event in sm.completed_events:
        start = event.get("start_time") or 0
        end = event.get("end_time") or start + 5
        
        # Find the frozen snapshot from the past that belongs to this event
        target_crops = []
        for snap_ts, snap_crops in pre_event_snapshots:
            if abs(snap_ts - start) <= 1.5:  # Match the snapshot to the event start
                target_crops = snap_crops
                break
        
        # Fallback just in case we didn't catch pre-frames (e.g. video started exactly as it tipped)
        if not target_crops:
            target_crops = [f["crop"] for f in event.get("frames", []) if f.get("crop")]

        votes = []
        if target_crops:
            for crop in target_crops:
                try:
                    pred = fullness_model(crop, device=device, verbose=False)[0]
                    raw_vote = pred.names[pred.probs.top1].lower()
                    # Safety translation just in case your Kaggle classes were French
                    clean_vote = "empty" if "vide" in raw_vote else raw_vote
                    clean_vote = "full" if "plein" in raw_vote else clean_vote
                    votes.append(clean_vote)
                except Exception:
                    votes.append("unknown")
            
            valid_votes = [v for v in votes if v in ["full", "empty"]]
            if valid_votes:
                vote_counts = Counter(valid_votes)
                fullness = vote_counts.most_common(1)[0][0]
            else:
                fullness = "unknown"
        else:
            fullness = "unknown"

        print(f"  Event #{event['event_id']} (t={start:.1f}s to {end:.1f}s):")
        print(f"     -> Pre-Emptying Frames Looked At : {len(target_crops)}")
        print(f"     -> Raw Pre-Event AI Votes        : {votes}")
        print(f"     -> Final Pre-Event Decision      : {fullness.upper()}")
        print("-" * 40)
        
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

    print("\nLoading models...")
    get_bin_models()
    get_emptying_models()
    print("All models loaded\n")

    s1_frames = _sample_frames(video_path, cfg.FRAME_INTERVAL, "S1 bins")
    if not s1_frames:
        print("No frames read from video")
        return {}

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

    # --- 🧠 YOUR ORIGINAL CYCLE ANALYSIS (Bin-Driven Regroupments) 🧠 ---
    frame_counts = s1.get("frame_counts", [])
    raw_events = s2["emptying_events"]
    
    cycles = []
    current_cycle = None
    GAP_THRESHOLD = 20.0

    for f in frame_counts:
        if f["count"] > 0:
            if current_cycle is None:
                current_cycle = {"start_time": f["timestamp"], "end_time": f["timestamp"], "frames": [f]}
            else:
                if f["timestamp"] - current_cycle["end_time"] <= GAP_THRESHOLD:
                    current_cycle["end_time"] = f["timestamp"]
                    current_cycle["frames"].append(f)
                else:
                    cycles.append(current_cycle)
                    current_cycle = {"start_time": f["timestamp"], "end_time": f["timestamp"], "frames": [f]}
    
    if current_cycle is not None:
        cycles.append(current_cycle)

    cycles = [c for c in cycles if len(c["frames"]) >= 2]

    # 2. Map the data to the cycles
    final_cycles = []
    print("\n--- 🛑 CYCLE LOGIC DEBUGGER 🛑 ---")
    
    for i, c in enumerate(cycles):
        cycle_events = [ev for ev in raw_events if c["start_time"] - 15.0 <= ev["start_time"] <= c["end_time"] + 15.0]
        
        # Get all active frames in the cycle
        active_counts = [f["count"] for f in c["frames"] if f["count"] > 0]
        
        if not active_counts:
            bins_present = 0
            print(f"Cycle {i+1}: No active bins detected.")
        else:
            # 🛑 YOUR EXACT FIX: Consecutive Physical Stability 🛑
            streaks = {}
            current_val = None
            current_streak = 0
            
            for val in active_counts:
                if val == current_val:
                    current_streak += 1
                else:
                    if current_val is not None:
                        streaks[current_val] = max(streaks.get(current_val, 0), current_streak)
                    current_val = val
                    current_streak = 1
            if current_val is not None:
                streaks[current_val] = max(streaks.get(current_val, 0), current_streak)
                
            required_streak = min(3, len(active_counts)) 
            
            valid_counts = [count for count, max_streak in streaks.items() if max_streak >= required_streak]
            
            print(f"Cycle {i+1}: Active frame array: {active_counts}")
            print(f"Cycle {i+1}: Max consecutive streaks: {streaks}")
            print(f"Cycle {i+1}: Valid stable counts (>= {required_streak} in a row): {valid_counts}")
            
            if valid_counts:
                bins_present = max(valid_counts) # Take the highest physically stable count
            else:
                bins_present = max(active_counts) # Absolute fallback
                
        # 🛑 SAFELY APPEND MATERIALS (Doesn't touch bin count math) 🛑
        cycle_p = 0
        cycle_m = 0
        for f in c["frames"]:
            if f["count"] == bins_present:
                mats = f.get("materials", [])
                cycle_p = mats.count("plastic") + mats.count("plastique")
                cycle_m = mats.count("metal")
                break # Just grabs the material text from the first frame that matches your confirmed bin count
        
        final_cycles.append({
            "cycle_id": i + 1,
            "start_time": round(c["start_time"], 1),
            "end_time": round(c["end_time"], 1),
            "total_bins_present": bins_present,
            "plastic_count": cycle_p,
            "metal_count": cycle_m,
            "emptied_count": len(cycle_events),
            "events": cycle_events
        })

    mapped_event_ids = [ev["event_id"] for c in final_cycles for ev in c["events"]]
    unmapped_events = [ev for ev in raw_events if ev["event_id"] not in mapped_event_ids]
    
    if unmapped_events:
        final_cycles.append({
            "cycle_id": len(final_cycles) + 1,
            "start_time": unmapped_events[0]["start_time"],
            "end_time": unmapped_events[-1]["end_time"],
            "total_bins_present": len(unmapped_events), 
            "emptied_count": len(unmapped_events),
            "events": unmapped_events
        })

    final_cycles = sorted(final_cycles, key=lambda x: x["start_time"])
    for i, c in enumerate(final_cycles):
        c["cycle_id"] = i + 1

    global_total_bacs = sum(c["total_bins_present"] for c in final_cycles)
    global_plastic = sum(c.get("plastic_count", 0) for c in final_cycles)
    global_metal = sum(c.get("metal_count", 0) for c in final_cycles)

    # -----------------------------------------------------------------------

    s1_sum = s1["summary"]
    s2_sum = s2["summary"]
    
    result = {
        "total_bacs": global_total_bacs,  
        "small_bacs": s1_sum.get("small_bacs", 0),
        "large_bacs": s1_sum.get("large_bacs", 0),
        "plastic_bacs": global_plastic, 
        "metal_bacs": global_metal,
        "empty_bacs": s2_sum["empty_count"],
        "full_bacs": s2_sum["full_count"],
        "emptying_events": s2_sum["total_emptying_events"],
        "event_details": s2["emptying_events"],
        "cycles": final_cycles,
        "aggregation": s1["aggregation_methods"],
    }

    total_time = time.time() - overall_start

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"   Total bins (Global) : {result['total_bacs']}")
    print(f"   Material            : {result['plastic_bacs']}P + {result['metal_bacs']}M")
    print(f"   Size                : {result['small_bacs']}S + {result['large_bacs']}L")
    print(f"   Emptying events     : {result['emptying_events']}")
    print(f"   Fullness            : {result['empty_bacs']}E + {result['full_bacs']}F")
    
    print(f"\n   --- CYCLE BREAKDOWN (Bins per Regroupment) ---")
    if not final_cycles:
        print("   No bin regroupments detected.")
    for c in final_cycles:
        print(f"   Cycle {c['cycle_id']} (t={c['start_time']}s to {c['end_time']}s):")
        print(f"      Bins on street : {c['total_bins_present']}")
        print(f"      Material       : {c.get('plastic_count', 0)}P + {c.get('metal_count', 0)}M")
        print(f"      Bins emptied   : {c['emptied_count']}")

    print(f"\n   Parallel time   : {parallel_time:.1f}s")
    print(f"   Total time      : {total_time:.1f}s")
    print(f"{'=' * 70}")

    os.makedirs("results", exist_ok=True)
    out_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join("results", f"{out_name}_result.json")

    save_result = {k: v for k, v in result.items() if k not in ["aggregation", "frame_counts"]}
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