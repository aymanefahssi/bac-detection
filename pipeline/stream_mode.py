"""
Stream pipeline -- IDLE → ACTIVE → END → repeat.

Fixes vs supervisor's version
-------------------------------
1. S2 state label was INVERTED (emptying never detected). Fixed.
2. Material labels not normalised ("plastique" → "plastic"). Fixed.
3. Fullness voted on in-event frames instead of pre-event crops. Fixed.
4. IDLE hot loop: wider frame interval, runs bin model only.
5. Exact Physical Counting: Strict Streak Logic confidently handles truck occlusions.
6. The Blindspot Fix: State Machine is now global and calibrates during IDLE to catch early events.
7. Zero-Crop Ghost Fix: Safely rejects phantom boundary events.
8. Current-Frame Fallback: Safely saves instant-start events from being deleted.
9. IDLE S2 Buffer: Runs emptying model during IDLE to populate pre-event crop buffer.
10. Minimum Crop Gate: Events with < MIN_PRE_CROPS crops are rejected entirely (no more "unknown" ghosts).
"""

import json
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
from PIL import Image

from config import cfg
from detection.bins import SimpleTrack, aggregate_bin_results
from detection.emptying import EmptyingStateMachine
from detection.models import get_bin_models, get_emptying_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil(frame) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


_MAT_MAP = {"plastique": "plastic", "metal": "metal"}


def _vote_fullness(
    crops: List,
    fullness_model,
    device: str,
) -> Tuple[str, List[str]]:
    """Run fullness model on a list of crops, return (winner, raw_votes)."""
    votes = []
    for crop in crops:
        try:
            pred = fullness_model(crop, device=device, verbose=False)[0]
            raw  = pred.names[pred.probs.top1].lower()
            v    = "empty" if "vide"  in raw else raw
            v    = "full"  if "plein" in v   else v
            votes.append(v)
        except Exception:
            votes.append("unknown")

    valid = [v for v in votes if v in ("full", "empty")]
    winner = Counter(valid).most_common(1)[0][0] if valid else "unknown"
    return winner, votes


def _iou(a: List[int], b: List[int]) -> float:
    """Intersection-over-union for two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


_CACHE_IOU_THRESH = 0.50


def _cache_lookup(
    box: List[int],
    cache: List[Tuple],
) -> Optional[Tuple[str, str]]:
    best_iou   = _CACHE_IOU_THRESH
    best_entry = None
    for cached_box, mat, sz in cache:
        score = _iou(box, cached_box)
        if score > best_iou:
            best_iou   = score
            best_entry = (mat, sz)
    return best_entry


def _save_cycle(result: Dict, source_name: str, cycle_id: int) -> str:
    os.makedirs(os.path.join("results", "stream"), exist_ok=True)
    path = os.path.join("results", "stream", f"{source_name}_cycle_{cycle_id}.json")
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(video_path: str) -> Dict:
    overall_start = time.time()

    # ── Config ──────────────────────────────────────────────────────────────
    FRAME_IV        = cfg.STREAM_FRAME_INTERVAL          # 0.25s active
    IDLE_FRAME_IV   = FRAME_IV * 4                       # 1.0s idle (4x faster)
    BIN_CONF        = 0.45              
    EMP_CONF        = 0.70     
    MIN_CONSEC      = cfg.STREAM_MIN_CONSECUTIVE         # 5
    NO_DET_TIMEOUT  = cfg.STREAM_NO_DETECTION_TIMEOUT    # 30s
    MIN_CYC_FRAMES  = cfg.STREAM_MIN_CYCLE_FRAMES        # 10
    MIN_PRE_CROPS   = 2   
    
    MAX_EV_DUR = getattr(cfg, 'MAX_EVENT_DURATION', 15.0)

    print("\n" + "=" * 70)
    print("STREAM MODE  (IDLE → ACTIVE → END → repeat)")
    print("=" * 70)
    print(f"  Frame interval (active) : {FRAME_IV}s")
    print(f"  Emptying confidence     : {EMP_CONF}")
    print("=" * 70)

    print("\nLoading models...")
    bin_model, material_model, size_model = get_bin_models()
    emptying_model, fullness_model        = get_emptying_models()
    print("All models loaded\n")

    device = cfg.DEVICE
    half   = cfg.HALF_PRECISION

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
    active_skip   = max(1, int(fps * FRAME_IV))
    idle_skip     = max(1, int(fps * IDLE_FRAME_IV))
    source_name   = os.path.splitext(os.path.basename(video_path))[0]

    all_cycles    = []
    cycle_count   = 0
    frame_idx     = 0

    # 🛑 GLOBAL STATE: Lives forever so it never loses calibration
    sm                  = EmptyingStateMachine()
    processed_event_ids = set()
    rolling_crops       = []
    in_emptying         = False
    pre_event_snapshots = []

    while frame_idx < total_frames:

        # =================================================================
        # IDLE — scan for bins
        # =================================================================
        print(f"\n{'─' * 60}")
        print(f"  IDLE  │ scanning from t={frame_idx/fps:.1f}s")
        print(f"{'─' * 60}")

        consecutive = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx += idle_skip
                continue

            ts  = frame_idx / fps
            pil = _pil(frame)

            n = len(bin_model.predict(
                pil, imgsz=640, conf=BIN_CONF,
                device=device, half=half, verbose=False
            )[0].boxes)

            # WARMUP PING: Feed SM during IDLE so it builds a "normal" baseline
            sm.process_frame("normal", ts, None, None, n, False)

            # 🔥 Feed the rolling crop buffer during IDLE with normal-state S2 crops
            if n > 0:
                s2_idle = emptying_model.predict(
                    pil, imgsz=640, conf=EMP_CONF,
                    device=device, half=half, verbose=False)
                if len(s2_idle[0].boxes) > 0:
                    s2_boxes_idle = s2_idle[0].boxes
                    best_idle = s2_boxes_idle.conf.argmax().item()
                    det_class_idle = s2_idle[0].names[int(s2_boxes_idle.cls[best_idle].item())]
                    if det_class_idle.lower() != "emptying":   # only normal-state crops
                        s2_bbox_idle = s2_boxes_idle.xyxy[best_idle].cpu().numpy().astype(int).tolist()
                        s2_crop_idle = pil.crop((s2_bbox_idle[0], s2_bbox_idle[1],
                                                 s2_bbox_idle[2], s2_bbox_idle[3]))
                        rolling_crops.append({"ts": ts, "crop": s2_crop_idle})
                        # keep only last 4 seconds
                        rolling_crops = [r for r in rolling_crops if ts - r["ts"] <= 4.0]

            if n > 0:
                consecutive += 1
                if consecutive >= MIN_CONSEC:
                    print(f"  → Bins confirmed — switching to ACTIVE")
                    break
            else:
                consecutive = 0

            frame_idx += idle_skip

        if frame_idx >= total_frames:
            break

        # =================================================================
        # ACTIVE — per-frame S1 + S2, full classification
        # =================================================================
        cycle_count      += 1
        cycle_start_time  = frame_idx / fps
        last_bin_ts       = cycle_start_time
        session_id        = f"{int(time.time())}_{cycle_count}"

        print(f"\n{'=' * 60}")
        print(f"  CYCLE {cycle_count} STARTED │ t={cycle_start_time:.1f}s")
        print(f"{'=' * 60}")

        tracks          = []
        next_id         = 0
        frame_history   = []
        frames_w_bins   = 0
        cycle_events    = []
        _classify_cache = []

        status_tick = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx += active_skip
                continue

            ts  = frame_idx / fps
            pil = _pil(frame)

            # ── S1: Bin detection ─────────────────────────────────────────
            s1    = bin_model.predict(
                pil, imgsz=640, conf=BIN_CONF,
                device=device, half=half, verbose=False)
            boxes     = s1[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
            bin_count = len(boxes)
            new_cache = []

            frame_data = {
                "frame_idx": frame_idx,
                "timestamp": ts,
                "bin_count": bin_count,
                "detections": []
            }

            for box in boxes:
                cached = _cache_lookup(box, _classify_cache)

                if cached is None:
                    crop = pil.crop((box[0], box[1], box[2], box[3]))
                    mat_pred = material_model(crop, device=device, verbose=False)[0]
                    raw_mat  = mat_pred.names[mat_pred.probs.top1]
                    material = _MAT_MAP.get(raw_mat, raw_mat)

                    sz_pred  = size_model(crop, device=device, verbose=False)[0]
                    size_lbl = sz_pred.names[sz_pred.probs.top1]
                else:
                    material, size_lbl = cached

                frame_data["detections"].append({"box": box, "material": material, "size": size_lbl})
                new_cache.append((box, material, size_lbl))

                matched = False
                for track in tracks:
                    if track.matches(box):
                        track.update(box, material, size_lbl, ts)
                        matched = True
                        break
                if not matched:
                    tracks.append(SimpleTrack(next_id, box, material, size_lbl, ts))
                    next_id += 1

            _classify_cache = new_cache
            frame_history.append(frame_data)

            if bin_count > 0:
                frames_w_bins += 1
                last_bin_ts    = ts

            # ── S2: Emptying detection ────────────────────────────────────
            emptying_state = "normal"
            s2_crop        = None
            s2_bbox        = None

            if bin_count > 0:
                s2 = emptying_model.predict(
                    pil, imgsz=640, conf=EMP_CONF,
                    device=device, half=half, verbose=False)

                if len(s2[0].boxes) > 0:
                    s2_boxes  = s2[0].boxes
                    best      = s2_boxes.conf.argmax().item()
                    det_class = s2[0].names[int(s2_boxes.cls[best].item())]
                    conf_val  = s2_boxes.conf[best].item()

                    s2_bbox = s2_boxes.xyxy[best].cpu().numpy().astype(int).tolist()
                    s2_crop = pil.crop((s2_bbox[0], s2_bbox[1], s2_bbox[2], s2_bbox[3]))

                    if det_class.lower() == "emptying":
                        x1, y1, x2, y2 = s2_bbox
                        w = float(x2 - x1)
                        h = float(y2 - y1) or 1.0
                        
                        print(f"  [S2 DETECT t={ts:.1f}s] conf={conf_val:.2f}", end="")

                        if y2 < pil.height * 0.50:
                            print(f" -> Rejected (Too high)")
                        elif w < h * 0.95:
                            print(f" -> Rejected (Too narrow)")
                        elif (w * h) / (pil.width * pil.height) < 0.04:
                            print(f" -> Rejected (Too small)")
                        else:
                            emptying_state = "emptying"
                            print(f" -> ACCEPTED!")

            if emptying_state == "emptying":
                if not in_emptying:
                    pre_event_snapshots.append(
                        (ts, [rc["crop"] for rc in rolling_crops])
                    )
                    in_emptying = True
            else:
                in_emptying = False
                if s2_crop is not None:
                    rolling_crops.append({"ts": ts, "crop": s2_crop})
                rolling_crops = [r for r in rolling_crops if ts - r["ts"] <= 4.0]

            sm.process_frame(
                emptying_state, ts, s2_crop, s2_bbox,
                num_bins=bin_count, is_model_detection=(s2_crop is not None),
            )

            # Harvest completed events
            for event in sm.completed_events:
                ev_id = event["event_id"]
                if ev_id in processed_event_ids:
                    continue
                processed_event_ids.add(ev_id)

                start_ev = event.get("start_time") or 0
                end_ev   = event.get("end_time")   or start_ev + 5

                target_crops = []
                for snap_ts, snap_crops in pre_event_snapshots:
                    if abs(snap_ts - start_ev) <= 1.5:
                        target_crops = snap_crops
                        break
                
                # Active Frame Fallback
                if not target_crops:
                    print(f"  [S2] Event #{ev_id}: Cycle started during emptying. Using active frames instead of pre-frames.")
                    target_crops = [f["crop"] for f in event.get("frames", []) if f.get("crop")]

                # Last-resort fallback to the current frame's S2 crop
                if not target_crops and s2_crop is not None:
                    print(f"  [S2] Event #{ev_id}: No pre/active frames. Falling back to the current S2 crop.")
                    target_crops = [s2_crop]

                # 🛑 MINIMUM CROP GATE: reject if fewer than MIN_PRE_CROPS
                if len(target_crops) < MIN_PRE_CROPS:
                    print(f"\n  [S2] EVENT #{ev_id} REJECTED: only {len(target_crops)} crop(s) available (min {MIN_PRE_CROPS}).")
                    continue

                duration_ev = end_ev - start_ev
                if event.get("force_ended") or duration_ev >= MAX_EV_DUR:
                    print(f"\n  [S2] EVENT #{ev_id} REJECTED: Hit max duration timeout ({duration_ev:.1f}s) - Likely false positive")
                    continue

                fullness, votes = _vote_fullness(target_crops, fullness_model, device)
                print(f"\n  [S2] EVENT #{ev_id} CONFIRMED: t={start_ev:.1f}s–{end_ev:.1f}s → {fullness.upper()} (votes: {votes})")

                cycle_events.append({
                    "event_id":   ev_id,
                    "start_time": round(start_ev, 2),
                    "end_time":   round(end_ev,   2),
                    "duration":   round(duration_ev, 2),
                    "fullness":   fullness,
                })

            # ── Timeout check ─────────────────────────────────────────────
            gap = ts - last_bin_ts
            status_tick += 1
            if status_tick % 4 == 0:
                print(f"  t={ts:.1f}s │ bins={bin_count} tracks={len(tracks)} events={len(cycle_events)} silent={gap:.1f}s")

            if gap >= NO_DET_TIMEOUT:
                print(f"\n  → No bins for {gap:.1f}s — closing cycle")
                sm.force_end_if_active(ts)
                
                # Check for one last flushed event
                for event in sm.completed_events:
                    ev_id = event["event_id"]
                    if ev_id in processed_event_ids: continue
                    processed_event_ids.add(ev_id)
                    start_ev = event.get("start_time") or 0
                    end_ev   = event.get("end_time")   or start_ev + 5
                    
                    target_crops = [f["crop"] for f in event.get("frames", []) if f.get("crop")]
                    
                    # 🛑 MINIMUM CROP GATE in flush
                    if len(target_crops) < MIN_PRE_CROPS:
                        continue

                    if not event.get("force_ended") and (end_ev - start_ev) < MAX_EV_DUR:
                        fullness, votes = _vote_fullness(target_crops, fullness_model, device)
                        print(f"\n  [S2] EVENT #{ev_id} CONFIRMED (on flush): t={start_ev:.1f}s–{end_ev:.1f}s → {fullness.upper()}")
                        cycle_events.append({
                            "event_id": ev_id, "start_time": round(start_ev, 2), 
                            "end_time": round(end_ev, 2), "duration": round(end_ev - start_ev, 2), "fullness": fullness,
                        })
                break

            frame_idx += active_skip

        # =================================================================
        # END CYCLE — aggregate + save
        # =================================================================
        cycle_end_time = last_bin_ts

        print(f"\n{'=' * 60}")
        print(f"  CYCLE {cycle_count} END  │ {cycle_start_time:.1f}s → {cycle_end_time:.1f}s")
        print(f"{'=' * 60}")

        if frames_w_bins < MIN_CYC_FRAMES:
            frame_idx += active_skip
            continue

        active_frames = [f for f in frame_history if f["timestamp"] <= last_bin_ts + 2.0]
        if not active_frames: active_frames = frame_history

        print(f"\n[S1] Aggregating {len(active_frames)} active frames, {len(tracks)} tracks...")
        s1 = aggregate_bin_results(active_frames, tracks)
        s1_sum = s1.get("summary", {})

        active_counts = [f["bin_count"] for f in active_frames if f["bin_count"] > 0]
        if not active_counts:
            bins_present = 0
        else:
            required_streak = max(3, int(1.5 / FRAME_IV)) 
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
                
            valid_counts = [count for count, max_streak in streaks.items() if max_streak >= required_streak]
            
            if valid_counts:
                bins_present = max(valid_counts) 
                print(f"  [S1] Strict Streak Logic firmly asserts: {bins_present} Bins")
            else:
                bins_present = s1_sum.get("total_bacs", 0)
                print(f"  [S1] No solid streak found. Falling back to S1 Mode: {bins_present} Bins")

        cycle_plastic = 0
        cycle_metal = 0
        target_frames = [f for f in active_frames if f["bin_count"] == bins_present]
        
        if target_frames:
            all_mats = []
            for f in target_frames:
                for d in f["detections"]:
                    all_mats.append(d.get("material", "unknown"))
            
            num_frames = len(target_frames)
            cycle_plastic = round((all_mats.count("plastic") + all_mats.count("plastique")) / num_frames) if num_frames > 0 else 0
            cycle_metal = round(all_mats.count("metal") / num_frames) if num_frames > 0 else 0
        else:
            cycle_plastic = s1_sum.get("plastic_bacs", s1_sum.get("plastique_bacs", 0))
            cycle_metal = s1_sum.get("metal_bacs", 0)

        empty_n = sum(1 for e in cycle_events if e["fullness"] == "empty")
        full_n  = sum(1 for e in cycle_events if e["fullness"] == "full")

        result = {
            "cycle_id":           cycle_count,
            "session_id":         session_id,
            "total_bacs":         bins_present,
            "small_bacs":         s1_sum.get("small_bacs", 0),
            "large_bacs":         s1_sum.get("large_bacs", 0),
            "plastic_bacs":       cycle_plastic,
            "metal_bacs":         cycle_metal,
            "empty_bacs":         empty_n,
            "full_bacs":          full_n,
            "emptying_events":    len(cycle_events),
            "event_details":      cycle_events,
            "aggregation":        s1.get("aggregation_methods", {}),
            "start_time":         round(cycle_start_time, 2),
            "end_time":           round(cycle_end_time, 2),
            "duration":           round(cycle_end_time - cycle_start_time, 2),
        }

        print(f"\nCYCLE {cycle_count} RESULTS:")
        print(f"  Bins     : {bins_present}  ({cycle_plastic}P + {cycle_metal}M)")
        print(f"  Size     : {s1_sum.get('small_bacs', 0)}S + {s1_sum.get('large_bacs', 0)}L")
        print(f"  Emptying : {len(cycle_events)} events  ({empty_n}E + {full_n}F)")

        path = _save_cycle(result, source_name, cycle_count)
        print(f"  💾 Saved  : {path}")

        all_cycles.append(result)
        frame_idx += active_skip

    # =========================================================================
    # Done
    # =========================================================================
    cap.release()
    elapsed = time.time() - overall_start

    total_bins     = sum(c["total_bacs"] for c in all_cycles)
    total_plastic  = sum(c["plastic_bacs"] for c in all_cycles)
    total_metal    = sum(c["metal_bacs"] for c in all_cycles)
    total_emptying = sum(c["emptying_events"] for c in all_cycles)
    total_empty    = sum(c["empty_bacs"] for c in all_cycles)
    total_full     = sum(c["full_bacs"] for c in all_cycles)

    print(f"\n{'=' * 70}")
    print("STREAM ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Cycles completed   : {len(all_cycles)}")
    print(f"  Total bins         : {total_bins} ({total_plastic}P + {total_metal}M)")
    print(f"  Total emptying     : {total_emptying} ({total_empty}E + {total_full}F)")
    for c in all_cycles:
        print(f"\n  Cycle {c['cycle_id']} (t={c['start_time']}s → {c['end_time']}s)")
        print(f"     Bins     : {c['total_bacs']} ({c['plastic_bacs']}P + {c['metal_bacs']}M)")
        print(f"     Size     : {c['small_bacs']}S + {c['large_bacs']}L")
        print(f"     Emptying : {c['emptying_events']} ({c['empty_bacs']}E + {c['full_bacs']}F)")
    print(f"\n  Wall-clock time    : {elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    return {
        "total_cycles":          len(all_cycles),
        "total_bins":            total_bins,
        "total_emptying_events": total_emptying,
        "cycles":                all_cycles,
        "processing_time":       round(elapsed, 2),
    }