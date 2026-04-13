"""
Bin detection: lightweight IoU tracker and multi-method aggregation.
"""

from collections import Counter
from typing import Dict, List


class SimpleTrack:
    """Lightweight IoU-based tracker for attribute assignment."""

    def __init__(self, track_id: int, box: List[int], material: str,
                 size: str, timestamp: float):
        self.id = track_id
        self.boxes = [box]
        self.materials = [material]
        self.sizes = [size]
        self.timestamps = [timestamp]
        self._frame_count = 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def update(self, box: List[int], material: str, size: str,
               timestamp: float):
        self.boxes.append(box)
        self.materials.append(material)
        self.sizes.append(size)
        self.timestamps.append(timestamp)
        self._frame_count += 1

    def matches(self, box: List[int], iou_threshold: float = 0.3) -> bool:
        last_box = self.boxes[-1]
        x1 = max(last_box[0], box[0])
        y1 = max(last_box[1], box[1])
        x2 = min(last_box[2], box[2])
        y2 = min(last_box[3], box[3])

        if x2 < x1 or y2 < y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1])
        area2 = (box[2] - box[0]) * (box[3] - box[1])
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou >= iou_threshold

    def get_dominant_attributes(self) -> tuple:
        material = Counter(self.materials).most_common(1)[0][0]
        size = Counter(self.sizes).most_common(1)[0][0]
        return material, size


# ---------------------------------------------------------------------------
# Smart decision engine
# ---------------------------------------------------------------------------
def _smart_decision(
    mode_count: int,
    mode_confidence: float,
    median_count: int,
    early_mode: int,
    early_avg: float,
    sorted_tracks: list,
    bin_counts: list,
) -> tuple:
    """
    Deterministic heuristics that mimic the LLM decision logic.

    Core idea -- "peak count": the highest bin count that appears in >= 15%
    of non-zero frames is likely the true count.  Lower counts are partial
    detections (occlusion); very rare higher counts are false positives.
    """
    nonzero_counts = [c for c in bin_counts if c > 0]
    n_nonzero = len(nonzero_counts)

    if mode_confidence >= 60:
        return mode_count, "mode (high confidence)"

    if not nonzero_counts:
        return 0, "no detections"

    nz_freq = Counter(nonzero_counts)
    print(f"Smart decision: mode={mode_count}@{mode_confidence:.0f}%, "
          f"median={median_count}, early_mode={early_mode}, "
          f"non-zero distribution={dict(nz_freq)}")

    # Peak count: highest count appearing in >= 15% of non-zero frames
    PEAK_THRESHOLD = 0.15
    peak_count = 0
    peak_pct = 0
    for count_val in sorted(nz_freq.keys(), reverse=True):
        fraction = nz_freq[count_val] / n_nonzero
        if fraction >= PEAK_THRESHOLD:
            peak_count = count_val
            peak_pct = fraction * 100
            break

    if peak_count == 0:
        nz_mode = nz_freq.most_common(1)[0][0]
        return nz_mode, "non-zero mode (no peak above threshold)"

    print(f"  Peak count: {peak_count} bins "
          f"(appears in {nz_freq[peak_count]}/{n_nonzero} non-zero frames "
          f"= {peak_pct:.0f}%)")

    n_tracks = len(sorted_tracks)
    if peak_count <= n_tracks:
        return peak_count, f"peak count ({peak_count}, {peak_pct:.0f}% of non-zero frames)"

    return min(peak_count, n_tracks), f"peak count capped by tracks ({n_tracks})"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_bin_results(frame_results: List[Dict],
                          tracks: List[SimpleTrack]) -> Dict:
    """
    Aggregate per-frame bin detections using 4 methods:
      1. Mode   2. Median   3. Early-frames   4. Top-N tracks
    Then apply smart decision engine for final count.
    """
    bin_counts = [r["bin_count"] for r in frame_results]

    if not bin_counts:
        return {
            "summary": {"total_bacs": 0, "small_bacs": 0, "large_bacs": 0,
                         "plastique_bacs": 0, "metal_bacs": 0},
            "aggregation_methods": {},
            "frames": [],
        }

    # Method 1: MODE
    count_freq = Counter(bin_counts)
    mode_count, mode_freq = count_freq.most_common(1)[0]
    mode_percentage = (mode_freq / len(bin_counts)) * 100

    print(f"Method 1 - MODE:")
    print(f"   Distribution: {dict(count_freq)}")
    print(f"   Mode: {mode_count} bins "
          f"({mode_freq}/{len(bin_counts)} frames = {mode_percentage:.0f}%)")

    # Method 2: MEDIAN
    sorted_counts = sorted(bin_counts)
    median_count = sorted_counts[len(sorted_counts) // 2]
    print(f"Method 2 - MEDIAN: {median_count} bins")

    # Method 3: EARLY FRAMES
    n_frames = len(bin_counts)
    early_third = bin_counts[: n_frames // 3]
    early_avg = sum(early_third) / len(early_third) if early_third else 0
    early_mode = (Counter(early_third).most_common(1)[0][0]
                  if early_third else 0)
    print(f"Method 3 - EARLY FRAMES:")
    print(f"   First {len(early_third)} frames avg: {early_avg:.1f}, mode: {early_mode}")

    # Method 4: TOP-N TRACKS
    sorted_tracks_list = sorted(tracks, key=lambda t: t.frame_count, reverse=True)

    print(f"Method 4 - TRACKS:")
    print(f"   Total tracks: {len(tracks)}")
    for t in sorted_tracks_list[:5]:
        m, s = t.get_dominant_attributes()
        print(f"     Track #{t.id}: {t.frame_count} frames -> {m}, {s}")

    # Smart decision
    final_count, decision_note = _smart_decision(
        mode_count, mode_percentage, median_count, early_mode, early_avg,
        sorted_tracks_list, bin_counts,
    )

    top_tracks = sorted_tracks_list[:final_count] if final_count > 0 else []
    material_counts = Counter(
        [t.get_dominant_attributes()[0] for t in top_tracks]
    )
    size_counts = Counter(
        [t.get_dominant_attributes()[1] for t in top_tracks]
    )
    plastic_bins = material_counts.get("plastic", 0)
    metal_bins = material_counts.get("metal", 0)
    small_bins = size_counts.get("small", 0)
    large_bins = size_counts.get("large", 0)

    print(f"FINAL DECISION ({decision_note}):")
    print(f"   Total bins: {final_count}")
    print(f"   Material: {plastic_bins}P + {metal_bins}M")
    print(f"   Size: {small_bins}S + {large_bins}L")

    all_track_details = [
        {"id": t.id, "frames": t.frame_count,
         "material": t.get_dominant_attributes()[0],
         "size": t.get_dominant_attributes()[1]}
        for t in sorted_tracks_list
    ]

    return {
        "summary": {
            "total_bacs": final_count,
            "small_bacs": small_bins,
            "large_bacs": large_bins,
            "plastique_bacs": plastic_bins,
            "metal_bacs": metal_bins,
        },
        "aggregation_methods": {
            "mode": {"count": mode_count, "confidence": mode_percentage,
                     "distribution": dict(count_freq)},
            "median": {"count": median_count},
            "early_frames": {"average": early_avg, "mode": early_mode},
            "tracking": {
                "total_tracks": len(sorted_tracks_list),
                "top_tracks_used": len(top_tracks),
                "track_details": all_track_details[:final_count],
                "all_tracks": all_track_details,
            },
            "decision_note": decision_note,
        },
        "frames": frame_results,
    }
