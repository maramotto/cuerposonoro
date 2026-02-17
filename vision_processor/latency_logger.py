"""
Latency logger for benchmarking the Cuerpo Sonoro pipeline.

Instruments each stage of the pipeline (capture, pose, features, send)
and generates CSV reports with per-frame timings and aggregate statistics.

Usage:
    logger = LatencyLogger(config={"camera": "C922", "model_complexity": 1})

    while running:
        logger.start_frame()
        frame = cap.read()
        logger.mark("capture")
        results = pose.estimate(frame)
        logger.mark("pose")
        features = extractor.calculate(landmarks, prev)
        logger.mark("features")
        sender.send_features(features)
        logger.mark("send")
        logger.end_frame(pose_detected=True)

    logger.save()  # writes raw CSV + summary CSV
"""

import os
import time
import csv
import statistics
from datetime import datetime
from typing import Optional


class LatencyLogger:
    """
    Records per-stage latencies for each frame and generates reports.

    Attributes:
        config: dict with hardware/software metadata for the session
        frames: list of per-frame timing records
    """

    def __init__(self, config: Optional[dict] = None, output_dir: str = "logs"):
        """
        Args:
            config: Metadata describing the test configuration, e.g.:
                {
                    "machine": "MacBook Pro 2020 i7",
                    "camera": "Logitech C922",
                    "resolution": "1280x720",
                    "model_complexity": 1,
                    "output_mode": "OSC",
                    "notes": "baseline test",
                }
            output_dir: Directory where CSV files are saved.
        """
        self.config = config or {}
        self.output_dir = output_dir

        # Per-frame storage
        self.frames: list[dict] = []

        # Current frame state
        self._frame_start: float = 0.0
        self._marks: list[tuple[str, float]] = []
        self._frame_number: int = 0

        # Stage names in order (populated from first frame)
        self._stage_names: list[str] = []

        # Session timestamp for filenames
        self._session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def start_frame(self):
        """Call at the very beginning of each pipeline iteration."""
        self._frame_start = time.perf_counter()
        self._marks = []

    def mark(self, stage_name: str):
        """
        Record a timestamp after a pipeline stage completes.

        Args:
            stage_name: Label for the stage, e.g. "capture", "pose",
                        "features", "send".
        """
        self._marks.append((stage_name, time.perf_counter()))

    def end_frame(self, pose_detected: bool = True):
        """
        Finalize the current frame and store its timing data.

        Args:
            pose_detected: Whether MediaPipe detected a pose this frame.
        """
        self._frame_number += 1
        frame_end = time.perf_counter()

        # Build timing record
        record = {
            "frame": self._frame_number,
            "pose_detected": pose_detected,
        }

        # Calculate per-stage durations
        prev_time = self._frame_start
        for stage_name, timestamp in self._marks:
            duration_ms = (timestamp - prev_time) * 1000
            record[f"{stage_name}_ms"] = round(duration_ms, 3)
            prev_time = timestamp

        # Total frame time
        record["total_ms"] = round((frame_end - self._frame_start) * 1000, 3)

        # FPS (instantaneous)
        if record["total_ms"] > 0:
            record["fps"] = round(1000.0 / record["total_ms"], 1)
        else:
            record["fps"] = 0.0

        self.frames.append(record)

        # Capture stage names from first frame
        if not self._stage_names:
            self._stage_names = [name for name, _ in self._marks]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def get_last_total_ms(self) -> float:
        """Return total_ms of the last recorded frame, or 0.0."""
        if self.frames:
            return self.frames[-1].get("total_ms", 0.0)
        return 0.0

    def save(self) -> tuple[str, str]:
        """
        Write raw data and summary statistics to CSV files.

        Returns:
            Tuple of (raw_csv_path, summary_csv_path).
        """
        os.makedirs(self.output_dir, exist_ok=True)

        raw_path = os.path.join(
            self.output_dir, f"latency_raw_{self._session_ts}.csv"
        )
        summary_path = os.path.join(
            self.output_dir, f"latency_summary_{self._session_ts}.csv"
        )

        self._write_raw_csv(raw_path)
        self._write_summary_csv(summary_path)

        print(f"\nLatency logs saved:")
        print(f"  Raw data:  {raw_path} ({len(self.frames)} frames)")
        print(f"  Summary:   {summary_path}")

        return raw_path, summary_path

    def print_summary(self):
        """Print a formatted summary table to the console."""
        if not self.frames:
            print("No frames recorded.")
            return

        stage_cols = [f"{s}_ms" for s in self._stage_names] + ["total_ms"]

        print("\n" + "=" * 65)
        print("  LATENCY BENCHMARK RESULTS")
        print("=" * 65)

        # Config
        if self.config:
            print("\n  Configuration:")
            for key, value in self.config.items():
                print(f"    {key}: {value}")

        print(f"\n  Frames: {len(self.frames)}")

        # Filter frames where pose was detected for accurate stats
        valid = [f for f in self.frames if f.get("pose_detected", True)]
        if len(valid) < len(self.frames):
            print(f"  Valid (pose detected): {len(valid)}")

        print(f"\n  {'Stage':<16} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
        print("  " + "-" * 56)

        for col in stage_cols:
            values = [f[col] for f in valid if col in f]
            if not values:
                continue

            label = col.replace("_ms", "")
            mean = statistics.mean(values)
            p50 = self._percentile(values, 50)
            p95 = self._percentile(values, 95)
            p99 = self._percentile(values, 99)
            mx = max(values)

            print(f"  {label:<16} {mean:>7.1f}ms {p50:>7.1f}ms "
                  f"{p95:>7.1f}ms {p99:>7.1f}ms {mx:>7.1f}ms")

        # FPS stats
        fps_values = [f["fps"] for f in valid if f.get("fps", 0) > 0]
        if fps_values:
            print(f"\n  FPS: mean={statistics.mean(fps_values):.1f}, "
                  f"min={min(fps_values):.1f}, "
                  f"max={max(fps_values):.1f}")

        # Target check
        total_vals = [f["total_ms"] for f in valid if "total_ms" in f]
        if total_vals:
            under_80 = sum(1 for v in total_vals if v <= 80) / len(total_vals) * 100
            print(f"\n  Frames under 80ms target: {under_80:.1f}%")

        print("=" * 65)

    # --- Private helpers ---

    def _write_raw_csv(self, path: str):
        """Write one row per frame with all stage timings."""
        if not self.frames:
            return

        fieldnames = list(self.frames[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.frames)

    def _write_summary_csv(self, path: str):
        """Write aggregate statistics + config metadata."""
        if not self.frames:
            return

        valid = [f for f in self.frames if f.get("pose_detected", True)]
        stage_cols = [f"{s}_ms" for s in self._stage_names] + ["total_ms"]

        rows = []

        # Config rows
        for key, value in self.config.items():
            rows.append({"metric": f"config_{key}", "value": str(value)})

        rows.append({"metric": "total_frames", "value": str(len(self.frames))})
        rows.append({"metric": "valid_frames", "value": str(len(valid))})

        if self.frames:
            detection_rate = len(valid) / len(self.frames) * 100
            rows.append({
                "metric": "detection_rate_pct", "value": f"{detection_rate:.1f}"
            })

        # Per-stage stats
        for col in stage_cols:
            values = [f[col] for f in valid if col in f]
            if not values:
                continue

            label = col.replace("_ms", "")
            rows.append({"metric": f"{label}_mean_ms", "value": f"{statistics.mean(values):.2f}"})
            rows.append({"metric": f"{label}_p50_ms", "value": f"{self._percentile(values, 50):.2f}"})
            rows.append({"metric": f"{label}_p95_ms", "value": f"{self._percentile(values, 95):.2f}"})
            rows.append({"metric": f"{label}_p99_ms", "value": f"{self._percentile(values, 99):.2f}"})
            rows.append({"metric": f"{label}_max_ms", "value": f"{max(values):.2f}"})

        # FPS stats
        fps_values = [f["fps"] for f in valid if f.get("fps", 0) > 0]
        if fps_values:
            rows.append({"metric": "fps_mean", "value": f"{statistics.mean(fps_values):.1f}"})
            rows.append({"metric": "fps_min", "value": f"{min(fps_values):.1f}"})
            rows.append({"metric": "fps_max", "value": f"{max(fps_values):.1f}"})

        # Target compliance
        total_vals = [f["total_ms"] for f in valid if "total_ms" in f]
        if total_vals:
            under_80 = sum(1 for v in total_vals if v <= 80) / len(total_vals) * 100
            rows.append({"metric": "pct_under_80ms", "value": f"{under_80:.1f}"})

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _percentile(values: list[float], pct: int) -> float:
        """Calculate percentile from a sorted list."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 0:
            return 0.0
        idx = int(pct / 100 * (n - 1))
        return sorted_vals[idx]
