# Latency Benchmarks

Systematic evaluation of the Cuerpo Sonoro pipeline latency across different hardware and software configurations. This data directly supports the **Testing and Evaluation** chapter of the TFG.

---

## Table of Contents

- [Methodology](#methodology)
- [Test Environment](#test-environment)
- [Configuration Matrix](#configuration-matrix)
- [Results](#results)
  - [Overview](#overview)
  - [Per-Stage Breakdown](#per-stage-breakdown)
  - [Pose Model Impact](#pose-model-impact)
  - [Key Findings](#key-findings)
- [Latency Distribution](#latency-distribution)
- [Conclusions](#conclusions)
- [Reproducing the Benchmarks](#reproducing-the-benchmarks)
  - [Running Benchmarks](#running-benchmarks)
  - [Analyzing Results](#analyzing-results)
  - [Directory Structure](#directory-structure)

---

## Methodology

The benchmarking system instruments the four stages of the real-time pipeline using `time.perf_counter()`:

```
Camera Capture → Pose Estimation → Feature Extraction → OSC/MIDI Send
    (capture)       (pose)            (features)          (send)
```

Each benchmark run:

1. Opens the camera and initializes all pipeline components for the given configuration.
2. Discards 30 warmup frames to let MediaPipe's internal tracking stabilize.
3. Measures 300 frames, recording per-stage timestamps for each frame.
4. Runs **headless** (no OpenCV display window) to avoid contaminating measurements with rendering overhead.
5. Outputs raw per-frame data and aggregate statistics (mean, P50, P95, P99, max) as CSV files.

An optional `--preview` mode opens a camera window with skeleton overlay *before* each configuration starts, allowing the operator to verify positioning, then closes it before headless measurement begins.

**What is measured:** Software-level latency from frame capture to OSC/MIDI send. This does not include the audio engine processing time (SuperCollider buffer, typically 3-6ms on CoreAudio, 1.5ms on JACK).

**What is not measured:** The final audio output latency (DAC buffer). For full end-to-end measurement, a high-speed camera recording both the skeleton display and audio output would be needed.

---

## Test Environment

| Component | Specification |
|-----------|--------------|
| Machine | MacBook Pro 2020 (Intel i7 Quad-Core 2.3 GHz, 32 GB RAM) |
| OS | macOS (Darwin x86_64) |
| Camera 1 | MacBook Pro Built-in (720p native) |
| Camera 2 | Logitech C922 (1080p native, connected via USB) |
| Python | 3.11 |
| MediaPipe | Pose (CPU inference, no GPU on macOS) |
| Frames per config | 300 (+ 30 warmup) |

---

## Configuration Matrix

The benchmark tests a full matrix of **36 configurations**:

```
2 cameras × 2 resolutions × 3 pose models × 3 output modes = 36
```

| Dimension | Values |
|-----------|--------|
| Camera | MacBook Pro Built-in (device 1), Logitech C922 (device 0) |
| Resolution | 480p (640×480), 720p (1280×720) |
| Pose model | Lite (complexity=0), Full (complexity=1), Heavy (complexity=2) |
| Output mode | OSC individual messages, OSC bundle, MIDI/MPE |

---

## Results

### Overview

All 36 configurations sorted by mean total latency. Lite and Full models comfortably meet the 80ms target across all cameras and output modes. Heavy consistently exceeds it.

![Benchmark comparison across all configurations](charts/benchmark_output_mean_latency.png)

The fastest configuration is `macbook_480p_lite_osc_bundle` at **33.3ms** mean latency. The slowest is `c922_720p_heavy_midi` at **99.2ms**. The full comparison table sorted by mean latency:

![Full comparison table](charts/benchmark_output_comparison-configurations.png)

### Per-Stage Breakdown

The per-stage breakdown reveals where time is spent in each configuration:

![Per-stage breakdown in milliseconds](charts/benchmark_output_per-stage-breakdown.png)

Across all configurations, the time distribution follows a consistent pattern: capture and pose dominate, while features and send are negligible:

![Per-stage percentage breakdown](charts/benchmark_output_comparison-configurations-percentage.png)

The stacked bar chart makes the pattern visually clear — Lite and Full configurations stay well under the 80ms red line, while Heavy configurations blow past it:

![Per-stage stacked bar chart](charts/chart_stages.png)

### Pose Model Impact

The pose model complexity is the single most impactful variable in the entire matrix. This grouped bar chart isolates the effect:

![Pose model complexity comparison](charts/chart_pose_comparison.png)

| Pose Model | Mean Latency | Pose Stage | FPS | Under 80ms |
|------------|-------------|-----------|-----|-----------|
| **Lite** (complexity=0) | 33.7ms | ~17ms | ~30 | 99–100% |
| **Full** (complexity=1) | 34.7ms | ~21ms | ~30 | 99–100% |
| **Heavy** (complexity=2) | 86.6ms | ~65ms | ~12 | 6–85% |

Critical observations:

- **Full costs only +1.0ms vs Lite.** This is the most important finding — Full provides better pose accuracy at virtually no latency cost, making it the clear recommendation for production use.
- **Heavy costs +51.9ms vs Full.** This nearly triples the pose estimation time and pushes most configurations above the 80ms target. Heavy is not viable for real-time interaction on this hardware.
- The +4ms difference between Lite and Full in the pose stage is partially offset by capture time differences across configurations.

### Key Findings

The automated analysis produces this summary:

![Key findings](charts/benchmark_output_key-findings.png)

Summary of findings across all 36 configurations:

| Variable | Finding | Impact |
|----------|---------|--------|
| **Pose model** | Bottleneck. Full ≈ Lite, Heavy 2.5× slower | **Critical** — choose Full |
| **Camera** | C922 1.7ms slower than Built-in on average | **Negligible** |
| **Resolution** | 480p vs 720p: ~1-3ms difference | **Negligible** (MediaPipe resizes internally to 256×256) |
| **Output mode** | OSC vs MIDI: ~3.5ms difference | **Negligible** |
| **OSC send mode** | Individual vs bundle: <1ms difference | **Negligible** |
| **Pipeline bottleneck** | Pose estimation: 67% of total time | Features + Send combined: <1ms |

---

## Latency Distribution

The box plot shows the full distribution of per-frame latency for all 36 configurations. The red dashed line marks the 80ms target:

![Latency distribution box plot](charts/chart_boxplot.png)

Lite and Full configurations cluster tightly around 33-35ms with occasional outliers. Heavy configurations show both higher medians (70-100ms) and wider variance, with P95 values reaching 110-115ms.

---

## Conclusions

1. **The 80ms target is met** by all Lite and Full configurations (99-100% of frames under target), on both cameras and all output modes.

2. **MediaPipe Full (complexity=1) is the recommended model** for production use. It provides better pose accuracy than Lite at essentially the same latency cost (+1ms). There is no reason to use Lite unless targeting significantly weaker hardware.

3. **MediaPipe Heavy (complexity=2) is not viable** for real-time interaction on this hardware. Only 6-85% of frames meet the 80ms target depending on configuration.

4. **Camera choice doesn't matter for latency.** The Logitech C922 and MacBook built-in camera perform within 2ms of each other. Camera choice should be driven by image quality, field of view, and mounting considerations.

5. **Output protocol doesn't matter for latency.** OSC and MIDI are both fire-and-forget at the send stage (<0.5ms). Bundle vs individual OSC messages make no measurable difference.

6. **Resolution doesn't matter for latency.** MediaPipe internally resizes input to 256×256 regardless of capture resolution. The 1-3ms difference is from capture buffer handling, not from pose inference.

7. **Pose estimation is the bottleneck** (67% of pipeline time). Any future optimization effort should focus on this stage — for example, GPU-accelerated inference or switching to a lighter model architecture.

---

## Reproducing the Benchmarks

### Running Benchmarks

```bash
# List all 36 combinations without running
python benchmarks/run_benchmark.py --list

# Run the full matrix with camera preview before each config
python benchmarks/run_benchmark.py --preview --session-name full-matrix

# Run a filtered subset
python benchmarks/run_benchmark.py --camera-profile c922 --pose full --output osc

# Custom frame count
python benchmarks/run_benchmark.py --frames 500 --warmup 50
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--list` | Print all combinations and exit |
| `--preview` | Show camera feed with skeleton before each run |
| `--camera-profile` | Filter by camera (e.g. `c922`, `macbook`) |
| `--pose` | Filter by pose model (e.g. `lite`, `full`, `heavy`) |
| `--output` | Filter by output mode (e.g. `osc`, `midi`) |
| `--frames N` | Frames to measure per config (default: 300) |
| `--warmup N` | Warmup frames to discard (default: 30) |
| `--session-name` | Custom name for the results folder |
| `--config-path` | Path to config.yaml if not at project root |

**Before running**, verify camera device IDs match `config.yaml`:

```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'  device {i}: {w}x{h}')
        cap.release()
"
```

**Tips for accurate measurements:** Close CPU-heavy applications, connect the power adapter (avoid thermal throttling), and stay visible in the camera frame throughout the run.

### Analyzing Results

Requires `pandas` and `matplotlib`:

```bash
pip install pandas matplotlib
```

```bash
# Analyze all sessions
python benchmarks/analyze_results.py

# Analyze a specific session
python benchmarks/analyze_results.py --session 2026-02-17_full-matrix

# Save charts as PNG to benchmarks/charts/
python benchmarks/analyze_results.py --save

# Filter by keyword
python benchmarks/analyze_results.py --filter c922

# Console output only, no charts
python benchmarks/analyze_results.py --no-charts

# List available sessions
python benchmarks/analyze_results.py --list-sessions
```

The analysis script outputs:

- **Comparison table** — all configurations sorted by mean latency with P50, P95, FPS, and % under 80ms
- **Per-stage breakdown** — mean latency and percentage per pipeline stage
- **Key findings** — automated identification of fastest/slowest configs, pose model impact, camera comparison, bottleneck analysis
- **Charts** (with `--save`):
  - `chart_stages.png` — stacked bar chart of per-stage latency
  - `chart_boxplot.png` — box plot of latency distribution
  - `chart_pose_comparison.png` — grouped bar comparing pose model complexities

### Directory Structure

```
benchmarks/
├── README.md                  ← This file
├── run_benchmark.py           ← Benchmark runner
├── analyze_results.py         ← Analysis with pandas + matplotlib
├── results/                   ← Session folders with raw data (gitignored)
│   └── 2026-02-17_full-matrix/
│       ├── latency_raw_20260217_*.csv
│       └── latency_summary_20260217_*.csv
└── charts/                    ← Generated charts and screenshots (gitignored)
    ├── chart_stages.png
    ├── chart_boxplot.png
    ├── chart_pose_comparison.png
    ├── benchmark_output_mean_latency.png
    ├── benchmark_output_per-stage-breakdown.png
    ├── benchmark_output_comparison-configurations.png
    ├── benchmark_output_comparison-configurations-percentage.png
    └── benchmark_output_key-findings.png
```

Raw CSV files and charts are gitignored. To reproduce: run the benchmarks, then run the analysis with `--save`.