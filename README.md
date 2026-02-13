# Cuerpo Sonoro

**Interactive installation that transforms human movement into real-time music.**

Cuerpo Sonoro captures human body movement through computer vision and translates it into musical expression in real time. The system uses AI pose estimation to extract motion features — energy, symmetry, smoothness, joint angles — and maps them to sound synthesis parameters, creating a dialogue between performer and machine.

> Final Degree Project (TFG) · Software Engineering · Universidad Rey Juan Carlos · 2025/2026

**Live Demo:** [cuerposonoro.art](https://cuerposonoro.art)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [MPE Features & Musical Mapping](#mpe-features--musical-mapping)
  - [Chords (Lower Body)](#chords-lower-body)
  - [Melody (Upper Body)](#melody-upper-body)
  - [Global Controls](#global-controls)
  - [Feature Reference Table](#feature-reference-table)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Running the System](#running-the-system)
- [Web Demo](#web-demo)
- [Performance Targets](#performance-targets)
- [Testing](#testing)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [Academic Context](#academic-context)
- [License](#license)
- [Author](#author)

---

## Overview

Cuerpo Sonoro explores how the human body can become a musical instrument. A camera captures the performer's movements, MediaPipe estimates body pose in real time, and a set of algorithms extract meaningful motion features. These features are then mapped to musical parameters and sent via OSC to SuperCollider for audio synthesis, or via MIDI/MPE to external synthesizers like Surge XT.

The system supports two modes of operation:

- **Local installation** — Full pipeline with SuperCollider and/or MIDI/MPE output, designed for live performance and installation contexts.
- **Web demo** — Browser-based version deployed at [cuerposonoro.art](https://cuerposonoro.art), using MediaPipe.js and the Web Audio API so anyone can try it from their webcam.

### Pipeline

```
Camera → Pose Estimation → Feature Extraction → Mapping → Audio Synthesis
          (MediaPipe)        (Python)           (OSC/MIDI)  (SuperCollider / Surge XT / Web Audio)
```

---

## Architecture

The system follows a merged 2-service Docker architecture, chosen over a 3-service separation to minimize latency (saves 5-15ms of inter-container serialization overhead).

```
┌─────────────────────────────────────────┐
│         Vision Processor                │  ← Docker Service 1
│  ┌─────────────────────────────────┐    │
│  │  Camera Capture (OpenCV)        │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  Pose Estimation (MediaPipe)    │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  Feature Extraction             │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  OSC Sender / MIDI Sender       │    │
│  └─────────────────────────────────┘    │
└──────────────────┬──────────────────────┘
                   │ OSC / MIDI messages
                   ▼
┌─────────────────────────────────────────┐
│         Audio Engine                    │  ← Docker Service 2
│  SuperCollider (scsynth + sclang)       │
└─────────────────────────────────────────┘
```

The vision processor uses a **camera abstraction layer**, making it trivial to switch between different camera hardware (e.g., webcam, Intel RealSense D435) with a single configuration change.

---

## MPE Features & Musical Mapping

The system extracts a set of kinematic and postural descriptors from body landmarks and maps them to musical parameters using the MPE (MIDI Polyphonic Expression) paradigm. The body is divided into three zones: lower body controls harmony, upper body controls melody, and head/global motion controls effects.

### Chords (Lower Body)

**1. Chord Selection — Foot position on X axis (`feetCenterX`)**

The feet determine which chord is playing. The camera's capture space is divided into 4 horizontal zones. Walking left or right changes the chord. The midpoint between both ankles is used to calculate position. C major scale with degrees I, IV, V and VI in triads.

**2. Chord Expression — Hip lateral tilt (`hipTilt`)**

The lateral tilt of the hips controls two things: pitch bend (subtle detuning that creates tension) and the addition of extension notes. When the tilt is moderate, pitch bend is applied. When extreme, 6th or 7th notes are added to the chord to enrich it harmonically.

**3. Chord Volume — Knee flexion angle (`kneeAngle`)**

The knee flexion angle controls volume. Extended legs mean maximum volume. Bending the knees progressively reduces volume. This allows natural crescendos and diminuendos with the body.

### Melody (Upper Body)

**4. Right hand melodic note — Right hand Y height (`rightHandY`)**

The vertical height of the right hand in space determines which note plays in the lower octave (C3 to B3). Hand down plays low notes, hand up plays high notes within that octave.

**5. Left hand melodic note — Left hand Y height (`leftHandY`)**

Same as the right hand but in the upper octave (C5 to B5). This enables playing two-voice melodies separated by two octaves.

**6. Note trigger — Sudden hand/wrist movement (`rightHandJerk`, `leftHandJerk`)**

Notes do not sound continuously. They are triggered only when a sudden movement (high jerk) is detected in the hand or wrist. This gives percussive, expressive control over when notes sound.

**7. Note intensity and duration — Arm speed (`rightArmVelocity`, `leftArmVelocity`)**

The speed at which the arm moves during a gesture determines two things: MIDI velocity (how loud the note sounds) and note duration. Fast movement produces loud, short notes (staccato). Slow movement produces soft, longer notes.

**8. Glissando and vibrato — Elbow-to-hip angle (`rightElbowHipAngle`, `leftElbowHipAngle`)**

The angle formed by the arm relative to the torso (measured between elbow and hip) controls pitch bend on melodic notes. Arm close to the body means a stable note. Extended arm applies glissando. Oscillating elbow movement generates vibrato.

### Global Controls

**9. Global frequency filter — Head tilt (`headTilt`)**

The lateral tilt of the head controls a global frequency filter that affects all sound. Head straight is neutral sound. Tilting to one side darkens the sound, tilting to the other makes it brighter.

**10. Textures and drones — Global energy (`energy`)**

The overall motion energy of the body is sent to SuperCollider to control background textures and drones that complement the melodic and harmonic MPE output.

### Feature Reference Table

Features implemented in `features.py`:

| Feature | Landmarks Used | Output Range |
|---------|---------------|-------------|
| `feetCenterX` | Ankles (27, 28) | 0.0 – 1.0 |
| `hipTilt` | Hips (23, 24) | -1.0 – 1.0 |
| `kneeAngle` | Hip, knee, ankle (23/24, 25/26, 27/28) | 0.0 – 1.0 |
| `rightHandY` | Right wrist (16) | 0.0 – 1.0 |
| `leftHandY` | Left wrist (15) | 0.0 – 1.0 |
| `rightHandJerk` | Right wrist (16) velocity | 0.0 – 1.0 |
| `leftHandJerk` | Left wrist (15) velocity | 0.0 – 1.0 |
| `rightArmVelocity` | Right wrist (16) | 0.0 – 1.0 |
| `leftArmVelocity` | Left wrist (15) | 0.0 – 1.0 |
| `rightElbowHipAngle` | Shoulder, elbow, hip (12, 14, 24) | 0.0 – 1.0 |
| `leftElbowHipAngle` | Shoulder, elbow, hip (11, 13, 23) | 0.0 – 1.0 |
| `headTilt` | Ears (7, 8) | -1.0 – 1.0 |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Pose Estimation | [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) | Real-time body landmark detection (33 points) |
| Audio Synthesis | [SuperCollider](https://supercollider.github.io/) | Algorithmic sound synthesis via OSC |
| MIDI/MPE Output | [python-rtmidi](https://github.com/SpotlightKid/python-rtmidi) | MIDI Polyphonic Expression for external synths |
| External Synth | [Surge XT](https://surge-synthesizer.github.io/) | Open-source MPE-compatible synthesizer |
| Video Capture | [OpenCV](https://opencv.org/) | Camera access and frame processing |
| Communication | [python-osc](https://github.com/attwad/python-osc) | OSC protocol for real-time control |
| Web Frontend | MediaPipe.js + Web Audio API | Browser-based pose detection and synthesis |
| Web Backend | [FastAPI](https://fastapi.tiangolo.com/) + WebSocket | Real-time feature extraction server |
| Containerization | [Docker](https://www.docker.com/) + Docker Compose | Portable, reproducible deployment |
| Infrastructure | Hetzner VPS + Nginx + Let's Encrypt | Cloud deployment with HTTPS |
| Language | Python 3.10+ | Core application logic |
| Dependencies | pip-tools | Reproducible dependency management |

---

## Project Structure

```
cuerposonoro/
├── vision_processor/           # Perception + feature extraction module
│   ├── __init__.py
│   ├── capture.py              # Camera abstraction layer
│   ├── pose.py                 # MediaPipe pose estimation wrapper
│   ├── features.py             # Motion feature extraction (17 features)
│   ├── osc_sender.py           # OSC communication to SuperCollider
│   └── midi_sender.py          # MIDI/MPE communication to Surge XT
├── audio_engine/               # SuperCollider audio synthesis
│   └── ...
├── supercollider/              # SuperCollider SynthDef files
│   └── ...
├── tests/
│   ├── unit/                   # Automated unit tests (pytest)
│   │   └── test_features.py
│   ├── integration/            # Automated integration tests (pytest)
│   │   └── test_integration.py
│   └── manual/                 # Interactive scripts (require hardware)
│       ├── README.md
│       ├── manual_camera.py
│       ├── manual_pose.py
│       ├── manual_osc.py
│       ├── manual_midi_sender.py
│       ├── manual_e2e_osc.py
│       └── manual_e2e_midi_debug.py
├── logs/                       # Session logs (CSV) for analysis
├── main.py                     # Application entry point
├── config.yaml                 # Centralized configuration
├── requirements.in             # Top-level dependencies
├── requirements.txt            # Pinned dependencies (pip-tools)
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **SuperCollider** — [Download](https://supercollider.github.io/downloads)
- **Webcam** (Logitech C922 recommended, any USB webcam works)
- **Docker** (optional, for containerized deployment)

For MIDI/MPE mode:
- **Surge XT** — [Download](https://surge-synthesizer.github.io/) (free, open-source, GPL3)

### Local Installation

1. **Clone the repository:**

```bash
git clone https://github.com/AmarilloBit/cuerposonoro.git
cd cuerposonoro
```

2. **Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

3. **Configure the system:**

Edit `config.yaml` to adjust camera settings, OSC ports, feature parameters, and more.

### Running the System

**With SuperCollider (OSC mode):**

1. Open SuperCollider and load the SynthDef files from the `supercollider/` directory.
2. Boot the SuperCollider server.
3. Run the vision processor:

```bash
python main.py
```

**With Surge XT (MIDI/MPE mode):**

1. Open Surge XT in standalone mode.
2. Enable MIDI input from "Cuerpo Sonoro" virtual port.
3. Run the vision processor with MIDI output enabled.

The system will open a camera window with a skeleton overlay for real-time debugging.

---

## Web Demo

A browser-based version is deployed at **[cuerposonoro.art](https://cuerposonoro.art)**, allowing anyone with a webcam to experience the installation without any software installation.

The web demo uses a different architecture optimized for the browser:

```
┌─────────────────────────────────────────────────┐
│                   Browser                       │
│  Camera → MediaPipe.js → WebSocket → Web Audio  │
└────────────────────────┬────────────────────────┘
                         │ landmarks (WebSocket)
                         ▼
┌─────────────────────────────────────────────────┐
│              Server (Docker)                    │
│  Nginx (static + reverse proxy)                 │
│  FastAPI (feature extraction via WebSocket)     │
└─────────────────────────────────────────────────┘
```

Pose detection runs client-side in the browser using MediaPipe.js. Landmarks are sent to the FastAPI backend via WebSocket for feature extraction, and the computed features are returned to drive browser-based audio synthesis through the Web Audio API.

---

## Testing

The project has a three-tier testing strategy:

**Automated tests (no hardware required):**
```bash
pytest tests/ -v                          # all automated (129 tests)
pytest tests/unit/ -v                     # unit tests only (80 tests)
pytest tests/integration/ -v              # integration tests only (~50 tests)
```

- **Unit tests** (`tests/unit/test_features.py`) — test all 17 feature extraction methods with synthetic landmark data. Cover output ranges, edge cases, temporal smoothing, and boundary conditions.
- **Integration tests** (`tests/integration/test_integration.py`) — test OSCSender, MidiSender, and full pipeline flows with mocked I/O (no SuperCollider or Surge XT needed).

**Manual tests (require hardware):**

Interactive scripts for validating the full system with real camera input and audio output. See [`tests/manual/README.md`](tests/manual/README.md) for prerequisites and usage instructions.
```bash
python tests/manual/manual_camera.py      # verify webcam
python tests/manual/manual_pose.py        # test pose detection
python tests/manual/manual_e2e_osc.py     # full pipeline → SuperCollider
python tests/manual/manual_e2e_midi_debug.py  # full pipeline → Surge XT
```

Manual E2E tests generate CSV logs in `tests/manual/logs/` with per-frame data for post-hoc analysis.
---

## Testing

The project includes integration tests that validate the full pipeline from camera capture to audio output.

```bash
# Run the end-to-end integration test
python tests/test-integration.py

# Run the MIDI/MPE end-to-end test with CSV logging
python tests/test-midi-e2e-debug-log.py
```

Tests generate CSV logs in `logs/` with per-frame data: timestamp, FPS, pose detection status, all feature values, and trigger events for later analysis.

---

## Deployment

### Docker (local)

```bash
docker compose up --build
```

### Cloud (Hetzner VPS)

The web demo is deployed on a Hetzner VPS with Docker Compose, Nginx as a reverse proxy, and Let's Encrypt for HTTPS (required for browser camera access).

```bash
ssh user@your-server
cd ~/cuerposonoro-webdemo
docker compose up -d --build
```

SSL certificates are managed with Certbot and automatically renewed.

---

## Future Work

- **Phase 2 — Depth sensing:** Integrate Intel RealSense D435 for 3D pose analysis, enabled by the existing camera abstraction layer.
- **Phase 3 — Visual projection:** Add real-time visual feedback using TouchDesigner or a Python-based solution, synchronized with audio output.
- **Mobile optimization:** Improve the web demo experience on mobile browsers.
- **Extended musical mappings:** Explore more complex harmonic and timbral mappings.

---

## Academic Context

This project is a Final Degree Project (Trabajo de Fin de Grado, TFG) for the **Software Engineering degree** at **Universidad Rey Juan Carlos (URJC)**, Madrid, Spain.

It bridges software engineering, artificial intelligence, and digital art, demonstrating competencies in computer vision, real-time data processing, audio synthesis, cloud deployment, and interactive system design.

### Evaluation Criteria

- **Quantitative:** FPS, end-to-end latency, pose detection robustness.
- **Qualitative:** User testing with 3–5 participants measuring perceived responsiveness, expressiveness, intuitiveness, engagement, and sense of control (Likert scale questionnaire).

### Ethical Considerations

- Only pose landmark coordinates are processed — no images or video are stored.
- GDPR-compliant: no personal or identifiable data is retained.
- User testing participants sign informed consent forms.

---

## License

This project is open source. See the [LICENSE](LICENSE) file for details.

- **Code:** [Unlicense](LICENSE)
- **Documentation:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

---

## Author

**Ana María Jurado Crespo**
- Web: [maramotto.com](https://maramotto.com)
- Github:  [@maramotto](https://github.com/maramotto)
- Email: am.juradoc@alumnos.urjc.es
- University: ETSII, Universidad Rey Juan Carlos
