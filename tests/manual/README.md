# Manual Tests

These are interactive test scripts that require hardware (camera, speakers) and/or external software (SuperCollider, Surge XT). They must be run manually by a person — they are **not** collected by pytest.

All commands should be run from the **project root** (`cuerposonoro/`).

---

## Scripts

### manual_camera.py — Camera Capture
Verifies webcam works and displays FPS.

**Requires:** Webcam  
**Run:** `python tests/manual/manual_camera.py`  
**Controls:** `q` quit, `s` save screenshot to `tests/manual/logs/`

---

### manual_pose.py — Pose Estimation
Tests MediaPipe pose detection with skeleton overlay.

**Requires:** Webcam  
**Run:** `python tests/manual/manual_pose.py`  
**Controls:** `q` quit  
**Shows:** FPS, detection rate, skeleton overlay

---

### manual_osc.py — OSC Communication
Sends test OSC messages to SuperCollider.

**Requires:** SuperCollider running with OSCdef listener  
**Setup:**
1. Open SuperCollider
2. Boot server: `s.boot;`
3. Run the OSCdef listener (see main project README)

**Run:** `python tests/manual/manual_osc.py`

---

### manual_e2e_osc.py — Full Pipeline (OSC)
End-to-end test: Camera → Pose → Features → OSC → SuperCollider.

**Requires:** Webcam + SuperCollider  
**Setup:**
1. Open SuperCollider
2. Run: `s.options.numInputBusChannels = 0; s.boot;`
3. Load the OSCdef listeners

**Run:** `python tests/manual/manual_e2e_osc.py`  
**Controls:** `q` quit  
**Shows:** FPS, feature bars, skeleton overlay, OSC status

---

### manual_e2e_midi_debug.py — Full Pipeline (MIDI/MPE)
End-to-end test: Camera → Pose → Features → MIDI → Surge XT, with CSV logging.

**Requires:** Webcam + Surge XT  
**Setup:**
1. Open Surge XT Standalone
2. Go to Menu → MIDI Settings
3. Select "Cuerpo Sonoro" as MIDI input
4. Enable MPE mode

**Run:** `python tests/manual/manual_e2e_midi_debug.py`  
**Controls:** `q` quit  
**Output:** CSV log saved to `tests/manual/logs/midi_e2e_debug_TIMESTAMP.csv`  
**Shows:** FPS, chord zone, jerk values, trigger alerts

---

### manual_midi_sender.py — MIDI Sender Unit Test
Tests MidiSender in isolation: chord changes, melody triggers, expression.

**Requires:** Surge XT  
**Setup:**
1. Open Surge XT Standalone
2. Go to Menu → MIDI Settings
3. Select "Cuerpo Sonoro" as MIDI input
4. Enable MPE mode

**Run:** `python tests/manual/manual_midi_sender.py`  
**Shows:** Chord changes, note triggers, expression values

## Logs

The `logs/` folder contains CSV files from previous MIDI debug sessions and screenshots. These are gitignored except for the directory structure.