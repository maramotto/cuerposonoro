"""
MIDI/MPE sender for communicating with Surge XT.
Translates motion features into expressive MIDI messages.

MPE Channel allocation:
- Channel 1:  Master (global messages)
- Channel 2:  Chord note 1 (root)
- Channel 3:  Chord note 2 (3rd)
- Channel 4:  Chord note 3 (5th)
- Channel 5:  Melody right hand (lower octave)
- Channel 6:  Melody left hand (higher octave)
"""

import mido
import time
from typing import Optional


class MidiSender:
    """Sends motion features to Surge XT via MPE/MIDI."""

    # Scale definition: C Major
    SCALE_NOTES = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B (semitones from root)

    # Chord definitions for C Major (MIDI note numbers)
    # Each chord is a tuple of 3 notes (triad)
    CHORDS = {
        "I":   (48, 52, 55),  # C3, E3, G3  (C Major)
        "IV":  (53, 57, 60),  # F3, A3, C4  (F Major)
        "V":   (55, 59, 62),  # G3, B3, D4  (G Major)
        "VI":  (57, 60, 64),  # A3, C4, E4  (A minor)
    }

    # Chord zones mapped to X position
    CHORD_ZONES = [
        (0.00, 0.25, "I"),
        (0.25, 0.50, "IV"),
        (0.50, 0.75, "V"),
        (0.75, 1.00, "VI"),
    ]

    # MPE Channel assignments
    CH_MASTER = 0        # Channel 1 (0-indexed in mido)
    CH_CHORD_ROOT = 1    # Channel 2
    CH_CHORD_THIRD = 2   # Channel 3
    CH_CHORD_FIFTH = 3   # Channel 4
    CH_MELODY_RIGHT = 4  # Channel 5
    CH_MELODY_LEFT = 5   # Channel 6

    # Melody note ranges
    MELODY_RIGHT_BASE = 48   # C3 (lower octave)
    MELODY_LEFT_BASE = 72    # C5 (higher octave)

    # Thresholds
    JERK_THRESHOLD = 0.4     # Minimum jerk to trigger a note
    HIP_TILT_THRESHOLD = 0.6 # Threshold for adding 6th/7th to chord

    def __init__(self, port_name: str = "Cuerpo Sonoro"):
        """
        Initialize MIDI sender with a virtual port.

        Args:
            port_name: Name of the virtual MIDI port to create
        """
        self.port_name = port_name
        self.port = None

        # State tracking
        self.current_chord = None
        self.current_chord_notes = []
        self.melody_right_note = None
        self.melody_left_note = None
        self.melody_right_note_time = 0
        self.melody_left_note_time = 0

        # Note duration settings (in seconds)
        self.base_note_duration = 0.3
        self.min_note_duration = 0.15
        self.max_note_duration = 0.6

        self._open_port()

    def _open_port(self):
        """Open virtual MIDI port."""
        try:
            self.port = mido.open_output(self.port_name, virtual=True)
            print(f"MIDI port opened: {self.port_name}")
            print("Connect Surge XT to this port to receive MIDI.")
        except Exception as e:
            print(f"Error opening MIDI port: {e}")
            print("Available ports:", mido.get_output_names())

    def close(self):
        """Close MIDI port and send all notes off."""
        if self.port:
            self._all_notes_off()
            self.port.close()
            print("MIDI port closed.")

    def update(self, features: dict):
        """
        Main update method called each frame.
        Routes features to appropriate handlers.

        Args:
            features: Dictionary with all extracted motion features
        """
        if not self.port:
            return

        # Update chords (lower body)
        self._update_chords(features)

        # Update melody (upper body)
        self._update_melody(features)

        # Update global expression
        self._update_global_expression(features)


    # CHORD HANDLING

    def _update_chords(self, features: dict):
        """
        Handle chord selection and expression based on lower body features.

        Features used:
        - feetCenterX: chord zone selection
        - hipTilt: pitch bend + chord extensions
        - kneeAngle: chord volume
        """
        feet_x = features.get("feetCenterX", 0.5)
        hip_tilt = features.get("hipTilt", 0.0)
        knee_angle = features.get("kneeAngle", 1.0)

        # Determine which chord zone we're in
        new_chord = self._get_chord_from_position(feet_x)

        # If chord changed, update notes
        if new_chord != self.current_chord:
            self._change_chord(new_chord, knee_angle)

        # Apply expression to current chord
        if self.current_chord:
            self._apply_chord_expression(hip_tilt, knee_angle)

    def _get_chord_from_position(self, x: float) -> str:
        """Determine chord based on horizontal position."""
        for min_x, max_x, chord in self.CHORD_ZONES:
            if min_x <= x < max_x:
                return chord
        return "I"  # Default

    def _change_chord(self, new_chord: str, velocity_factor: float):
        """
        Change to a new chord.

        Args:
            new_chord: Chord name (I, IV, V, VI)
            velocity_factor: 0.0-1.0 from knee angle
        """
        # Turn off old chord notes
        for note, channel in zip(self.current_chord_notes,
                                  [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]):
            self._note_off(note, channel)

        # Get new chord notes
        self.current_chord_notes = list(self.CHORDS[new_chord])
        self.current_chord = new_chord

        # Calculate velocity from knee angle (bent = quiet, straight = loud)
        velocity = int(40 + velocity_factor * 87)  # Range: 40-127
        velocity = max(1, min(127, velocity))

        # Turn on new chord notes
        channels = [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]
        for note, channel in zip(self.current_chord_notes, channels):
            self._note_on(note, velocity, channel)

        print(f"Chord: {new_chord} (velocity: {velocity})")

    def _apply_chord_expression(self, hip_tilt: float, knee_angle: float):
        """
        Apply expression to current chord.

        Args:
            hip_tilt: -1.0 to 1.0 for pitch bend
            knee_angle: 0.0 to 1.0 for volume (via aftertouch)
        """
        # Apply pitch bend based on hip tilt
        # MPE pitch bend range: 0-16383, center = 8192
        pitch_bend = int(8192 + hip_tilt * 4096)  # ±4096 from center
        pitch_bend = max(0, min(16383, pitch_bend))

        channels = [self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH]
        for channel in channels:
            self._pitch_bend(pitch_bend, channel)

        # Apply aftertouch (pressure) based on knee angle
        pressure = int(knee_angle * 127)
        for channel in channels:
            self._channel_pressure(pressure, channel)


    # MELODY HANDLING

    def _update_melody(self, features: dict):
        """
        Handle melody notes based on upper body features.

        Features used:
        - rightHandY / leftHandY: note selection
        - rightHandJerk / leftHandJerk: note triggering
        - rightArmVelocity / leftArmVelocity: velocity and duration
        - rightElbowHipAngle / leftElbowHipAngle: pitch bend
        """
        current_time = time.time()

        # Right hand melody
        self._update_melody_hand(
            features,
            side="right",
            hand_y=features.get("rightHandY", 0.5),
            jerk=features.get("rightHandJerk", 0.0),
            arm_velocity=features.get("rightArmVelocity", 0.0),
            elbow_angle=features.get("rightElbowHipAngle", 0.0),
            base_note=self.MELODY_RIGHT_BASE,
            channel=self.CH_MELODY_RIGHT,
            current_time=current_time
        )

        # Left hand melody
        self._update_melody_hand(
            features,
            side="left",
            hand_y=features.get("leftHandY", 0.5),
            jerk=features.get("leftHandJerk", 0.0),
            arm_velocity=features.get("leftArmVelocity", 0.0),
            elbow_angle=features.get("leftElbowHipAngle", 0.0),
            base_note=self.MELODY_LEFT_BASE,
            channel=self.CH_MELODY_LEFT,
            current_time=current_time
        )

    def _update_melody_hand(self, features: dict, side: str, hand_y: float,
                            jerk: float, arm_velocity: float, elbow_angle: float,
                            base_note: int, channel: int, current_time: float):
        """
        Update melody for one hand.

        Args:
            side: "right" or "left"
            hand_y: vertical position (0-1)
            jerk: sudden movement amount (0-1)
            arm_velocity: arm speed (0-1)
            elbow_angle: elbow-hip angle (0-1)
            base_note: MIDI base note for this hand's octave
            channel: MIDI channel for this hand
            current_time: current timestamp
        """
        # Get current note state
        if side == "right":
            current_note = self.melody_right_note
            note_start_time = self.melody_right_note_time
        else:
            current_note = self.melody_left_note
            note_start_time = self.melody_left_note_time

        # Calculate target note from hand position
        target_note = self._hand_y_to_note(hand_y, base_note)

        # Check if we should trigger a new note (jerk exceeds threshold)
        if jerk > self.JERK_THRESHOLD:
            # Calculate velocity from arm velocity
            velocity = int(60 + arm_velocity * 67)  # Range: 60-127
            velocity = max(1, min(127, velocity))

            # Calculate note duration from arm velocity
            # Fast movement = short note, slow = longer note
            duration = self.base_note_duration - (arm_velocity * 0.15)
            duration = max(self.min_note_duration, min(self.max_note_duration, duration))

            # Turn off current note if any
            if current_note is not None:
                self._note_off(current_note, channel)

            # Turn on new note
            self._note_on(target_note, velocity, channel)

            # Update state
            if side == "right":
                self.melody_right_note = target_note
                self.melody_right_note_time = current_time
            else:
                self.melody_left_note = target_note
                self.melody_left_note_time = current_time

            print(f"Melody {side}: note {target_note} (vel: {velocity}, dur: {duration:.2f}s)")

        # Check if current note should end (duration exceeded)
        elif current_note is not None:
            # Use velocity-based duration
            note_duration = self.base_note_duration
            if current_time - note_start_time > note_duration:
                self._note_off(current_note, channel)
                if side == "right":
                    self.melody_right_note = None
                else:
                    self.melody_left_note = None

        # Apply pitch bend from elbow angle (if note is playing)
        if current_note is not None:
            # Elbow angle controls pitch bend for glissando/vibrato
            pitch_bend = int(8192 + elbow_angle * 2048)  # Smaller range than chords
            pitch_bend = max(0, min(16383, pitch_bend))
            self._pitch_bend(pitch_bend, channel)

    def _hand_y_to_note(self, hand_y: float, base_note: int) -> int:
        """
        Convert hand Y position to a note in the scale.

        Args:
            hand_y: 0.0 (low) to 1.0 (high)
            base_note: Base MIDI note for this octave

        Returns:
            MIDI note number (constrained to C Major scale)
        """
        # Map hand_y to scale index (0-7 for 8 notes in an octave)
        scale_index = int(hand_y * 7.99)  # 0-7
        scale_index = max(0, min(7, scale_index))

        # Handle octave wrapping
        if scale_index == 7:
            # High C of next octave
            return base_note + 12

        # Get semitone offset from scale
        semitone_offset = self.SCALE_NOTES[scale_index]

        return base_note + semitone_offset


    # GLOBAL EXPRESSION

    def _update_global_expression(self, features: dict):
        """
        Handle global expression parameters.

        Features used:
        - headTilt: global filter (CC74 on master channel)
        """
        head_tilt = features.get("headTilt", 0.0)

        # Map head tilt to CC74 (Slide/Brightness)
        # -1.0 = 0 (dark), 0.0 = 64 (neutral), 1.0 = 127 (bright)
        cc_value = int(64 + head_tilt * 63)
        cc_value = max(0, min(127, cc_value))

        # Send on master channel to affect all sounds
        self._control_change(74, cc_value, self.CH_MASTER)


    # LOW-LEVEL MIDI METHODS

    def _note_on(self, note: int, velocity: int, channel: int):
        """Send MIDI Note On message."""
        msg = mido.Message('note_on', note=note, velocity=velocity, channel=channel)
        self.port.send(msg)

    def _note_off(self, note: int, channel: int):
        """Send MIDI Note Off message."""
        msg = mido.Message('note_off', note=note, velocity=0, channel=channel)
        self.port.send(msg)

    def _pitch_bend(self, value: int, channel: int):
        """
        Send MIDI Pitch Bend message.

        Args:
            value: 0-16383 (8192 = center/no bend)
            channel: MIDI channel
        """
        msg = mido.Message('pitchwheel', pitch=value - 8192, channel=channel)
        self.port.send(msg)

    def _channel_pressure(self, value: int, channel: int):
        """
        Send MIDI Channel Pressure (Aftertouch) message.

        Args:
            value: 0-127
            channel: MIDI channel
        """
        msg = mido.Message('aftertouch', value=value, channel=channel)
        self.port.send(msg)

    def _control_change(self, control: int, value: int, channel: int):
        """
        Send MIDI Control Change message.

        Args:
            control: CC number (e.g., 74 for Slide)
            value: 0-127
            channel: MIDI channel
        """
        msg = mido.Message('control_change', control=control, value=value, channel=channel)
        self.port.send(msg)

    def _all_notes_off(self):
        """Send All Notes Off on all used channels."""
        channels = [
            self.CH_CHORD_ROOT, self.CH_CHORD_THIRD, self.CH_CHORD_FIFTH,
            self.CH_MELODY_RIGHT, self.CH_MELODY_LEFT
        ]
        for channel in channels:
            # CC 123 = All Notes Off
            self._control_change(123, 0, channel)
