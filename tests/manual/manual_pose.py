#!/usr/bin/env python3
"""
Test pose estimation with skeleton overlay.

Usage:
    python tests/test_pose.py

Controls:
    q - Quit (click window first)
"""

import cv2
import time
import sys

sys.path.insert(0, '.')

from vision_processor.pose import PoseEstimator


def main():
    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Initializing pose estimator...")
    pose_estimator = PoseEstimator(model_complexity=0)

    print("Running! Press 'q' to quit (click window first)\n")

    # FPS tracking
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    valid_frames = 0
    total_frames = 0

    frame_count = 0
    results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_count += 1

        # Process pose every 2 frames for better performance
        if frame_count % 2 == 0:
            results = pose_estimator.estimate(frame)
            if results and results.pose_landmarks:
                valid_frames += 1

        # Draw skeleton (uses last results)
        if results:
            frame = pose_estimator.draw_skeleton(frame, results)

        # Calculate FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # Draw stats
        detection_rate = (valid_frames / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Detection: {detection_rate:.0f}%", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Status indicator
        has_pose = results and results.pose_landmarks
        status = "POSE DETECTED" if has_pose else "NO POSE"
        color = (0, 255, 0) if has_pose else (0, 0, 255)
        cv2.putText(frame, status, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Cuerpo Sonoro - Pose Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    pose_estimator.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nResults:")
    print(f"  Final FPS: {current_fps:.1f}")
    print(f"  Detection rate: {detection_rate:.1f}%")
    print(f"  Valid frames: {valid_frames}/{total_frames}")


if __name__ == "__main__":
    main()
