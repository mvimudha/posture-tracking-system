# main.py (fixed & improved)
import cv2
import time
import numpy as np
from ultralytics import YOLO
from angles import compute_angles
from classifier import PostureClassifier
from tracker import MultiPersonTracker
from config import WIN, THRESHOLDS, OVERREACH_MIN_SEC, INACTIVITY_SEC, BALANCE_JITTER
import mediapipe as mp

mp_pose_module = mp.solutions.pose

def draw_skeleton(vis, landmarks, w, h):
    """Draws skeleton and joints on a frame."""
    for connection in mp_pose_module.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x1, y1 = int(start[0] * w), int(start[1] * h)
            x2, y2 = int(end[0] * w), int(end[1] * h)
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in landmarks:
        x, y = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    try:
        yolo = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please run: pip install ultralytics")
        return

    tracker = MultiPersonTracker(iou_thresh=0.35, max_age=20)
    per_id_classifier = {}
    custom_labels = {}

    mp_pose = mp_pose_module.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1,
        smooth_landmarks=True
    )

    print("Starting multi-person posture analysis...")
    print("Press 'q' or ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        h, w = frame.shape[:2]
        vis = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.time()

        # YOLO detection
        try:
            yolo_results = yolo(frame, verbose=False)[0]
            detections = []
            for box in yolo_results.boxes:
                if int(box.cls[0]) != 0:  # only keep people
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2])
        except Exception as e:
            print(f"YOLO detection error: {e}")
            continue

        tracks = tracker.update(detections)

        for person_id, (x1, y1, x2, y2) in tracks.items():
            try:
                crop_rgb = rgb[y1:y2, x1:x2]
                if crop_rgb.size == 0:
                    continue

                pose_results = mp_pose.process(crop_rgb)
                if not pose_results.pose_landmarks:
                    continue

                # landmarks normalized in crop frame
                landmarks = [[lm.x, lm.y] for lm in pose_results.pose_landmarks.landmark]
                angle_features = compute_angles(pose_results.pose_landmarks.landmark)

                if angle_features is None:
                    continue  # skip person if core angles not available

                # Combine left/right arm for classifier
                arm_elev = max(
                    [v for v in [angle_features.get("l_arm_elev"), angle_features.get("r_arm_elev")] if v is not None],
                    default=None
                )


                # Build features dict for classifier
                features = {
                    "neck_tilt": angle_features.get("neck_tilt"),
                    "spine_bend": angle_features.get("spine_bend"),
                    "arm_elev": arm_elev,
                    "knee_bend": angle_features.get("knee_bend"),
                    "twist": angle_features.get("twist"),
                    "shoulder_mid": angle_features.get("shoulder_mid"),
                    "hip_mid": angle_features.get("hip_mid"),
                    "l_ankle": angle_features.get("l_ankle"),
                    "r_ankle": angle_features.get("r_ankle")
                }

                # Instantiate classifier per person if not exists
                if person_id not in per_id_classifier:
                    per_id_classifier[person_id] = PostureClassifier()
                    custom_labels[person_id] = f"Worker {person_id}"

                classifier = per_id_classifier[person_id]
                classifier.update(features, now)

                # Get classification and smoothed angles
                decisions, angles = classifier.classify()

                crop_vis = vis[y1:y2, x1:x2]
                draw_skeleton(crop_vis, landmarks, x2 - x1, y2 - y1)

                # Display ID and posture
                person_title = custom_labels.get(person_id, f"ID {person_id}")
                cv2.putText(vis, person_title, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 255), 2)

                # Display posture decisions
                y_text = y1 + 20
                for label, risk in decisions:
                    if risk == "Safe":
                        color = (0, 200, 0)
                    elif risk == "High Injury Risk":
                        color = (0, 0, 255)
                    else:
                        color = (0, 165, 255)
                    cv2.putText(vis, f"{label} [{risk}]", (x1 + 5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_text += 22

                # Display smoothed angles
                angle_text = (
                    f"N:{int(angles.get('neck_tilt') or -1)}  "
                    f"S:{int(angles.get('spine_bend') or -1)}  "
                    f"A:{int(angles.get('arm_elev') or -1)}  "
                    f"K:{int(angles.get('knee_bend') or -1)}  "
                    f"T:{int(angles.get('twist') or -1)}"
                )
                cv2.putText(vis, angle_text, (x1 + 5, min(y_text, y2 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

            except Exception as e:
                print(f"Error processing person {person_id}: {e}")
                continue

        cv2.imshow("Multi-Person Posture Analysis (OSHA-guided)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_pose.close()
    print("Application closed")

if __name__ == "__main__":
    main()
