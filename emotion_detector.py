# emotion_detector.py
import cv2
from deepface import DeepFace
import datetime
from collections import Counter

def get_majority_emotion(emotions):
    if not emotions:
        return None
    count = Counter(emotions)
    return count.most_common(1)[0][0]

def analyze_emotions_from_webcam(frames_to_collect=5):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    batch_emotions = []

    while frame_count < frames_to_collect:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=True,
                detector_backend='opencv'
            )

            result = [result] if not isinstance(result, list) else result
            result.sort(key=lambda x: x['region']['w'] * x['region']['h'], reverse=True)
            primary = result[0]
            dominant_emotion = primary['dominant_emotion']
            batch_emotions.append(dominant_emotion)
            frame_count += 1

        except Exception as e:
            print("[Warning] Detection failed:", e)

    cap.release()

    majority_emotion = get_majority_emotion(batch_emotions)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": timestamp,
        "majority_emotion": majority_emotion,
        "individual_emotions": batch_emotions
    }
