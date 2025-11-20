import cv2
import mediapipe as mp
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def count_fingers(hand_landmarks, handedness_str=None, img_w=1, img_h=1):
    lm = hand_landmarks.landmark
    fingers_open = [False]*5

    try:
        if handedness_str is None:
            thumb_is_open = lm[4].x < lm[3].x
        else:
            if handedness_str.lower().startswith("right"):
                thumb_is_open = lm[4].x < lm[3].x
            else:
                thumb_is_open = lm[4].x > lm[3].x
        fingers_open[0] = bool(thumb_is_open)
    except Exception:
        fingers_open[0] = False

    tips_vs_pips = [(8,6), (12,10), (16,14), (20,18)]
    for i, (tip, pip) in enumerate(tips_vs_pips, start=1):
        try:
            fingers_open[i] = lm[tip].y < lm[pip].y
        except Exception:
            fingers_open[i] = False

    return sum(fingers_open), fingers_open

def main():
    pass

if __name__ == "__main__":
    pass