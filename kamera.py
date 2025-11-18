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
            if lm[4].x < lm[3].x:
                thumb_is_open = True
            else:
                thumb_is_open = False
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam (index 0). Coba periksa koneksi kamera atau ganti index.")
        return

    prev_time = 0
    fps_deque = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: frame tidak terbaca dari kamera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands_model.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = None
                if results.multi_handedness and len(results.multi_handedness) > hand_idx:
                    handedness_label = results.multi_handedness[hand_idx].classification[0].label  # 'Left' or 'Right'

                count, fingers_open_list = count_fingers(hand_landmarks, handedness_label, w, h)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))

                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs)*w)-10, int(max(xs)*w)+10
                y_min, y_max = int(min(ys)*h)-10, int(max(ys)*h)+10
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                label = f"Tangan terdeteksi ({handedness_label if handedness_label else 'Unknown'})"
                cv2.putText(frame, label, (x_min, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Jari terbuka: {count}", (x_min, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

                names = ["Thumb","Index","Middle","Ring","Pinky"]
                info = ", ".join(f"{n}:{'1' if o else '0'}" for n,o in zip(names, fingers_open_list))
                cv2.putText(frame, info, (x_min, y_max+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

        else:
            cv2.putText(frame, "Tidak ada tangan terdeteksi", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,180,255), 2)

        curr_time = time.time()
        fps = 1/(curr_time - prev_time) if prev_time != 0 else 0.0
        prev_time = curr_time
        fps_deque.append(fps)
        avg_fps = sum(fps_deque)/len(fps_deque)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20,frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Hand Detector - tekan 'q' untuk keluar", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()