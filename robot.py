# robot.py
import threading
import time
import serial
import sys

import cv2
import kamera  # gunakan modul kamera.py yang sudah ada (memanfaatkan mediapipe)
import speech_recognition as sr

# ---------------- USER CONFIG ----------------
SERIAL_PORT = "COM7"   # <-- Ganti sesuai port Arduino (contoh "COM3" di Windows)
SERIAL_BAUDRATE = 9600

DIST_THRESHOLD_CM = 30         # jarak untuk berhenti & servo
CENTER_TOLERANCE = 0.08       # toleransi center (normalized, 0..0.5)
SEARCH_REPEAT_DELAY = 1.0     # delay antar langkah search
# ---------------------------------------------

# shared state
state_lock = threading.Lock()
audio_trigger = threading.Event()   # set ketika kata "sampah" terdeteksi
hand_present = False
hand_center_x = 0.5   # normalized [0..1], 0.5 = center
fingers_open_count = 0
current_distance = 9999   # cm
running = True

# Serial port (initialized later)
ser = None

def serial_writer_loop():
    # optional: keep alive or debug - empty now
    pass

def serial_reader_loop():
    global current_distance, ser, running
    buf = ""
    while running:
        try:
            if ser is None:
                time.sleep(0.1)
                continue
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue
            # expecting lines like: DIST:123
            if line.startswith("DIST:"):
                try:
                    val = int(line.split(":",1)[1])
                    with state_lock:
                        current_distance = val
                except:
                    pass
            else:
                # can print ACKs for debugging
                print("[ARDUINO]", line)
        except Exception as e:
            print("Serial read error:", e)
            time.sleep(0.5)

def audio_listen_loop():
    """Thread: listen continuously for keyword 'sampah'."""
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("Audio thread: kalibrasi mikrofon, jangan bergerak... (1s)")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
    print("Audio ready. Mulai mendengar kata 'sampah'...")
    while running:
        try:
            with mic as source:
                audio = r.listen(source, phrase_time_limit=3)
            text = ""
            try:
                text = r.recognize_google(audio, language="id-ID")
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                print("Speech API error (offline?)")
                continue

            if "sampah" in text.lower():
                print("Audio trigger: 'sampah' terdeteksi.")
                audio_trigger.set()
        except Exception as e:
            print("Audio thread error:", e)
            time.sleep(0.5)

def camera_thread_loop():
    """Thread: capture frames and update hand_present, hand_center_x, fingers_open_count."""
    global hand_present, hand_center_x, fingers_open_count, running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Kamera tidak terbuka")
        running = False
        return

    prev_time = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame failed")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = kamera.hands_model.process(img_rgb)

        local_hand_present = False
        local_center_x = 0.5
        local_fingers = 0

        if results.multi_hand_landmarks:
            # pick the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = None
            if results.multi_handedness and len(results.multi_handedness) > 0:
                handedness_label = results.multi_handedness[0].classification[0].label

            cnt, fingers_open_list = kamera.count_fingers(hand_landmarks, handedness_label, w, h)
            local_fingers = cnt

            xs = [lm.x for lm in hand_landmarks.landmark]
            x_min, x_max = min(xs), max(xs)
            cx_norm = (x_min + x_max) / 2.0  # normalized center x in 0..1 relative to frame width
            local_center_x = cx_norm
            local_hand_present = True

            # debug drawing (optional)
            kamera.mp_drawing.draw_landmarks(frame, hand_landmarks, kamera.mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (int(cx_norm*w), int(h*0.2)), 6, (0,255,0), -1)
            cv2.putText(frame, f"Fingers:{cnt}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        else:
            cv2.putText(frame, "Tidak ada tangan", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255),2)

        # publish to shared variables
        with state_lock:
            hand_present = local_hand_present
            hand_center_x = local_center_x
            fingers_open_count = local_fingers

        # show frame (for debug)
        cv2.putText(frame, f"Dist: {current_distance} cm", (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.imshow("Robot Camera (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

def send_cmd(cmd):
    """Send command to Arduino with newline."""
    global ser
    if ser is None:
        print("Serial not connected, cmd:", cmd)
        return
    try:
        line = (cmd + "\n").encode()
        ser.write(line)
    except Exception as e:
        print("Failed to send cmd:", e)

def main_loop():
    """
    State machine implementing rules:
    1) On audio trigger -> TURN_RIGHT continuous until camera finds open hand.
    2) After camera sees an open hand -> approach the hand by moving forward and steering toward center.
    3) Robot stops only if ultrasonic <= 30 cm AND camera still sees hand centered.
    4) When stopped and both conditions met -> ask Arduino to run SERVO, then wait until next audio trigger.
    5) If ultrasonic detects something (<30cm) but hand NOT detected -> run search: rotate right -> forward -> left, keep searching for hand, then resume rule 3 when found.
    """
    global audio_trigger, hand_present, hand_center_x, fingers_open_count, current_distance, running

    print("Main loop started.")
    state = "IDLE"

    while running:
        # check serial distance quickly
        with state_lock:
            hp = hand_present
            hx = hand_center_x
            fingers = fingers_open_count
            dist = current_distance

        if state == "IDLE":
            if audio_trigger.is_set():
                audio_trigger.clear()
                print("Audio trigger received -> send TURN_RIGHT")
                send_cmd("TURN_RIGHT")
                state = "TURNING_RIGHT"
            else:
                time.sleep(0.05)

        elif state == "TURNING_RIGHT":
            # keep turning right until camera finds an open hand (all fingers open OR fingers >=4 considered open)
            if hp and fingers >= 4:
                print("Hand with many fingers detected -> start tracking")
                send_cmd("FORWARD")
                state = "TRACKING"
            else:
                # continue turning
                time.sleep(0.05)

        elif state == "TRACKING":
            # compute offset from center
            offset = hx - 0.5  # negative: left, positive: right
            abs_offset = abs(offset)

            # if distance < threshold AND hand centered => STOP & SERVO
            if dist > 0 and dist <= DIST_THRESHOLD_CM and abs_offset <= CENTER_TOLERANCE and hp:
                print(f"Close object detected ({dist} cm) and hand centered -> STOP and SERVO")
                send_cmd("STOP")
                # request servo action
                send_cmd("SERVO")
                state = "WAIT_AFTER_SERVO"
                # after servo, robot must remain stopped until next audio 'sampah'
            else:
                # if something ahead but no hand -> go to SEARCH mode (rule 5)
                if dist > 0 and dist <= DIST_THRESHOLD_CM and not hp:
                    print("Close obstacle detected but no hand -> SEARCH mode")
                    state = "SEARCHING"
                    continue

                # steering logic: if hand offset beyond tolerance, do small turn, else forward
                if abs_offset <= CENTER_TOLERANCE:
                    # roughly centered
                    send_cmd("FORWARD")
                else:
                    # steer slightly: use short turn pulses to re-center
                    if offset > 0:
                        # hand to right -> turn right briefly
                        send_cmd("TURN_RIGHT")
                        time.sleep(0.12)
                        send_cmd("FORWARD")
                    else:
                        # hand to left -> turn left briefly
                        send_cmd("TURN_LEFT")
                        time.sleep(0.12)
                        send_cmd("FORWARD")
                time.sleep(0.05)

        elif state == "SEARCHING":
            # perform search routine: right -> forward -> left, scanning for hand
            print("Executing search pattern step: TURN RIGHT short")
            send_cmd("TURN_RIGHT")
            time.sleep(0.7)
            send_cmd("FORWARD")
            time.sleep(0.4)
            send_cmd("TURN_LEFT")
            time.sleep(0.7)
            send_cmd("STOP")
            # after each search pattern, check if hand found
            with state_lock:
                hp = hand_present
                fingers = fingers_open_count
            if hp and fingers >= 4:
                print("Hand found during search -> TRACKING")
                send_cmd("FORWARD")
                state = "TRACKING"
            else:
                print("No hand found, repeat search after short pause")
                time.sleep(0.5)

        elif state == "WAIT_AFTER_SERVO":
            # Arduino already ran servo; remain stopped until next "sampah"
            with state_lock:
                hp = hand_present
            print("Stopped after servo. Waiting for next audio trigger.")
            # keep motors stopped
            send_cmd("STOP")
            # wait for audio trigger
            # Block but allow running flag to break
            while running and not audio_trigger.is_set():
                time.sleep(0.2)
            if not running:
                break
            # new audio, reset and start again
            audio_trigger.clear()
            print("New audio trigger after servo -> TURN_RIGHT")
            send_cmd("TURN_RIGHT")
            state = "TURNING_RIGHT"

        else:
            # fallback
            send_cmd("STOP")
            state = "IDLE"
            time.sleep(0.1)

    # cleanup on exit
    send_cmd("STOP")
    print("Main loop exiting.")

if __name__ == "__main__":
    # open serial
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        time.sleep(2)  # wait for Arduino reset
        print("Connected to serial", SERIAL_PORT)
    except Exception as e:
        print("Failed to open serial port:", e)
        ser = None

    # threads
    threads = []
    t_serial_reader = threading.Thread(target=serial_reader_loop, daemon=True)
    threads.append(t_serial_reader)
    t_serial_reader.start()

    t_camera = threading.Thread(target=camera_thread_loop, daemon=True)
    threads.append(t_camera)
    t_camera.start()

    t_audio = threading.Thread(target=audio_listen_loop, daemon=True)
    threads.append(t_audio)
    t_audio.start()

    try:
        main_loop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt -> shutting down")
    finally:
        running = False
        time.sleep(0.5)
        if ser:
            ser.close()
        print("Program terminated.")
        sys.exit(0)
