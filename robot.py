import threading
import time
import serial
import sys

import cv2
import kamera  # gunakan modul kamera.py yang sudah ada (memanfaatkan mediapipe)
import speech_recognition as sr

# ---------------- USER CONFIG ----------------
SERIAL_PORT = "/dev/cu.usbmodem11301"   # <-- Ganti sesuai port Arduino (contoh "COM3" di Windows)
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
    print("[SERIAL] Reader thread started")
    while running:
        if ser is None:
            print("[SERIAL] Serial port not initialized, waiting...")
            time.sleep(1)
            continue
        try:
            if ser.in_waiting > 0:  # Check if data available
                line = ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue
                # expecting lines like: DIST:123
                if line.startswith("DIST:"):
                    try:
                        val = int(line.split(":",1)[1])
                        with state_lock:
                            current_distance = val
                        print(f"[ARDUINO-DIST] Jarak: {val} cm")
                    except ValueError as e:
                        print(f"[SERIAL] Parse error: {e}")
                else:
                    # can print ACKs for debugging
                    print("[ARDUINO]", line)
            else:
                time.sleep(0.01)  # Small delay to prevent CPU spinning
        except serial.SerialException as e:
            print(f"[SERIAL] Connection error: {e}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[SERIAL] Read error: {e}")
            time.sleep(0.5)

def audio_listen_loop():
    """Thread: listen continuously for keyword 'sampah'."""
    global running
    print("[AUDIO] Thread starting...")
    r = sr.Recognizer()
    
    try:
        mic = sr.Microphone()
        print("[AUDIO] Kalibrasi mikrofon, jangan bergerak... (1s)")
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=1)
        print("[AUDIO] Ready. Mulai mendengar kata 'sampah'...")
    except Exception as e:
        print(f"[AUDIO] ERROR: Tidak dapat mengakses mikrofon: {e}")
        print("[AUDIO] Thread akan dinonaktifkan. Gunakan keyboard untuk testing.")
        return
    
    while running:
        try:
            with mic as source:
                audio = r.listen(source, phrase_time_limit=3)
            text = ""
            try:
                text = r.recognize_google(audio, language="id-ID")
                print(f"[AUDIO] Terdengar: '{text}'")
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[AUDIO] Speech API error: {e}")
                time.sleep(1)
                continue

            if "sampah" in text.lower():
                print("[AUDIO] *** TRIGGER: 'sampah' terdeteksi! ***")
                audio_trigger.set()
        except Exception as e:
            print(f"[AUDIO] Thread error: {e}")
            time.sleep(0.5)

def camera_thread_loop():
    """Thread: capture frames and update hand_present, hand_center_x, fingers_open_count."""
    global hand_present, hand_center_x, fingers_open_count, running
    print("[CAMERA] Thread starting...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CAMERA] ERROR: Kamera tidak terbuka")
        running = False
        return

    print("[CAMERA] Kamera berhasil dibuka")
    
    while running:
        ret, frame = cap.read()
        if not ret:
            print("[CAMERA] Frame capture failed")
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

            # debug drawing
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
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[CAMERA] ESC pressed, shutting down...")
            running = False
            break
        elif key == ord('s'):  # Press 's' to simulate audio trigger
            print("[CAMERA] Manual trigger with 's' key")
            audio_trigger.set()

    cap.release()
    cv2.destroyAllWindows()
    print("[CAMERA] Thread stopped")

def send_cmd(cmd):
    """Send command to Arduino with newline."""
    global ser
    if ser is None or not ser.is_open:
        print(f"[CMD] Serial not connected, cmd: {cmd}")
        return
    try:
        line = (cmd + "\n").encode()
        ser.write(line)
        print(f"[CMD] Sent: {cmd}")
    except Exception as e:
        print(f"[CMD] Failed to send '{cmd}': {e}")

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

    print("[MAIN] Main loop started.")
    print("[MAIN] Tekan 's' di jendela kamera untuk simulasi trigger audio")
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
                print("[STATE] IDLE -> TURNING_RIGHT (audio trigger)")
                send_cmd("TURN_RIGHT")
                state = "TURNING_RIGHT"
            else:
                time.sleep(0.05)

        elif state == "TURNING_RIGHT":
            # keep turning right until camera finds an open hand (all fingers open OR fingers >=4 considered open)
            if hp and fingers >= 4:
                print(f"[STATE] Hand detected with {fingers} fingers -> TRACKING")
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
                print(f"[STATE] Target reached! Dist={dist}cm, centered -> STOP & SERVO")
                send_cmd("STOP")
                time.sleep(0.2)
                send_cmd("SERVO")
                state = "WAIT_AFTER_SERVO"
            else:
                # if something ahead but no hand -> go to SEARCH mode (rule 5)
                if dist > 0 and dist <= DIST_THRESHOLD_CM and not hp:
                    print(f"[STATE] Obstacle at {dist}cm but no hand -> SEARCHING")
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
            print("[STATE] Executing search pattern...")
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
                print("[STATE] Hand found during search -> TRACKING")
                send_cmd("FORWARD")
                state = "TRACKING"
            else:
                print("[STATE] No hand found, repeat search...")
                time.sleep(0.5)

        elif state == "WAIT_AFTER_SERVO":
            print("[STATE] Servo complete. Waiting for next audio trigger...")
            send_cmd("STOP")
            
            # wait for audio trigger
            while running and not audio_trigger.is_set():
                time.sleep(0.2)
            
            if not running:
                break
            
            # new audio, reset and start again
            audio_trigger.clear()
            print("[STATE] New audio trigger -> TURN_RIGHT")
            send_cmd("TURN_RIGHT")
            state = "TURNING_RIGHT"

        else:
            # fallback
            print(f"[STATE] Unknown state '{state}' -> IDLE")
            send_cmd("STOP")
            state = "IDLE"
            time.sleep(0.1)

    # cleanup on exit
    send_cmd("STOP")
    print("[MAIN] Main loop exiting.")

if __name__ == "__main__":
    print("="*50)
    print("ROBOT CONTROL SYSTEM")
    print("="*50)
    
    # open serial
    try:
        print(f"[SERIAL] Connecting to {SERIAL_PORT} at {SERIAL_BAUDRATE} baud...")
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        time.sleep(2)  # wait for Arduino reset
        print(f"[SERIAL] Connected successfully!")
    except Exception as e:
        print(f"[SERIAL] Failed to open serial port: {e}")
        print("[SERIAL] Program akan tetap berjalan tanpa koneksi serial (debug mode)")
        ser = None

    # Check kamera.py availability
    try:
        print(f"[INIT] Checking kamera module...")
        print(f"[INIT] - hands_model: {hasattr(kamera, 'hands_model')}")
        print(f"[INIT] - count_fingers: {hasattr(kamera, 'count_fingers')}")
        print(f"[INIT] - mp_drawing: {hasattr(kamera, 'mp_drawing')}")
        print(f"[INIT] - mp_hands: {hasattr(kamera, 'mp_hands')}")
    except Exception as e:
        print(f"[INIT] ERROR: Problem with kamera module: {e}")
        sys.exit(1)

    # threads
    threads = []
    
    t_serial_reader = threading.Thread(target=serial_reader_loop, daemon=True, name="SerialReader")
    threads.append(t_serial_reader)
    t_serial_reader.start()

    t_camera = threading.Thread(target=camera_thread_loop, daemon=True, name="Camera")
    threads.append(t_camera)
    t_camera.start()

    t_audio = threading.Thread(target=audio_listen_loop, daemon=True, name="Audio")
    threads.append(t_audio)
    t_audio.start()

    print("[INIT] All threads started")
    print("[INIT] Controls:")
    print("  - Say 'sampah' to trigger robot")
    print("  - Press 's' in camera window to manually trigger")
    print("  - Press ESC in camera window to quit")
    print("="*50)

    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt -> shutting down")
    finally:
        running = False
        time.sleep(0.5)
        if ser and ser.is_open:
            send_cmd("STOP")
            ser.close()
            print("[SERIAL] Connection closed")
        print("[MAIN] Program terminated.")
        sys.exit(0)