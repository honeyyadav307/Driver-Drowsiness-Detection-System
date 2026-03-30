import cv2
import mediapipe as mp
import pygame
import os

# ================== SOUND SETUP ==================
pygame.mixer.init()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(BASE_DIR, "alarm.mp3")

if not os.path.exists(sound_path):
    print(" alarm.mp3 NOT FOUND in folder")
    print("Files present:", os.listdir(BASE_DIR))
    exit()

pygame.mixer.music.load(sound_path)

# ================== MEDIAPIPE ==================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ================== CAMERA ==================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(" Camera not opening")
    exit()
else:
    print(" Camera opened")

# ================== VARIABLES ==================
COUNTER = 0
THRESHOLD = 15
ALARM_ON = False

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()

    if not ret:
        print(" Frame not received")
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Left eye
            left_top = face_landmarks.landmark[159].y
            left_bottom = face_landmarks.landmark[145].y

            # Right eye
            right_top = face_landmarks.landmark[386].y
            right_bottom = face_landmarks.landmark[374].y

            # Eye openness
            left_eye_open = abs(left_top - left_bottom)
            right_eye_open = abs(right_top - right_bottom)

            eye_ratio = (left_eye_open + right_eye_open) / 2

            # Drowsiness logic
            if eye_ratio < 0.015:
                COUNTER += 1

                if COUNTER > THRESHOLD:
                    if not ALARM_ON:
                        pygame.mixer.music.play(-1)
                        ALARM_ON = True

                    cv2.putText(frame, "DROWSY ALERT!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                COUNTER = 0
                if ALARM_ON:
                    pygame.mixer.music.stop()
                    ALARM_ON = False

            # Display eye ratio
            cv2.putText(frame, f"Eye Ratio: {eye_ratio:.3f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show window
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()