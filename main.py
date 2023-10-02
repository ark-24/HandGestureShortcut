import time
import webbrowser
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import subprocess

# Function to extract keypoints from the hand landmarks
def extract_keypoints(results):
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([right_hand, left_hand])

# Initialize variables
global a
a = 0
global check
check = True
cap = cv2.VideoCapture(0)

model = load_model('actions.h5')
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
global sequence
global sentence
threshold = 0.9
actions = np.array(['okay', 'peace', 'thumbsUp','thumbsDown', 'salute', 'spiderman'])
global actionTrack

# Function to switch between main menu and processing
def loop():
    global a
    if a == 0:
        main_screen()
    elif a == 1:
        processing()

# Function to display the main menu
def main_screen():
    global a

    _, frame = cap.read()

    frame_height, frame_width, _ = frame.shape

    # Load and resize background image
    background_image = cv2.imread('backgroundImage.jpg')
    background_image = cv2.resize(background_image, (frame_width, frame_height))

    # Create a new image for text overlay
    background = background_image.copy()

    cv2.putText(background, "Welcome to the Hand Gesture Shortcut Program!", (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.82, (0, 0, 255), 2, cv2.LINE_AA)

    # Display menu options
    cv2.putText(background, "Press s to start or press q to exit", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(background, "Okay to open Google", (10, 280), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(background, "Thumbs Up to open Calculator", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(background, "Peace sign to open Notepad", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(background, "Salute to open Volume Settings", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(background, "Spiderman pose to open Facebook", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)

    # Display the background image with text
    cv2.imshow("Output", background)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
    elif cv2.waitKey(1) == ord('s'):
        a = 1

# Function to process hand gestures and open applications
def processing():
    global a
    global check
    global sequence
    global sentence
    global actionTrack
    actionTrack = []
    sentence = []
    sequence = []
    check = True

    # Initialize holistic model for hand tracking
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            frame_height, frame_width, _ = frame.shape

            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Rendering hand landmarks
            if results:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(50, 22, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 0), thickness=2, circle_radius=2))
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(50, 22, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 0), thickness=2, circle_radius=2))

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                actionTrack.append(actions[np.argmax(res)])

                if len(actionTrack) > 25 and len(set(actionTrack[-25:])) == 1:
                    selectedAction = actionTrack[-1]
                    actionTrack = []

                    # Handle different actions
                    match selectedAction:
                        case "okay":
                            selectedAction = ''
                            webbrowser.open('www.google.com')
                            selectedAction = ''
                            # time.sleep(2)
                            cv2.putText(image, "Is this your desired program (press Y or N):", (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Hand Tracking', image)
                            check = True
                            while check == True:
                                temp = cv2.waitKey(0)
                                if temp == ord('y'):
                                    a = 0
                                    cv2.waitKey(1000)
                                    check = False
                                    # time.sleep(1)
                                elif temp == ord('n'):
                                    # time.sleep(2)
                                    CREATE_NO_WINDOW = 0x08000000
                                    subprocess.call('taskkill /F /IM CalculatorApp.exe', creationflags=CREATE_NO_WINDOW)
                                    check = False

                        case "spiderman":
                            selectedAction = ''
                            webbrowser.open('www.facebook.com')
                            selectedAction = ''
                            # time.sleep(2)
                            cv2.putText(image, "Is this your desired program (press Y or N):", (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Hand Tracking', image)
                            check = True

                            while check == True:
                                temp = cv2.waitKey(0)
                                if temp == ord('y'):
                                    a = 0
                                    cv2.waitKey(1000)
                                    check = False
                                elif temp == ord('n'):
                                    check = False
                                    # time.sleep(1)

                        case "peace":
                            selectedAction = ''
                            p = subprocess.Popen('C:\\Windows\\System32\\notepad.exe')
                            cv2.putText(image, "Is this your desired program (press Y or N):", (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Hand Tracking', image)
                            check = True

                            while check == True:
                                temp = cv2.waitKey(0)
                                if temp == ord('y'):
                                    a = 0
                                    cv2.waitKey(1000)
                                    check = False
                                    # time.sleep(1)
                                elif temp == ord('n'):
                                    CREATE_NO_WINDOW = 0x08000000
                                    subprocess.call('taskkill /F /IM Notepad.exe', creationflags=CREATE_NO_WINDOW)
                                    check = False
                                    # time.sleep(2)

                        case "thumbsUp":
                            selectedAction = ''
                            p = subprocess.Popen('C:\\Windows\\System32\\calc.exe')
                            cv2.putText(image, "Is this your desired program (press Y or N):", (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Hand Tracking', image)
                            check = True

                            while check == True:
                                temp = cv2.waitKey(0)
                                if temp == ord('y'):
                                    a = 0
                                    cv2.waitKey(1000)
                                    check = False
                                    # time.sleep(1)
                                elif temp == ord('n'):
                                    # time.sleep(2)
                                    CREATE_NO_WINDOW = 0x08000000
                                    subprocess.call('taskkill /F /IM CalculatorApp.exe', creationflags=CREATE_NO_WINDOW)
                                    check = False

                        case "salute":
                            selectedAction = ''
                            p = subprocess.Popen('C:\\Windows\\System32\\SndVol.exe')
                            cv2.putText(image, "Is this your desired program (press Y or N):", (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Hand Tracking', image)
                            check = True

                            while check == True:
                                temp = cv2.waitKey(0)
                                if temp == ord('y'):
                                    a = 0
                                    cv2.waitKey(1000)
                                    check = False
                                    time.sleep(1)
                                elif temp == ord('n'):
                                    # time.sleep(2)
                                    CREATE_NO_WINDOW = 0x08000000
                                    subprocess.call('taskkill /F /IM SndVol.exe', creationflags=CREATE_NO_WINDOW)
                                    check = False
                                    # time.sleep(1)

                        case _:
                            selectedAction = ''

            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()

# Run the main loop
while True:
    loop()
