import time
import webbrowser
import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
from tensorflow.keras.models import load_model



def extract_keypoints(results):
    left_hand =np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand =np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([right_hand,left_hand])


model = load_model('actions.h5')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic # Holistic model
sequence = []
sentence = []
threshold = 0.9
actions = np.array(['okay', 'peace', 'thumbsUp', 'thumbsDown', 'salute', 'spiderman' ])
actionTrack = []

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #RENDERING
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
            res = model.predict(np.expand_dims(sequence,axis=0))[0]
            actionTrack.append(actions[np.argmax(res)])  
            if len(actionTrack) > 15 and len(set(actionTrack[-15:])) == 1:
                selectedAction = actionTrack[-1]
                actionTrack=[]
                print(selectedAction)
                match selectedAction:
                    case "okay":
                        webbrowser.open('www.google.ca')
                        selectedAction='jjjjjljn'
                        time.sleep(2)

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
