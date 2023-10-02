import cv2
import mediapipe as mp
import numpy as np
import uuid
import os

# Import required modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# Define data paths and actions
DATA_PATH = os.path.join('Landmark_Data')
actions = np.array(['okay', 'peace', 'thumbsUp', 'thumbsDown', 'salute', 'spiderman'])

# Define the number of sequences and sequence length
no_sequences = 40  # 40 clips
sequence_length = 30  # 30 frames per clip

# Function to extract keypoints from results
def extract_keypoints(results):
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([right_hand, left_hand])

# Create directories for data storage
for action in actions:
    for sequence in range(no_sequences):
        try:
            sequence_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(sequence_path)
        except:
            pass

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Initialize holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for snapshot in range(sequence_length):
                # Capture frame from webcam
                ret, frame = cap.read()

                # Convert frame to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame with holistic model
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw hand landmarks on the frame
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(50, 22, 255), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(250, 44, 0), thickness=2, circle_radius=2))

                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(50, 22, 255), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(250, 44, 0), thickness=2, circle_radius=2))

                # Display information on the frame
                if snapshot == 0:
                    cv2.putText(image, 'Starting Collection of data', (120, 200), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collect data for {} clip {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Hand Data collection', image)

                    cv2.waitKey(1500)
                else:
                    cv2.putText(image, 'Collect data for {} clip {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Hand Data collection', image)

                # Extract keypoints and save data
                keypointData = extract_keypoints(results)
                savePath = os.path.join(DATA_PATH, action, str(sequence), str(snapshot))
                np.save(savePath, keypointData)

                # Exit if 'q' key is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
