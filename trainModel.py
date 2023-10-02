# Import required libraries and modules
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os

# Define paths and directories
DATA_PATH = os.path.join('Landmark_Data')
log_dir = os.path.join('Logs')
tb_callBack = TensorBoard(log_dir=log_dir)

# Define possible actions (gestures)
actions = np.array(['okay', 'peace', 'thumbsUp', 'thumbsDown', 'salute', 'spiderman' ])

# Create a mapping of actions to numeric labels
label_map = {label: num for num, label in enumerate(actions)}

# Define the number of sequences and sequence length
no_seq, seq_length = 30, 30

# Initialize lists to store sequences and labels
sequences, labels = [], []

# Loop through each action and sequence to collect data
for action in actions:
    for seq in range(no_seq):
        window = []
        for clip in range(seq_length):
            # Load the saved keypoints data for each frame
            res = np.load(os.path.join(DATA_PATH, action, str(seq), "{}.npy".format(clip)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert lists to numpy arrays
x = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05) 

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model with training data
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callBack])

# Save the trained model
model.save('actions.h5')
