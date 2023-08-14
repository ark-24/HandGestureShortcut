from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import os

DATA_PATH = os.path.join('Landmark_Data')
log_dir = os.path.join('Logs')
tb_callBack = TensorBoard(log_dir=log_dir)

actions = np.array(['okay', 'peace', 'thumbsUp', 'thumbsDown', 'salute', 'spiderman' ])
label_map = {label:num for num,label in enumerate(actions)}  
no_seq,seq_length = 30, 30
sequences, labels =[], []
for action in actions:
    for seq in range(no_seq):
        window =[]
        for clip in range(seq_length):
            res = np.load(os.path.join(DATA_PATH,action, str(seq), "{}.npy".format(clip)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
x= np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.05) 

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32 , activation='relu'))
model.add(Dense(actions.shape[0] , activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=250, callbacks=[tb_callBack])
model.save('actions.h5')





