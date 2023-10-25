from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import os
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical


signs = np.array(['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I','J', 'K', 'L','M', 'N', 'O','P', 'Q', 'R','S', 'T', 'U','V', 'W', 'X', 'Y', 'Z', 'Air', 'Delete', 'Fine', 'Go', 'Good', 'Hearing', 'My', 'Name', 'Person', 'Separate', 'Stay', 'Study', 'Talk', 'Time', 'Us', 'Walk', 'What', 'Wrong', 'You', 'Your'])
num_of_videos = 100
num_of_frames = 30
DATA_PATH = os.path.join('HandsData')     # Path to load data

#-----Preprocessing ----------#

label_map = {label:num for num, label in enumerate(signs)}  #converting each sign to number
print(label_map)

sequences, labels = [], []                      # X , y
for sign in signs:
    for video_num in range(num_of_videos):
        window = []                             #To combine all frames, (1 frame is of 1662 1D Array), so window is (30, 1662) 2D Array
        for frame_num in range(num_of_frames):
            res = np.load(os.path.join(DATA_PATH, sign, str(video_num), "{}.npy".format(frame_num)))        #loading each frame
            window.append(res)
        sequences.append(window)                # Combining all videos         (3D array of (num_of_signs*num_of_video, num_of_frames, keypoints in each frame))
        labels.append(label_map[sign])          # Corresponding label to the sign 

X = np.array(sequences)
y = to_categorical(labels).astype(int)

#Model Setup

model = Sequential()

model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,126)))     
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])   #kullback_leibler_divergence, sparse_categorical_crossentropy

model.load_weights('sign88.h5')

yhat = model.predict(X)

ytrue = np.argmax(y, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))