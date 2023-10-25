#Imports

import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

#-------Global Variables-------#

DATA_PATH = os.path.join('HandsData')     # Path to load data
num_of_videos = 100                  # 100 videos per action
num_of_frames = 30                # One Sign in 30 frames of video 
signs = np.array(['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I','J', 'K', 'L','M', 'N', 'O','P', 'Q', 'R','S', 'T', 'U','V', 'W', 'X', 'Y', 'Z', 'Air', 'Delete', 'Fine', 'Go', 'Good', 'Hearing', 'My', 'Name', 'Person', 'Separate', 'Stay', 'Study', 'Talk', 'Time', 'Us', 'Walk', 'What', 'Wrong', 'You', 'Your'])

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # 85%train, 15%test


#--------Training-----------#

model = Sequential()
# model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,126)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(signs.shape[0], activation='softmax'))

model.add(LSTM(2048, return_sequences=True, activation='selu', input_shape=(30,126)))
model.add(LSTM(1024, return_sequences=True, activation='tanh'))
model.add(LSTM(768, return_sequences=True, activation='leaky_relu'))

model.add(LSTM(512, return_sequences=False, activation='leaky_relu'))

model.add(Dense(512, activation='leaky_relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='relu'))

model.add(Dense(signs.shape[0], activation='softmax'))



print (model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])   #kullback_leibler_divergence, sparse_categorical_crossentropy

model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=100)

#Predictions#

res = model.predict(X_test)
print(signs[np.argmax(res[9])])
print(signs[np.argmax(y_test[9])])

print(signs[np.argmax(res[20])])
print(signs[np.argmax(y_test[20])])
#Saving Weights#

model.save('sign.h5')

#Evaluation

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))
