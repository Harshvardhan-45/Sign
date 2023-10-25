#Imports

import os
import numpy as np
import keras_tuner
from tensorflow import keras

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,LeakyReLU
# from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score,classification_report

#-------Global Variables-------#

DATA_PATH = os.path.join('HandsData')     # Path to load data
num_of_videos = 100                  # 100 videos per action
num_of_frames = 30                # One Sign in 30 frames of video 

#-----Preprocessing ----------#

#signs = np.array(['Talk'])          # All Sign Action
signs = np.array(['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I','J', 'K', 'L','M', 'N', 'O','P', 'Q', 'R','S', 'T', 'U','V', 'W', 'X', 'Y', 'Z', 'Air', 'Delete', 'Fine', 'Go', 'Good', 'Hearing', 'My', 'Name', 'Person', 'Separate', 'Stay', 'Study', 'Talk', 'Time', 'Us', 'Walk', 'What', 'Wrong', 'You', 'Your'])

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

def build_model(hp):
  model = Sequential()

  model.add(LSTM(hp.Int('1stLunits', min_value=64, max_value=512, step=32),             #First layer
                 return_sequences=True, 
                 activation=hp.Choice('1stactivation',values=['relu','tanh','selu','leaky_relu']),
                 input_shape=(30,126))) 

  for i in range(hp.Int('lstm_layers',min_value=1, max_value=5)):
    model.add(
       LSTM(hp.Int('Lunits'+str(i), min_value=32, max_value=512, step=32), 
            return_sequences=True, 
            activation=hp.Choice('Lactivation' + str(i),values=['relu','tanh','selu','leaky_relu'])
            )
        )

  model.add(LSTM(hp.Int('MLunits', min_value=32, max_value=512, step=32),             #Middle layer
                 return_sequences=False, 
                 activation=hp.Choice('MLactivation',values=['relu','tanh','selu','leaky_relu'])
                )) 

  for i in range(hp.Int('dense_layers',min_value=1, max_value=5)):
    model.add(
       Dense(hp.Int('Dunits'+str(i), min_value=32, max_value=512, step=32), 
            activation=hp.Choice('Dactivation' + str(i),values=['relu','tanh','selu','leaky_relu'])
            )
        )

  model.add(Dense(signs.shape[0], activation='softmax'))                    #Output Layer

  opt = hp.Choice('optimizer', values=['rmsprop', 'adam', 'nadam', 'adadelta'])
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])  
   
  return model

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=15)

tuner.search(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
# best_model = tuner.get_best_models()[0]
 
print(tuner.get_best_hyperparameters()[0].values) 