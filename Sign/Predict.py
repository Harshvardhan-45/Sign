#Imports

import cv2
from tensorflow.keras.models import load_model
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
import mediapipe as mp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#TTS Module

from gtts import gTTS
from playsound import playsound

def T2S(str):
    tts = gTTS(str)
    tts.save('audio.mp3')
    playsound('audio.mp3')

#Model Setup

signs = np.array(['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I','J', 'K', 'L','M', 'N', 'O','P', 'Q', 'R','S', 'T', 'U','V', 'W', 'X', 'Y', 'Z', 'Air', 'Delete', 'Fine', 'Go', 'Good', 'Hearing', 'My', 'Name', 'Person', 'Separate', 'Stay', 'Study', 'Talk', 'Time', 'Us', 'Walk', 'What', 'Wrong', 'You', 'Your'])
num_of_videos = 100
num_of_frames = 30

model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))     #change 128 to 64
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])   #kullback_leibler_divergence, sparse_categorical_crossentropy

model.load_weights('sign.h5')


#-------Extraing the Key Point-------#

mp_holistic = mp.solutions.holistic                 # Holistic model
mp_drawing = mp.solutions.drawing_utils             # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR to RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB to BGR
    return image, results
    

def draw_landmarks(image, results):
    """mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(204,228,248), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(188,214,244), thickness=1, circle_radius=1)
                             ) """
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(178,176,0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(206,204,82), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,52,21), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(148,114,68), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(31,23,127), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(45,33,182), thickness=2, circle_radius=2)
                             ) 


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

#-----------Predicting-----------#

sequence = []               #To collect last 30 frames
sentence = []               #To store predicted text
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        
        draw_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results) 
        sequence.append(keypoints)
        sequence = sequence[-30:]               #last 30 frame keypoints
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]   #predicting last sequence
            print(signs[np.argmax(res)])
            predictions.append(np.argmax(res))
            
        # Visualization logic
            if np.unique(predictions[-10:])[0]==np.argmax(res):  #last 10 frames must be same word
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if signs[np.argmax(res)] != sentence[-1]:   #if last word is not same as predicted
                            sentence.append(signs[np.argmax(res)])
                            T2S(signs[np.argmax(res)])
                    else:
                        sentence.append(signs[np.argmax(res)])         #if 1st word
                        T2S(signs[np.argmax(res)])



            if len(sentence) > 5:                   #if sentence greater than 5, display last 5
                sentence = sentence[-5:]
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
