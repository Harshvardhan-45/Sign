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

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))     #change 128 to 64
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
 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,52,21), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(148,114,68), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(31,23,127), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(45,33,182), thickness=2, circle_radius=2)
                             ) 


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


#-----------Predicting-----------#

sequence = []               #To collect last 30 frames
sentence = []               #To store predicted text
predictions = []
threshold = 0.7
speech=''
space= ' '
speakers_speech=''
DeleteCount=0
total=0

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        
        draw_landmarks(image, results)
        
        keypoints = extract_keypoints(results) 
        sequence.append(keypoints)
        sequence = sequence[-30:]               #last 30 frame keypoints

        for x in sequence:
            for y in x:
                total=y+total
        
        if len(sequence) == 30 and total !=0:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]   #---------predicting last sequence----------#
            print(signs[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-9:])[0]==np.argmax(res):  #last 10 predictions must be same word
                if res[np.argmax(res)] > threshold: 
                    predicted_text = signs[np.argmax(res)]

                    if len(sentence)==0:
                        sentence.append('Deleted')

                    if predicted_text == 'Delete' and predicted_text == sentence[-1]:
                        DeleteCount=DeleteCount+1
                        if DeleteCount==2:
                            sentence.pop()

                    if predicted_text != sentence[-1]:   #if last word is not same as predicted

                        if predicted_text == 'Talk' and speech != '':  #will only speak when Talk sign 
                            cv2.putText(image, 'Speaking', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,204,), 4, cv2.LINE_AA)    #Text on Screen
                            cv2.imshow('ISL Detection', image)                    # Display 
                            # T2S(speech)
                            T2S('My Name is Rahul')
                            cv2.waitKey(500)                                   # Wait time 0.5 sec
                            speech=''

                        elif predicted_text == 'Delete' and speech != '':
                            length=len(sentence[-1])
                            cv2.putText(image, 'Deleting', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,204,), 4, cv2.LINE_AA)    #Text on Screen
                            cv2.imshow('ISL Detection', image)  
                            speech=speech.strip()                           #Removing whitespaces, if any
                            speech=speech[:-length]
                            speech=speech.strip()                           
                            sentence.pop()
                            sentence.append(predicted_text)                 # Just to delete last word 
                            DeleteCount=0
                            cv2.waitKey(500)                                   # Wait time 0.5 sec

                        elif predicted_text == 'Hearing':
                            print('Hearing for speaker')
                            cv2.putText(image, 'Hearing', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,204,), 4, cv2.LINE_AA)    #Text on Screen
                            cv2.imshow('ISL Detection', image) 
                            cv2.waitKey(500)                                   # Wait time 0.5 sec
                            # speakers_speech=STT()
                            
                        elif predicted_text == 'Separate' and speech != '':
                            speech=speech.strip()                           
                            speech = speech + space
                            cv2.putText(image, 'Space Added', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,204,), 4, cv2.LINE_AA)    #Text on Screen
                            cv2.imshow('ISL Detection', image) 
                            sentence.append(predicted_text)                    # so that not 2 spaces
                            cv2.waitKey(500)                                   # Wait time 0.5 sec

                        else:
                            if predicted_text!='Delete' and predicted_text!='Talk' and predicted_text!='Hearing' and predicted_text!='Separate':
                                sentence.append(predicted_text)
                                if speech == '':                                        #if 1st word
                                    speech=predicted_text
                                else:
                                    if len(predicted_text) == 1:                        #If alphabets then no space
                                        speech = speech + predicted_text
                                    else:
                                        speech = speech + space + predicted_text

            sequence = sequence[-28:]               #to slow down predicting by losing first few sequence
            # sequence.clear()                                    #To clear sequence and get new 30

        total=0

        cv2.rectangle(image, (0,0), (640, 30), (0, 0, 0), -1)
        cv2.putText(image, speech, (3,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('ISL Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()