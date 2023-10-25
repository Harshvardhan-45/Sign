#Imports

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

#-------Global Variables-------#

DATA_PATH = os.path.join('Data')     # Path to store data
signs = np.array(['Talk'])          # Sign Actions to Collect
num_of_videos = 100                  # 100 videos per action
num_of_frames = 30                # One Sign in 30 frames of video 

#-------Extraing the Key Point-------#

mp_holistic = mp.solutions.holistic                 # Holistic model
mp_drawing = mp.solutions.drawing_utils             # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CHANNEL CONVERSION BGR to RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CHANNEL COVERSION RGB to BGR
    return image, results
    

def draw_landmarks(image, results): 
         
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,                  # On pose             
                             mp_drawing.DrawingSpec(color=(178,176,0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(206,204,82), thickness=2, circle_radius=2))      # Connection Color
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,             #On left hand 
                             mp_drawing.DrawingSpec(color=(80,52,21), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(148,114,68), thickness=2, circle_radius=2)) 
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,            #On right hand 
                             mp_drawing.DrawingSpec(color=(31,23,127), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(45,33,182), thickness=2, circle_radius=2)) 
    
    
def extract_keypoints(results):                         # Returning keypoints in np array of size 1662
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)             #add some logic to stop predicting if no hand
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])
            
#------Saving data------#

for sign in signs: 
    for video_num in range(num_of_videos):                #To add more data to existing sign, edit the loop condition
        try: 
            os.makedirs(os.path.join(DATA_PATH, sign, str(video_num)))          # Making Directories 
        except:
            pass
            
cap = cv2.VideoCapture(0)                               #Accessing Webcam 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:     # Mediapipe model 
    
    for sign in signs:                                                  #Loop through signs

        for video_num in range(num_of_videos):                           #Loop through videos

            for frame_num in range(num_of_frames):                      #Loop through video length/frames

                ret, frame = cap.read()                                 # Reading cap feed

                image, results = mediapipe_detection(frame, holistic)   # Making detections using MP model on captured frame

                draw_landmarks(image, results)                          #Drawing Lanndmarks over the image
                
                #  Wait logic
                if frame_num == 0:                                      #If 1st frame then wait
                    cv2.putText(image, 'STARTING COLLECTION IN 2 sec', (90,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,204,), 4, cv2.LINE_AA)    #Text on Screen
                    cv2.imshow('OpenCV Feed', image)                    # Display 
                    cv2.waitKey(2000)                                   # Wait time 2 sec

                else: 
                    cv2.putText(image, 'Collecting {}, Video Number {}, Frame Number {}'.format(sign, video_num, frame_num), (20,15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (153,255,153), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)               
                
                keypoints = extract_keypoints(results)                  # Extracting and Saving keypoints
                npy_path = os.path.join(DATA_PATH, sign, str(video_num), str(frame_num))
                np.save(npy_path, keypoints)
               
                if cv2.waitKey(10) & 0xFF == ord('x'):                  # Breaking the loop by pressing x
                    break

    cap.release()
    cv2.destroyAllWindows()                # After finishing all signs Exit    