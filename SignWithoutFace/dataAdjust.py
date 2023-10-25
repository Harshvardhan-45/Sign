#Removing face, 1662 feature to 258

import numpy as np
import os

DATA_PATH = os.path.join('CompleteData')     # Path to load data
NEW_PATH = os.path.join('PoseHandsData')         #to save
num_of_videos = 100                  # 100 videos per action
num_of_frames = 30


signs = np.array(['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I','J', 'K', 'L','M', 'N', 'O','P', 'Q', 'R','S', 'T', 'U','V', 'W', 'X', 'Y', 'Z', 'Air', 'Delete', 'Fine', 'Go', 'Good', 'Hearing', 'My', 'Name', 'Person', 'Separate', 'Stay', 'Study', 'Talk', 'Time', 'Us', 'Walk', 'What', 'Wrong', 'You', 'Your'])

for sign in signs: 
    for video_num in range(num_of_videos):                #To add more data to existing sign, edit the loop condition
        try: 
            os.makedirs(os.path.join(NEW_PATH, sign, str(video_num)))          # Making Directories 
        except:
            pass

for sign in signs:
    print(sign)
    for video_num in range(num_of_videos):
        for frame_num in range(num_of_frames):
            a = np.load(os.path.join(DATA_PATH, sign, str(video_num), "{}.npy".format(frame_num)))        #loading each frame
            pose= a[0:132]
            hands= a[1536:]
            b=np.concatenate([pose, hands])

            npy_path = os.path.join(NEW_PATH, sign, str(video_num), str(frame_num))
            np.save(npy_path, b)


print('Finished')

"""
a=np.load(os.path.join(DATA_PATH, 'Talk', str(0), "{}.npy".format(0)))        #loading each frame
# a=np.array(a)
print(a.shape) 

pose= a[0:132]
hands= a[1536:]
print(pose.shape) 
print(hands.shape) 

b=np.concatenate([pose, hands])

print(b.shape) 

npy_path = os.path.join(DATA_PATH,str(0))
np.save(npy_path, b)
"""