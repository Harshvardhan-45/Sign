# from gtts import gTTS
# import os
# from playsound import playsound
import pyttsx3

str = 'My Name is Rahul'
# def usingGTTS(str) :
#     speech = gTTS(str)
#     speech.save('A.mp3')
#     playsound('A.mp3')


# def usingpyttsx3(str):
audio = pyttsx3.init()
audio.setProperty('rate', 125)
audio.setProperty('volume', 0.8)

"""VOICE"""
# voices = audio.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
# audio.setProperty('voice', voices[0].id)   #changing index, changes voices. 1 for female

audio.say(str)

audio.runAndWait()


# usingpyttsx3(str)
