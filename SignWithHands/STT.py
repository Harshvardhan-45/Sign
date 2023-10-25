import speech_recognition as sr

def STT():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        
        print("Please say something...")
        
        audio=r.listen(source)
        
        try:
            #print("you have said : \n "+r.recognize_google(audio))
            return r.recognize_google(audio)
            
        except Exception as e:
            print("Error:"+str(e))

speech=STT()
print(speech)
