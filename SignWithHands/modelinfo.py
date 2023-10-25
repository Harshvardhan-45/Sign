from tensorflow.keras.models import load_model

model=load_model('sign88.h5')

print(model.summary())
