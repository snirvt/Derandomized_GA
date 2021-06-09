import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,Sequential

def get_model():
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(50,activation="sigmoid"))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model



