from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import json
import numpy as np
import pickle


def harmonize_json(json_melody: str, overfit=False)->str:
    pass


def harmonize(melody: np.ndarray, overfit=False):
    if overfit:
        overfit_model = load_model('overfit.h5')
        return overfit_model.predict(melody)
    model: Model = load_model('final.h5')
    return model.predict(melody)
