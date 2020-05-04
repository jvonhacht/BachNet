import numpy as np

def midi_to_one_hot(note):
    

data = np.load("data/Jsb16thSeparated.npz", allow_pickle=True, encoding='latin1')

x_train = [[harmony[0] for harmony in piece] for piece in data['train']]
y_train = [[harmony[1:] for harmony in piece] for piece in data['train']]
x_val = [[harmony[0] for harmony in piece] for piece in data['val']]
y_val = [[harmony[1:] for harmony in piece] for piece in data['val']]
x_test = [[harmony[0] for harmony in piece] for piece in data['test']]
y_test = [[harmony[1:] for harmony in piece] for piece in data['test']]


pass