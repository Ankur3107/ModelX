from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model, Sequential
import numpy as np

class DenseLayerModel():
    def __init__(self, hidden_layer_size=[1024,512,256,128, 64], activation_function='relu'):
        self.hidden_layer_size = hidden_layer_size
        self.activation_function = activation_function
    
    def __call__(self,pre_layer):
        for i in range(len(self.hidden_layer_size)):
            pre_layer = Dense(self.hidden_layer_size[i], activation=self.activation_function)(pre_layer)
        return pre_layer