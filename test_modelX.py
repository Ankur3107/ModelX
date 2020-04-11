from model_X.bilstm_architecture import BiLSTMModel
from model_X.dense_architecture import DenseLayerModel
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model, Sequential
import numpy as np

def test():
    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = BiLSTMModel(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())
