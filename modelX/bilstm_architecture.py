from keras.layers import *
from keras.optimizers import *
from keras.models import Model, Sequential
import numpy as np

class BiLSTMModel():
    def __init__(self, nb_words, embedding_size, embedding_matrix=None, is_embedding_trainable=False):
        
        if embedding_matrix is None:
            embedding_matrix = np.zeros((nb_words, embedding_size))

        self.nb_words = nb_words
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
    
    def __call__(self,pre_layer):
        x = Embedding(self.nb_words, self.embedding_size, weights=[self.embedding_matrix], trainable=self.is_embedding_trainable)(pre_layer)
        x = SpatialDropout1D(0.3)(x)
        x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
        x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)
        max_pool1 = GlobalMaxPooling1D()(x1)
        max_pool2 = GlobalMaxPooling1D()(x2)
        conc = Concatenate()([max_pool1, max_pool2])
        return conc