from keras.layers import *
from keras.optimizers import *
from keras.models import Model, Sequential
import numpy as np

class BiLSTMGRUSpatialDropout1D():
    def __init__(self, nb_words, embedding_size, embedding_matrix=None, is_embedding_trainable=False, h_lstm=256, h_gru=128):
        
        if embedding_matrix is None:
            embedding_matrix = np.zeros((nb_words, embedding_size))

        self.nb_words = nb_words
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
        self.h_lstm = h_lstm
        self.h_gru = h_gru
        
    def __call__(self,pre_layer):
        x = Embedding(self.nb_words, self.embedding_size, weights=[self.embedding_matrix], trainable=self.is_embedding_trainable)(pre_layer)
        x = SpatialDropout1D(0.3)(x)
        x1 = Bidirectional(CuDNNLSTM(self.h_lstm, return_sequences=True))(x)
        x2 = Bidirectional(CuDNNGRU(self.h_gru, return_sequences=True))(x1)
        max_pool1 = GlobalMaxPooling1D()(x1)
        max_pool2 = GlobalMaxPooling1D()(x2)
        conc = Concatenate()([max_pool1, max_pool2])
        return conc


class BiLSTMGRUSelfAttention():
    def __init__(self, nb_words, embedding_size, embedding_matrix=None, is_embedding_trainable=False, h_lstm=256, h_gru=128):
        
        if embedding_matrix is None:
            embedding_matrix = np.zeros((nb_words, embedding_size))

        self.nb_words = nb_words
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
        self.h_lstm = h_lstm
        self.h_gru = h_gru
        
    def __call__(self,pre_layer):
        x = Embedding(self.nb_words, self.embedding_size, weights=[self.embedding_matrix], trainable=self.is_embedding_trainable)(pre_layer)
        x = SpatialDropout1D(0.3)(x)
        
        x1 = Bidirectional(CuDNNLSTM(self.h_lstm, return_sequences=True))(x)
        x1_self = SeqSelfAttention(attention_activation='sigmoid')(x1)
    
        x2 = Bidirectional(CuDNNGRU(self.h_gru, return_sequences=True))(x1)
        x2_self = SeqSelfAttention(attention_activation='sigmoid')(x2)
    
        max_pool1 = GlobalMaxPooling1D()(x1_self)
        max_pool2 = GlobalMaxPooling1D()(x2_self)
        conc = Concatenate()([max_pool1, max_pool2])
        return conc

