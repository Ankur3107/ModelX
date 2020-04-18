from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model, Sequential
import numpy as np
from .utils import MultiHead
from .utils import MultiHeadAttention
from .utils import SeqSelfAttention
from .utils import ScaledDotProductAttention
from .utils import SeqWeightedAttention
import tensorflow as tf

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
        x1 = Bidirectional(LSTM(self.h_lstm, return_sequences=True))(x)
        x2 = Bidirectional(GRU(self.h_gru, return_sequences=True))(x1)
        max_pool1 = GlobalMaxPooling1D()(x1)
        max_pool2 = GlobalMaxPooling1D()(x2)
        conc = Concatenate()([max_pool1, max_pool2])
        return conc

class BiLSTMGRUAttention():
    def __init__(self, nb_words, embedding_size, embedding_matrix=None, is_embedding_trainable=False, h_lstm=256, h_gru=128, attention_type='SeqSelfAttention'):
        
        if embedding_matrix is None:
            embedding_matrix = np.zeros((nb_words, embedding_size))

        self.nb_words = nb_words
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
        self.h_lstm = h_lstm
        self.h_gru = h_gru
        self.attention_type = attention_type
        
    def __call__(self,pre_layer):
        x = Embedding(self.nb_words, self.embedding_size, weights=[self.embedding_matrix], trainable=self.is_embedding_trainable)(pre_layer)
        x = SpatialDropout1D(0.3)(x)
        
        x1 = Bidirectional(LSTM(self.h_lstm, return_sequences=True))(x)
        if self.attention_type == 'SeqSelfAttention':
            x1_self = SeqSelfAttention(attention_activation='sigmoid')(x1)

        elif self.attention_type == 'ScaledDotProductAttention':
            x1_self = ScaledDotProductAttention()(x1)

        else:
            print('Attention Type', self.attention_type,' not found !')
    

        x2 = Bidirectional(GRU(self.h_gru, return_sequences=True))(x1)
        if self.attention_type == 'SeqSelfAttention':
            x2_self = SeqSelfAttention(attention_activation='sigmoid')(x2)

        elif self.attention_type == 'ScaledDotProductAttention':
            x2_self = ScaledDotProductAttention()(x2)
            
        else:
            print('Attention Type', self.attention_type,' not found !')
    
        max_pool1 = GlobalMaxPooling1D()(x1_self)
        max_pool2 = GlobalMaxPooling1D()(x2_self)
        conc = Concatenate()([max_pool1, max_pool2])
        return conc

class BiLSTMGRUMultiHeadAttention():
    def __init__(self, nb_words, embedding_size, embedding_matrix=None, is_embedding_trainable=False, h_lstm=256, h_gru=128,head_num=3):
        
        if embedding_matrix is None:
            embedding_matrix = np.zeros((nb_words, embedding_size))

        self.nb_words = nb_words
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
        self.h_lstm = h_lstm
        self.h_gru = h_gru
        self.head_num = head_num
        
    def __call__(self,pre_layer):
        x = Embedding(self.nb_words, self.embedding_size, weights=[self.embedding_matrix], trainable=self.is_embedding_trainable)(pre_layer)
        x = SpatialDropout1D(0.3)(x)
        
        x1 = MultiHead(Bidirectional(LSTM(self.h_lstm, return_sequences=False)),layer_num=3)(x)
        x1_self = MultiHeadAttention(head_num=self.head_num)(x1)
    
        x2 = MultiHead(Bidirectional(GRU(self.h_gru, return_sequences=False)),layer_num=3)(x)
        x2_self = MultiHeadAttention(head_num=self.head_num)(x2)
    
        max_pool1 = GlobalMaxPooling1D()(x1_self)
        max_pool2 = GlobalMaxPooling1D()(x2_self)
        conc = Concatenate()([max_pool1, max_pool2])
        return conc

class SplitBiLSTMGRUSpatialDropout1D():
    def __init__(self, nb_words, embedding_size, embedding_matrix=None, is_embedding_trainable=False, h_lstm1=256, h_lstm2=512, h_gru=128):
        
        if embedding_matrix is None:
            embedding_matrix = np.zeros((nb_words, embedding_size))

        self.nb_words = nb_words
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
        self.h_lstm1 = h_lstm1
        self.h_lstm2 = h_lstm2
        self.h_gru = h_gru
        
    def __call__(self,pre_layer):
        x = Embedding(self.nb_words, self.embedding_size, weights=[self.embedding_matrix], trainable=self.is_embedding_trainable)(pre_layer)
        x = SpatialDropout1D(0.3)(x)
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=2))(x)

        x1 = Bidirectional(LSTM(self.h_lstm1, return_sequences=True))(splits[0])
        x2 = Bidirectional(LSTM(self.h_lstm2, return_sequences=True))(splits[1])

        conc = Concatenate()([x1, x2])
        conct_layer = Bidirectional(GRU(self.h_gru, return_sequences=True))(conc)
        max_pool_layer = GlobalMaxPooling1D()(conct_layer)

        return max_pool_layer

class SplitBiLSTMGRU():
    def __init__(self, nb_words, embedding_size, embedding_matrix=None, is_embedding_trainable=False, h_lstm1=256, h_lstm2=512, h_gru=128):
        
        if embedding_matrix is None:
            embedding_matrix = np.zeros((nb_words, embedding_size))

        self.nb_words = nb_words
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
        self.h_lstm1 = h_lstm1
        self.h_lstm2 = h_lstm2
        self.h_gru = h_gru
        
    def __call__(self,pre_layer):
        x = Embedding(self.nb_words, self.embedding_size, weights=[self.embedding_matrix], trainable=self.is_embedding_trainable)(pre_layer)
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=2))(x)

        x1 = Bidirectional(LSTM(self.h_lstm1, return_sequences=True))(splits[0])
        x2 = Bidirectional(LSTM(self.h_lstm2, return_sequences=True))(splits[1])

        conc = Concatenate()([x1, x2])
        conct_layer = Bidirectional(GRU(self.h_gru, return_sequences=True))(conc)
        max_pool_layer = GlobalMaxPooling1D()(conct_layer)

        return max_pool_layer