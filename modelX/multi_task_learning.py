from keras.layers import *
from keras.optimizers import *
from keras.models import Model, Sequential, load_model
from keras_self_attention import SeqSelfAttention
from keras_transformer import get_model
import tensorflow as tf
import numpy as np
import os
from keras_bert.loader import load_trained_model_from_checkpoint
from keras.engine.topology import Layer

class GenericClassifier():
    
    def __init__(self, max_length, nb_words, embedding_size, embedding_matrix, is_embedding_trainable=False):
        #self.input_text = input_text
        self.max_length = max_length
        self.nb_words = nb_words
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
        
    def get_current_model_config(self):
        
        self.config = {}
        
        self.config['representation_model'] = self._model_based_upon
        self.config['max_length'] = self.max_length
        self.config['nb_words'] = self.nb_words
        self.config['embedding_size'] = self.embedding_size
        self.config['is_embedding_trainable'] = self.is_embedding_trainable
        
        return self.config
    
    def _get_transformer_based_representation_model(self, encoder_num=3, decoder_num=2, head_num=3, hidden_dim=120):
        
        self._transformer_representational_model = get_model(
            token_num=self.nb_words,
            embed_dim=self.embedding_size,
            encoder_num=encoder_num,
            decoder_num=decoder_num,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation='relu',
            feed_forward_activation='relu',
            dropout_rate=0.05,
            embed_weights=self.embedding_matrix
        )
        return self._transformer_representational_model
    
    def _get_transformer_representation_layer(self):
        
        layer1 = self._transformer_representational_model.get_layer('Encoder-1-FeedForward-Norm').output
        layer2 = self._transformer_representational_model.get_layer('Encoder-2-FeedForward-Norm').output
        concatenate_layer = concatenate([layer1, layer2])
        return concatenate_layer
    
    def _get_bilstm_based_representation_model(self):
        
        text_input = Input(shape=(self.max_length,))
        x = Embedding(self.nb_words, self.embedding_size, weights=[self.embedding_matrix], trainable=self.is_embedding_trainable)(text_input)
        x = SpatialDropout1D(0.3)(x)
        x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
        x1_self = SeqSelfAttention(attention_activation='sigmoid')(x1)
    
        x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)
        x2_self = SeqSelfAttention(attention_activation='sigmoid')(x2)
    
        max_pool1 = GlobalMaxPooling1D()(x1_self)
        max_pool2 = GlobalMaxPooling1D()(x2_self)
        conc = Concatenate()([max_pool1, max_pool2])
        self._bilstm_representational_model = Model(inputs=text_input, outputs=conc)
        
        return self._bilstm_representational_model
    
    def _get_pretrained_model(self, BERT_PRETRAINED_DIR, is_bert_training=True):
        
        config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
        checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
        model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=is_bert_training,seq_len=self.max_length)

        return model
    
    def _get_bert_based_representation_model(self, BERT_PRETRAINED_DIR):
        
        bert_model = self._get_pretrained_model(BERT_PRETRAINED_DIR)
        
        self._bert_representational_model = Model(inputs=[bert_model.input[0],bert_model.input[1]], outputs=bert_model.get_layer('Extract').output)
        
        return self._bert_representational_model
        
    def build_text_representation_model(self, model_based_upon='bilstm', BERT_PRETRAINED_DIR=None):
        
        if model_based_upon == 'transformer':
            self._text_representation_model = self._get_transformer_based_representation_model()
            self._model_based_upon = model_based_upon
            
        elif model_based_upon == 'bilstm':
            self._text_representation_model = self._get_bilstm_based_representation_model()
            self._model_based_upon = model_based_upon
        elif model_based_upon == 'bert':
            self._text_representation_model = self._get_bert_based_representation_model(BERT_PRETRAINED_DIR)
            self._model_based_upon = model_based_upon
        else:
            print('Invalid Model Name :', model_based_upon)
            self._text_representation_model = None
        
        return self._text_representation_model
    
    def load_text_representation_model(self, weight_file_path):
        
        self._text_representation_model = load_model(weight_file_path, custom_objects={'SeqSelfAttention':SeqSelfAttention})
        self._model_based_upon = 'bilstm'
        
    def _get_output_layer(self, n_output=9, activation_function='softmax'):
        
        output_layer = Dense(n_output, activation=activation_function)
        return output_layer
    
    # Not used so far
    def _get_dense_downward_triangle_layers(self, input_shape = (100,), hidden_layer_size=[1024,512,256,128, 64], activation_function='relu'):
        dense_model = Sequential()

        for i in range(len(hidden_layer_size)):

            if i==0:
                dense_layer = Dense(hidden_layer_size[i], activation=activation_function, input_shape=input_shape)
            else:
                dense_layer = Dense(hidden_layer_size[i], activation=activation_function)

            dense_model.add(dense_layer)

        return dense_model

    def build_l1_classifier(self, n_output, output_activation):
        
        if self._model_based_upon is 'transformer':
            represented_conc = self._get_transformer_representation_layer()
        
        if self._model_based_upon is 'bilstm':
            represented_conc = self._text_representation_model.output
            
        if self._model_based_upon is 'bert':
            represented_conc = self._text_representation_model.output
        
        l1_classifier_input = self._text_representation_model.input
        
        dense1 = Dense(1024, activation='relu')(represented_conc)
        dense1 = Dropout(0.5)(dense1)
        
        dense2 = Dense(256, activation='relu')(dense1)
        dense2 = Dropout(0.4)(dense2)
        
        dense3 = Dense(64, activation='relu')(dense2)
        dense3 = Dropout(0.2)(dense3)
        
        output = self._get_output_layer(n_output, output_activation)(dense3)
        
        self._l1_classifier = Model(inputs=l1_classifier_input, outputs=output)
        return self._l1_classifier
    
    def load_l1_classifier(self, weight_file_path):
        
        self._l1_classifier = load_model(weight_file_path, custom_objects={'SeqSelfAttention':SeqSelfAttention})
    
    def build_l1_based_sentiment_classifier(self, l1_len, n_output, output_activation):
        
        if self._model_based_upon is 'transformer':
            represented_conc = self._get_transformer_representation_layer()
        
        if self._model_based_upon is 'bilstm':
            represented_conc = self._text_representation_model.output
            
        if self._model_based_upon is 'bert':
            represented_conc = self._text_representation_model.output
        
        text_input = self._text_representation_model.input
        l1_input = Input(shape=(l1_len,))
        
        l1_conc = Concatenate()([l1_input, represented_conc])
        
        dense1 = Dense(1024, activation='relu')(l1_conc)
        dense1 = Dropout(0.5)(dense1)
        
        dense2 = Dense(256, activation='relu')(dense1)
        dense2 = Dropout(0.4)(dense2)
        
        dense3 = Dense(64, activation='relu')(dense2)
        dense3 = Dropout(0.2)(dense3)
        
        output = self._get_output_layer(n_output, output_activation)(dense3)
        
        model_input = self._get_calculated_model_input(text_input, l1_input)
        
        self._l1_based_sentiment_classifier = Model(inputs=model_input, outputs=output)
        return self._l1_based_sentiment_classifier
    
    def _get_calculated_model_input(self, text_input, l1_input):
        
        result = []
        if type(text_input)==type([]) and len(text_input)>1:
            result = text_input
            result.append(l1_input)
        else:
            result.append(text_input)
            result.append(l1_input)
            
        return result
    
    def load_l1_based_sentiment_classifier(self, weight_file_path):
        
        self._l1_based_sentiment_classifier = load_model(weight_file_path, custom_objects={'SeqSelfAttention':SeqSelfAttention})
        
    def build_l1_based_l2_classifier(self, l1_len, n_output, output_activation):
        
        if self._model_based_upon is 'transformer':
            represented_conc = self._get_transformer_representation_layer()
        
        if self._model_based_upon is 'bilstm':
            represented_conc = self._text_representation_model.output
        
        if self._model_based_upon is 'bert':
            represented_conc = self._text_representation_model.output
        
        text_input = self._text_representation_model.input
        l1_input = Input(shape=(l1_len,))
        
        l1_conc = Concatenate(name='concat')([l1_input, represented_conc])
        
        dense1 = Dense(1024, activation='relu')(l1_conc)
        dense1 = Dropout(0.5)(dense1)
        
        dense2 = Dense(256, activation='relu')(dense1)
        dense2 = Dropout(0.4)(dense2)
        
        dense3 = Dense(64, activation='relu')(dense2)
        dense3 = Dropout(0.2)(dense3)
        
        output = self._get_output_layer(n_output, output_activation)(dense3)
        
        model_input = self._get_calculated_model_input(text_input, l1_input)
        
        self._l1_based_l2_classifier = Model(inputs=model_input, outputs=output)
        return self._l1_based_l2_classifier
    
    def load_l1_based_l2_classifier(self, weight_file_path):
        
        self._l1_based_l2_classifier = load_model(weight_file_path, custom_objects={'SeqSelfAttention':SeqSelfAttention})