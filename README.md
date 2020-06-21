# Model_X

[![Downloads](https://pepy.tech/badge/model-x)](https://pepy.tech/project/model-x)
[![Downloads](https://pepy.tech/badge/model-x/month)](https://pepy.tech/project/model-x/month)
[![Downloads](https://pepy.tech/badge/model-x/week)](https://pepy.tech/project/model-x/week)

Model_X package is a collection of different NLP architecture models.

# Implementation

## 1. BiLSTM+BiGRU Architectures

### a. BiLSTMGRUSpatialDropout1D

    from model_X.bilstm_architectures import *
    from model_X.dense_architectures import DenseLayerModel
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = BiLSTMGRUSpatialDropout1D(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

### b. BiLSTMGRUSelfAttention

    from model_X.bilstm_architectures import *
    from model_X.dense_architectures import DenseLayerModel
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = BiLSTMGRUAttention(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

### c.  BiLSTMGRUMultiHeadAttention

    from model_X.bilstm_architectures import *
    from model_X.dense_architectures import DenseLayerModel
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = BiLSTMGRUMultiHeadAttention(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

### d.  SplitBiLSTMGRUSpatialDropout1D

    from model_X.bilstm_architectures import *
    from model_X.dense_architectures import DenseLayerModel
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = SplitBiLSTMGRUSpatialDropout1D(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

### e.  SplitBiLSTMGRU

    from model_X.bilstm_architectures import *
    from model_X.dense_architectures import DenseLayerModel
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = SplitBiLSTMGRU(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

## 2. Dense Architectures


### a. DenseLayerModel

    from model_X.dense_architectures import DenseLayerModel
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    dense_layers = DenseLayerModel()(model_input)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())


## 3 Transformer Architectures

### a. VanillaTransformer

    from transformers_architectures import *
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model
    import argparse

    config = argparse.Namespace(vocab_size=1000,
                            embed_dim=512,
                            ff_dim=32,
                            num_heads=8,
                            rate=0.1,
                            maxlen=128)

    inputs = tf.keras.layers.Input(shape=(config.maxlen,))
    pooled_output,sequence_output = VanillaTransformer(config)(inputs)
    output = Dense(3, activation='softmax')(pooled_output)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())
