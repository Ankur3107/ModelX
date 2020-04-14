# Model_X

Model_X package is a collection of different NLP architecture models.

# Implementation

## 1. BiLSTM+BiGRU Architectures

### a. BiLSTMGRUSpatialDropout1D

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = BiLSTMGRUSpatialDropout1D(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

### b. BiLSTMGRUSelfAttention

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = BiLSTMGRUSelfAttention(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

### c.  BiLSTMGRUMultiHeadAttention

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = BiLSTMGRUMultiHeadAttention(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

### d.  SplitBiLSTMGRUSpatialDropout1D

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = SplitBiLSTMGRUSpatialDropout1D(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

### e.  SplitBiLSTMGRU

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    bilstm_layers = SplitBiLSTMGRU(10, 100)(model_input)
    dense_layers = DenseLayerModel()(bilstm_layers)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())

## 2. Dense Architectures


### a. DenseLayerModel

    input_shape = (100,)
    model_input = Input(shape=input_shape)
    dense_layers = DenseLayerModel()(model_input)
    output = Dense(3, activation='softmax')(dense_layers)
    full_model = Model(inputs=model_input, outputs=output)
    print(full_model.summary())


