import tensorflow as tf
from .transformers_utils import *

class VanillaTransformer():
    def __init__(self, config, embedding_matrix=None, is_embedding_trainable=False, is_position_embedding_trainable=False):
        if embedding_matrix is None:
            embedding_matrix = np.zeros((config.vocab_size, config.embed_dim))

        self.config = config
        self.embedding_matrix = embedding_matrix
        self.is_embedding_trainable = is_embedding_trainable
        self.is_position_embedding_trainable = is_position_embedding_trainable

        self.pooler_transform = tf.keras.layers.Dense(
                                units=self.config.embed_dim,
                                activation="tanh",
                                name="pooler_transform")

    def __call__(self, pre_layer):
        embedding_layer = TokenAndPositionEmbedding(self.config, self.embedding_matrix, self.is_embedding_trainable, self.is_position_embedding_trainable)(pre_layer)
        sequence_output = TransformerBlock(self.config)(embedding_layer)
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = self.pooler_transform(first_token_tensor)
        return (pooled_output,sequence_output)