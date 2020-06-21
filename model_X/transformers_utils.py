import tensorflow as tf
import numpy as np


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding dimension = {self.embed_dim} should be divisible by number of heads = {self.num_heads}"
            )
        self.projection_dim = self.embed_dim // self.num_heads
        self.query_dense = tf.keras.layers.Dense(self.embed_dim)
        self.key_dense = tf.keras.layers.Dense(self.embed_dim)
        self.value_dense = tf.keras.layers.Dense(self.embed_dim)
        self.combine_heads = tf.keras.layers.Dense(self.embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config):#embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(config)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(config.ff_dim, activation="relu"), tf.keras.layers.Dense(config.embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(config.rate)
        self.dropout2 = tf.keras.layers.Dropout(config.rate)

    def __call__(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, config, embedding_matrix, is_embedding_trainable, is_position_embedding_trainable):#maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.embed_dim, trainable=is_embedding_trainable, weights=[embedding_matrix])
        self.pos_emb = tf.keras.layers.Embedding(input_dim=config.maxlen, output_dim=config.embed_dim, trainable=is_position_embedding_trainable, weights=[get_pos_encoding_matrix(config.maxlen, config.embed_dim)])

    def __call__(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_pos_encoding_matrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc