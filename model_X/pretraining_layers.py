import tensorflow as tf
import tf_utils
import copy

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions.
  Args:
      sequence_tensor: Sequence output of `BertModel` layer of shape
        (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
        hidden units of `BertModel` layer.
      positions: Positions ids of tokens in sequence to mask for pretraining of
        with dimension (batch_size, max_predictions_per_seq) where
        `max_predictions_per_seq` is maximum number of tokens to mask out and
        predict per each sequence.
  Returns:
      Masked out sequence tensor of shape (batch_size * max_predictions_per_seq,
      num_hidden).
  """
  sequence_shape = tf_utils.get_shape_list(
      sequence_tensor, name='sequence_output_tensor')
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.keras.backend.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.keras.backend.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.keras.backend.reshape(
      sequence_tensor, [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

  return output_tensor

class PretrainLayer(tf.keras.layers.Layer):
  """Wrapper layer for pre-training a ALBERT model.
  This layer wraps an existing `albert_layer` which is a Keras Layer.
  It outputs `sequence_output` from TransformerBlock sub-layer and
  `sentence_output` which are suitable for feeding into a ALBertPretrainLoss
  layer. This layer can be used along with an unsupervised input to
  pre-train the embeddings for `albert_layer`.
  """

  def __init__(self,
               config,
               embedding_table,
               initializer=None,
               float_type=tf.float32,
               **kwargs):
    super(PretrainLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)
    self.float_type = float_type

    self.embedding_table = embedding_table
    if initializer:
      self.initializer = initializer
    else:
      self.initializer = tf.keras.initializers.TruncatedNormal(
          stddev=self.config.initializer_range)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.output_bias = self.add_weight(
        shape=[self.config.vocab_size],
        name='predictions/output_bias',
        initializer=tf.keras.initializers.Zeros())
    self.lm_dense = tf.keras.layers.Dense(
        self.config.embedding_size,
        activation=tf_utils.get_activation(self.config.hidden_act),
        kernel_initializer=self.initializer,
        name='predictions/transform/dense')
    self.lm_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='predictions/transform/LayerNorm')

    super(PretrainLayer, self).build(unused_input_shapes)

  def __call__(self,
               pooled_output,
               sequence_output=None,
               masked_lm_positions=None,
               **kwargs):
    inputs = tf_utils.pack_inputs(
        [pooled_output, sequence_output, masked_lm_positions])
    return super(PretrainLayer, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    pooled_output = unpacked_inputs[0]
    sequence_output = unpacked_inputs[1]
    masked_lm_positions = unpacked_inputs[2]

    mask_lm_input_tensor = gather_indexes(sequence_output, masked_lm_positions)
    lm_output = self.lm_dense(mask_lm_input_tensor)
    lm_output = self.lm_layer_norm(lm_output)
    lm_output = tf.matmul(lm_output, self.embedding_table, transpose_b=True)
    lm_output = tf.nn.bias_add(lm_output, self.output_bias)
    lm_output = tf.nn.log_softmax(lm_output, axis=-1)

    return lm_output

class PretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def __init__(self, bert_config, **kwargs):
    super(PretrainLossAndMetricLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(bert_config)

  def __call__(self,
               lm_output,
               lm_label_ids=None,
               lm_label_weights=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([
        lm_output, lm_label_ids, lm_label_weights
    ])
    return super(PretrainLossAndMetricLayer, self).__call__(
        inputs, **kwargs)

  def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                   lm_per_example_loss):
    """Adds metrics."""
    masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        lm_labels, lm_output)
    masked_lm_accuracy = tf.reduce_mean(masked_lm_accuracy * lm_label_weights)
    self.add_metric(
        masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')

    lm_example_loss = tf.reshape(lm_per_example_loss, [-1])
    lm_example_loss = tf.reduce_mean(lm_example_loss * lm_label_weights)
    self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    lm_output = unpacked_inputs[0]
    lm_label_ids = unpacked_inputs[1]
    lm_label_ids = tf.keras.backend.reshape(lm_label_ids, [-1])
    lm_label_ids_one_hot = tf.keras.backend.one_hot(lm_label_ids,
                                                    self.config.vocab_size)
    lm_label_weights = tf.keras.backend.cast(unpacked_inputs[2], tf.float32)
    lm_label_weights = tf.keras.backend.reshape(lm_label_weights, [-1])
    lm_per_example_loss = -tf.keras.backend.sum(
        lm_output * lm_label_ids_one_hot, axis=[-1])
    numerator = tf.keras.backend.sum(lm_label_weights * lm_per_example_loss)
    denominator = tf.keras.backend.sum(lm_label_weights) + 1e-5
    mask_label_loss = numerator / denominator

    final_loss = mask_label_loss

    self._add_metrics(lm_output, lm_label_ids, lm_label_weights,
                      lm_per_example_loss)
    return final_loss