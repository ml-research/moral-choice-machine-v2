from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub

from sentenceEmbedding.skip_thoughts.ops import input_ops_use as input_ops


def random_orthonormal_initializer(shape, dtype=tf.float32,
                                   partition_info=None):  # pylint: disable=unused-argument
    """Variable initializer that produces a random orthonormal matrix."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u


class SkipThoughtsModel(object):
    """Skip-thoughts model."""

    def __init__(self, config, mode="train", input_reader=None, input_reader_snli=None):
        """Basic setup. The actual TensorFlow graph is constructed in build().

        Args:
          config: Object containing configuration parameters.
          mode: "train", "eval" or "encode".
          input_reader: Subclass of tf.ReaderBase for reading the input serialized
            tf.Example protocol buffers. Defaults to TFRecordReader.

        Raises:
          ValueError: If mode is invalid.
        """
        if mode not in ["train", "eval", "encode"]:
            raise ValueError("Unrecognized mode: %s" % mode)

        self.config = config
        self.mode = mode
        self.reader = input_reader if input_reader else tf.TFRecordReader()
        self.reader_snli = input_reader_snli if input_reader_snli else tf.TFRecordReader()
        # Initializer used for non-recurrent weights.
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-self.config.uniform_init_scale,
            maxval=self.config.uniform_init_scale)

        # Input sentences represented as sequences of word ids. "encode" is the
        # source sentence, "decode_pre" is the previous sentence and "decode_post"
        # is the next sentence.

        # Input sentences represented as sentence string.
        # A string Tensor with shape [batch_size,].
        self.encode_sent = None
        self.encode_pre_sent = None
        self.encode_post_sent = None

        self.snli_input1 = None
        self.snli_input2 = None
        self.snli_labels = None
        # The output sentences represented as sentence embeddings. embdim=512
        # Each is a float32 Tensor with shape [batch_size, emb_dim].
        self.decode_pre_sent_emb = None
        self.decode_post_sent_emb = None

        self.decode_snli_emb1 = None
        self.decode_snli_emb2 = None

        # The output from the sentence encoder.
        # A float32 Tensor with shape [batch_size, emb_dim].
        self.thought_vectors = None

        # The cross entropy losses and corresponding weights of the decoders. Used
        # for evaluation.
        self.target_cross_entropy_losses = []

        # The total loss to optimize.
        self.total_loss = None

        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3",
                                trainable=mode == 'train',
                                name='use')
        self.global_step = 0

    def build_inputs(self):
        """Builds the ops for reading input data.

        Outputs:
          self.encode_sent
          self.decode_pre_sent
          self.decode_post_sent
        """
        if self.mode == "encode":
            # Word embeddings are fed from an external vocabulary which has possibly
            # been expanded (see vocabulary_expansion.py).

            encode_sent = tf.placeholder(tf.string, (None,), name="messages_to_encode")
            encode_pre_sent = None
            encode_post_sent = None
        else:
            # Prefetch serialized tf.Example protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                shuffle=self.config.shuffle_input_data,
                capacity=self.config.input_queue_capacity,
                num_reader_threads=self.config.num_input_reader_threads)

            # Deserialize a batch.
            serialized = input_queue.dequeue_many(self.config.batch_size)
            encode_sent, encode_pre_sent, encode_post_sent = input_ops.parse_example_batch(serialized)

        self.encode_sent = encode_sent
        self.encode_pre_sent = encode_pre_sent
        self.encode_post_sent = encode_post_sent

    def build_inputs_snli(self):
        """Builds the ops for reading input data.

        Outputs:
          self.encode_sent
          self.decode_pre_sent
          self.decode_post_sent
        """
        if self.mode == "encode":
            # Word embeddings are fed from an external vocabulary which has possibly
            # been expanded (see vocabulary_expansion.py).

            encode_snli1 = None
            encode_snli2 = None
            labels_snli = None
        else:
            # Prefetch serialized tf.Example protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader_snli,
                self.config.input_file_pattern_snli,
                shuffle=self.config.shuffle_input_data,
                capacity=self.config.input_queue_capacity,
                num_reader_threads=self.config.num_input_reader_threads)

            # Deserialize a batch.
            serialized = input_queue.dequeue_many(self.config.batch_size)
            encode_snli1, encode_snli2, labels_snli = input_ops.parse_example_batch_snli(serialized)

        self.snli_input1 = encode_snli1
        self.snli_input2 = encode_snli2
        self.snli_labels = labels_snli

    def build_sent_embeddings(self):
        """Builds the word embeddings.

        Inputs:
          self.decode_pre_ids
          self.decode_post_ids

        Outputs:
          self.decode_pre_sent_emb
          self.decode_post_sent_emb
        """
        if self.mode == "encode":
            # Word embeddings are fed from an external vocabulary which has possibly
            # been expanded (see vocabulary_expansion.py).
            # No sequences to decode.
            decode_pre_sent_emb = None
            decode_post_sent_emb = None
            decode_snli_emb1 = None
            decode_snli_emb2 = None
        else:
            decode_pre_sent_emb = self.embed(self.encode_pre_sent)
            decode_post_sent_emb = self.embed(self.encode_post_sent)

            decode_snli_emb1 = self.embed(self.snli_input1)
            decode_snli_emb2 = self.embed(self.snli_input2)

        self.decode_pre_sent_emb = decode_pre_sent_emb
        self.decode_post_sent_emb = decode_post_sent_emb
        self.decode_snli_emb1 = decode_snli_emb1
        self.decode_snli_emb2 = decode_snli_emb2

    def build_encoder(self):
        """Builds the sentence encoder.

        Inputs:
          self.encode_sent

        Outputs:
          self.thought_vectors

        """
        thought_vectors = self.embed(self.encode_sent)
        self.thought_vectors = tf.identity(thought_vectors, name="thought_vectors")

    def _build_decoder_dnn(self, name, sent_embeddings, thought_vectors,
                       reuse_logits):
        """Builds a sentence decoder.

        Args:
          name: Decoder name.
          sent_embeddings: Batch of sentences to decode; a float32 Tensor with shape
            [batch_size, emb_dim].
          targets: Batch of target word ids; an int64 Tensor with shape
            [batch_size, padded_length].
          mask: A 0/1 Tensor with shape [batch_size, padded_length].
          initial_state: Initial state of the GRU. A float32 Tensor with shape
            [batch_size, num_gru_cells].
          reuse_logits: Whether to reuse the logits weights.
        """
        cos_similarity = tf.keras.layers.Dot(axes=-1, normalize=True)
        # Decoder RNN.
        with tf.variable_scope(name) as scope:
            decoder_input_1 = tf.identity(sent_embeddings, name="input")

            decoder_layer1 = tf.contrib.layers.fully_connected(
                inputs=decoder_input_1,
                num_outputs=512,
                activation_fn=tf.nn.relu,
                weights_initializer=self.uniform_initializer,
                scope=scope)

        # Logits.
        with tf.variable_scope("logits", reuse=reuse_logits) as scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=decoder_layer1,
                num_outputs=512,
                activation_fn=None,
                weights_initializer=self.uniform_initializer,
                scope=scope)

        losses = -cos_similarity([thought_vectors, logits])

        batch_loss = tf.reduce_mean(losses)
        tf.losses.add_loss(0.1*batch_loss)

        tf.summary.scalar("losses/" + name, batch_loss)

        self.target_cross_entropy_losses.append(losses)

    def _build_classifier_fc(self, name, sent_embedding1, sent_embedding2, targets):
        """Builds a sentence decoder.

        Args:
          name: Decoder name.
          sent_embeddings: Batch of sentences to decode; a float32 Tensor with shape
            [batch_size, emb_dim].
          targets: Batch of target word ids; an int64 Tensor with shape
            [batch_size, padded_length].
          mask: A 0/1 Tensor with shape [batch_size, padded_length].
          initial_state: Initial state of the GRU. A float32 Tensor with shape
            [batch_size, num_gru_cells].
          reuse_logits: Whether to reuse the logits weights.
        """
        fc_input_1 = tf.concat([sent_embedding1, sent_embedding2, tf.abs(sent_embedding1 - sent_embedding2),
                               (sent_embedding1 * sent_embedding2)], -1)
        # Decoder RNN.
        with tf.variable_scope(name) as scope:

            decoder_layer1 = tf.contrib.layers.fully_connected(
                inputs=fc_input_1,  # 512+512+512+512
                num_outputs=100,
                activation_fn=tf.nn.relu,
                weights_initializer=self.uniform_initializer,
                scope=scope)

        # Logits.
        with tf.variable_scope(name + "_logits") as scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=decoder_layer1,
                num_outputs=3,
                activation_fn=None,
                weights_initializer=self.uniform_initializer,
                scope=scope)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)
        batch_loss = tf.reduce_mean(losses)
        tf.losses.add_loss(1.*batch_loss)

        tf.summary.scalar("losses/" + name, batch_loss)

        self.target_cross_entropy_losses.append(losses)

    def build_decoders(self):
        """Builds the sentence decoders.

        Inputs:
          self.decode_pre_emb
          self.decode_post_emb
          self.decode_pre_ids
          self.decode_post_ids
          self.decode_pre_mask
          self.decode_post_mask
          self.thought_vectors

        Outputs:
          self.target_cross_entropy_losses
        """
        if self.mode != "encode":
            # Pre-sentence decoder.
            self._build_decoder_dnn("decoder_pre", self.decode_pre_sent_emb,
                                self.thought_vectors, False)

            # Post-sentence decoder. Logits weights are reused.
            self._build_decoder_dnn("decoder_post", self.decode_post_sent_emb,
                                    self.thought_vectors, True)

            self._build_classifier_fc("classifier_snli", self.decode_snli_emb1,
                                      self.decode_snli_emb2, self.snli_labels)

    def build_loss(self):
        """Builds the loss Tensor.

        Outputs:
          self.total_loss
        """
        if self.mode != "encode":
            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar("losses/total", total_loss)

            self.total_loss = total_loss

    def build_global_step(self):
        """Builds the global step Tensor.

        Outputs:
          self.global_step
        """
        self.global_step = tf.contrib.framework.create_global_step()

    def build(self):
        """Creates all ops for training, evaluation or encoding."""
        self.build_inputs()
        self.build_inputs_snli()

        self.build_sent_embeddings()
        self.build_encoder()

        self.build_decoders()
        self.build_loss()
        self.build_global_step()
