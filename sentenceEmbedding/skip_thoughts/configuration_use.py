from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class _HParams(object):
    """Wrapper for configuration parameters."""
    pass


def model_config(input_file_pattern=None,
                 input_file_pattern_snli=None,
                 input_queue_capacity=640000,
                 num_input_reader_threads=1,
                 shuffle_input_data=True,
                 uniform_init_scale=0.1,
                 vocab_size=200001,  # USE: vocab_size = 200003 - 2 (start/stop token)
                 batch_size=32,
                 encoder_dim=512):
    """Creates a model configuration object.

    Args:
      input_file_pattern: File pattern of sharded TFRecord files containing
        tf.Example protobufs for similarity.
      input_file_pattern_snli: File pattern of sharded TFRecord files containing
        tf.Example protobufs for sentiment.
      input_queue_capacity: Number of examples to keep in the input queue.
      num_input_reader_threads: Number of threads for prefetching input
        tf.Examples.
      shuffle_input_data: Whether to shuffle the input data.
      uniform_init_scale: Scale of random uniform initializer.
      vocab_size: Number of unique words in the vocab.
      batch_size: Batch size (training and evaluation only).
      encoder_dim: Number of output dimensions of the sentence encoder.

    Returns:
      An object containing model configuration parameters.
    """
    config = _HParams()
    config.input_file_pattern = input_file_pattern
    config.input_file_pattern_snli = input_file_pattern_snli
    config.input_queue_capacity = input_queue_capacity
    config.num_input_reader_threads = num_input_reader_threads
    config.shuffle_input_data = shuffle_input_data
    config.uniform_init_scale = uniform_init_scale
    config.vocab_size = vocab_size
    config.batch_size = batch_size
    config.encoder_dim = encoder_dim
    return config


def training_config(learning_rate=0.001,
                    learning_rate_decay_factor=0.5,
                    learning_rate_decay_steps=400000,
                    number_of_steps=500000,  # 500000
                    clip_gradient_norm=5.0,
                    save_model_secs=60,
                    save_summaries_secs=60):
    """Creates a training configuration object.

    Args:
      learning_rate: Initial learning rate.
      learning_rate_decay_factor: If > 0, the learning rate decay factor.
      learning_rate_decay_steps: The number of steps before the learning rate
        decays by learning_rate_decay_factor.
      number_of_steps: The total number of training steps to run. Passing None
        will cause the training script to run indefinitely.
      clip_gradient_norm: If not None, then clip gradients to this value.
      save_model_secs: How often (in seconds) to save model checkpoints.
      save_summaries_secs: How often (in seconds) to save model summaries.

    Returns:
      An object containing training configuration parameters.

    Raises:
      ValueError: If learning_rate_decay_factor is set and
        learning_rate_decay_steps is unset.
    """
    if learning_rate_decay_factor and not learning_rate_decay_steps:
        raise ValueError(
            "learning_rate_decay_factor requires learning_rate_decay_steps.")

    config = _HParams()
    config.learning_rate = learning_rate
    config.learning_rate_decay_factor = learning_rate_decay_factor
    config.learning_rate_decay_steps = learning_rate_decay_steps
    config.number_of_steps = number_of_steps
    config.clip_gradient_norm = clip_gradient_norm
    config.save_model_secs = save_model_secs
    config.save_summaries_secs = save_summaries_secs
    return config
