from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf


def _string_feature(value):
    """Helper for creating an Int64 Feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=value))


def parse_example_batch(serialized):
    """Parses a batch of tf.Example protos.

    Args:
      serialized: A 1-D string Tensor; a batch of serialized tf.Example protos.
    Returns:
      encode: A SentenceBatch of encode sentences.
      decode_pre: A SentenceBatch of "previous" sentences to decode.
      decode_post: A SentenceBatch of "post" sentences to decode.
    """
    features = tf.parse_example(
        serialized,
        features={
            "sent": tf.FixedLenFeature([], dtype=tf.string),#tf.FixedLenFeature([], tf.string, default_value=''),
            "sent_pre": tf.FixedLenFeature([], dtype=tf.string),# tf.FixedLenFeature([], tf.string, default_value=''),
            "sent_post": tf.FixedLenFeature([], dtype=tf.string),# tf.FixedLenFeature([], tf.string, default_value=''),
        })
    """
    features = tf.parse_example(
        serialized,
        features={
            "sent": tf.VarLenFeature(tf.string),
            "sent_pre": tf.VarLenFeature(tf.string),
            "sent_post": tf.VarLenFeature(tf.string),
        })
    """
    print(features)
    output_names = ("sent", "sent_pre", "sent_post")
    res = tuple(features[x] for x in output_names)

    return res


def parse_example_batch_snli(serialized):
    """Parses a batch of tf.Example protos.

    Args:
      serialized: A 1-D string Tensor; a batch of serialized tf.Example protos.
    Returns:
      input1: A SentenceBatch of .
      input2: A SentenceBatch of .
      labels: A SentenceBatch of labels.
    """
    features = tf.parse_example(
        serialized,
        features={
            "input1": tf.FixedLenFeature([], dtype=tf.string),#tf.FixedLenFeature([], tf.string, default_value=''),
            "input2": tf.FixedLenFeature([], dtype=tf.string),# tf.FixedLenFeature([], tf.string, default_value=''),
            "labels": tf.FixedLenFeature([], dtype=tf.int64),# tf.FixedLenFeature([], tf.string, default_value=''),
        })

    print(features)
    output_names = ("input1", "input2", "labels")
    res = tuple(features[x] for x in output_names)

    return res


def prefetch_input_data(reader,
                        file_pattern,
                        shuffle,
                        capacity,
                        num_reader_threads=1):
    """Prefetches string values from disk into an input queue.

    Args:
      reader: Instance of tf.ReaderBase.
      file_pattern: Comma-separated list of file patterns (e.g.
          "/tmp/train_data-?????-of-00100", where '?' acts as a wildcard that
          matches any character).
      shuffle: Boolean; whether to randomly shuffle the input data.
      capacity: Queue capacity (number of records).
      num_reader_threads: Number of reader threads feeding into the queue.

    Returns:
      A Queue containing prefetched string values.
    """
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=shuffle, capacity=16, name="filename_queue")

    if shuffle:
        min_after_dequeue = int(0.6 * capacity)
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            dtypes=[tf.string],
            shapes=[[]],
            name="random_input_queue")
    else:
        values_queue = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[tf.string],
            shapes=[[]],
            name="fifo_input_queue")

    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(
        tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    tf.summary.scalar("queue/%s/fraction_of_%d_full" % (values_queue.name,
                                                        capacity),
                      tf.cast(values_queue.size(), tf.float32) * (1.0 / capacity))

    return values_queue
