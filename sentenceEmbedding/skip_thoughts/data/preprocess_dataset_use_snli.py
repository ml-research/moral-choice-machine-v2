from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf
import pandas as pd

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_dir", None,
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")

tf.flags.DEFINE_string("output_dir", None, "Output directory.")

tf.flags.DEFINE_integer("train_output_shards", 2146,
                        "Number of output shards for the training set.")

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.logging.set_verbosity(tf.logging.INFO)


def _string_feature(value):
    value_bytes = str.encode(value)
    """Helper for creating an string Feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value_bytes]))


def _label_feature(value):
    #['entailment', 'neutral', 'contradiction']
    if value == 'entailment':
        one_hot_idx = 0
    if value == 'neutral':
        one_hot_idx = 1
    if value == 'contradiction':
        one_hot_idx = 2

    """Helper for creating an string Feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=[one_hot_idx]))


def _create_serialized_example(input1, input2, label):
    """Helper for creating a serialized Example proto."""
    input1_str = _string_feature(input1)
    input2_str = _string_feature(input2)
    label_str = _label_feature(label)
    example = tf.train.Example(features=tf.train.Features(feature={
        "input1": input1_str,
        "input2": input2_str,
        "labels": label_str,
    }))
    return example.SerializeToString()


def _process_input_file(filename, stats):
    """Processes the sentences in an input file.

    Args:
      filename: Path to a pre-tokenized input .txt file.
      vocab: A dictionary of word to id.
      stats: A Counter object for statistics.

    Returns:
      processed: A list of serialized Example protos
    """
    tf.logging.info("Processing input file: %s", filename)
    processed = []

    dataset = pd.read_csv(filename, delimiter="\t")
    for sample_idx in range(dataset.shape[0]):
        sample = dataset.iloc[sample_idx]
        stats.update(["sentences_seen"])
        # The first 2 sentences per file will be skipped.

        input1 = sample.sentence1
        input2 = sample.sentence2
        label = sample.gold_label
        stats.update(["sentences_considered"])
        if isinstance(input1, str) and isinstance(input2, str) and label in ['entailment', 'neutral', 'contradiction']:
            serialized = _create_serialized_example(input1, input2, label)
            processed.append(serialized)
            stats.update(["sentences_output"])
        else:
            stats.update(["invalid label"])
    tf.logging.info("Completed processing file %s", filename)
    return processed


def _write_shard(filename, dataset, indices):
    """Writes a TFRecord shard."""
    with tf.python_io.TFRecordWriter(filename) as writer:
        for j in indices:
            writer.write(dataset[j])


def _write_dataset(name, dataset, indices, num_shards):
    """Writes a sharded TFRecord dataset.

    Args:
      name: Name of the dataset (e.g. "train").
      dataset: List of serialized Example protos.
      indices: List of indices of 'dataset' to be written.
      num_shards: The number of output shards.
    """
    tf.logging.info("Writing dataset %s", name)
    borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
    for i in range(num_shards):
        filename = os.path.join(FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                       num_shards))
        shard_indices = indices[borders[i]:borders[i + 1]]
        _write_shard(filename, dataset, shard_indices)
        tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                        borders[i], borders[i + 1], filename)
    tf.logging.info("Finished writing %d sentences in dataset %s.",
                    len(indices), name)


def main(unused_argv):
    print(FLAGS.input_dir)
    if not FLAGS.input_dir:
        raise ValueError("--input_dir or dir is required.")
    if not FLAGS.output_dir:
        raise ValueError("--output_dir is required.")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    tf.logging.info("Found input files.")
    tf.logging.info("Generating dataset.")
    stats = collections.Counter()
    dataset = []

    dataset.extend(_process_input_file(FLAGS.input_dir + 'snli_1.0_test.txt', stats))
    #dataset.extend(_process_input_file(FLAGS.input_dir + 'snli_1.0_dev.txt', stats))
    num_validation_sentences = len(dataset)
    tf.logging.info("Generated dataset with %d sentences.", len(dataset))
    for k, v in stats.items():
        tf.logging.info("%s: %d", k, v)
    dataset.extend(_process_input_file(FLAGS.input_dir + 'snli_1.0_train.txt', stats))

    tf.logging.info("Generated dataset with %d sentences.", len(dataset))
    for k, v in stats.items():
        tf.logging.info("%s: %d", k, v)

    np.random.seed(123)
    shuffled_indices = np.arange(len(dataset))
    val_indices = shuffled_indices[:num_validation_sentences]
    train_indices = shuffled_indices[num_validation_sentences:]
    _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
    _write_dataset("validation", dataset, val_indices,
                   FLAGS.validation_output_shards)


if __name__ == "__main__":
    tf.app.run()
