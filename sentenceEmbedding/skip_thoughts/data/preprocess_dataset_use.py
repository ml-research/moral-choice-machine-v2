from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf
import glob

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_dir", None,
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")

tf.flags.DEFINE_string("input_files", None,
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")

tf.flags.DEFINE_string("output_dir", None, "Output directory.")

tf.flags.DEFINE_integer("train_output_shards", 83879,
                        "Number of output shards for the training set.") #39072

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("num_validation_sentences", 2385866,
                        "Number of output shards for the validation set.") #1205820

tf.flags.DEFINE_integer("max_sentences", 0,
                        "If > 0, the maximum number of sentences to output.")

tf.flags.DEFINE_integer("max_sentence_length", 0,
                        "If > 0, exclude sentences whose encode, decode_pre OR"
                        "decode_post sentence exceeds this length.")

tf.logging.set_verbosity(tf.logging.INFO)


def _string_feature(value):
    value_bytes = str.encode(value)
    """Helper for creating an string Feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value_bytes]))


def _create_serialized_example(predecessor, current, successor):
    """Helper for creating a serialized Example proto."""
    predecessor_str = _string_feature(predecessor)
    current_str = _string_feature(current)
    successor_str = _string_feature(successor)
    example = tf.train.Example(features=tf.train.Features(feature={
        "sent_pre": predecessor_str,
        "sent": current_str,
        "sent_post": successor_str,
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

    predecessor = None  # Predecessor sentence (list of words).
    current = None  # Current sentence (list of words).
    successor = None  # Successor sentence (list of words).

    for successor_str in tf.gfile.FastGFile(filename):
        stats.update(["sentences_seen"])
        successor = successor_str#.split()
        # The first 2 sentences per file will be skipped.
        if predecessor and current and successor:
            stats.update(["sentences_considered"])

            # Note that we are going to insert <EOS> later, so we only allow
            # sentences with strictly less than max_sentence_length to pass.
            if FLAGS.max_sentence_length and (
                    len(predecessor) >= FLAGS.max_sentence_length or len(current) >=
                    FLAGS.max_sentence_length or len(successor) >=
                    FLAGS.max_sentence_length):
                stats.update(["sentences_too_long"])
            else:
                serialized = _create_serialized_example(predecessor, current, successor)
                processed.append(serialized)
                stats.update(["sentences_output"])

        predecessor = current
        current = successor

        sentences_seen = stats["sentences_seen"]
        sentences_output = stats["sentences_output"]
        if sentences_seen and sentences_seen % 100000 == 0:
            tf.logging.info("Processed %d sentences (%d output)", sentences_seen,
                            sentences_output)
        if FLAGS.max_sentences and sentences_output >= FLAGS.max_sentences:
            break

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
    if not FLAGS.input_files and not FLAGS.input_dir:
        raise ValueError("--input_files or dir is required.")
    if not FLAGS.output_dir:
        raise ValueError("--output_dir is required.")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    if FLAGS.input_files:
        for pattern in FLAGS.input_files.split(","):
            match = tf.gfile.Glob(FLAGS.input_files)
            if not match:
                raise ValueError("Found no files matching %s" % pattern)
            input_files.extend(match)
    else:
        for input_dir in FLAGS.input_dir.split(","):
            for filename in glob.glob(input_dir + '*.txt'):
                input_files.extend([filename])
    tf.logging.info("Found %d input files.", len(input_files))
    tf.logging.info("Generating dataset.")
    stats = collections.Counter()
    dataset = []
    for filename in input_files:
        dataset.extend(_process_input_file(filename, stats))
        if FLAGS.max_sentences and stats["sentences_output"] >= FLAGS.max_sentences:
            break

    tf.logging.info("Generated dataset with %d sentences.", len(dataset))
    for k, v in stats.items():
        tf.logging.info("%s: %d", k, v)

    tf.logging.info("Shuffling dataset.")
    np.random.seed(123)
    shuffled_indices = np.random.permutation(len(dataset))
    val_indices = shuffled_indices[:FLAGS.num_validation_sentences]
    train_indices = shuffled_indices[FLAGS.num_validation_sentences:]

    _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
    _write_dataset("validation", dataset, val_indices,
                   FLAGS.validation_output_shards)


if __name__ == "__main__":
    tf.app.run()
