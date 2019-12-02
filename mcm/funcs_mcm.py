from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import sentenceEmbedding.skip_thoughts.configuration_use as configuration
import sentenceEmbedding.skip_thoughts.skip_thoughts_model_use as skip_thoughts_model

from scipy import spatial
from mcm.util_use import chunks
import numpy as np
import logging
import os
from tqdm import tqdm


def get_sen_embedding_from(network, checkpoint_path=None):
    if network == "use_hub":
        return model_use_hub.get_sen_embedding
    elif "use" in network:
        model_use_skipthoughts.checkpoint_path = checkpoint_path
        return model_use_skipthoughts.get_sen_embedding
    else:
        raise ValueError("embedding network not supported. Options are use_hub, use_rcv1, ..., bert")


class USE_SkipThoughts:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.model = None
        self.checkpoint_path = ''

    def _load_embedding(self):
        model_config = configuration.model_config()
        self.model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="encode")

    def get_sen_embedding(self, messages):
        tf.logging.info("Building training graph.")
        g = tf.Graph()

        # checkpoint_path = '/home/deepspacebim/PycharmProjects/data/train_rcv1/'
        with g.as_default():
            self._load_embedding()
            self.model.build()
            saver = tf.train.Saver()

            if tf.gfile.IsDirectory(self.checkpoint_path):
                latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                if not latest_checkpoint:
                    raise ValueError("No checkpoint file found in: %s" % self.checkpoint_path)
                checkpoint_path = latest_checkpoint
            else:
                checkpoint_path = self.checkpoint_path

            def _restore_fn(sess):
                tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
                saver.restore(sess, checkpoint_path)
                tf.logging.info("Successfully loaded checkpoint: %s",
                                os.path.basename(checkpoint_path))

            # model.build_debug()
            with tf.Session() as sess:
                sess.run(tf.tables_initializer())
                _restore_fn(sess)

                feed_dict = {
                    "messages_to_encode:0": messages,
                }

                ret = sess.run("thought_vectors:0", feed_dict=feed_dict)
                sess.close()
        return ret.tolist()


class USE_Hub:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.embed = None
        self.session = None

    def _load_embedding(self):
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

    def get_sen_embedding(self, messages):
        if self.embed is None:
            self._load_embedding()
            self.session = tf.Session()
            self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # Set logging output.
        tf.logging.set_verbosity(tf.logging.INFO)
        # Import the Universal Sentence Encoder's TF Hub module
        logging.info('Import done!')

        ret = []
        # with tf.Session() as session:
        #    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #    message_embeddings = session.run(self.embed(messages))
        #
        #    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        #        ret.append(message_embedding)
        #    session.close()
        # logging.info('get_sen_embedding --- end')

        message_embeddings = self.session.run(self.embed(messages))
        #
        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            ret.append(message_embedding)
        return ret


model_use_skipthoughts = USE_SkipThoughts()
model_use_hub = USE_Hub()


def best_and_worst_question(biases, overall_bias):
    diffs = [(abs(x[0] - overall_bias), x[1]) for x in biases]
    diffs_high_and_low = [(x[0], x[1]) for x in biases]
    diffs.sort(key=lambda x: x[0])
    diffs_high_and_low.sort(key=lambda x: -x[0])

    # if biases[0][1] == "Is it ok to smile?":
    #    print("Overall bias", overall_bias)
    #    print(biases)
    #    print("Best and worst fitting question", (diffs[0], diffs[-1]))
    #    print("Highest and lowest question", (diffs_high_and_low[0], diffs_high_and_low[-1]))

    return [(diffs[0], diffs[-1]), (diffs_high_and_low[0], diffs_high_and_low[-1])]


def mcm_template_quests(template, insert, network, checkpoint_path=None):
    dataMCM = list([x[0].format(i), x[1], x[2]] for i in insert for x in template)
    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in dataMCM:
        merged.extend(elem)

    get_sen_embedding_ = get_sen_embedding_from(network, checkpoint_path)
    res = get_sen_embedding_(merged)
    # print(res[:10])

    embed_list = chunks(res, 3)
    ret = []

    for i, line in tqdm(enumerate(dataMCM)):
        embedding = embed_list[i]

        q_a1_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))
        q_a2_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[2]))

        bias = q_a2_dist - q_a1_dist

        ret.append([bias, line[0], line[1], line[2]])

    d = ret
    res = []
    for chunk in chunks(d, n=len(template)):  # n = number of sentences in the template
        biases = [x[0] for x in chunk]  # get only biases
        questions = [x[1] for x in chunk]
        overall_bias = (round(np.mean(biases), 4))

        best_and_worst = best_and_worst_question(chunk, overall_bias)

        res.append([overall_bias, questions, best_and_worst])

    j = 0
    for i in insert:
        overall_bias = res[j][0]
        questions = res[j][1]
        best_and_worst_quest = res[j][2]
        res[j] = [overall_bias, i, best_and_worst_quest]

        if False in [i in q for q in questions]:
            logging.error('mcm_overall --- wrong value considered for overall bias')
        j += 1
    # logging.info('mcm_overall --- end')
    return res


def mcm_template_quests_biasDistiance(template, insert, network, checkpoint_path=None):
    dataMCM = list([x[0].format(i), x[1], x[2]] for i in insert for x in template)
    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in dataMCM:
        merged.extend(elem)

    get_sen_embedding_ = get_sen_embedding_from(network, checkpoint_path)
    res = get_sen_embedding_(merged)
    # print(res[:10])

    embed_list = chunks(res, 3)
    ret = []

    for i, line in tqdm(enumerate(dataMCM)):
        embedding = embed_list[i]

        q_a1_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))
        q_a2_dist = 1 - spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[2]))

        bias = q_a2_dist - q_a1_dist

        if bias > 0:
            ret.append([np.array(embedding[0]) - np.array(embedding[1]), line[0], line[1], line[2]])
        else:
            ret.append([np.array(embedding[0]) - np.array(embedding[2]), line[0], line[1], line[2]])

    d = ret
    res = []
    for chunk in chunks(d, n=len(template)):  # n = number of sentences in the template
        biases = [x[0] for x in chunk]  # get only biases
        questions = [x[1] for x in chunk]
        overall_bias = np.mean(biases, axis=0)

        # print(np.array(biases).shape)
        # print(np.array(overall_bias).shape)

        res.append([overall_bias, questions, None])

    j = 0
    for i in insert:
        overall_bias = res[j][0]
        questions = res[j][1]
        res[j] = overall_bias

        if False in [i in q for q in questions]:
            logging.error('mcm_overall --- wrong value considered for overall bias')
        j += 1
    # logging.info('mcm_overall --- end')
    return res


def mcm_words_quests(actions, words, checkpoint_path, network="use_hub"):
    data = list()

    for i, e_actions in enumerate(actions):
        data += list([x, 'Should I ' + e_actions] for x in words)

    # merge strings to single list (to compute all embeddings with a single session)
    merged = []
    for elem in data:
        merged.extend(elem)

    # calculate embeddings and split in lists of the form [q,a,a] again
    get_sen_embedding_ = get_sen_embedding_from(network, checkpoint_path)

    embed_list = chunks(get_sen_embedding_(merged), 2)

    ret = []

    for i, line in enumerate(data):
        embedding = embed_list[i]

        q_a1_dist = spatial.distance.cosine(np.array(embedding[0]), np.array(embedding[1]))

        bias = q_a1_dist

        ret.append([bias, line[0], line[1]])

    ret_chunks = chunks(ret, len(words))

    return ret_chunks


def normalizeBias3(bias_list, maxData=None, minData=None):
    bias_list = [[float(x[0]) / abs(maxData) if float(x[0]) >= 0 else float(x[0]) / abs(minData), x[1]] for x in
                 bias_list]

    return bias_list


def getThresholdMean(biasList):
    biasList_ = np.array(biasList, dtype=np.float32)
    threshold = np.mean(biasList_)
    print("Threshold:", threshold)
    return threshold


def getThresholdMedian(biasList):
    biasList_ = np.array(biasList, dtype=np.float32)
    threshold = biasList_[int(len(biasList_) // 2)]
    print("Threshold:", threshold)
    return threshold
