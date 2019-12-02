import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import json
from tqdm import tqdm
from experiments.dataMoral import *
from nltk.corpus import wordnet
from multiprocessing import Pool
from bs4 import BeautifulSoup ##pip install BeautifulSoup4
import glob
import argparse
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--sentence_path', default=None, type=str,
                    help='data name', required=True)
parser.add_argument('--sentence_years', default=None, type=str,
                    help='data name', required=True)


def get_base_verb(verbs):
    verbsBase = [WordNetLemmatizer().lemmatize(word, 'v') for word in verbs]
    verb_cleaned = []
    for w in verbsBase:
        pos_l = set()
        for tmp in wn.synsets(w):
            if tmp.name().split('.')[0] == w:
                pos_l.add(tmp.pos())
        if "v" in pos_l:
            verb_cleaned.append(w)
    return verb_cleaned


def get_verbs(sent):
    tokens = word_tokenize(sent)
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    tagged_words = nltk.pos_tag(words)
    verbs = [word[0].lower() for word in tagged_words if 'VB' in word[1]]
    return get_base_verb(verbs)


def count_occurrences(textRow):
    verbList = dict()
    sentenceList = sent_tokenize(textRow)
    for sent in sentenceList:
        verbsInSent = get_verbs(sent)

        for verb in verbsInSent:
            if verb in verbList.keys():
                verbList[verb] += 1
            else:
                verbList[verb] = 1
    return verbList


def combine_dicts(dict_list):
    verbList = dict()

    for verbList_tmp in dict_list:
        for key in verbList_tmp.keys():
            if key in verbList.keys():
                verbList[key] += verbList_tmp[key]
            else:
                verbList[key] = verbList_tmp[key]
    return verbList


if __name__ == '__main__':
    args = parser.parse_args()

    sentence_files = glob.glob('{}sentenceData_{}/*.txt'.format(args.sentence_path, args.sentence_years))
    print(len(sentence_files))

    error_count = 0

    # cnt = 0
    textArray = list()
    for file in tqdm(sentence_files):
        try:
            f = open(file, 'r')
            data = f.readlines()

            textRow = ''
            for line in data:
                textRow += line.strip('\n')

            textArray.append(textRow)

            # cnt += 1
            # if cnt == 2:
            #     break
            f.close()
        except UnicodeDecodeError:
            error_count += 1
            print("UnicodeDecodeError: ", file)

    # print(textArray)

    pool = Pool(processes=30)
    output_dict_list = pool.map(count_occurrences, textArray)
    verbList = combine_dicts(output_dict_list)
    # print(output_dict_list)
    print(verbList)

    with open('./News/extracted/extractedVerbs_{}.json'.format(args.sentence_years), 'w') as fp:
        json.dump(verbList, fp)
    print("#Errors: ", error_count)