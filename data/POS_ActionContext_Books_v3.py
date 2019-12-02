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

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--book_path', default=None, type=str,
                    help='data name', required=True)
parser.add_argument('--book_years', default=None, type=str,
                    help='data name', required=True)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def get_word_list(sent):
    # for w in word_tokenize(sent):
    #     print(get_wordnet_pos(w))
    return [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(sent)]


def getBaseVerb(verbs):
    verbsBase = [WordNetLemmatizer().lemmatize(word, 'v') for word in verbs]
    return verbsBase


def getVerbs(sent):
    text = word_tokenize(sent)
    # print(text)
    tagged_words = nltk.pos_tag(text)
    verbs = [word[0].lower() for word in tagged_words if 'VB' in word[1]]
    return getBaseVerb(verbs)


def count_occurrences(textRow):
    verbList = dict()
    contextList = dict()
    verbContextList = dict()
    sentenceList = sent_tokenize(textRow)

    prevSent = ([], [])
    for sent in sentenceList:
        verbsInSent = getVerbs(sent)
        wordsInSent = get_word_list(sent)

        # check if action in current sent
        for action in actionDic.keys():
            context_belonging_action = actionDic[action]
            if action in verbsInSent:
                if action in verbList.keys():
                    verbList[action] += 1
                else:
                    verbList[action] = 1

                # check if action and any context is occurring in same sentence or prev sentence
                for word in context_belonging_action:
                    if len(word.split(" ")) == 1:
                        if word in wordsInSent + prevSent[1]:
                            # increase counter of occurring at the same time
                            if action + '_' + word in verbContextList.keys():
                                verbContextList[action + '_' + word] += 1
                            else:
                                verbContextList[action + '_' + word] = 1
                    else:
                        wordTuple = word.split(" ")
                        if (wordTuple[0] in wordsInSent + prevSent[1]) and (
                                wordTuple[1] in wordsInSent + prevSent[1]):
                            # increase counter of occurring at the same time
                            if action + '_' + wordTuple[0] + '_' + wordTuple[1] in verbContextList.keys():
                                verbContextList[action + '_' + wordTuple[0] + '_' + wordTuple[1]] += 1
                            else:
                                verbContextList[action + '_' + wordTuple[0] + '_' + wordTuple[1]] = 1

            # check if the context belonging to the action in current sent
            for word in context_belonging_action:
                if len(word.split(" ")) == 1:
                    if word in wordsInSent:
                        if word in contextList.keys():
                            contextList[word] += 1
                        else:
                            contextList[word] = 1
                        # check if the action is in prev sent
                        if action in prevSent[0]:
                            if action + '_' + word in verbContextList.keys():
                                verbContextList[action + '_' + word] += 1
                            else:
                                verbContextList[action + '_' + word] = 1
                else:
                    wordTuple = word.split(" ")
                    if (wordTuple[0] in wordsInSent) and (wordTuple[1] in wordsInSent):
                        if wordTuple[0] + '_' + wordTuple[1] in contextList.keys():
                            contextList[wordTuple[0] + '_' + wordTuple[1]] += 1
                        else:
                            contextList[wordTuple[0] + '_' + wordTuple[1]] = 1
                        # check if the action is in prev sent
                        if action in prevSent[0]:
                            # increase counter of occurring at the same time
                            if action + '_' + wordTuple[0] + '_' + wordTuple[1] in verbContextList.keys():
                                verbContextList[action + '_' + wordTuple[0] + '_' + wordTuple[1]] += 1
                            else:
                                verbContextList[action + '_' + wordTuple[0] + '_' + wordTuple[1]] = 1

        prevSent = (verbsInSent, wordsInSent)
    return verbList, contextList, verbContextList


def combine_dicts(dict_list):
    verbList = dict()
    contextList = dict()
    verbContextList = dict()

    for dict_array in dict_list:
        (verbList_tmp, contextList_tmp, verbContextList_tmp) = dict_array
        for key in verbList_tmp.keys():
            if key in verbList.keys():
                verbList[key] += verbList_tmp[key]
            else:
                verbList[key] = verbList_tmp[key]

        for key in contextList_tmp.keys():
            if key in contextList.keys():
                contextList[key] += contextList_tmp[key]
            else:
                contextList[key] = contextList_tmp[key]

        for key in verbContextList_tmp.keys():
            if key in verbContextList.keys():
                verbContextList[key] += verbContextList_tmp[key]
            else:
                verbContextList[key] = verbContextList_tmp[key]

    return verbList, contextList, verbContextList

if __name__ == '__main__':
    args = parser.parse_args()

    # reuterFiles = glob.glob('/home/cgdmtrn/PycharmProjects/Reuter/rcv1/' + '//*/*.xml')
    reuterFiles = glob.glob('{}sentenceData_{}/*.txt'.format(args.book_path, args.book_years))

    print(len(reuterFiles))

    error_count = 0

    # actionsWithContextIns = [
    #     ("go", "sleep"),
    # ]
    # actionsWithContextIns = [
    #     ("have", "gun kill people")
    # ]
    # actionsWithContextIns = [
    #     ("go", "school"),
    #     ("eat", "bread,animal product")
    # ]
    # actionsWithContextIns = [
    #     ("eat", "animal product")
    # ]
    # actionsWithContextIns = [
    #     ("trust", "machine"),
    #     ("eat", "animal product"),
    #     ("have", "gun hunt animal")
    # ]
    actionsWithContextIns = [
        ("go", "school"),
        ("eat", "animal product,bread")
    ]

    actionVerbs = getBaseVerb([a for a, c in actionsWithContextIns])
    context_tmp = [c for a, c in actionsWithContextIns]

    actionDic = dict()
    for a, c in actionsWithContextIns:
        actionDic[a] = c.split(",")

    print(actionDic)

    cnt = 0
    textArray = list()
    for file in tqdm(reuterFiles):
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

    pool = Pool(processes=15)
    output_dict_list = pool.map(count_occurrences, textArray)
    (verbList, contextList, verbContextList) = combine_dicts(output_dict_list)
    # print(output_dict_list)
    print(verbList)
    print(contextList)
    print(verbContextList)

    with open('./News/extractedNew/extractedActions_{}.json'.format(args.book_years), 'w') as fp:
        json.dump(verbList, fp)
    with open('./News/extractedNew/extractedContext_{}.json'.format(args.book_years), 'w') as fp:
        json.dump(contextList, fp)
    with open('./News/extractedNew/extractedActionContext_{}.json'.format(args.book_years), 'w') as fp:
        json.dump(verbContextList, fp)
    print("#Errors: ", error_count)