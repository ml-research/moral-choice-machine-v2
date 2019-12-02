from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import re
import glob
import os
from tqdm import tqdm
from data.books.detect_language import detect_language
import argparse

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--data_dir', default=None, type=str,
                    help='data name', required=True)
parser.add_argument('--out_dir', default=None, type=str,
                    help='data name', required=True)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    args = parser.parse_args()
    sentListAll = []
    countSent = 0

    bookFiles = glob.glob(args.data_dir + '//*/*.txt')
    # reuterFiles = glob.glob('./News/data'+'//*.*')
    print(len(bookFiles))
    ensure_dir(args.out_dir)

    for file in tqdm(bookFiles):
        nameFile = file.split('/')
        # print(nameFile[8])

        f = open(file, 'r')
        content = f.read()
        language = detect_language(content)
        if language != "english":
            continue
        sentenceList = sent_tokenize(content)

        fS = open(os.path.join(args.out_dir, nameFile[-1]), 'w+')

        for sent in sentenceList:
            countSent += 1
            # print(sent)
            fS.write(sent + '\n')
        fS.close()
        f.close()

    print(countSent)
