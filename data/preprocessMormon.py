from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import requests

fileName = '/Users/ml-cturan/Documents/Corpora/religious/guthenberg/preprocessed/mormon.txt'
f = open(fileName)
data = f.readlines()

saveName = '/Users/ml-cturan/Documents/Corpora/religious/sentenceData/religious/mormon.txt'

#data = data[0:30]
#print(data)
sentListAll = []
sentenceIns = []
i = 0
for line in data:
    if line is '\n':
        if len(sentenceIns) == 0:
            #sentenceIns =[]
            continue
        elif len(sentenceIns[0]) < 20:
            sentenceIns = ' '.join(sentenceIns[1:])
        else:
            sentenceIns = ' '.join(sentenceIns)

        if len(sentenceIns.split()) == 0:
            sentenceIns = []
            continue
        if sentenceIns.split()[0].isdigit():
            lenDigit = len(sentenceIns.split()[0])
            sentenceIns = sentenceIns[lenDigit+1:]

        sentenceList = sent_tokenize(sentenceIns)
        for sent in sentenceList:
            if not (sent is ''):
                # print(repr(sentCheck))
                sentListAll.append(sent)
                # print(sent)
                i += 1
        sentenceIns = []
        continue
    line = line.replace('\n', '')
    line = line.strip()
    sentenceIns.append(line)

print(i)
fS = open(saveName, 'w+')
fS.writelines('\n'.join(sentListAll))
fS.close()