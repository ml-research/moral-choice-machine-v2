from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import requests

fileName = '/Users/ml-cturan/Documents/Corpora/religious/guthenberg/preprocessed/kingjamesbible.txt'
f = open(fileName)
data = f.readlines()
#data = data[0:100]
#print(data)
saveName = '/Users/ml-cturan/Documents/Corpora/religious/sentenceData/religious/bible.txt'


sentListAll = []
sentenceIns = []
i = 0
for line in data:
    if line is '\n':
        if len(sentenceIns) == 0:
            #sentenceIns = []
            continue
        else:
            sentenceIns = ' '.join(sentenceIns)
        sentenceIns = sentenceIns.strip()
        if sentenceIns.split()[0].split(':')[0].isdigit():
            #print(sentenceIns.split()[0].split(':')[1])
            lenDigit = len(sentenceIns.split()[0])
            sentenceIns = sentenceIns[lenDigit+1:]
        if len(sentenceIns.split(':')) > 1:
            lenSent = len(sentenceIns)
            start = 0
            while True:
                indices = sentenceIns.find(':',start,lenSent)
                if indices == -1:
                    break
                else:
                    if sentenceIns[indices-1].isdigit():
                        sentenceInsNew = sentenceIns[0:indices-2] + sentenceIns[indices+3:]
                        sentenceIns = sentenceInsNew
                    else:
                        start = indices + 1

        #print(sentenceIns)
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
#print(sentListAll)

fS = open(saveName, 'w+')
fS.writelines('\n'.join(sentListAll))
fS.close()