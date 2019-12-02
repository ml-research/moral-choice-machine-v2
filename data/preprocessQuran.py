from nltk.tokenize import sent_tokenize

fileName = '/Users/ml-cturan/Documents/Corpora/religious/guthenberg/preprocessed/quran.txt'
f = open(fileName)
data = f.readlines()
#data = data[0:150]
#print(data)
saveName = '/Users/ml-cturan/Documents/Corpora/religious/sentenceData/religious/quran.txt'

flagSkip = False
sentenceIns = []
sentListAll = []
i = 0
for line in data:
    if line is '\n':
        continue
    elif "SURA" in line:
        flagSkip = True

        sentenceIns = ' '.join(sentenceIns)
        sentenceIns = sentenceIns.strip()
        sentenceList = sent_tokenize(sentenceIns)
        for sent in sentenceList:
            if not (sent is ''):
                # print(repr(sentCheck))
                sentListAll.append(sent)
                # print(sent)
                i += 1
        sentenceIns = []
        continue

    if flagSkip:
        flagSkip = False
        continue

    line = line.replace('\n', '')
    line = line.replace('\x93', '')
    line = line.replace('\x92', '')
    line = line.replace('\x94', '')
    line = line.strip()
    sentenceIns.append(line)

#print(sentenceIns)
print(i)

fS = open(saveName, 'w+')
fS.writelines('\n'.join(sentListAll))
fS.close()