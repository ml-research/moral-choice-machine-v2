from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import requests

listName = '/Users/ml-cturan/Documents/Corpora/religious/linkList.txt'
savePath = '/Users/ml-cturan/Documents/Corpora/religious/Constitution/{}'
f = open(listName)
linkList = f.readlines()
#print(linkList)
i = 0
for link in linkList:
    link = link.replace('\n', ' ')
    print(link)
    linkIns = link.split('/')
    saveName = savePath.format(linkIns[-1].split('?')[0] + '.txt')
    fS = open(saveName, 'w+')

    r = requests.get(link)
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    contents = soup.findAll('p')  # find all body tags

    print(len(contents))
    sentListAll = []
    for content in contents:
        # print(content.text)
        sentenceList = sent_tokenize(content.text)
        for sent in sentenceList:
            sent = sent.replace('\n', ' ')
            sent = sent.strip()
            sentCheck2 = sent.split('.')[0]
            if not (sent is '') and not (sentCheck2.isdigit()):
                # print(repr(sentCheck))
                sentListAll.append(sent)
                #print(sent)
                i += 1
    fS.writelines('\n'.join(sentListAll))
    fS.close()

print('total sentence is ' + str(i))
