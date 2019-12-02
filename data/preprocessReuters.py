from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import re
import glob
reuterFiles = glob.glob('./News/data'+'//*.*')
print(reuterFiles)
i = 0
for file in reuterFiles:
    nameData = file.replace('/','.').split('.')[4]
    print(nameData)
    #nameData = "reut2-000"
    fS = open('./News/sentenceData/' + nameData + '.txt','w+')
    f = open('./News/data/' + nameData + '.sgm', 'r', errors="ignore")
    data = f.read()
    soup = BeautifulSoup(data,'html.parser')
    contents= soup.findAll('body') # find all body tags
    print(len(contents)) # print number of body tags in sgm file
    pattern = re.compile("reuter", re.IGNORECASE)

    sentListAll = []
    for content in contents:
        sentenceList = sent_tokenize(content.text)
        for sent in sentenceList:
            sent = sent.replace('\n', ' ')
            sent = pattern.sub(" ", sent)
            #sent = sent.replace('reuter', ' ')
            sentCheck = sent.strip()
            sentCheck = sentCheck.replace('\x03','')
            if not (sentCheck is ''):
                #print(repr(sentCheck))
                sentListAll.append(sent)
                i += 1
            #print(repr(sent))
    fS.writelines('\n'.join(sentListAll))
        #print(content.text)

    f.close()
    fS.close()
print(i)