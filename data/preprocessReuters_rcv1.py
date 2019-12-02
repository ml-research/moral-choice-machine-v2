from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import re
import glob
import os

sentListAll = []
nameData = 'reut_rcv1_'
countFileSave = 1

reuterFiles = glob.glob('/Users/ml-cturan/Documents/Corpora/Reuter/rcv1/rcv1/' + '//*/*.xml')
#reuterFiles = glob.glob('./News/data'+'//*.*')
print(len(reuterFiles))

fS = open('./News/sentenceData_rcv1/' + nameData + str(countFileSave) + '.txt','w+')
error_count = 0
for i, file in enumerate(reuterFiles):
    try:
        print(i)
        f = open(file,'r')
        data = f.read()
        soup = BeautifulSoup(data, 'html.parser')
        contents = soup.findAll('p')  # find all body tags
        #print(len(contents))  # print number of body tags in sgm file

        for content in contents:
            sentenceList = sent_tokenize(content.text)
            for sent in sentenceList:
                pass
                #fS.write(sent + '\n')

        f.close()
        if (i+1) % 10000 == 0:
            fS.close()
            countFileSave += 1
            fS = open('./News/sentenceData_rcv1/' + nameData + str(countFileSave) + '.txt', 'w+')
    except UnicodeDecodeError:
        error_count += 1
        print("UnicodeDecodeError: ", file)
print("#Errors: ", error_count)
1/0


f = open("/Users/ml-cturan/Documents/Corpora/Reuter/rcv1/rcv1/19970222/396114newsML.xml", 'r')
data = f.read()
soup = BeautifulSoup(data,'html.parser')
contents= soup.findAll('p') # find all body tags
print(len(contents)) # print number of body tags in sgm file


for content in contents:
    sentenceList = sent_tokenize(content.text)
    for sent in sentenceList:
        fS.write(sent + '\n')
        #print('**'+repr(sent))


1/0
for root, dirs, files in os.walk("/Users/ml-cturan/Documents/Corpora/Reuter/rcv1/rcv1", topdown=False):
    print(root)

1/0

reuterFolders = [x[0] for x in os.walk("/Users/ml-cturan/Documents/Corpora/Reuter/rcv1/rcv1")]
print(reuterFolders)
