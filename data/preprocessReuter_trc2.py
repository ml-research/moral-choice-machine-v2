from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import re
import glob
import os
import pandas as pd
import numpy as np

fileName = '/Users/ml-cturan/Documents/Corpora/Reuter/trc2/headlines-docs.csv'

nameData = 'reut_trc2_'
countFileSave = 1
print(countFileSave)
error_count = 0

fS = open('./News/sentenceData_trc2/' + nameData + str(countFileSave) + '.txt','w+')
for i, df in enumerate(pd.read_csv(fileName, sep=',', header=None, chunksize=1, encoding="ISO-8859-1")):
    #print(i)
    try:
        textRow = df.iat[0,2]
        #print(repr(textRow))

        if pd.isnull(textRow):
           continue

        sentenceList = sent_tokenize(textRow)
        for sent in sentenceList:
            fS.write(sent + '\n')

        if (i + 1) % 10000 == 0:
            fS.close()
            countFileSave += 1
            print(countFileSave)
            fS = open('./News/sentenceData_trc2/' + nameData + str(countFileSave) + '.txt', 'w+')
    except UnicodeDecodeError:
        error_count += 1
        print("UnicodeDecodeError: row ", i)
print("#Errors: ", error_count)
fS.close()
