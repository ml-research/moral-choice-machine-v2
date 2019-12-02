import json
from nltk.corpus import wordnet as wn

verbList = []
save = True

with open('extractedVerbs_trc2.json') as json_file:
    data = json.load(json_file)
    for i, verb in enumerate(data):
        verbList.append(verb)

print(len(verbList))

pos_all = dict()
for i, w in enumerate(verbList):
    pos_l = set()
    for tmp in wn.synsets(w):
        if tmp.name().split('.')[0] == w:
            pos_l.add(tmp.pos())
    if 'v' in pos_l:
        pos_all[w] = pos_l

verbListRevised = [*pos_all]
print(len(verbListRevised))
#print(verbListRevised)

if save:
    with open('verbList_trc2.txt', 'w') as f:
        for item in verbListRevised:
            f.write("%s\n" % item)