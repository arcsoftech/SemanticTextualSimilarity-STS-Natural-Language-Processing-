from nltk.corpus import wordnet as wn

a=  wn.synsets('stack')
b=  wn.synsets('queue')
sim =[]
for x in a:
    for y in b:
        p=x.wup_similarity(y)
        if p is not None:
            sim.append(p)
# print(sim)
print(sum(sim)/len(sim))

