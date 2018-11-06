#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:13:06 2018

@author: majiga

Here we create Word2Vec model
Dataset: WAMEX entities only file
"""

import gensim


FILE_TO_READ = r"/Users/majiga/Documents/wamex/WAMEX_geological_entities_allfiles.txt"
#FILE_TO_READ = "C:/wamex/data/WAMEX_geological_entities_allfiles.txt"

with open(FILE_TO_READ) as f:
    lines = f.readlines()

#tokens = [word for sent in nltk.sent_tokenize(input_text) for word in nltk.word_tokenize(sent)]

# A sentence here contains entities of a wamex report separated by comma
sentences = []
count = 0 # counts tokens
count_entities = 0
# A line represents a wamex report here
for line in lines:
    tokens = []
    for term in line.split(','):    # entities are separated with comma
        term_stripped = term.strip()
        if (term_stripped == ''):
            continue
        tokens.append(term_stripped.replace(' ', '_'))
        count += len(term_stripped.split())
        count_entities += 1
    sentences.append(tokens)

print(str(count_entities))
#print(sentences[0][0])
    
# remove whitespace characters like `\n` at the end of each line
#reports = [x.strip() for x in reports] 

#word2vec_model = gensim.models.Word2Vec(report2kw, size=100, window=5, min_count=20, workers=4)
# The training params are: embedding size: 300, negative samples: 5, window_size: 5,
#                       starting learning rate: 0.025 (a dynamic learning rate was used)
#                       minimum frequency: 10 (only word occurs more than 10 times were considered).
# class gensim.models.word2vec.Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5,
#                           max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001,
#                           sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>, iter=5,
#                           null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=())
word2vec_model_SG = gensim.models.Word2Vec(sentences, window=5, size=100, workers=4, min_count=100, sg=1) 
#print(list(word2vec_model_SG.wv.vocab))
print(len(word2vec_model_SG.wv.vocab))  # 837 100+freq

word2vec_model_SG.wv.most_similar('gold')
word2vec_model_SG.wv.most_similar('iron_ore')
word2vec_model_SG.wv.most_similar('iron', topn=10)
word2vec_model_SG.wv.most_similar(positive=['kalgoorlie','iron_ore'],negative=['gold'])

#word2vec_model.wv.most_similar('kalgoorlie')

# min_count = 30
#word2vec_model_SG_30 = gensim.models.Word2Vec(sentences, window=5, size=100, workers=4, min_count=30, sg=1) 
#print(len(word2vec_model_SG.wv.vocab))


word2vec_model_SG.save(r"/Users/majiga/Documents/wamex_Word2Vec_CSG_1freq.model")
#word2vec_model_SG.save("C:/wamex/data/wamex_Word2Vec_CSG_min_freq_30.model")

#word2vec_model_BOW = gensim.models.Word2Vec(sentences, window=5, size=100, workers=4, min_count=300) 
#print(list(word2vec_model_BOW.wv.vocab))
#print(len(word2vec_model_BOW.wv.vocab))
#word2vec_model_BOW.save(r"/Users/majiga/Documents/wamex_Word2Vec_CBOW_min_freq_300.model")
#word2vec_model_SG.save("C:/wamex/data/wamex_Word2Vec_CBOW_min_freq_30.model")





model = gensim.models.Word2Vec.load(r"/Users/majiga/Documents/wamex_Word2Vec_CSG_min_freq_300.model")

#model = gensim.models.Word2Vec.load(r"/Users/majiga/Documents/wamex_Word2Vec_CBOW_min_freq_300.model")

from scipy.spatial import distance

def euclidian_distance(term1, term2):
    a = model.wv[term1]
    b = model.wv[term1]
    return distance.euclidean(a,b) 
    

# Similarity tasks
model.wv.most_similar('gold', topn=10)
print(str(euclidian_distance('gold', 'kalgoorlie')))


model.wv.most_similar('silver', topn=5)
model.wv.most_similar('iron', topn=10)
model.wv.most_similar('iron ore', topn=10)
model.wv.most_similar('banded iron formation', topn=10)
model.wv.most_similar('nickel', topn=10)

model.wv.most_similar('greenstone belt', topn=10)
model.wv.most_similar('diamond', topn=5)
model.wv.most_similar('volcanic rock', topn=5)



model.wv.most_similar('newman', topn=5)
model.wv.most_similar('pilbara', topn=5)
model.wv.most_similar('kalgoorlie', topn=5)
model.wv.most_similar('karratha', topn=5)
model.wv.most_similar('jimblebar', topn=5)
model.wv.most_similar('kimberley', topn=5)

model.wv.most_similar('tom price', topn=5)
model.wv.most_similar('tom price', topn=5)

# Analogy tasks

#print(model.most_similar(positive=['woman', 'king'], negative=['man']))

model.wv.most_similar(positive=['kalgoorlie','iron ore'],negative=['gold'])
model.wv.most_similar(positive=['hematite', 'iron ore'], negative=['gold'])

model.wv.most_similar(positive=['iron', 'iron ore'],negative=['gold'])
model.wv.most_similar(positive=['gold', 'iron ore'],negative=['iron'])

model.wv.most_similar(positive=['gold','iron ore'],negative=['hematite'])

model.wv.most_similar(positive=['gold','greenstone belt'],negative=['iron ore'])


model.wv.most_similar(positive=['komatiite', 'nickel'],negative=['gold'], topn = 5)
model.wv.most_similar(positive=['nickel','komatiite'],negative=['iron'], topn = 5)

model.wv.most_similar(positive=['gold','telluride'],negative=['iron'])
model.wv.most_similar(positive=['iron ore', 'hamersley'],negative=['gold'])

model.wv.most_similar(positive=['nickel','iron ore'],negative=['hematite'])
model.wv.most_similar(positive=['hematite', 'iron_ore'], negative=['gold'])

model['gold'] 

model.wv.similarity("gold","kalgoorlie")
model.wv.similarity("gold","iron")
model.wv.similarity("gold","iron ore")


#word2vec_model.doesnt_match("duncan macbeth scotland banquo".split())
