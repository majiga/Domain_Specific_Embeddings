#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 19:55:21 2018

@author: majiga

Dataset: Pre-trained FastText embedding

"""

import io
import numpy as np
import tqdm

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

# Load vectors from file to vectors dictionary
vectors = load_vectors(r"/Users/majiga/Downloads/crawl-300d-2M.vec")

for x in vectors:
    if (('iron' in x.lower()) & ('ore' in x.lower())):
        print (x)
    if (x.lower() == 'kalgoorlie'):
        print (x)
    if (x.lower() == 'gold'):
        print (x)
"""
gold
Gold
GOLD
GOld
gOld
Iron-ore
iron-ore
Kalgoorlie
kalgoorlie
"""


"""
Load Fas

"""


import gensim

#vec_list, vec_word_dict = read_embedding_list (r"/Users/majiga/Downloads/crawl-300d-2M.vec") 

# Load FastText vectors into GenSim model
model = gensim.models.KeyedVectors.load_word2vec_format(r"/Users/majiga/Downloads/crawl-300d-2M.vec")


model.wv.most_similar('gold'\\
                      model.wv.most_similar('Gold')
model.wv.most_similar('GOLD')
model.wv.most_similar('GOld')
model.wv.most_similar('gOld')   

model.wv.most_similar('kalgoorlie')
model.wv.most_similar('Kalgoorlie')

model.wv.most_similar('iron-ore')
model.wv.most_similar('Iron-ore')

model.wv.most_similar('iron', topn=20)


model.wv.most_similar(positive=['kalgoorlie','iron-ore'],negative=['gold'])
model.wv.most_similar(positive=['Kalgoorlie','iron-ore'],negative=['gold'])
model.wv.most_similar(positive=['kalgoorlie','Iron-ore'],negative=['gold'])
model.wv.most_similar(positive=['Kalgoorlie','Iron-ore'],negative=['gold'])


model.wv.most_similar(positive=['king','woman'],negative=['man'])

