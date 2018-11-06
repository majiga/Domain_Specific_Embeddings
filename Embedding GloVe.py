#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:08:31 2018

@author: majiga
""" 
 
from glove import Corpus, Glove

FILE_TO_READ = r"/Users/majiga/Documents/WAMEX_geological_entities_allfiles.txt"

with open(FILE_TO_READ) as f:
    sentences = f.readlines()
    #data = f.read()
    #print(len(data))


#sentences = list(itertools.islice(Text8Corpus('text8'),None))
 
corpus = Corpus()
 
corpus.fit(sentences, window=5)
 
glove = Glove(no_components=100, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
#Performing 30 training epochs with 4 threads

glove.most_similar('man')


glove.most_similar('queen', number=10)