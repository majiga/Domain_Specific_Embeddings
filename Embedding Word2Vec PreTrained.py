#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:34:03 2018

@author: majiga
"""

from gensim.models import KeyedVectors

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
#model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)
modelg = KeyedVectors.load_word2vec_format(r'C:/WAMEX_project/Vectors/GoogleNews-vectors-negative300.bin', binary=True)

dog = modelg['dog']
print(dog.shape)
#print(dog[:10])

# Deal with an out of dictionary word: Михаил (Michail)
if 'gold' in modelg:
    print(modelg['gold'].shape)
else:
    print('{0} is an out of dictionary word'.format('gold'))

# Some predefined functions that show content related information for given words
print(modelg.most_similar(positive=['king', 'woman'], negative=['man']))

#print(model.doesnt_match("breakfast cereal dinner lunch".split()))

print(modelg.most_similar('iron'))
print(modelg.most_similar('iron_ore'))
print(modelg.most_similar('Kalgoorlie'))

print(modelg.most_similar('western_australia'))
print(modelg.most_similar('Western_Australia'))
print(modelg.most_similar('WESTERN_AUSTRALIA'))


print(modelg.most_similar('gold'))
print(modelg.most_similar('hematite'))
print(modelg.most_similar('knowledge'))


print(modelg.similarity('gold', 'silver'))

print(modelg.most_similar(positive=['Kalgoorlie', 'iron_ore'], negative=['gold']))
print(modelg.most_similar(positive=['hematite', 'iron_ore'], negative=['gold']))


len(modelg.vocab)
print(list(modelg.vocab))

print(modelg.similarity('funny', 'woman'))
print(modelg.similarity('funny', 'man'))
print(modelg.similarity('sad', 'woman'))
print(modelg.similarity('sad', 'man'))

print(modelg.similarity('funny', 'women'))
print(modelg.similarity('funny', 'men'))
print(modelg.similarity('sad', 'women'))
print(modelg.similarity('sad', 'men'))
