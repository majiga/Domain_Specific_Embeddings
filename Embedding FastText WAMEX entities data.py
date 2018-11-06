# -*- coding: utf-8 -*-
"""
Spyder Editor
Majiga 23/06/2018

Dataset: All entities in a file: train the embedding

"""

## GENSIM 

from gensim.models import FastText

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info('START - FastText Embeddings for entities only file')

# Open file
#FILE_TO_READ = r"/Users/majiga/Documents/wamex/WAMEX_fasttext_entities.txt"

FILE_TO_READ = r"/Users/majiga/Documents/wamex/WAMEX_geological_entities_allfiles.txt"
#FILE_TO_READ = "C:/1ADMA2018-Word Embeddings/WAMEX_geological_entities_allfiles.txt"

FILE_FASTTEXT_VECTORS_ENTITIES = r"/Users/majiga/Documents/wamex/model_fasttext_entities_100freq.model"

sentences = ""
with open(FILE_TO_READ, 'r') as f:
    sentences = f.readlines()
print(len(sentences))   # 31328 files=lines of 28910989 words=tokens

data = []
count_terms = 0
for s in sentences:
    arr = s.split(', ')
    words = []
    for w in arr:
        a = w.strip().replace(' ', '-')
        words.append(a)
    data.append(words)
    count_terms += len(words) # 2,772,122

print(count_terms)

logging.info("START - Train the fastText model")
# Skip gram model FastText model 
# min_count of 100 --- min number of word occurrence
# number of negatives sampled [5]
model_fasttext = FastText(data, size=100, window=5, min_count=100, workers=4, sg=1)
logging.info('END - FastText Embeddings for entities')

# save model
model_fasttext.save(FILE_FASTTEXT_VECTORS_ENTITIES)

print(model_fasttext.wv.vocab)
print(len(model_fasttext.wv.vocab))
# 838 words 100+ freq

model_fasttext.wv.most_similar('gold')
model_fasttext.wv.most_similar('iron-ore')
model_fasttext.wv.most_similar('iron', topn=10)
model_fasttext.wv.most_similar(positive=['kalgoorlie','iron-ore'],negative=['gold'])

model_fasttext.wv.most_similar('kalgoorlie')
print(model_fasttext['gold'])


# LOADING MODEL
model = FastText.load_model(FILE_FASTTEXT_VECTORS_ENTITIES)
print(model['gold'])
print(model.words)

"""
import fasttext

# Open file
FILE_TO_READ = r"/Users/majiga/Documents/wamex/WAMEX_geological_entities_allfiles.txt"
#FILE_TO_READ = "C:/1ADMA2018-Word Embeddings/WAMEX_geological_entities_allfiles.txt"

FILE_FASTTEXT_ENTITIES = r"/Users/majiga/Documents/wamex/WAMEX_fasttext_entities.txt"
FILE_FASTTEXT_MODELS = r"/Users/majiga/Documents/wamex/WAMEX_fasttext_entities.mnodel"


data = ""
with open(FILE_TO_READ, 'r') as f:
    data = f.read()
print(len(data))   # 28,910,989 words=tokens

# Add hyphen - to make words as one word and remove commas,
data_fasttext = []
for s in data.split(','):
    data_fasttext.append(s.strip().replace(' ','-'))

with open(FILE_FASTTEXT_ENTITIES, 'w') as file:
    file.write(' '.join(data_fasttext))


model = fasttext.skipgram(FILE_FASTTEXT_ENTITIES, FILE_FASTTEXT_MODELS)


print(len(model.words))
print(model.words)


model.most_similar('gold')

model.wv.most_similar('iron-ore')

model.wv.most_similar('iron', topn=10)

model.wv.most_similar('kalgoorlie')


model.wv.most_similar(positive=['kalgoorlie','iron-ore'],negative=['gold'])
"""

