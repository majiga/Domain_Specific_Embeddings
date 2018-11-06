# -*- coding: utf-8 -*-
"""
23 June 2018

Dataset: wamex_xml folder / All WAMEX reports
Word2Vec training on pre-processed dataset
@author: Majiga
"""

import re
import codecs
import glob, os
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info('START - FastText Embeddings for all wamex reports')


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from stop_words import get_stop_words
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = list(get_stop_words('en'))         # 174 stopwords
nltk_words = list(stopwords.words('english'))   # 153 stopwords
stop_words.extend(nltk_words)                   # 353 in total

from nltk.stem.wordnet import WordNetLemmatizer
wordnet = WordNetLemmatizer()

WAMEX_DATA_FOLDER = r"/Users/majiga/Documents/wamex/data/wamex_xml/"

MODEL_FILE = r"/Users/majiga/Documents/wamex/word2vec_wamex_all_300freq.model" # bin file

"""
# Read *.txt file
def get_vocabulary_from_TXT_file(txt_filename):    
    with open(txt_filename, 'r') as f:
        data = f.read()
        data = data.lower()
        data = data.replace('-',' ')        
        file_terms = data.split('\n')
        terms_list = [item.strip() for item in file_terms]
    terms = list(filter(None, terms_list))
    return terms
"""


def tokenize_and_lemmatize(input_text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token    
    tokens = [word for sent in nltk.sent_tokenize(input_text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z-]', token):
            filtered_tokens.append(token)
    # Lemma
    lemmas = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return ' '.join(lemmas)


"""
Read a txt file and return sentences
"""
def read_clean_file(filename):
       
    with codecs.open(filename, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
    if (len(data) < 10):
        return None
    
    
    data_cleaned = [] 
    # remove stop words
    data = data.lower()
    #data = data.replace('-', ' ')
    data = data.replace(',', ' ')
    data = data.replace('\\', ' ')
    data = data.replace('/', ' ')
    
    for w in data.split():
        if (w not in stop_words):
            data_cleaned.append(w)
    
    #print('CLEAN DATA')
    #print(' '.join(data_cleaned))
    
    # lemmatize words in each sentences
    data_lemmatized = tokenize_and_lemmatize(' '.join(data_cleaned))
    
    return data_lemmatized

"""
def string_found(string_find, string_long):
    if re.search(r"\b" + re.escape(string_find) + r"\b", string_long):
        #print(string_find + " : " + string_long)
        return True
    return False


def save_text_only(text, filename):
    
    open(filename, 'w').close()
            
    # Write sentences annotated with Geological Terms to txt file
    with codecs.open(filename, "w",encoding='utf-8', errors='ignore') as fp:
        fp.write(text)
"""


# READ WAMEX REPORTS FROM THE DATA FOLDER
logging.info('START - Read cleaned wamex reports in ' + WAMEX_DATA_FOLDER)
reports_data = []
for filename in glob.glob(os.path.join(WAMEX_DATA_FOLDER, '*.json')):
    # do your stuff
    if (read_clean_file(filename) is not None):
        reports_data.append(read_clean_file(filename).split())
logging.info('END - Read wamex reports')


# PRE-PROCESS: CLEAN AND REMOVE STOP WORDS AND NUMBERS # 31328 files are not empty (has more that 5 chars)
#logging.info('START - Clean list of strings that contain wamex reports. Number of reports: ' + str(len(reports_data)))
#reports_data_clean = []
#for d in reports_data:
#    reports_data_clean.append(remove_words(clean_text(d)))
#logging.info('END - Clean list of strings')


#sentences = []
#for r in reports_data:
#    sentences.append(tokenize_and_lemmatize(r))

# Count the number of tokens
count_tokens = 0
for data in reports_data:
    count_tokens += len(data)
print(count_tokens)
# 42650553 tokens / 42279106 - without removing hyphens

logging.info("START - Train the word2vec model")
# The training params are: embedding size: 300, negative samples: 5, window_size: 5,
#                       starting learning rate: 0.025 (a dynamic learning rate was used)
#                       minimum frequency: 10 (only word occurs more than 10 times were considered).
# class gensim.models.word2vec.Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5,
#                           max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001,
#                           sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>, iter=5,
#                           null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=())
word2vec_model = gensim.models.Word2Vec(reports_data, window=5, size=100, workers=4, min_count=300, sg=1) 

print(len(word2vec_model.wv.vocab)) #  8730 tokens
print(word2vec_model.wv.vocab)

# min_count=100 => training on a 246019540 raw words (182291433 effective words) took 817.9s, 222873 effective words/s if 100+ frequency


logging.info('END - word2vec Embeddings for all wamex reports')

word2vec_model.save(MODEL_FILE)

# load model
new_model = gensim.models.Word2Vec.load(MODEL_FILE)
print(new_model)

word2vec_model.wv.most_similar('gold')

#word2vec_model.wv.most_similar('kalgoorlie')

word2vec_model.wv.most_similar('iron ore')
word2vec_model.wv.most_similar('iron-ore')

word2vec_model.wv.most_similar('iron', topn=10)


word2vec_model.wv.most_similar(positive=['kalgoorlie','iron-ore'],negative=['gold'])


word2vec_model.wv.most_similar(positive=['king','woman'],negative=['man'])





