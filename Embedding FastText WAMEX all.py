# -*- coding: utf-8 -*-
"""
Created on 21 June 2018

@author: 20230326 Majiga


Dataset: All WAMEX reports: preprocess them and train the embedding
"""

from gensim.models import FastText

import nltk
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer('english')
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

import re
import os, glob, codecs
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info('START - FastText Embeddings for all wamex reports')

WAMEX_DATA_FOLDER = r"/Users/majiga/Documents/wamex/data/wamex_xml"
#r'C:/wamex/data/wamex_xml_5000'

MODEL_FILE = r"/Users/majiga/Documents/wamex/fastText_wamex_all_300freq.model" # bin file
#"C:/1WAMEX/word2vec_wamex_5000.model"

"""
    step one:
    extract keywords per paper from Reports
    text clean -> tokenize -> stem -> tfidf -> keywords. 
"""
"""
# Removes numbers and special characters
# Leave only letters and - hyphen or minus sign
def clean_text(text):
    list_of_cleaning_signs = ['\x0c','\n']
    for sign in list_of_cleaning_signs:
        text = text.replace(sign, ' ')
    clean_text = re.sub('[^a-zA-Z-]+',' ',text)
    return clean_text.lower()

#def remove_stopwords:
    

def remove_words(text):    
    stop_words_list = ['pdf', 'txt', 'ass', 'wasg', 'specifi', 'text', 'fix', 'file', 'report', 'type', 'km', 'annual',
                       'pty', 'ending', 'information']
    
    filtered_words = []
    for word in text.split(' '):
        if (word not in stop_words_list):
                filtered_words.append(word)
    
    text_removed_strangestrings = []
    for w in filtered_words:
        if ((len(w)>2) and (len(w)<20)):
            text_removed_strangestrings.append(w)
    return " ".join(text_removed_strangestrings)

def tokenize_and_lemmatize(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token    
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    # stem
    #stems = [stemmer.stem(t) for t in filtered_tokens]
    #return stems
    # Lemma
    lemmas = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return lemmas
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
    data = data.replace('-', ' ')
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



# READ WAMEX REPORTS FROM THE DATA FOLDER
logging.info('START - Read cleaned wamex reports in ' + WAMEX_DATA_FOLDER)
reports_data = []
for filename in glob.glob(os.path.join(WAMEX_DATA_FOLDER, '*.json')):
    # do your stuff
    if (read_clean_file(filename) is not None):
        reports_data.append(read_clean_file(filename).split())
logging.info('END - Read wamex reports')

#
## READ WAMEX REPORTS FROM THE DATA FOLDER
#logging.info('START - Read wamex reports in ' + WAMEX_DATA_FOLDER)
#reports_data = []
#for filename in glob.glob(os.path.join(WAMEX_DATA_FOLDER, '*.json')):
#    # do your stuff
#    with codecs.open(filename, "r", encoding='utf-8', errors='ignore') as f:
#        data = f.read()
#        if (len(data) > 5):
#            reports_data.append(data)
#logging.info('END - Read wamex reports')


# PRE-PROCESS: CLEAN AND REMOVE STOP WORDS AND NUMBERS # 31328 files are not empty (has more that 5 chars)
#logging.info('START - Clean list of strings that contain wamex reports. Number of reports: ' + str(len(reports_data)))
#reports_data_clean = []
#for d in reports_data:
#    reports_data_clean.append(remove_words(clean_text(d)))
#logging.info('END - Clean list of strings')


#sentences = []
#for r in reports_data_clean:
#    sentences.append(tokenize_and_lemmatize(r))
#
## Count the number of tokens
#count_tokens = 0
#for sent in sentences:
#    count_tokens += len(sent)
#print(count_tokens)
# 49203908 tokens

count_tokens = 0
for data in reports_data:
    count_tokens += len(data)
print(count_tokens)
# 42650553 tokens


logging.info("START - Train the fastText model")
model_fasttext = FastText(reports_data, size=100, window=5, min_count=300, workers=4, sg=1)

# min_count=100 => training on a 246019540 raw words (182291433 effective words) took 817.9s, 222873 effective words/s if 100+ frequency


logging.info('END - FastText Embeddings for all wamex reports')

model_fasttext.wv.most_similar("gold")

# Count the words in the vocabulary
print(len(model_fasttext.wv.vocab)) # 8562

# save model
model_fasttext.save(MODEL_FILE)

# load model
new_model = FastText.load(MODEL_FILE)
print(new_model)

model_fasttext.wv.most_similar('gold')

# model_fasttext.wv.most_similar('kalgoorlie')

model_fasttext.wv.most_similar('iron-ore')

model_fasttext.wv.most_similar('iron', topn=10)


model_fasttext.wv.most_similar(positive=['kalgoorlie','iron-ore'],negative=['gold'])


#model_fasttext.wv.most_similar(positive=['king','woman'],negative=['man'])


new_model.wv['gold']  # numpy vector of a word

words = list(model_fasttext.wv.vocab)
print(words)


"""
for x in model_fasttext.wv.vocab:
    if (('iron' in x.lower()) & ('ore' in x.lower())):
        print (x)
    if (x.lower() == 'kalgoorlie'):
        print (x)
    if (x.lower() == 'gold'):
        print (x)
"""




