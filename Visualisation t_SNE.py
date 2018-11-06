# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:44:46 2018

@author: 20230326
"""
import gensim
from sklearn.manifold import TSNE


"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
"""

model = gensim.models.Word2Vec.load(r"/Users/majiga/Documents/wamex_Word2Vec_CSG_min_freq_300.model")

#model = gensim.models.Word2Vec.load('C:\\1WAMEX\\word2vec_cooccurrence_graph.model')
model.wv['gold']  # raw NumPy vector of a word


print(len(model.wv.vocab))


# HIERARCHICAL CLUSTERING
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

#l = linkage(model.wv.vectors, method='complete', metric='seuclidean')
l = linkage(model.wv.syn0, method='complete', metric='cosine')
#l = linkage(model.wv.syn0, method='ward', metric='euclidean')

# calculate full dendrogram
plt.figure(figsize=(300, 40))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('word')
plt.xlabel('distance')

dendrogram(
    l,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    orientation='top',
    leaf_label_func=lambda v: str(model.wv.index2word[v])
)
plt.show()


"""
# In[]
# GET THE GEOLOGICAL TERMS FROM DICTIONARY FILES
FOLDER_TO_PROCESS = "C:\1WAMEX\\"

FILE_VOCABULARY = "Geological_Vocabulary.json"
geological_vocabulary = []
with open(FILE_VOCABULARY, 'r') as f:
    text = f.read()
    text = text.lower()
    geological_vocabulary = json.loads(text)
print(geological_vocabulary)
"""

# In[]
# DIMENSIONALITY REDUCTION T-SNE
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        #if (word in geological_vocabulary):   # choose the words in the domain vocabulary
        tokens.append(model.wv[word])
        labels.append(word)
    
    #tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23, early_exaggeration=20, color='label')
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23, early_exaggeration=20)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(100, 100)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)

# In[]

model['gold']

model.most_similar('gold')
model.most_similar('iron ore')
model.most_similar('iron')

model.most_similar('nickel')
model.most_similar('copper')
model.most_similar('silver')

model.most_similar('capricorn')
model.most_similar('kalgoorlie')
model.most_similar('kimberley')
model.most_similar('hamersley')


model.similarity("gold","iron")

model.similarity("lake","rock")

model.most_similar(positive=['gold','kalgoorlie'],negative=['iron ore'])
model.most_similar(positive=['kalgoorlie', 'gold'],negative=['iron ore'])

model.most_similar(positive=['iron','iron formation'],negative=['gold'])

model.most_similar(positive=['iron','iron ore'],negative=['diamond'])

model.most_similar(positive=['komatiite', 'nickel'],negative=['gold'])
model.most_similar(positive=['komatiite', 'nickel'],negative=['iron ore'])

#model.doesnt_match("duncan gold scotland banquo".split())





"""
#the vector dictionary of the model
word2vec_dict={}
for i in model.wv.vocab.keys():
    try:
        word2vec_dict[i]=model[i]
    except:    
        pass

#This is also interesting to try with Ward Hierarchical Clustering
clusters = MiniBatchKMeans(n_clusters=10, max_iter=10,batch_size=200,
                        n_init=1,init_size=2000)
X = np.array([i for i in word2vec_dict.items()])
y = [i for i in word2vec_dict.iterkeys()]
clusters.fit(X)

from collections import defaultdict
cluster_dict=defaultdict(list)
for word,label in zip(y,clusters.labels_):
    cluster_dict[label].append(word)
"""