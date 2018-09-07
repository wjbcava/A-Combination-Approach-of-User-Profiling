
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

from jieba.analyse import *
from nltk.corpus import stopwords
import pke
import scipy
import nltk
import networkx
import sklearn
import unidecode
import future

from matplotlib.ticker import NullFormatter
from sklearn import metrics
from sklearn import preprocessing 
from sklearn import manifold, datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale,StandardScaler
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA, KernelPCA
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers


# In[ ]:


def process_text(df):
    df = df.apply(lambda x: x.str.lower()) 
    df = df.apply(lambda x: x.str.replace('[^\w\s]',' '))   

    for column in df.columns:
        if df[column].dtypes == np.object:
            df[column] = df[column].apply(lambda sen:" ".join(x for x in sen.split() if x not in stop))

    return df


# In[ ]:


Auto_K = KMeans(n_clusters = 4)
clf = KMeans(n_clusters = 4,random_state = 0) 
y_Auto_K = clf.fit_predict(reduced_X)

cluster = pd.DataFrame(data=y_Auto_K, columns=['cluster'],index=df.index)
result = df.join(cluster)
result['cluster'].value_counts()
dff = pd.read_csv("group_for_nlp.csv")
dff = dff.set_index(['MediaGammaImpression'])

dff = dff.loc[:, dff.dtypes == np.object].astype(str)
dff = dff.join(result['cluster'])

str_df = dff.groupby('cluster').agg(lambda x: ''.join(set(x)))

data = process_text(str_df)

data.to_csv("data for keyphrase extraction.csv")
data = pd.read_csv("data for keyphrase extraction.csv")


# In[ ]:


def p_website(data):
    if "petspyjama com travel" in data[0]:
        print("www.petspyjamas.com/travel/dog-friendly/")
    if "petspyjama com product" or "petspyjama com product" in data[0]  :
        print("www.petspyjamas.com/pet-accessories/")
    else:
        print ("www.petspyjamas.com")


# In[ ]:



extractor = pke.unsupervised.TfIdf()
extractor = pke.unsupervised.SingleRank()
extractor = pke.unsupervised.TopicRank()
extractor = pke.unsupervised.MultipartiteRank()
# load the content of the document.
input_text = dataset['href'][0]
input_text = dataset['href'][1]
input_text = dataset['href'][2]
input_text = dataset['href'][3]

extractor.read_text(input_text)

extractor.candidate_selection()
extractor.candidate_weighting()
#  get the 50 highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=50)
keyphrase, score = zip(*keyphrases)
p_website(keyphrase)


# In[ ]:


stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['petspyjamasshop','petspyjamasdog','petspyjamasdog friendli','friendli',
                'dog petspyjamasshop','dog petspyjamasdog','dog petspyjamasdog',
                'har dog petspyjamasshop','har dog petspyjamasdog','petspyjamasshop dog',
                'petspyjamasshop collar','petspyjamaswelcom','pet','peopl petspyjamasdog',
                'peopl petspyjamasdog friendli','petspyjamaspetspyjama','result',
                'petspyjamaspetspyjama travel','travel search result','petspyjamassign',
                'petspyjamascheckout','petspyjamasmi','dog','friendli','cat','search',
                'gor','peopl','holiday','housewelcom','har','petspyjamasproduct','hors',
                'petspyjamascomfort','petspyjamascock','petspyjamasregistr','foam',
                'petspyjamaschoos','petspyjamasl','petspyjamasdusti','petspyjamasmemori',
                'payment','le','petspyjamasflanno','petspyjamasmini','petspyjamashom',
                'address','puppi','place','hot','accessori','customis','cavali','cockerel', 
                'fontain'  ,'franc','tan', 'petspyjamaswag','petspyjamasrattan','cockapoo',
                'deliveri','petspyjamasdusti','petspyjamasmemori','petspyjamascollap',
                'petspyjamasoatm','petspyjamasskul','petspyjamasmaritim','petspyjamaslean',
                'petspyjamasknotti','petspyjamasplush','burgundi','rac'               
               ]
stopwords.extend(newStopWords)


# In[ ]:


def p_product(data):   
    filtered = [w for w in data if (w not in stopwords)]
    
    for word in filtered[0:20]:
        wordss = word.split()
        words = [w for w in wordss if(w not in stopwords)]
        
    for i in range(len(words)):
        
        elif  words[i]=='cottag':
            words[i]='cottage'
        elif  words[i]=='accessori':
            words[i]=='accessory'
        elif  words[i]=='wale':
            words[i]=='Wales'
        elif  words[i]=='cornwal':
            words[i]=='cornwall'
        elif  words[i]=='friendli':
            words[i]='friendly'
        elif words[i]=='bow':
            words[i]=='bowl'
            
    print(' '.join(words))


# In[ ]:



extractor = pke.unsupervised.TfIdf()
extractor = pke.unsupervised.SingleRank()
extractor = pke.unsupervised.TopicRank()
extractor = pke.unsupervised.MultipartiteRank()

input_text = dataset['document.title'][0]
input_text = dataset['document.title'][1]
input_text = dataset['document.title'][2]
input_text = dataset['document.title'][3]

extractor.read_text(input_text)

extractor.candidate_selection()
extractor.candidate_weighting()

keyphrases = extractor.get_n_best(n=50)
keyphrase, score = zip(*keyphrases)
p_product(keyphrase)

