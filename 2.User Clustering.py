
# coding: utf-8

# In[ ]:



import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn import manifold, datasets
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans,AgglomerativeClustering,MiniBatchKMeans

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers


# In[ ]:


df = pd.read_csv("data_for_clustering.csv")
df = df.set_index(['MediaGammaImpression'])


# In[ ]:


df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

df_map = {}  # 保存映射关系
cols = df.columns.values
for col in cols:
    if df[col].dtype != np.int64 and df[col].dtype != np.float64:
        temp = {}
        x = 0
        for ele in set(df[col].values.tolist()):
            if ele not in temp:
                temp[ele] = x
                x += 1
 
        df_map[df[col].name] = temp
        df[col] = list(map(lambda val:temp[val], df[col]))
        


# In[ ]:


######################################################################
X = StandardScaler().fit_transform(df)
reduced_X = PCA(n_components=2).fit_transform(X)

######################################################################
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)
######################################################################


# In[ ]:


######################################################################
encoding_dim = 2

# this is our input placeholder
input_dim = Input(shape=(X.shape[1],))

# encoder layers
encoded = Dense(16, activation='relu')(input_dim)
encoded = Dense(8, activation='relu')(encoded)
encoded = Dense(4, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(4, activation='relu')(encoder_output)
decoded = Dense(8, activation='relu')(decoded)
decoded = Dense(16, activation='relu')(decoded)
decoded = Dense(X.shape[1], activation='sigmoid')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_dim, output=decoded)



encoder = Model(input = input_dim, output = encoder_output)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X,
                nb_epoch=100,#100
                batch_size=256,
                shuffle=True)
encoded_X = encoder.predict(X)
######################################################################


# In[ ]:


######################################################################
def clustering_indicator(estimator, name, data):
    estimator.fit(data)
    print(name,
             metrics.calinski_harabaz_score(data, estimator.labels_),  
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',sample_size=None))   
#kMeans
clustering_indicator(KMeans(n_clusters = n_clusters), name="Kmeans", data=X)
#ward 
clustering_indicator(AgglomerativeClustering(linkage='ward', n_clusters=4), name="AggHier", data=X)
#minibatch kmean
clustering_indicator(MiniBatchKMeans(init='k-means++', n_clusters=4, 
                      n_init=10, max_no_improvement=10, verbose=0), name="MiniBatchKMeans", data=X)
#SpectralClustering
clustering_indicator(cluster.SpectralClustering(
        n_clusters=4, eigen_solver='arpack',
        affinity="nearest_neighbors"), name="SpectralClustering", data=X)

#birch
clustering_indicator(cluster.Birch(n_clusters=4), name="SpectralClustering", data=X)
######################################################################


# In[ ]:


def clustering_metrics(data, n_clusters):
    #metrics = pd.DataFrame(columns=['method','chs','ss'])
    Kmeans = cluster.KMeans(n_clusters = n_clusters)
    GMM = mixture.GaussianMixture(covariance_type='full',  n_components= n_clusters)
    Aggcluster = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",
                                                 n_clusters=n_clusters)
    Ward = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    MiniBatchKMeans = cluster.MiniBatchKMeans(init='k-means++', n_clusters = n_clusters, 
                      n_init=10, max_no_improvement=10, verbose=0)
    Spectral = cluster.SpectralClustering(n_clusters= n_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
    Birch = cluster.Birch(n_clusters=n_clusters)
    
    
    clustering_algorithms = (
        ('Kmeans', Kmeans),
        ('GaussianMixture', GMM),
        ('AgglomerativeClustering', Aggcluster),
        ('Ward', Ward),
        ('MiniBatchKMeans', MiniBatchKMeans),
        #('SpectralClustering', Spectral),
        #('Ward', ward),
        ('Birch', Birch))
    

    for name, algorithm in clustering_algorithms:
        algorithm.fit(data)
        if hasattr(algorithm, 'labels_'):
            y_labels = algorithm.labels_.astype(np.int)
        else:
            y_labels = algorithm.predict(data)
            
        print(name, 

                 #estimator.inertia_,  #越小越好
                metrics.calinski_harabaz_score(data, y_labels),  #越大越好
                metrics.silhouette_score(data, y_labels,
                                      metric='euclidean',sample_size=None)) 
    


# In[ ]:


n_clusters = 4
clustering_metrics(X,n_clusters)
clustering_metrics(reduced_X,n_clusters)
clustering_metrics(encoded_X,n_clusters) 


# In[ ]:


tsne = TSNE(n_components=2)
Y_K = tsne.fit_transform(X)
y_R = tsne.fit_transform(reduced_X)
y_E = tsne.fit_transform(encoded_X)

clf = KMeans(n_clusters = n_clusters,random_state = 0)  #max_iter=300,
y = clf.fit_predict(X)
y_ = clf.fit_predict(reduced_X)
y_e = clf.fit_predict(encoded_X)


# In[ ]:



fig = plt.figure(figsize=plt.figaspect(0.25))
plt.subplot(1,3,1)
plt.scatter(Y_K[:, 0], Y_K[:, 1], 20, y, cmap=plt.cm.Spectral)
plt.colorbar(ticks=range(n))
plt.subplot(1,3,2)
plt.scatter(y_R[:, 0], y_R[:, 1], 20, y_, cmap=plt.cm.Spectral)
plt.colorbar(ticks=range(n))
plt.subplot(1,3,3)
plt.scatter(y_E[:, 0], y_E[:, 1], 20, y_e, cmap=plt.cm.Spectral)
plt.colorbar(ticks=range(n))
plt.show()


# In[ ]:


Auto_K = KMeans(n_clusters = 4)
clf = KMeans(n_clusters = 4,random_state = 0) 
y_Auto_K = clf.fit_predict(reduced_X)

cluster = pd.DataFrame(data=y_Auto_K, columns=['cluster'],index=df.index)
result = df.join(cluster)
result['cluster'].value_counts()


# In[ ]:


from jieba.analyse import *
from nltk.corpus import stopwords

def process_text(df):
    df = df.apply(lambda x: x.str.lower()) 
    df = df.apply(lambda x: x.str.replace('[^\w\s]',' '))   

    for column in df.columns:
        if df[column].dtypes == np.object:
            df[column] = df[column].apply(lambda sen:" ".join(x for x in sen.split() if x not in stop))

    return df


#######################################################################
dff = pd.read_csv("group_for_nlp.csv")
dff = dff.set_index(['MediaGammaImpression'])

dff = dff.loc[:, dff.dtypes == np.object].astype(str)
dff = dff.join(result['cluster'])

str_df = dff.groupby('cluster').agg(lambda x: ''.join(set(x)))

data = process_text(str_df)

data.to_csv("data for keyphrase extraction.csv")

