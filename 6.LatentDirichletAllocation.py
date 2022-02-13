# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 09:09:43 2021

@author: Adrian Ramos
"""
from time import time
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

#from sklearn.datasets import fetch_20newsgroups

def plot_top_words(model, feature_names, n_top_words, title):
    # Malla para graficar 2 filas 5 columnas
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    # model.components_ hace referencia a la matriz phi
    for topic_idx, topic in enumerate(model.components_):
        # La hacemos distribución
        topic /= topic.sum()
        # Seleccionamos los índices más relevantes del tópico
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        # Obtenemos las palabras
        top_features = [feature_names[i] for i in top_features_ind]
        # Obtenemos la ponderación
        weights = topic[top_features_ind]
        
        # Graficamos barras horizontales por tópicos con las palabras más relevantes
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


#%% DATA RETRIEVAL
print("Loading dataset...")
t0 = time()
#data, _ = fetch_20newsgroups(shuffle=True, random_state=1, remove= ('headers', 'footers', 'quotes'), return_X_y=True)
data = pd.read_csv('ag_news.csv')
print("done in %0.3fs." % (time() - t0))



#%% TEXT VECTORIZATION

docs_no, topics_no, words_no = 3000, 10, 1000

vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=words_no, stop_words='english')
# Vectorized documents matrix
corpus = vectorizer.fit_transform(data[:docs_no])


#%% LATENT DIRICHLET ALLOCATION
lda = LatentDirichletAllocation(n_components=topics_no)
lda.fit(corpus)

plot_top_words(lda, vectorizer.get_feature_names(), 10, 'Topics in LDA model')

#%% TEXT CATEGORIZATION
#print(data[53])
#lda.transform(corpus[53])

