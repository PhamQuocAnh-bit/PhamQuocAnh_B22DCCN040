# %%
import pandas as pd 
import numpy as np 

df = pd.read_csv('../part1/results.csv') 
data = df[df.columns[4:]].copy()
data.dropna(axis='columns', inplace=True)


data = ((data - data.min()) / (data.max() - data.min())) * 9 + 1



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output


from sklearn.cluster import KMeans 

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data) 

def optimise_k_means(data_2d, max_k): 
    means = [] 
    inertias = [] 

    for k in range(1, max_k + 1): 
        kmeans = KMeans(n_clusters=k) 
        kmeans.fit(data) 

        means.append(k) 
        inertias.append(kmeans.inertia_) 

    plt.plot(means, inertias, 'o-')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.savefig("../../images/optimise k.png")
    plt.show() 

optimise_k_means(data_2d, 10)