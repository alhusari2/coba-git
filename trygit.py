import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Wholesalecustomersdata.csv')
from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='average'))
plt.axhline(y=0.6, color='r', linestyle='--')
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
HC=cluster.fit_predict(data_scaled)
print("Hasil Clustering")
plt.show()