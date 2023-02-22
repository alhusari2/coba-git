import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import dbscan
from sklearn.datasets import make_moons
from queue import Queue
# Create Blobs
blobs = make_moons(500, noise=0.055)[0]
# Standardized the data
for x in range(2):
  m = blobs[:,x].mean()
  s = blobs[:,x].std()
  for y in range(len(blobs)):
    blobs[y,x] = (blobs[y,x] - m)/s
# Scikit-Learn DBSCAN
preds = dbscan(X=blobs, eps=0.2, min_samples=5)[1]
dbscan_blob = np.append(blobs, preds.reshape(-1,1), axis=1)
pd.DataFrame(dbscan_blob).plot(x=1, y=0, kind="scatter", c=2, colorbar=False, title= "DBSCAN (eps=0.2, min_points=5)", marker="o", colormap="tab20b")
plt.show()