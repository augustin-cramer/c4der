#%%
import pandas as pd
import plotly.express as px
import itertools

#%%

df = pd.read_csv("/Users/g0bel1n/c4der/20220628_treated.csv", index_col=0)
df["timestamps"] = pd.to_datetime(df["timestamps"])

df1 = pd.read_csv("/Users/g0bel1n/c4der/20220627_treated.csv", index_col=0)
df1["timestamps"] = pd.to_datetime(df["timestamps"])
#%%

px.histogram(df.x)

# %%
px.histogram(df.y)

#%%

px.density_contour(x=df.x, y=df.y)
# %%

import numpy as np
from sklearn.decomposition import PCA

#%%
pca = PCA(n_components=2)

x1 = pca.fit_transform(df[["x", "y"]])
# %%
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

p = RandomForestClassifier()
# %%
df
p.fit(X=df.drop(["labels", "timestamps"], axis=1), y=df["labels"])
# %%
p.score(X=df1.drop(["labels", "timestamps"], axis=1), y=df1["labels"])
# %
# %%
count, divisions = pd.np.histogram(df.y, bins=20)
# %%

#%%
frequencies = list(zip(count, itertools.pairwise(divisions)))
# %%
frequencies.sort(key=lambda x: x[0], reverse=True)
# %%
frequencies
# %%
seats = np.array([el for _, el in frequencies[:3]])
# %%
seats = np.mean(seats, axis=1)

#%%

seats
# %%
from scipy.spatial.distance import cdist

# %%
d = cdist(df.y, seats)
# %%
d.shape
# %%
df["mass_center"] = df.y.apply(lambda x: np.argmin(np.abs(seats - x)))
# %%

px.scatter_3d(x=df.y, y=df.x, z=df.timestamps, color=df.mass_center)
# %%
df["mass_center"]
# %%
d
# %%
x1
# %%
px.scatter_3d(x=x1[:, 1], y=x1[:, 0], z=df.timestamps, color=df.labels)

# %%
px.scatter_3d(x=df.y, y=df.x, z=df.timestamps, color=df.labels)

# %%
