import shift_kmeans

from sklearn.cluster import KMeans    
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
import numpy as np
import torch

pl.close('all')

nx = 30
ny = 30

x = np.linspace(-1, 1, 30)
y = np.linspace(-1, 1, 30)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()
v = 1.0 * np.sin(np.pi * X) * np.cos(np.pi * Y)

wav = np.linspace(-5, 5, 100)
y1 = np.exp(-(wav[None, :] - v[:, None])**2 * 2.0)
y2 = np.exp(-(wav[None, :] - v[:, None])**2 * 3.0) + 0.5 * np.exp(-((wav[None, :] - v[:, None]) - 2)**2 * 2.0)

y = np.concatenate([y1[0:450, :], y2[450:, :]], axis=0)

k = 2

kmeans_np = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(y)

y = torch.tensor(y.astype('float32'))
centroids, labels, vout = shift_kmeans.shift_kmeans(y, k, max_iters=250, lr=10.0, gpu=0, infer_v=True)

limited_colors = pl.cm.tab10(np.linspace(0, k/10.0-0.1, k))
limited_cmap = ListedColormap(limited_colors)

fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(12, 6))
for i in range(k):
    ax[0].plot(centroids[i, :].cpu().numpy(), label=f'Centroid {i}')
ax[0].legend()
ax[0].set_title('KMeans Centroids (PyTorch)')

for i in range(k):
    ax[1].plot(kmeans_np.cluster_centers_[i, :], label=f'Cluster {i}')
ax[1].set_title('KMeans Centroids (Scikit-learn)')

fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(12, 12))
im = ax[0, 0].imshow(labels.reshape(nx, ny).cpu().numpy(), cmap=limited_cmap)
pl.colorbar(im, ax=ax[0, 0])
ax[0, 0].set_title('KMeans Labels (PyTorch)')

im = ax[0, 1].imshow(kmeans_np.labels_.reshape(nx, ny), cmap=limited_cmap)
pl.colorbar(im, ax=ax[0, 1])
ax[0, 1].set_title('KMeans Labels (Scikit-learn)')

im = ax[1, 0].imshow(vout.reshape(nx, ny).cpu().numpy())
pl.colorbar(im, ax=ax[1, 0])    
ax[1, 0].set_title('Subpixel Shift v')

im = ax[1, 1].imshow(v.reshape(nx, ny))
pl.colorbar(im, ax=ax[1, 1])
ax[1, 1].set_title('v')