# shift-kmeans

## Overview

`shift-kmeans` is a Python implementation of a k-means clustering algorithm with a shift 
mechanism to improve cluster assignments. Every sample will have an associated shift that
will be inferred by the algorithm. This shift is used to improve the clusterization 
abilities of kmeans.

## Features

- Includes a shift mechanism for refining cluster centers.
- Supports standard k-means clustering.


## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/aasensio/shift-kmeans.git
cd shift-kmeans
pip install -r requirements.txt
```

## Usage

To use it, provided you have data with dimensions (n_samples, n_features), just call the 

```python

from kmeans_shift import kmeans_shift

centroids, labels, vout = kmeans_shift(y, k, max_iters=250, lr=10.0, gpu=0, infer_v=True)
```

It returns the `centroids` with the features of the `k` clusters, the `labels` of each 
sample in the training set, and the shift `vout` of each sample.

## Parameters

- `data (torch.Tensor)`: The input data tensor (n_samples, n_features).
- `k (int)`: The number of clusters.
- `max_iters (int)`: Maximum number of iterations.
- `lr (float)`: Initial learning rate for the optimizer (best results are obtained with large values)
- `gpu (int)`: GPU index to use (-1 for CPU).
- `infer_v (bool)`: If True, the subpixel shift 'v' is optimized during clustering.
- `init (str)`: Initialization method for centroids (`random` or `kmeans++`).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please submit issues or pull requests to improve the project.