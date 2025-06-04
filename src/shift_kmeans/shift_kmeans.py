import torch
import torch.optim as optim
import torch.nn.functional as F

def subpixel_shift_1d_batched_grid_sample(input_tensor, s):
    """
    Shifts a batched 1D tensor by a subpixel distance 's' using grid_sample.

    Args:
        input_tensor (torch.Tensor): Batched 1D tensor of shape (B, N).
        s (torch.Tensor or float): Subpixel shift distance(s).

    Returns:
        torch.Tensor: Shifted batched 1D tensor of shape (B, N).
    """
    B, N = input_tensor.shape

    # 1. Reshape to (batch_size, num_channels, height, width)
    input_tensor_reshaped = input_tensor.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)

    # 2. Create the grid
    # Output x-coordinates (normalized to [-1, 1])
    x_out = torch.linspace(-1, 1, N, device=input_tensor.device)

    # Handle both scalar and per-batch shifts
    if isinstance(s, float):
        shift_tensor = torch.full((B,), s, device=input_tensor.device)
    elif isinstance(s, torch.Tensor) and s.ndim == 1 and s.shape[0] == B:
        shift_tensor = s
    else:
        raise ValueError("Shift 's' must be a float or a 1D tensor with batch size.")

    # Normalize the shift tensor for grid_sample
    normalized_shift = shift_tensor * 2 / (N - 1) if N > 1 else shift_tensor * 0

    # Create the grid (batch_size, height, width, num_spatial_dims=2)
    # Output y-coordinates (constant at 0, normalized)
    y_out = torch.zeros(N, device=input_tensor.device)
    x_out_batched = x_out.unsqueeze(0).repeat(B, 1)  # (B, N)
    x_in = x_out_batched - normalized_shift.unsqueeze(1)  # (B, N)

    # Stack x and y coordinates, then reshape for grid_sample
    grid = torch.stack([x_in, y_out.unsqueeze(0).repeat(B, 1)], dim=-1).unsqueeze(1) # (B, 1, N, 2)

    # 3. Use grid_sample
    shifted_tensor = F.grid_sample(input_tensor_reshaped, grid, mode='bilinear', padding_mode='reflection', align_corners=True)

    # Reshape back to (B, N)
    return shifted_tensor.squeeze(1).squeeze(1)


def kmeans(data, k, max_iters=100, gpu=-1):
    """
    Performs standard k-means clustering using Euclidean distance.

    Args:
        data (torch.Tensor): The input data tensor (n_samples, n_features).
        k (int): The number of clusters.
        max_iters (int): Maximum number of iterations.

    Returns:
        tuple: A tuple containing:
            - centroids (torch.Tensor): The final cluster centroids (k, n_features).
            - labels (torch.Tensor): The cluster assignment for each data point (n_samples).
    """

    device = torch.device("cpu")
    if torch.cuda.is_available():
        if gpu == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{gpu}")

    data = data.to(device)

    n_samples, n_features = data.shape

    # Initialize centroids randomly from the data
    indices = torch.randperm(n_samples)[:k]
    centroids = data[indices].clone().float().to(device)

    for _ in range(max_iters):
        # Assign each data point to the closest centroid based on Euclidean distance
        distances = torch.cdist(data.float(), centroids, p=2)  # (n_samples, k)
        labels = torch.argmin(distances, dim=1)  # (n_samples)

        # Update centroids as the mean of the data points in each cluster (vectorized)
        # Create a mask where mask[i, j] is 1 if data[i] belongs to cluster j
        cluster_mask = (labels.unsqueeze(1) == torch.arange(k).to(device)).float()  # (n_samples, k)

        # Calculate the number of points in each cluster
        cluster_counts = torch.sum(cluster_mask, dim=0, keepdim=True).transpose(0, 1)  # (k, 1)

        # Calculate the sum of data points for each cluster
        new_centroids_sum = torch.matmul(cluster_mask.transpose(0, 1), data)  # (k, n_features)

        # Handle empty clusters by re-initializing their centroids
        empty_clusters = (cluster_counts == 0)
        if torch.any(empty_clusters):
            print(f"Warning: Empty clusters detected at iteration {iteration + 1}. Re-initializing.")
            random_indices = torch.randperm(n_samples)[:torch.sum(empty_clusters)].to(device)
            new_centroids_sum[empty_clusters.squeeze()] = data[random_indices].float().sum(dim=0)
            cluster_counts[empty_clusters] = 1.0

        new_centroids = new_centroids_sum / (cluster_counts + 1e-8)

        # Check for convergence
        if torch.all(new_centroids == centroids):
            break
        centroids = new_centroids

        loss = torch.sum((data - new_centroids[labels]) ** 2)
        print(f"Iteration {_ + 1}, Loss: {loss.item()}")

        

    return centroids, labels

def shift_kmeans(data, k, max_iters=100, lr=0.1, gpu=-1, infer_v=True, init='random'):
    """
    Performs shift-independent k-means clustering using Euclidean distance

    Args:
        data (torch.Tensor): The input data tensor (n_samples, n_features).
        k (int): The number of clusters.
        max_iters (int): Maximum number of iterations.
        lr (float): Initial learning rate for the optimizer (best results are obtained with large values)
        gpu (int): GPU index to use (-1 for CPU).
        infer_v (bool): If True, the subpixel shift 'v' is optimized during clustering.
        init (str): Initialization method for centroids ('random' or 'kmeans++').

    Returns:
        tuple: A tuple containing:
            - centroids (torch.Tensor): The final cluster centroids (k, n_features).
            - labels (torch.Tensor): The cluster assignment for each data point (n_samples).
    """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if gpu == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{gpu}")

    data = data.to(device)

    n_samples, n_features = data.shape

    if init == 'random':
        indices = torch.randperm(n_samples)[:k]
        centroids = data[indices].clone()

    if init == 'kmeans++':

        # kmeans++ initialization
        centroids = torch.zeros(k, n_features, dtype=data.dtype, device=device)

        # 1. Choose the first centroid randomly from the data points
        first_centroid_index = torch.randint(0, n_samples, (1,), device=device)
        centroids[0] = data[first_centroid_index].clone()

        for i in range(1, k):
            # 2. Calculate the squared distance between each data point and the nearest existing centroid
            distances = torch.cdist(data, centroids[:i], p=2)  # (n_samples, i)
            min_distances_sq = torch.min(distances, dim=1)[0] ** 2  # (n_samples,)

            # 3. Choose the next centroid with probability proportional to the squared distance
            probabilities = min_distances_sq / torch.sum(min_distances_sq)
            cumulative_probabilities = torch.cumsum(probabilities, dim=0)
            random_value = torch.rand((1,), device=device)

            # Find the index where the random value falls in the cumulative probabilities
            next_centroid_index = torch.searchsorted(cumulative_probabilities, random_value).item()

            # Select the next centroid
            centroids[i] = data[next_centroid_index].clone()

    # Initialize the subpixel shift vector v
    v = torch.zeros(n_samples, dtype=torch.float32, requires_grad=True, device=device)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW([v], lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    for _ in range(max_iters):
        

        optimizer.zero_grad()

        # Shift data according to the subpixel shift v
        stokesi_shift = subpixel_shift_1d_batched_grid_sample(data, v)
        
        distances = torch.cdist(stokesi_shift, centroids, p=2)  # (n_samples, k)
        labels = torch.argmin(distances, dim=1)  # (n_samples)

        # Recompute centroids based on the shifted data and not backpropagate through the process
        with torch.no_grad():

            # Update centroids as the mean of the data points in each cluster (vectorized)
            # Create a mask where mask[i, j] is 1 if data[i] belongs to cluster j
            cluster_mask = (labels.unsqueeze(1) == torch.arange(k).to(device)).float()  # (n_samples, k)

            # Calculate the number of points in each cluster
            cluster_counts = torch.sum(cluster_mask, dim=0, keepdim=True).transpose(0, 1)  # (k, 1)

            # Calculate the sum of data points for each cluster
            new_centroids_sum = torch.matmul(cluster_mask.transpose(0, 1), stokesi_shift)  # (k, n_features)

            # Handle empty clusters by re-initializing their centroids
            empty_clusters = (cluster_counts == 0)
            if torch.any(empty_clusters):
                print(f"Warning: Empty clusters detected at iteration {iteration + 1}. Re-initializing.")
                random_indices = torch.randperm(n_samples)[:torch.sum(empty_clusters)].to(device)
                new_centroids_sum[empty_clusters.squeeze()] = stokesi_shift[random_indices].float().sum(dim=0)
                cluster_counts[empty_clusters] = 1.0

            new_centroids = new_centroids_sum / (cluster_counts + 1e-8)
        
        # Compute the loss as the sum of squared distances to the centroids
        loss = torch.mean((stokesi_shift - new_centroids[labels]) ** 2)
        
        if infer_v:
            loss.backward()
            optimizer.step()
            scheduler.step()

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        print(f"Iteration {_ + 1}, Loss: {loss.item()} - lr: {current_lr} - v: {torch.min(v):.4f} to {torch.max(v):.4f}")

        # Check for convergence
        if torch.all(new_centroids == centroids):
            break
        centroids = new_centroids

    centroids = new_centroids.detach()
    v = v.detach()  # Detach v to avoid accumulating gradients
    
    return centroids, labels, v

if __name__ == '__main__':

    from sklearn.cluster import KMeans    
    import matplotlib.pyplot as pl
    from matplotlib.colors import ListedColormap
    import numpy as np
    
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
    centroids, labels, vout = shift_kmeans(y, k, max_iters=250, lr=10.0, gpu=0, infer_v=True)

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