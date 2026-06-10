import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple
import torchvision
import torchvision.transforms as transforms


def dirichlet_partition(dataset, num_clients, beta, num_classes=10, seed=42):
    rng = np.random.default_rng(seed)
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets.numpy()
        else:
            labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    class_indices = {k: np.where(labels == k)[0] for k in range(num_classes)}
    client_indices = [[] for _ in range(num_clients)]
    client_class_counts = {i: np.zeros(num_classes, dtype=np.int64) for i in range(num_clients)}
    for k in range(num_classes):
        indices_k = class_indices[k].copy()
        rng.shuffle(indices_k)
        proportions = rng.dirichlet(np.ones(num_clients) * beta)
        counts = (proportions * len(indices_k)).astype(int)
        remainder = len(indices_k) - counts.sum()
        if remainder > 0:
            extra = rng.choice(num_clients, size=remainder, replace=False)
            for c in extra:
                counts[c] += 1
        start = 0
        for cid in range(num_clients):
            end = start + counts[cid]
            client_indices[cid].extend(indices_k[start:end].tolist())
            client_class_counts[cid][k] = counts[cid]
            start = end
    client_datasets = [Subset(dataset, idx) for idx in client_indices]
    return client_datasets, client_class_counts


def get_client_classes(client_class_counts):
    return {cid: np.where(counts > 0)[0].tolist() for cid, counts in client_class_counts.items()}


def load_dataset(name, data_dir="./data", train=True):
    if name.upper() == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        return torchvision.datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    elif name.upper() == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        return torchvision.datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")
