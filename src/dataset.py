import torch
import torchvision
import torchvision.transforms.v2 as T
from pathlib import Path
from torch.utils.data import ConcatDataset

supported_datasets = ["mnist", "cifar10"]

def load_dataset(ds_config, combine_train_and_val_ds=False):
    assert ds_config.name in supported_datasets, f"Unknown dataset. Supported datasets: {supported_datasets}"

    cache_dir = Path("./dataset-cache") / ds_config.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # https://docs.pytorch.org/vision/0.24/transforms.html#performance-considerations
    transform = T.Compose(
        transforms=[
            T.ToImage(), # [C, H, W]
            T.ToDtype(torch.uint8, scale=True),
            T.Resize(ds_config.img_size),
            T.RandomHorizontalFlip(p=0.5 if ds_config.h_flip_aug else 0.0),
            T.ToDtype(torch.float32, scale=True),
            T.Lambda(lambda x: x*2 - 1) # scale between [-1., 1.]
        ]
    )

    if ds_config.name == "mnist":
        train_ds = torchvision.datasets.MNIST(cache_dir, train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.MNIST(cache_dir, train=False, download=True, transform=transform)
        torch_ds = ConcatDataset([train_ds, val_ds]) if combine_train_and_val_ds else train_ds
    elif ds_config.name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(cache_dir, train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.CIFAR10(cache_dir, train=False, download=True, transform=transform)
        torch_ds = ConcatDataset([train_ds, val_ds]) if combine_train_and_val_ds else train_ds
    return torch_ds