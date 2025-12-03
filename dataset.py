# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class RawCachingImageFolder(Dataset):
    def __init__(self, root, num_workers=8):
        raw = datasets.ImageFolder(root)
        self.samples = raw.samples
        self.classes = raw.classes
        self.class_to_idx = raw.class_to_idx

        self._pil_cache = {}

        print(f"Caching {len(self.samples)} images with {num_workers} workers…")
        with ThreadPoolExecutor(num_workers) as ex:
            for path, pil in tqdm(ex.map(_load, self.samples),total=len(self.samples)):
                self._pil_cache[path] = pil

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = self._pil_cache[path]
        return pil, label


class AugmentedDataset(Dataset):
    def __init__(self, raw_dataset, transform, times=1):
        self.raw_ds = raw_dataset
        self.transform = transform
        self.times = times

    def __len__(self):
        return len(self.raw_ds) * self.times

    def __getitem__(self, idx):
        real_idx = idx % len(self.raw_ds)
        img, label = self.raw_ds[real_idx]

        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        img_tensor = augmented["image"]

        return img_tensor, label


class DeviceDataLoader:
    def __init__(self, dl, dev):
        self.dl = dl
        self.dev = dev

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for x, y in batches:
            yield x.to(self.dev), y.to(self.dev)


def _load(path_label):
    path,_ = path_label
    return path, Image.open(path).convert("RGB")
