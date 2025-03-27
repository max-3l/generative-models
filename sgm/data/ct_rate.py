from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fcntl
import hashlib
import multiprocessing
import pickle
import traceback
from typing import Any, Iterable, Literal, Optional
import zipfile
from einops import rearrange
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import default_collate
import tqdm

from sgm.util import instantiate_from_config

class TestDataset(Dataset):
    def __init__(self, dataset_size: int, dimension: int, text_dimension: int, tokens_per_text: int):
        self.dataset_size = dataset_size
        self.dimension = dimension
        self.text_dimension = text_dimension
        self.tokens_per_text = tokens_per_text

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index):
        ct = torch.randn(1, self.dimension, self.dimension, self.dimension)
        xray = torch.randn(2, 1, self.dimension, self.dimension)
        text = torch.randn(self.tokens_per_text, self.text_dimension)
        return {
                "ct": ct,
                "xrays": xray,
                "text": text,
                "scan_id": [str(index)]
                }

from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

class CachedXCT_H5_dataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str, optimize_zip: bool, *args, **kwargs):
        base_dataset = XCT_H5_dataset
        self.optimize_zip = optimize_zip
        self.cache_dir = Path(cache_dir)
        self.cache_hash = self._compute_hash(base_dataset, args, kwargs)
        self.cache_path = self.cache_dir / self.cache_hash
        print("Dataset cache path:", self.cache_path)
        self.status_file = self.cache_path / "_status.txt"
        self.check_and_build_cache(base_dataset, args, kwargs)
        self.length = self.compute_length()

    def _compute_hash(self, base_dataset, args, kwargs):
        hash_input = str(base_dataset) + str(args) + str(kwargs)
        if self.optimize_zip:
            hash_input += "zip"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _build_cache(self, base_dataset, args, kwargs):
        self.cache_path.mkdir(parents=True, exist_ok=True)
        dataset = base_dataset(*args, **kwargs)

        def process_item(idx):
            item = dataset[idx]
            item_path = self.cache_path / f"{idx}"
            if self.optimize_zip:
                with zipfile.ZipFile(item_path.with_suffix(".zip"), 'w', compression=zipfile.ZIP_DEFLATED) as z:
                    with z.open('data.pkl', 'w') as f:
                        pickle.dump(item, f)
            else:
                with open(item_path.with_suffix(".pkl"), 'wb') as f:
                    pickle.dump(item, f)

        with ThreadPoolExecutor() as executor:
            list(tqdm.tqdm(executor.map(process_item, range(len(dataset))), total=len(dataset), desc=f"Building cache {self.cache_path}"))

        with open(self.status_file, 'w') as f:
            f.write("done")
        with open(self.cache_path / "_hash.txt", 'w') as f:
            f.write(self.cache_hash)
        with open(self.cache_path / "_args.txt", 'w') as f:
            f.write(str(args))
        with open(self.cache_path / "_kwargs.txt", 'w') as f:
            f.write(str(kwargs))

    def check_and_build_cache(self, base_dataset, args, kwargs):
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True, exist_ok=True)
        lock_file = self.cache_path / "_lockfile"
        with open(lock_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if not self.status_file.exists():
                self._build_cache(base_dataset, args, kwargs)
            fcntl.flock(f, fcntl.LOCK_UN)

    def compute_length(self):
        if self.optimize_zip:
            return len(list(self.cache_path.glob("*.zip")))
        return len(list(self.cache_path.glob("*.pkl")))

    def __getitem__(self, idx):
        if self.optimize_zip:
            item_path = self.cache_path / f"{idx}.zip"
            with zipfile.ZipFile(item_path, 'r') as z:
                with z.open('data.pkl', 'r') as f:
                    item = pickle.load(f)
        else:
            item_path = self.cache_path / f"{idx}.pkl"
            with open(item_path, 'rb') as f:
                item = pickle.load(f)
        return item

    def __len__(self):
        return self.length

class Dataloader(LightningDataModule):
    def __init__(self,
        train: DictConfig,
        validation: DictConfig,
    ):
        super().__init__()
        self.train = train
        self.validation = validation

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = instantiate_from_config(self.train.dataset)
        self.validation_dataset = instantiate_from_config(self.validation.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.batch_size, shuffle=True, num_workers=self.train.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.batch_size, shuffle=False, num_workers=self.validation.num_workers)

class IndividualXrayWrapper(Dataset):
    def __init__(self, dataset: Dataset, num_projections: int):
        self.dataset = dataset
        self.num_projections = num_projections

    def __len__(self):
        return len(self.dataset) * self.num_projections

    def __getitem__(self, idx):
        dataset_item = self.dataset[idx // self.num_projections]
        projection_idx = idx % self.num_projections
        return dataset_item | { "projection_idx": projection_idx, "xray": dataset_item["xrays"][projection_idx] }

class IndividualCTSliceWrapper(Dataset):
    def __init__(self, dataset: Dataset, num_slices: int):
        self.dataset = dataset
        self.num_slices = num_slices

    def __len__(self):
        return len(self.dataset) * self.num_slices

    def __getitem__(self, idx):
        dataset_item = self.dataset[idx // self.num_slices]
        slice_idx = idx % self.num_slices
        dataset = dataset_item | { "slice_idx": slice_idx, "ct_slice": dataset_item["ct"][..., slice_idx] }
        return dataset

class CTVideoWrapper(Dataset):
    def __init__(self, dataset: Dataset, train=True):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset_item = self.dataset[idx]
        ct_video = rearrange(dataset_item["ct"], "c h w d -> d c h w")
        # assert ct_video.shape == torch.Size((256, 1, 256, 256)), f"CT video shape is {ct_video.shape}"
        xrays = dataset_item["xrays"]
        cond_aug = torch.tensor([1e-5]) # 1e-5
        xrays_noisy = torch.randn_like(xrays) * cond_aug + xrays
        return dataset_item | {
            "ct_video": ct_video,
            "xrays_noisy": xrays_noisy,
            "cond_aug": cond_aug.repeat(ct_video.shape[0]),
            "image_only_indicator": torch.zeros(ct_video.shape[0]),
        }

class ZAsDepthWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset_item = self.dataset[idx]
        return dataset_item | { "ct_z_depth": rearrange(dataset_item["ct"], "c h w d -> c d h w").squeeze(0) }

class IndividualXrayDataloader(LightningDataModule):
    def __init__(self,
        num_projections: int,
        train: DictConfig,
        validation: DictConfig,
    ):
        super().__init__()
        self.train = train
        self.validation = validation
        self.num_projections = num_projections

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = IndividualXrayWrapper(instantiate_from_config(self.train.dataset), num_projections=self.num_projections)
        self.validation_dataset = IndividualXrayWrapper(instantiate_from_config(self.validation.dataset), num_projections=self.num_projections)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=True, num_workers=self.train.loader.num_workers, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=False, num_workers=self.validation.loader.num_workers)

class IndividualCtSliceDataloader(LightningDataModule):
    def __init__(self,
        num_slices: int,
        train: DictConfig,
        validation: DictConfig,
    ):
        super().__init__()
        self.train = train
        self.validation = validation
        self.num_slices = num_slices

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = IndividualCTSliceWrapper(instantiate_from_config(self.train.dataset), num_slices=self.num_slices)
        self.validation_dataset = IndividualCTSliceWrapper(instantiate_from_config(self.validation.dataset), num_slices=self.num_slices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

class CTAsDepthDataloader(LightningDataModule):
    def __init__(self,
        train: DictConfig,
        validation: DictConfig,
    ):
        super().__init__()
        self.train = train
        self.validation = validation

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = ZAsDepthWrapper(instantiate_from_config(self.train.dataset))
        self.validation_dataset = ZAsDepthWrapper(instantiate_from_config(self.validation.dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

class CT128Dataloader(LightningDataModule):
    def __init__(self,
        train: DictConfig,
        validation: DictConfig,
        dtype: str,
    ):
        super().__init__()
        self.train = train
        self.validation = validation
        self.dtype = dtype

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = instantiate_from_config(self.train.dataset)
        self.validation_dataset = instantiate_from_config(self.validation.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)


class IndividualCtSliceDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 16,
        shuffle: bool = True,
        distribution: Literal["normal", "uniform"] = "normal",
        sample_dim_size: int = 128,
        prefetch_factor = 1,
        num_workers = 5
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = instantiate_from_config(dataset) if isinstance(dataset, DictConfig) else dataset
        if distribution == "normal":
            self.distribution = scipy.stats.norm.pdf(list(range(0,sample_dim_size)), sample_dim_size//2, sample_dim_size//2)
            self.distribution = self.distribution + ((1 - self.distribution.sum()) / len(self.distribution))
        elif distribution == "uniform":
            self.distribution = torch.ones(sample_dim_size) / sample_dim_size
        else:
            raise NotImplementedError(f"Slice sampling distribution {distribution} is not implemented.")

        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers

    def __getitem__(self, index: int) -> dict:
        data = self.dataset[index]
        random_indices = np.random.choice(np.arange(data["ct"].shape[-1]), self.batch_size, p=self.distribution)
        slices = torch.from_numpy(random_indices).to(dtype=torch.long)
        cts = data["ct"][..., slices]
        return { "ct_slice": rearrange(cts, "d x y z -> z d x y"), "slice_indices": slices } | default_collate([{ k: v for k, v in data.items() if k != "ct"}] * len(slices))

    def __len__(self):
        return len(self.dataset)
    
    def __worker_fn(self, fetch_index_queue: multiprocessing.Queue, yield_index_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
        current_items = {}
        try:
            while True:
                fetch_index = fetch_index_queue.get()
                if fetch_index == "done":
                    break
                if fetch_index is not None:
                    current_items[fetch_index] = self[fetch_index] | { "index": fetch_index }
                while True:
                    yield_index = yield_index_queue.get()
                    if yield_index == "prefetch":
                        break
                    if yield_index in current_items:
                        result_queue.put(current_items[yield_index])
                        del current_items[yield_index]
                        break
                    else:
                        # Ignore next fetch_index
                        _ = fetch_index_queue.get()
        except:
            print(f"{multiprocessing.current_process().name}:", traceback.format_exc())
            raise

    def __iter__(self):
        print("Starting dataset iterator")
        indices = torch.randperm(len(self)).tolist() if self.shuffle else torch.arange(len(self)).tolist()
        for index in indices:
            yield self[index]

    def iter__(self):
        print("starting iter")
        if self.shuffle:
            indices = torch.randperm(len(self)).tolist()
        else:
            indices = torch.arange(len(self)).tolist()
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()
        fetch_index_queues = tuple(manager.Queue() for _ in range(self.num_workers))
        yield_index_queues = tuple(manager.Queue() for _ in range(self.num_workers))
        fetch_index = 0
        workers = (
            torch.multiprocessing.Process(target=self.__worker_fn, args=(fetch_index_queue, yield_index_queue, result_queue), name=f"IndividualCtSliceDataset Worker {worker_index}", daemon=True) for worker_index, fetch_index_queue, yield_index_queue in zip(range(self.num_workers), fetch_index_queues, yield_index_queues))
        for worker in workers:
            print("starting worker")
            worker.start()
        print("Started all workers")
        try:
            # Seed fetch index queue
            for i in range(min(self.prefetch_factor * self.num_workers, len(indices))):
                if (i // self.num_workers) != 0:
                    yield_index_queues[(i - 1) % self.num_workers].put("prefetch")
                fetch_index_queue_index = i % self.num_workers
                fetch_index_queue = fetch_index_queues[fetch_index_queue_index]
                fetch_index_queue.put(indices[fetch_index])
                fetch_index += 1

            for yield_index in indices:
                for yield_index_queue in yield_index_queues:
                    yield_index_queue.put(yield_index)
                result = result_queue.get()
                if fetch_index < len(indices):
                    for fetch_index_queue in fetch_index_queues:
                        fetch_index_queue.put(indices[fetch_index])
                    fetch_index += 1
                else:
                    for fetch_index_queue in fetch_index_queues:
                        fetch_index_queue.put(None)
                print("yielding item")
                yield result

            for fetch_index_queue in fetch_index_queues:
                fetch_index_queue.put("done")

        finally:
            for worker in workers:
                worker.terminate()
            for worker in workers:
                worker.join(1)

class IndividualCtSliceDataset2(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 16,
        shuffle: bool = True,
        distribution: Literal["normal", "uniform"] = "normal",
        sample_dim_size: int = 128,
        prefetch_count = 2,
        num_workers = 2
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        if distribution == "normal":
            self.distribution = scipy.stats.norm.pdf(list(range(0,sample_dim_size)), sample_dim_size//2, sample_dim_size//2)
            self.distribution = self.distribution + ((1 - self.distribution.sum()) / len(self.distribution))
        elif distribution == "uniform":
            self.distribution = torch.ones(sample_dim_size) / sample_dim_size
        else:
            raise NotImplementedError(f"Slice sampling distribution {distribution} is not implemented.")

        self.prefetch_count = prefetch_count
        self.num_workers = num_workers

    def __getitem__(self, index: int) -> dict:
        data = self.dataset[index]
        random_indices = np.random.choice(np.arange(data["ct"].shape[-1]), self.batch_size, p=self.distribution)
        slices = torch.from_numpy(random_indices).to(dtype=torch.long)
        cts = data["ct"][..., slices]
        return { "ct": rearrange(cts, "d x y z -> z d x y"), "slice_indices": slices } | default_collate([{ k: v for k, v in data.items() if k != "ct"}] * len(slices))

    def __len__(self):
        return len(self.dataset)
    
    def __worker_fn(self, fetch_index_queue: multiprocessing.Queue, result_dict: dict, finished_dict: dict):
        while True:
            fetch_index = fetch_index_queue.get()
            if fetch_index == "done":
                return
            result_dict[fetch_index] = self[fetch_index] | { "index": fetch_index }
            finished_dict[fetch_index].set()

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self)).tolist()
        else:
            indices = torch.arange(len(self)).tolist()
        manager = multiprocessing.Manager()
        result_dict = manager.dict()
        finished_dict = manager.dict()
        fetch_index_queue = manager.Queue()
        fetch_index = 0
        workers = (
            multiprocessing.Process(target=self.__worker_fn, args=(fetch_index_queue, result_dict, finished_dict), name=f"IndividualCtSliceDataset Worker {worker_index}", daemon=True) for worker_index in range(self.num_workers))
        for worker in workers:
            worker.start()
        try:
            # Seed fetch index queue
            for _ in range(min(self.prefetch_count * self.num_workers, len(indices))):
                finished_dict[indices[fetch_index]] = manager.Event()
                fetch_index_queue.put(indices[fetch_index])
                fetch_index += 1

            for yield_index in indices:
                finished_dict[yield_index].wait()
                result = result_dict[yield_index]
                del result_dict[yield_index]
                del finished_dict[yield_index]

                if fetch_index < len(indices):
                    finished_dict[indices[fetch_index]] = manager.Event()
                    fetch_index_queue.put(indices[fetch_index])
                    fetch_index += 1
                else:
                    fetch_index_queue.put("done")
                yield result

        finally:
            for worker in workers:
                worker.terminate()
            for worker in workers:
                worker.join(1)

class IndividualCtSlice128Dataloader(LightningDataModule):
    def __init__(self,
        num_slices: int,
        train: DictConfig,
        validation: DictConfig,
    ):
        super().__init__()
        self.train = train
        self.validation = validation
        self.num_slices = num_slices

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = IndividualCTSliceWrapper(instantiate_from_config(self.train.dataset), num_slices=self.num_slices)
        self.validation_dataset = IndividualCTSliceWrapper(instantiate_from_config(self.validation.dataset), num_slices=self.num_slices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

class ConditionCacheEnricher(Dataset):
    def __init__(self, dataset: Dataset, condition_cache: dict, split):
        self.dataset = dataset
        self.condition_cache = Path(condition_cache) / split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        scan_id = item["scan_id"]
        condition_info = torch.load(self.condition_cache / (scan_id + ".pt"), map_location="cpu")
        assert scan_id == condition_info["scan_id"], f"Scan ID mismatch: {scan_id} != {condition_info['scan_id']}"
        return condition_info | item

class CTVideoDataloader(LightningDataModule):
    def __init__(self,
        num_slices: int,
        train: DictConfig,
        validation: DictConfig,
        condition_cache_path: Optional[str] = None
    ):
        super().__init__()
        self.train = train
        self.validation = validation
        self.num_slices = num_slices
        self.condition_cache_path = Path(condition_cache_path) if condition_cache_path is not None else None

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = CTVideoWrapper(instantiate_from_config(self.train.dataset))
        self.validation_dataset = CTVideoWrapper(instantiate_from_config(self.validation.dataset))
        if self.condition_cache_path is not None:
            self.train_dataset = ConditionCacheEnricher(self.train_dataset, self.condition_cache_path, split="train")
            self.validation_dataset = ConditionCacheEnricher(self.validation_dataset, self.condition_cache_path, split="valid")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

class CTVideo128Dataloader(LightningDataModule):
    def __init__(self,
        num_slices: int,
        train: DictConfig,
        validation: DictConfig,
        condition_cache_path: Optional[str] = None
    ):
        super().__init__()
        self.train = train
        self.validation = validation
        self.num_slices = num_slices
        self.condition_cache_path = Path(condition_cache_path) if condition_cache_path is not None else None

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = CTVideoWrapper(instantiate_from_config(self.train.dataset))
        self.validation_dataset = CTVideoWrapper(instantiate_from_config(self.validation.dataset))
        if self.condition_cache_path is not None:
            self.train_dataset = ConditionCacheEnricher(self.train_dataset, self.condition_cache_path, split="train")
            self.validation_dataset = ConditionCacheEnricher(self.validation_dataset, self.condition_cache_path, split="valid")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

class CTIndividualSliceSingleCTDataloader(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        validation: DictConfig,
    ):
        super().__init__()
        self.train = train
        self.validation = validation

    def setup(self, stage):
        self.train_dataset = instantiate_from_config(self.train.dataset)
        self.validation_dataset = instantiate_from_config(self.validation.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.train.loader.num_workers, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, num_workers=self.validation.loader.num_workers, batch_size=None)
