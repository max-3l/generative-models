from typing import Optional
from einops import rearrange
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=True, num_workers=self.train.loader.num_workers, drop_last=True)

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
