from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import pickle
from typing import Optional
import zipfile
from einops import rearrange
import h5py
import hdf5plugin
import json
import logging
import os
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import tqdm

from sgm.util import instantiate_from_config

# from functools import cache


from .utils.transforms import Normalization, Normalization_min_max, ReturnIdentity, ToTensor

def center_crop_ct(image: torch.Tensor, size: int):
    x, y, z = image.size(-3), image.size(-2), image.size(-1)
    x_lower, y_lower, z_lower = (x - size) // 2, (y - size) // 2, (z - size) // 2
    return image[..., max(0,x_lower):min(x, x_lower+size), max(0, y_lower):min(y, y_lower+size), max(0, z_lower):min(z, z_lower+size)]

def center_crop_drr(image: torch.Tensor, size: int):
    x, y = image.size(-2), image.size(-1)
    x_lower, y_lower = (x - size) // 2, (y - size) // 2
    return image[..., max(0,x_lower):min(x, x_lower+size), max(0, y_lower):min(y, y_lower+size)]

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


class XCT_H5_dataset(Dataset):
    """
    A PyTorch dataset class for loading XCT (X-ray Computed Tomography) data from an H5 file.

    Args:
        h5file_path (Path): The path to the H5 file.
        h5_group_path (str): The path to the group within the H5 file where the data is stored.
        split_file (Path): The path to the split file containing the train/val split information.
        train (bool): Whether to load the training split or the validation split.
        load_res (optional): The resolution at which to load the data. Defaults to None.
        scale (int): The target scale for resizing the CT scans and X-ray images. Defaults to 256.
        scale_ct (bool): Whether to resize the CT scans. Defaults to False.
        use_f16 (bool): Whether to use half-precision floating-point format for the data. Defaults to True.
        crop_size (int): The size for cropping the CT scans and X-ray images. Defaults to -1 (no cropping).
        ct_only (bool): Whether to load only CT data. Defaults to False.
        xray_only (bool): Whether to load only X-ray data. Defaults to False.
        num_projections (int): The number of X-ray projections to load. Defaults to 2.
        is_preprocessed (bool): Whether the data is already preprocessed. Defaults to False.
    """
    def __init__(
            self,
            h5_dataset_path: str,
            h5_ct_group_path: str,
            h5_xray_group_path: str,
            h5_text_dataset_path: str,
            h5_text_group_path: str,
            split_file: str,
            train: bool,
            load_res=None, 
            scale=256,
            scale_images=False,
            crop_size=-1,
            use_f16=True,
            ct_only=False,
            xray_only=False,
            load_text=False,
            num_projections=2,
            is_preprocessed=False,
        ):
        assert not (ct_only and xray_only), "Error: ct_only and xray_only cannot be True at the same time!"
        assert os.path.exists(h5_dataset_path), f"Error: {h5_dataset_path} not found!"
        assert os.path.exists(split_file), f"Error: {split_file} not found!"

        self.h5file_path = Path(h5_dataset_path)
        self.h5_ct_group_path = h5_ct_group_path.split('/')
        self.h5_xray_group_path = h5_xray_group_path.split('/')

        self.load_text = load_text
        if load_text:
            logging.info("Loading text data from %s. Using group at %s", str(h5_text_dataset_path), (h5_text_group_path))
            assert os.path.exists(h5_text_dataset_path), f"Error: {h5_text_dataset_path} not found!"
            self.h5_text_dataset_path = Path(h5_text_dataset_path)
            self.h5_text_group_path = h5_text_group_path.split('/')
        else:
            logging.info("Not loading text data.")

        self.ct_only = ct_only
        self.xray_only = xray_only
        if self.ct_only:
            logging.info("Loading only CT data.")
        if self.xray_only:
            logging.info("Loading only X-ray data.")

        split = 'train' if train else 'val'

        with open(split_file, 'r') as file:
            self.names = json.load(file)[split]
        self.load_res = load_res

        self.f16 = use_f16
        crop_ct = crop_size > 0

        if crop_ct:
            logging.info(f"Cropping CT scans to {crop_size}x{crop_size}x{crop_size}")
        if scale_images:
            logging.info(f"Resizing CT scans to {scale}x{scale}x{scale}")

        self.crop_size = crop_size
        self.scale = scale
        self.ct_tx = [
            Normalization_min_max(0., 1., remove_noise=False), 
            ToTensor(f16=False),
            self.center_crop_ct if crop_ct else nn.Identity(),
            self.interpolate_ct if scale_images else nn.Identity(),
            self.to_half if self.f16 else nn.Identity()
        ]

        self.xray_tx = [
            ToTensor(f16=False),
            self.center_crop_drr if crop_ct else ReturnIdentity(),
            transforms.Resize(scale) if scale_images else ReturnIdentity(), # antialias=True
            Normalization(0, 255),
            self.to_half if self.f16 else nn.Identity()
        ]

        self.ct_mask_transform = [
            ToTensor(f16=False),
            self.center_crop_ct if crop_ct else nn.Identity(),
            self.interpolate_ct if scale_images else nn.Identity(),
            self.to_half
        ]

        self.num_projections = num_projections

        self.is_preprocessed = is_preprocessed

    def center_crop_ct(self, x):
        return center_crop_ct(x, self.crop_size)

    def center_crop_drr(self, x):
        return center_crop_drr(x, self.crop_size)

    def interpolate_ct(self, x):
        return nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0), size=(self.scale, self.scale, self.scale), mode='trilinear').squeeze(0).squeeze(0)

    def to_half(self, x):
        return x.half() if self.f16 else x

    def load_xray_array(self, name: str, index: list = None) -> dict:
        """
        Load X-ray data from the H5 file.

        Args:
            name (str): The name of the dataset in the H5 file.
            index (list, optional): The indices of the X-ray projections to load. Defaults to None (load all).

        Returns:
            dict: A dictionary containing the loaded X-ray data.
        """
        # Load mha
        with h5py.File(self.h5file_path, 'r') as h5:
            group = h5
            for comp in self.h5_xray_group_path:
                group = group[comp]
            xray_group = group[name]
            if index is None:
                index = list(range(self.num_projections))
            xray = [np.array(xray_group[str(i)]).astype(np.float32) for i in index]
        return xray

    def load_xray_data(self, name: str, index: list = None) -> dict:
        """
        Load and preprocess X-ray data from the H5 file.

        Args:
            name (str): The name of the dataset in the H5 file.
            index (list, optional): The indices of the X-ray projections to load. Defaults to None (load all).

        Returns:
            dict: A dictionary containing the loaded and preprocessed X-ray data.
        """
        xray_data = self.load_xray_array(name, index=index)
        if not self.is_preprocessed:
            for transf in self.xray_tx:
                for i in range(len(xray_data)):
                    xray_data[i] = transf(xray_data[i][np.newaxis, np.newaxis, :])[0, 0]
        else:
           xray_data = [torch.from_numpy(xray) for xray in xray_data]
        # Create new channel axis
        xray_data = torch.stack(xray_data, 0).unsqueeze(1)
        return {
            "xrays": xray_data,
        }

    def load_ct_array(self, ct_name: str) -> np.ndarray:
        """
        Load CT scan data from the H5 file.

        Args:
            ct_name (str): The name of the CT scan dataset in the H5 file.

        Returns:
            np.ndarray: The loaded CT scan data.
        """
        with h5py.File(self.h5file_path, 'r') as h5:
            group = h5
            for comp in self.h5_ct_group_path:
                group = group[comp]
            el = group[ct_name]
            if isinstance(group[ct_name], h5py.Group):
                ct = el[list(el.keys())[0]][()].astype(np.float32)
            ct = group[ct_name][()].astype(np.float32)
        return ct

    def load_ct_mask(self, ct_name: str, ct_shape: np.ndarray) -> np.ndarray:
        """
        Load CT scan mask data from the H5 file.

        Args:
            ct_name (str): The name of the CT scan dataset in the H5 file.
            ct_shape (np.ndarray): The shape of the CT scan.

        Returns:
            np.ndarray: The loaded CT scan mask data.
        """
        # Load mha
        with h5py.File(self.h5file_path, 'r') as h5:
            group = h5
            for comp in self.h5_ct_group_path:
                group = group[comp]
            dataset = group[ct_name]
            padding_mask = dataset.attrs['padding']
        data_mask = np.zeros(ct_shape)
        # .item() is needed here to index the numpy array
        # with the integer value from the tensor. Indexing with
        # a slice defined by one element tensors is not supported.
        data_mask[
            padding_mask[0][0].item():-padding_mask[0][1].item(),
            padding_mask[1][0].item():-padding_mask[1][1].item(),
            padding_mask[2][0].item():-padding_mask[2][1].item()] = 1
        return data_mask, padding_mask

    def load_embedding_views(self, name: str) -> dict:
        """
        Load and preprocess embedding views from the H5 file.

        Args:
            name (str): The name of the dataset in the H5 file.

        Returns:
            dict: A dictionary containing the loaded and preprocessed embedding views.
        """
        with h5py.File("/data/shared/x2ct/backup-maximilian.schulze/maximilian.schulze/masterthesis/vqgan_embeddings.h5", 'r') as h5:
            group = h5[name]
            return {
                "first_view_emb": group["first_view_emb"].__array__(dtype=np.float32),
                "second_view_emb": group["second_view_emb"].__array__(dtype=np.float32),
                "third_view_emb": group["third_view_emb"].__array__(dtype=np.float32)
            }

    # @cache
    def load_ct_data(self, name: str) -> dict:
        """
        Load and preprocess CT scan data from the H5 file.

        Args:
            name (str): The name of the CT scan dataset in the H5 file.

        Returns:
            dict: A dictionary containing the loaded and preprocessed CT scan data.
        """
        ct_scan = self.load_ct_array(name)
        # data_mask, padding_mask = self.load_ct_mask(name, ct_scan.shape)
        data_mask = np.ones(ct_scan.shape)
        # padding_mask = np.zeros(ct_scan.shape)

        if not self.is_preprocessed:
            for transf in self.ct_tx:
                ct_scan = transf(ct_scan)
            
            for transf in self.ct_mask_transform:
                data_mask = transf(data_mask)
        else:
            ct_scan = torch.from_numpy(ct_scan)
            data_mask = torch.from_numpy(data_mask)

        ct_scan = ct_scan.unsqueeze(0)
        data_mask = data_mask.unsqueeze(0)

        data = {
            "ct": ct_scan,
            "ct_data_mask": data_mask,
            "scan_id": name,
            "clip_embeddings": self.load_embedding_views(name)
            # "padding_mask": padding_mask.astype(np.int16)
        }

        return data

    def load_text_data(self, name: str) -> dict:
        """
        Load text embeddings from the H5 file. If the item name points to a `Dataset` the
        data is loaded as is. If the item name points to a `Group` the data is loaded
        as a stack of multiple embeddings.

        The format is 1xN for a single token and MxN for multiple embeddings, where N is the
        token embedding dimension and M is the number of embeddings in the group.

        Args:
            name (str): The name of the item to load in the H5 file.

        Returns:
            dict: A dictionary containing the loaded and preprocessed text data.
        """
        with h5py.File(self.h5_text_dataset_path, 'r') as h5:
            group = h5
            for comp in self.h5_text_group_path:
                group = group[comp]
            group = group[name]
            if isinstance(group, h5py.Dataset):
                # If the data is a single dataset, return it as is and add a new axis.
                # This allows us to iterate over the data in the same way as if it was a group of multiple embeddings.
                text_data = np.array(group).astype(np.float32)[np.newaxis, :]
                if len(text_data.shape) == 2:
                    text_data = text_data[np.newaxis, :]
            else:
                text_embeddings = [np.array(group[name]).astype(np.float32) for name in group]
                # text_data = np.stack(text_embeddings, axis=0)
                text_data = text_embeddings
        return { 'text': text_data }

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        data = {
            "name": name
        }
        t1 = time.perf_counter()
        if self.load_text:
            data = self.load_text_data(name)
            logging.debug(f"Text data loading time: {time.perf_counter() - t1}")
            t1 = time.perf_counter()
        if self.ct_only:
            ct_data = self.load_ct_data(name)
            logging.debug(f"CT data loading time: {time.perf_counter() - t1}")
            return data | ct_data
        if self.xray_only:
            xray_data = self.load_xray_data(name)
            logging.debug(f"X-ray data loading time: {time.perf_counter() - t1}")
            return data | xray_data

        ct_data = self.load_ct_data(name)
        logging.debug(f"CT data loading time: {time.perf_counter() - t1}")
        t1 = time.perf_counter()
        xray_data = self.load_xray_data(name)
        logging.debug(f"X-ray data loading time: {time.perf_counter() - t1}")
        # return data | ct_data | xray_data
        return { **data, **ct_data, **xray_data }

class CT_Rate_dataset(XCT_H5_dataset):
    def __init__(
            self,
            h5_dataset_path: str,
            h5_ct_group_path: str,
            h5_xray_group_path: str,
            h5_text_dataset_path: str,
            h5_text_group_path: str,
            train: bool,
            load_res=None, 
            scale=256,
            scale_images=False,
            crop_size=-1,
            use_f16=True,
            ct_only=False,
            xray_only=False,
            load_text=False,
            num_projections=2,
            is_preprocessed=False,
            **kwargs
        ):
        logging.warning("Ignording kwargs: %s",  ", ".join(kwargs.keys()))
        assert not (ct_only and xray_only), "Error: ct_only and xray_only cannot be True at the same time!"
        assert os.path.exists(h5_dataset_path), f"Error: {h5_dataset_path} not found!"

        self.split = 'train' if train else 'valid'

        self.h5file_path = Path(h5_dataset_path)
        self.h5_ct_group_path = h5_ct_group_path.split('/')
        self.h5_xray_group_path = h5_xray_group_path.split('/')

        self.load_text = load_text
        if load_text:
            logging.info("Loading text data from %s. Using group at %s", str(h5_text_dataset_path), (h5_text_group_path))
            assert os.path.exists(h5_text_dataset_path), f"Error: {h5_text_dataset_path} not found!"
            self.h5_text_dataset_path = Path(h5_text_dataset_path)
            self.h5_text_group_path = h5_text_group_path.split('/')
            raw_text_path = "/home/maximilian.schulze/datasets/CT-RATE/dataset/radiology_text_reports/"
            raw_text_path += "train_reports.csv" if train else "validation_reports.csv"
            self.raw_text_csv = pd.read_csv(raw_text_path, index_col="VolumeName")
        else:
            logging.info("Not loading text data.")

        self.ct_only = ct_only
        self.xray_only = xray_only
        if self.ct_only:
            logging.info("Loading only CT data.")
        if self.xray_only:
            logging.info("Loading only X-ray data.")

        with h5py.File(self.h5file_path, 'r') as h5:
            group = h5
            for comp in self.h5_ct_group_path:
                group = group[comp]
            group = group[self.split]
            self.names = list(group.keys())

        self.load_res = load_res

        self.f16 = use_f16
        crop_ct = crop_size > 0

        if crop_ct:
            logging.info(f"Cropping CT scans to {crop_size}x{crop_size}x{crop_size}")
        if scale_images:
            logging.info(f"Resizing CT scans to {scale}x{scale}x{scale}")
        self.crop_size = crop_size
        self.scale = scale
        self.ct_tx = [
            Normalization_min_max(0., 1., remove_noise=False), 
            ToTensor(f16=False),
            self.center_crop_ct if crop_ct else nn.Identity(),
            self.interpolate_ct if scale_images else nn.Identity(),
            self.to_half if self.f16 else nn.Identity()
        ]

        self.xray_tx = [
            ToTensor(f16=False),
            self.center_crop_drr if crop_ct else ReturnIdentity(),
            transforms.Resize(scale, antialias=True) if scale_images else ReturnIdentity(),
            Normalization(0, 255),
            self.to_half if self.f16 else nn.Identity()
        ]

        self.ct_mask_transform = [
            ToTensor(f16=False),
            self.center_crop_ct if crop_ct else nn.Identity(),
            self.interpolate_ct if scale_images else nn.Identity(),
            self.to_half if self.f16 else nn.Identity()
        ]

        self.num_projections = num_projections

        self.is_preprocessed = is_preprocessed

    def center_crop_ct(self, x):
        return center_crop_ct(x, self.crop_size)
    
    def center_crop_drr(self, x):
        return center_crop_drr(x, self.crop_size)

    def interpolate_ct(self, x):
        return nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0), size=(self.scale, self.scale, self.scale), mode='trilinear').squeeze(0).squeeze(0)

    def to_half(self, x):
        return x.half() if self.f16 else x

    def load_ct_array(self, ct_name: str) -> np.ndarray:
        """
        Load CT scan data from the H5 file.

        Args:
            ct_name (str): The name of the CT scan dataset in the H5 file.

        Returns:
            np.ndarray: The loaded CT scan data.
        """
        with h5py.File(self.h5file_path, 'r') as h5:
            group = h5
            for comp in self.h5_ct_group_path:
                group = group[comp]
            el = group[self.split][ct_name]
            while isinstance(el, h5py.Group):
                el = el[sorted(list(el.keys()))[0]]
            ct = el.__array__().astype(np.float32)
        return ct

    # @cache
    def load_ct_data(self, name: str) -> dict:
        """
        Load and preprocess CT scan data from the H5 file.

        Args:
            name (str): The name of the CT scan dataset in the H5 file.

        Returns:
            dict: A dictionary containing the loaded and preprocessed CT scan data.
        """
        ct_scan = self.load_ct_array(name)
        # data_mask, padding_mask = self.load_ct_mask(name, ct_scan.shape)
        data_mask = np.ones(ct_scan.shape)
        # padding_mask = np.zeros(ct_scan.shape)

        if not self.is_preprocessed:
            for transf in self.ct_tx:
                ct_scan = transf(ct_scan)
            
            for transf in self.ct_mask_transform:
                data_mask = transf(data_mask)
        else:
            ct_scan = torch.from_numpy(ct_scan)
            data_mask = torch.from_numpy(data_mask)

        ct_scan = ct_scan.unsqueeze(0)
        data_mask = data_mask.unsqueeze(0)

        data = {
            "ct": ct_scan,
            "ct_data_mask": data_mask,
            "scan_id": name,
            # "clip_embeddings": self.load_embedding_views(name)
            # "padding_mask": padding_mask.astype(np.int16)
        }
        return data

    def load_xray_array(self, name: str, index: list = None) -> dict:
        """
        Load X-ray data from the H5 file.

        Args:
            name (str): The name of the dataset in the H5 file.
            index (list, optional): The indices of the X-ray projections to load. Defaults to None (load all).

        Returns:
            dict: A dictionary containing the loaded X-ray data.
        """
        # Load mha
        with h5py.File(self.h5file_path, 'r') as h5:
            group = h5
            for comp in self.h5_xray_group_path:
                group = group[comp]
            xray_group = xray_group[self.split]
            xray_group = group[name]
            if "0" not in xray_group:
                xray_group = xray_group[sorted(list(xray_group.keys()))[0]]
            if index is None:
                index = list(range(self.num_projections))
            xray = [np.array(xray_group[str(i)]).astype(np.float32) for i in index]
        return xray
    

    def load_text_data(self, name: str) -> dict:
        """
        Load text embeddings from the H5 file. If the item name points to a `Dataset` the
        data is loaded as is. If the item name points to a `Group` the data is loaded
        as a stack of multiple embeddings.

        The format is 1xN for a single token and MxN for multiple embeddings, where N is the
        token embedding dimension and M is the number of embeddings in the group.

        Args:
            name (str): The name of the item to load in the H5 file.

        Returns:
            dict: A dictionary containing the loaded and preprocessed text data.
        """
        with h5py.File(self.h5_text_dataset_path, 'r') as h5:
            group = h5
            name = name + "_a_1.nii.gz"
            for comp in self.h5_text_group_path:
                group = group[comp.replace("{split}", self.split).replace("{name}", name)]
            if isinstance(group, h5py.Dataset):
                # If the data is a single dataset, return it as is and add a new axis.
                # This allows us to iterate over the data in the same way as if it was a group of multiple embeddings.
                text_data = np.array(group).astype(np.float32)
                if len(text_data.shape) == 2:
                    text_data = text_data[np.newaxis, :]
            else:
                text_embeddings = [np.array(group[name]).astype(np.float32) for name in group]
                # text_data = np.stack(text_embeddings, axis=0)
                text_data = text_embeddings

            group = h5
            for comp in self.h5_ct_clip_normalized_text_group_path:
                group = group[comp.replace("{split}", self.split).replace("{name}", name)]
            normalized_ct_clip_text = np.array(group).astype(np.float32).flatten()
            raw_text = self.raw_text_csv.loc[name]["Findings_EN"]

        return { 'text': text_data, 'normalized_ct_clip_text': normalized_ct_clip_text, "raw_text": raw_text }

class XRAY_dataset(XCT_H5_dataset):
    """
    A dataset class for X-ray data.

    This class extends the `XCT_H5_dataset` class and provides additional functionality
    specific to X-ray data. It loads each X-ray projection separately suitable for training
    models that require single X-ray projections as input.
    """
    def __init__(
            self,
            h5_dataset_path: str,
            h5_xray_group_path: str,
            h5_text_dataset_path: str,
            h5_text_group_path: str,
            split_file: str,
            train: bool,
            load_res=None, 
            scale=256,
            scale_images=False,
            crop_size=-1,
            use_f16=True,
            load_text=False,
            num_projections=2):
        super().__init__(
            h5_dataset_path=h5_dataset_path,
            h5_ct_group_path="",
            h5_xray_group_path=h5_xray_group_path,
            h5_text_dataset_path=h5_text_dataset_path,
            h5_text_group_path=h5_text_group_path,
            split_file=split_file,
            train=train,
            load_res=load_res,
            scale=scale,
            scale_images=scale_images,
            crop_size=crop_size,
            use_f16=use_f16,
            ct_only=False,
            xray_only=True,
            load_text=load_text,
            num_projections=num_projections
        )

    def __len__(self):
        return len(self.names) * self.num_projections

    def __getitem__(self, idx):
        name = self.names[idx // self.num_projections]
        data = {}
        if self.load_text:
            data = self.load_text_data(name)
        xray_data = self.load_xray_data(name, [idx % self.num_projections])
        return data | xray_data

class CachedXCT_H5_dataset(XCT_H5_dataset):
    def __init__(self, cache_dir: str, optimize_zip: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

# class CachedXCT_H5_dataset(torch.utils.data.Dataset):
#     def __init__(self, cache_dir: str, optimize_zip: bool, *args, **kwargs):
#         base_dataset = XCT_H5_dataset
#         self.optimize_zip = optimize_zip
#         self.cache_dir = Path(cache_dir)
#         self.cache_hash = self._compute_hash(base_dataset, args, kwargs)
#         self.cache_path = self.cache_dir / self.cache_hash
#         print("Dataset cache path:", self.cache_path)
#         self.status_file = self.cache_path / "_status.txt"
#         self.check_and_build_cache(base_dataset, args, kwargs)
#         self.length = self.compute_length()

#     def _compute_hash(self, base_dataset, args, kwargs):
#         hash_input = str(base_dataset) + str(args) + str(kwargs)
#         if self.optimize_zip:
#             hash_input += "zip"
#         return hashlib.md5(hash_input.encode()).hexdigest()

#     def _build_cache(self, base_dataset, args, kwargs):
#         self.cache_path.mkdir(parents=True, exist_ok=True)
#         dataset = base_dataset(*args, **kwargs)

#         def process_item(idx):
#             item = dataset[idx]
#             item_path = self.cache_path / f"{idx}"
#             if self.optimize_zip:
#                 with zipfile.ZipFile(item_path.with_suffix(".zip"), 'w', compression=zipfile.ZIP_DEFLATED) as z:
#                     with z.open('data.pkl', 'w') as f:
#                         pickle.dump(item, f)
#             else:
#                 with open(item_path.with_suffix(".pkl"), 'wb') as f:
#                     pickle.dump(item, f)

#         with ThreadPoolExecutor() as executor:
#             list(tqdm.tqdm(executor.map(process_item, range(len(dataset))), total=len(dataset), desc=f"Building cache {self.cache_path}"))
        
#         with open(self.status_file, 'w') as f:
#             f.write("done")
#         with open(self.cache_path / "_hash.txt", 'w') as f:
#             f.write(self.cache_hash)
#         with open(self.cache_path / "_args.txt", 'w') as f:
#             f.write(str(args))
#         with open(self.cache_path / "_kwargs.txt", 'w') as f:
#             f.write(str(kwargs))

#     def check_and_build_cache(self, base_dataset, args, kwargs):
#         if not self.cache_path.exists():
#             self.cache_path.mkdir(parents=True, exist_ok=True)
#         lock_file = self.cache_path / "_lockfile"
#         with open(lock_file, 'w') as f:
#             fcntl.flock(f, fcntl.LOCK_EX)
#             if not self.status_file.exists():
#                 self._build_cache(base_dataset, args, kwargs)
#             fcntl.flock(f, fcntl.LOCK_UN)

#     def compute_length(self):
#         if self.optimize_zip:
#             return len(list(self.cache_path.glob("*.zip")))
#         return len(list(self.cache_path.glob("*.pkl")))

#     def __getitem__(self, idx):
#         if self.optimize_zip:
#             item_path = self.cache_path / f"{idx}.zip"
#             with zipfile.ZipFile(item_path, 'r') as z:
#                 with z.open('data.pkl', 'r') as f:
#                     item = pickle.load(f)
#         else:
#             item_path = self.cache_path / f"{idx}.pkl"
#             with open(item_path, 'rb') as f:
#                 item = pickle.load(f)
#         return item

#     def __len__(self):
#         return self.length

from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

class RadchestDataloader(LightningDataModule):
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
        # text_path = "/raid/shared/x2ct/ct-rate-models/train_reports.csv" if train else "/raid/shared/x2ct/ct-rate-models/validation_reports.csv"
        text_path = "/data/shared/x2ct/backup-maximilian.schulze/maximilian.schulze/masterthesis/radchest-label-generator/python/radchest-labels-one-no-negations.json"
        raw_text_path = "/home/maximilian.schulze/datasets/CT-RATE/dataset/radiology_text_reports/"
        raw_text_path += "train_reports.csv" if train else "validation_reports.csv"
        self.raw_text_csv = pd.read_csv(raw_text_path, index_col="VolumeName")
        with open(text_path, "r") as f:
            self.raw_text_json_radchest = json.load(f)

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
            "raw_text": self.raw_text_json_radchest[dataset_item["scan_id"]]
        }

class ZAsDepthWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset_item = self.dataset[idx]
        return dataset_item | { "ct_z_depth": rearrange(dataset_item["ct"], "c h w d -> c d h w").squeeze(0).float() }

class Rescaler(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if "ct" in item:
            item["ct"] = item["ct"] * 2 - 1
        if "xrays" in item:
            item["xrays"] = item["xrays"] * 2 - 1
        return item

class RadchestIndividualXrayDataloader(LightningDataModule):
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
        self.train_dataset = IndividualXrayWrapper(Rescaler(instantiate_from_config(self.train.dataset)), num_projections=self.num_projections)
        self.validation_dataset = IndividualXrayWrapper(Rescaler(instantiate_from_config(self.validation.dataset)), num_projections=self.num_projections)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=True, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=False, num_workers=self.validation.loader.num_workers)

class RadchestIndividualCtSliceDataloader(LightningDataModule):
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
        self.train_dataset = IndividualCTSliceWrapper(Rescaler(instantiate_from_config(self.train.dataset)), num_slices=self.num_slices)
        self.validation_dataset = IndividualCTSliceWrapper(Rescaler(instantiate_from_config(self.validation.dataset)), num_slices=self.num_slices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

class RadchestCTAsDepthDataloader(LightningDataModule):
    def __init__(self,
        train: DictConfig,
        validation: DictConfig,
    ):
        super().__init__()
        self.train = train
        self.validation = validation

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        self.train_dataset = ZAsDepthWrapper(Rescaler(instantiate_from_config(self.train.dataset)))
        self.validation_dataset = ZAsDepthWrapper(Rescaler(instantiate_from_config(self.validation.dataset)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

class Resize128(Dataset):
    def __init__(self, dataset: Dataset, dtype: str = "float32"):
        self.dataset = dataset
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if "ct" in item:
            item["ct"] = nn.functional.interpolate(item["ct"].unsqueeze(0), size=(128, 128, 128), mode='trilinear').squeeze(0).to(dtype=self.dtype)
        if "xrays" in item:
            item["xrays"] = nn.functional.interpolate(item["xrays"], size=(128, 128), mode='bilinear').to(dtype=self.dtype)
        return item

class RadchestCT128Dataloader(LightningDataModule):
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
        self.train_dataset = Resize128(Rescaler(instantiate_from_config(self.train.dataset)), self.dtype)
        self.validation_dataset = Resize128(Rescaler(instantiate_from_config(self.validation.dataset)), self.dtype)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

class RadchestIndividualCtSlice128Dataloader(LightningDataModule):
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
        self.train_dataset = IndividualCTSliceWrapper(Resize128(Rescaler(instantiate_from_config(self.train.dataset))), num_slices=self.num_slices)
        self.validation_dataset = IndividualCTSliceWrapper(Resize128(Rescaler(instantiate_from_config(self.validation.dataset))), num_slices=self.num_slices)

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

class RadchestCTVideoDataloader(LightningDataModule):
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
        self.train_dataset = CTVideoWrapper(Rescaler(instantiate_from_config(self.train.dataset)))
        self.validation_dataset = CTVideoWrapper(Rescaler(instantiate_from_config(self.validation.dataset)))
        if self.condition_cache_path is not None:
            self.train_dataset = ConditionCacheEnricher(self.train_dataset, self.condition_cache_path, split="train")
            self.validation_dataset = ConditionCacheEnricher(self.validation_dataset, self.condition_cache_path, split="valid")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)

from sgm.data.utils.dataloader import TimedDataLoader

class RadchestCTVideo128Dataloader(LightningDataModule):
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
        self.train_dataset = CTVideoWrapper(Resize128(Rescaler(instantiate_from_config(self.train.dataset))))
        self.validation_dataset = CTVideoWrapper(Resize128(Rescaler(instantiate_from_config(self.validation.dataset))))
        if self.condition_cache_path is not None:
            self.train_dataset = ConditionCacheEnricher(self.train_dataset, self.condition_cache_path, split="train")
            self.validation_dataset = ConditionCacheEnricher(self.validation_dataset, self.condition_cache_path, split="valid")

    def train_dataloader(self):
        return TimedDataLoader(self.train_dataset, batch_size=self.train.loader.batch_size, shuffle=self.train.loader.shuffle, num_workers=self.train.loader.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.validation.loader.batch_size, shuffle=self.validation.loader.shuffle, num_workers=self.validation.loader.num_workers)
