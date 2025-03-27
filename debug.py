import torch
from x2ct_datasets.ct_rate.dataset import CTRATEDatasetWithText
from tqdm import tqdm
from sgm.data.ct_rate import IndividualXrayDataloader
from omegaconf import DictConfig

def test():
    loader = IndividualXrayDataloader(num_projections=2, train=DictConfig({ "loader": { "batch_size": 32, "num_workers": 8 }, "dataset": { "target": "x2ct_datasets.ct_rate.dataset.CTRATEDatasetWithText", "params": { "split": "train", "load_xray": True, "load_ct": False, "downsample_size": 128, "output_dtype": "bfloat16" } } }), validation=DictConfig({ "loader": { "batch_size": 32, "num_workers": 8 }, "dataset": { "target": "x2ct_datasets.ct_rate.dataset.CTRATEDatasetWithText", "params": { "split": "val", "load_xray": True, "load_ct": False, "downsample_size": 128, "output_dtype": "bfloat16" } } }))
    loader.setup("")
    train_loader = loader.train_dataloader()
    val_loader = loader.val_dataloader()
    for epoch in range(10):
        for element in tqdm(train_loader):
            el = element
    for epoch in range(10):
        for element in tqdm(val_loader):
            el = element

if __name__ == "__main__":
    test()
