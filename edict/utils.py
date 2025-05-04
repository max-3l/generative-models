import torch
from pathlib import Path
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
import json
from torch.utils.data import default_collate

def load_model_and_data_and_config(
        run_name="2025-02-28T14-19-35_x2ct-ct-sv3d-diffusion-ct-cached-128",
        chkpt_path= "/data/shared/x2ct/backup-maximilian.schulze/maximilian.schulze/generative-models/logs/2025-02-28T14-19-35_x2ct-ct-sv3d-diffusion-ct-cached-128/checkpoints/epoch=000601.ckpt"
    ):
    configs = list(Path(f"logs/{run_name}/configs").glob("*.yaml"))
    configs = [OmegaConf.load(c) for c in configs]
    config = OmegaConf.merge(*configs)
    assert Path(chkpt_path).exists(), f"Checkpoint {chkpt_path} does not exist"
    assert Path(chkpt_path).is_file(), f"Checkpoint {chkpt_path} is not a file"
    config.model.params.first_stage_config.params.ckpt_engine = config.model.params.first_stage_config.params.ckpt_engine.replace("/raid/maximilian.schulze", "/data/shared/x2ct/backup-maximilian.schulze/maximilian.schulze")
    config.model.params.conditioner_config.params.emb_models[1].params.encoder_config.params.ckpt_engine = config.model.params.conditioner_config.params.emb_models[1].params.encoder_config.params.ckpt_engine.replace("/raid/maximilian.schulze", "/data/shared/x2ct/backup-maximilian.schulze/maximilian.schulze")
    config.model.params.ckpt_path = chkpt_path
    model = instantiate_from_config(config.model)
    data_config = json.loads(
        json.dumps(OmegaConf.to_container(config.data)).replace("/raid/maximilian.schulze/dataset_cache", "/home/maximilian.schulze/dataset_cache").replace("/raid/shared/x2ct/radchest", "/data/shared/x2ct/backup-radchest-dataset/radchest").replace("/raid/maximilian.schulze", "/data/shared/x2ct/backup-maximilian.schulze/maximilian.schulze")
    )
    data = instantiate_from_config(OmegaConf.create(data_config))
    data.setup("")
    model = model.eval().cuda()
    return model, data, config

def load_idx(data, idx, device):
    element = data.validation_dataset[idx]
    batch_element = default_collate([element])
    for key in list(batch_element.keys()):
        if isinstance(batch_element[key], torch.Tensor):
            batch_element[key] = batch_element[key].to(device)
    return batch_element

def clone_batch_element(batch_element):
    batch_element_copy = {}
    for key, value in batch_element.items():
        if isinstance(value, torch.Tensor):
            batch_element_copy[key] = value.clone()
        elif isinstance(value, list):
            batch_element_copy[key] = [v.clone() if isinstance(v, torch.Tensor) else v for v in value]
        else:
            batch_element_copy[key] = value
    return batch_element_copy
