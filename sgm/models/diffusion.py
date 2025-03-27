import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from einops import rearrange, repeat
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)



# 3D SSIM Averaged over Depth, Height, and Width
def compute_3d_psnr_ssim(ground_truth, reconstructed, data_range=(0, 1)):
    """
    Compute 3D SSIM between two 3D images (volumes) averaged over depth, height, and width.

    Args:
    - image1: A tensor or numpy array of shape (B, C, D, H, W).
    - image2: A tensor or numpy array of shape (B, C, D, H, W).
    - data_range: The dynamic range of the images (optional).

    Re<turns:
    - ssim_value: The average SSIM value across all axes.
    """

    assert ground_truth.shape == reconstructed.shape, "Input images must have the same shape"
    assert len(ground_truth.shape) == 5, "Input images must have 5 dimensions (B, C, D, H, W)"

    # Initialize a list to hold SSIM values
    ssim_values = []
    psnr_values = []

    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=data_range, reduction="none").cuda()
    # psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=data_range, reduction="none").cuda()
    psnr = lambda x, y: 10 * torch.log10(1 / F.mse_loss(x, y, reduction='none').mean(dim=(1,2,3)))

    # Compute SSIM along each of the three axes (depth, height, width)
    for axis in range(3):
        # Iterate along the current axis
        for i in range(ground_truth.shape[2 + axis]):
            if axis == 0:
                slice_ssim = ssim(ground_truth[:, :, i, :, :], reconstructed[:, :, i, :, :])
                slice_psnr = psnr(ground_truth[:, :, i, :, :], reconstructed[:, :, i, :, :])
            elif axis == 1:
                slice_ssim = ssim(ground_truth[:, :, :, i, :], reconstructed[:, :, :, i, :])
                slice_psnr = psnr(ground_truth[:, :, :, i, :], reconstructed[:, :, :, i, :])
            elif axis == 2:
                slice_ssim = ssim(ground_truth[:, :, :, :, i], reconstructed[:, :, :, :, i])
                slice_psnr = psnr(ground_truth[:, :, :, :, i], reconstructed[:, :, :, :, i])

            ssim_values.append(slice_ssim)
            psnr_values.append(slice_psnr)

    # Return the average SSIM across all axes and slices
    return torch.stack(psnr_values).mean(dim=0), torch.stack(ssim_values).mean(dim=0)


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        num_frames=None,
        network_dtype="float32"
    ):
        super().__init__()
        dtype = "float32"
        if network_dtype == "float32":
            dtype = torch.float32
        elif network_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"network_dtype must be either float32 or bfloat16.")
        
        if ckpt_path is not None:
            print(f"Restoring diffusion from {ckpt_path}")
        self.num_frames = num_frames
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        if num_frames is not None:
            self.model = get_obj_from_str(network_wrapper)(
                model, num_frames=num_frames, compile_model=compile_model
            )
        else:
            self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
                model, compile_model=compile_model
            )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        ).to()

        self.model = self.model.to(dtype)
        self.loss_fn = self.loss_fn.to(dtype)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log


        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored diffusion module from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return rearrange(batch[self.input_key], "b t c h w -> (b t) c h w")

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict
    
    def cache_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        conditioning = self.conditioner.forward_for_cache(batch)
        return x, conditioning

    def validation_step(self, batch, batch_idx):
        metrics = self.batch_sample_metrics(batch, return_images=False)
        metrics = { f"val/{key}": el for key, el in metrics.items() }
        with self.ema_scope():
            ema_metrics = self.batch_sample_metrics(batch, return_images=False)
        ema_metrics = { f"val_ema/{key}": el for key, el in metrics.items() }
        return metrics | ema_metrics

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                if embedder.input_key == "xrays":
                    # xrays are a special case
                    log[embedder.input_key] =  rearrange(batch[embedder.input_key], "b t c h w -> b c h (t w)")[:n]
                    continue
                else:
                    x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = batch[self.input_key]
        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        x = x.flatten(0, 1)

        if "cond_aug" in batch:
            batch["cond_aug"] = batch["cond_aug"][:N]

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        sampling_kwargs["image_only_indicator"] = repeat(batch["image_only_indicator"][:N], "b ... -> (2 b) ...")

        log["video_inputs"] = x.reshape(N, -1, *x.shape[1:])
        z = self.encode_first_stage(x)
        log["video_reconstructions"] = self.decode_first_stage(z.to(dtype=self.first_stage_model.decoder.conv_in.weight.dtype))
        log["video_reconstructions"] = log["video_reconstructions"].reshape(N, -1, *log["video_reconstructions"].shape[1:])

        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor) and (k != "vector"):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=z.shape[0], **sampling_kwargs
                )
            samples = self.decode_first_stage(samples.to(dtype=self.first_stage_model.decoder.conv_in.weight.dtype))
            log["video_samples"] = samples.reshape(N, -1, *samples.shape[1:])

        video_reconstructions = (log["video_reconstructions"].clamp(-1, 1) + 1) / 2
        video_inputs = (log["video_inputs"].clamp(-1, 1) + 1) / 2
        samples = (log["video_samples"].clamp(-1, 1) + 1) / 2
        differences_input = torch.abs(samples - video_inputs)
        differences_reconstruction = torch.abs(samples - video_reconstructions)

        video_comparison = torch.cat([video_inputs, video_reconstructions, differences_reconstruction, differences_input], dim=-1)
        w_per_img = video_comparison.shape[-1] // 4
        h_per_img = 20
        text = log_txt_as_img((w_per_img, h_per_img), ["Input", "Reconstruction", "Difference Reconstruction", "Difference Input"], size=8)
        text = text.mean(dim=1)
        text = rearrange(text, "b h w -> h (w b)")
        text = repeat(text, "h w ->  b t c h w", b=N, t=video_comparison.shape[1], c=1)
        video_comparison = torch.cat([text, video_comparison.cpu()], dim=-2)
        video_comparison = rearrange(video_comparison, "b t c h w -> b (t c) h w")
        log["video_comparison"] = video_comparison

        return log
    
    @torch.no_grad()
    def batch_sample_metrics(
        self,
        batch: Dict,
        ucg_keys: List[str] = None,
        return_images = True,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        

        x = batch[self.input_key]
        x = x.to(self.device)
        N = x.shape[0]
        x = x.flatten(0, 1)

        if "cond_aug" in batch:
            batch["cond_aug"] = batch["cond_aug"]

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        sampling_kwargs["image_only_indicator"] = repeat(batch["image_only_indicator"], "b ... -> (2 b) ...")

        video_inputs = x.reshape(N, -1, *x.shape[1:])
        z = self.encode_first_stage(x)

        for k in c:
            if isinstance(c[k], torch.Tensor) and (k != "vector"):
                c[k], uc[k] = map(lambda y: y[k].to(self.device), (c, uc))

        with self.ema_scope("Plotting"):
            samples = self.sample(
                c, shape=z.shape[1:], uc=uc, batch_size=z.shape[0], **sampling_kwargs
            )
        samples = self.decode_first_stage(samples.to(dtype=self.first_stage_model.decoder.conv_in.weight.dtype))
        video_samples = samples.reshape(N, -1, *samples.shape[1:])
        video_samples = video_samples.clamp(-1, 1)
        video_inputs = video_inputs.clamp(-1, 1)
        video_samples = rearrange(video_samples, "b t c h w -> b c t h w")
        video_inputs = rearrange(video_inputs, "b t c h w -> b c t h w")
        psnr, ssim = compute_3d_psnr_ssim(video_inputs, video_samples, data_range=(-1, 1))
        output = { "psnr": psnr, "ssim": ssim, "mse": (video_samples - video_inputs).pow(2).flatten(start_dim=1).mean(dim=1) }
        if return_images:
            output |= { "video_samples": video_samples, "video_inputs": video_inputs }
        return output
