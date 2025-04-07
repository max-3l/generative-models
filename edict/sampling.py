from matplotlib.figure import Figure
import torch
from typing import List, cast
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from edict.edict import coupled_stablediffusion
from sgm.models.diffusion import compute_3d_psnr_ssim
from sgm.util import append_dims
from edict.utils import load_idx
from dataclasses import dataclass
from typing import List, Tuple


@torch.no_grad()
def sample_default(model, data, element_idx):
    element = load_idx(data, element_idx, model.device)
    conditioner_input_keys = [e.input_key for e in model.conditioner.embedders]
    ucg_keys = conditioner_input_keys
    x = element[model.input_key]
    x = x.to(model.device)
    N = x.shape[0]
    x = x.flatten(0, 1)
    c, uc = model.conditioner.get_unconditional_conditioning(
        element,
        force_uc_zero_embeddings=ucg_keys
        if len(model.conditioner.embedders) > 0
        else [],
    )
    sampled_latent_steps = []
    sampling_kwargs = {}
    sampling_kwargs["image_only_indicator"] = repeat(element["image_only_indicator"], "b ... -> (2 b) ...")
    video_inputs = x.reshape(N, -1, *x.shape[1:])
    z_first_stage = model.encode_first_stage(x)
    for k in c:
        if isinstance(c[k], torch.Tensor) and (k != "vector"):
            c[k], uc[k] = map(lambda y: y[k].to(model.device), (c, uc))
    with model.ema_scope("Plotting"):
        randn = torch.randn(z_first_stage.shape[0], *z_first_stage.shape[1:], device=model.device)
        denoiser = lambda input, sigma, c: model.denoiser(
            model.model, input, sigma, c, **sampling_kwargs
        )
        sample_latents = model.sampler(denoiser, randn, c, uc=uc)
        sampled_latent_steps.append(sample_latents.cpu())
    samples = model.decode_first_stage(sample_latents.to(dtype=model.first_stage_model.decoder.conv_in.weight.dtype))
    psnr, ssim = compute_3d_psnr_ssim(rearrange(x, "d c w h -> 1 c w h d"), rearrange(samples, "d c w h -> 1 c w h d"), (-1.0, 1.0))
    return samples, z_first_stage, sampled_latent_steps, psnr, ssim

@torch.no_grad()
def sample_default_from_closest(model, data, element_idx, element_idx_closest):
    first_element = load_idx(data, element_idx, model.device)
    closest_element = load_idx(data, element_idx_closest, model.device)
    conditioner_input_keys = [e.input_key for e in model.conditioner.embedders]
    ucg_keys = conditioner_input_keys
    x = first_element[model.input_key]
    x_closest = closest_element[model.input_key]
    x = x.to(model.device)
    x_closest = x_closest.to(model.device)
    N = x.shape[0]
    x = x.flatten(0, 1)
    x_closest = x_closest.flatten(0, 1)
    c, uc = model.conditioner.get_unconditional_conditioning(
        first_element,
        force_uc_zero_embeddings=ucg_keys
        if len(model.conditioner.embedders) > 0
        else [],
    )
    sampling_kwargs = {}
    sampling_kwargs["image_only_indicator"] = repeat(first_element["image_only_indicator"], "b ... -> (2 b) ...")
    video_inputs = x.reshape(N, -1, *x.shape[1:])
    z_first_stage = model.encode_first_stage(x)
    z_first_stage_closest = model.encode_first_stage(x_closest)
    sampled_latents_steps = []
    for k in c:
        if isinstance(c[k], torch.Tensor) and (k != "vector"):
            c[k], uc[k] = map(lambda y: y[k].to(model.device), (c, uc))
    with model.ema_scope("Plotting"):
        noise = torch.randn(z_first_stage.shape[0], *z_first_stage.shape[1:], device=model.device)
        sigmas = model.sampler.discretization(model.sampler.num_steps)
        sigma = sigmas[0].to(z_first_stage_closest.device)
        noised_z = z_first_stage_closest + noise * append_dims(sigma, z_first_stage_closest.ndim)
        noised_z = noised_z / torch.sqrt(
                    1.0 + sigmas[0] ** 2.0
                )  # Note: hardcoded to DDPM-like scaling. need to generalize later.
        denoiser = lambda input, sigma, c: model.denoiser(
            model.model, input, sigma, c, **sampling_kwargs
        )
        sample_latents_from_closest = model.sampler(denoiser, noised_z, c, uc=uc)
        sampled_latents_steps.append(sample_latents_from_closest.cpu())
    samples_from_closest = model.decode_first_stage(sample_latents_from_closest.to(dtype=model.first_stage_model.decoder.conv_in.weight.dtype))
    psnr, ssim = compute_3d_psnr_ssim(rearrange(x, "d c w h -> 1 c w h d"), rearrange(samples_from_closest, "d c w h -> 1 c w h d"), (-1.0, 1.0))
    return samples_from_closest, z_first_stage, sampled_latents_steps, psnr, ssim

def decode_latents(model, latents):
    decoded = []
    for latent in latents:
        latent = latent.to(model.device)
        latent = latent.to(dtype=model.first_stage_model.decoder.conv_in.weight.dtype)
        decoded.append(model.decode_first_stage(latent))
    return decoded

def decode_latents_and_metrics(model, latents, x):
    psnrs = []
    ssims = []
    decoded = decode_latents(model, latents)
    for latent in decoded:
        psnr, ssim = compute_3d_psnr_ssim(rearrange(x, "d c w h -> 1 c w h d"), rearrange(latent, "d c w h -> 1 c w h d"), (-1.0, 1.0))
        psnrs.append(psnr)
        ssims.append(ssim)
    return decoded, psnrs, ssims

def plot_sampling_results(psnrs: List[float], ssims: List[float]) -> Figure:
    """
    Plot the PSNR and SSIM values over the sampling steps.
    Args:
        psnrs (List[float]): List of PSNR values.
        ssims (List[float]): List of SSIM values.
    Returns:
        plt.Figure: The figure containing the plots.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax = cast(list[plt.Axes], ax)
    ax[0].plot(psnrs, label='PSNR')
    ax[0].set_title('PSNR over Sampling Steps')
    ax[0].set_xlabel('Sampling Step')
    ax[0].set_ylabel('PSNR')
    ax[0].legend()

    ax[1].plot(ssims, label='SSIM', color='orange')
    ax[1].set_title('SSIM over Sampling Steps')
    ax[1].set_xlabel('Sampling Step')
    ax[1].set_ylabel('SSIM')
    ax[1].legend()

    fig.tight_layout()
    return fig

@dataclass
class EdictClosestSample:
    samples_reverse_noised: List[torch.Tensor]
    samples_reversed_denoised: List[torch.Tensor]
    z_first_stage: torch.Tensor
    z_first_stage_closest: torch.Tensor
    latents_reversed_denoised_closest: List[torch.Tensor]
    x: torch.Tensor
    x_closest: torch.Tensor
    psnr_noised: Tuple[float, ...]
    ssim_noised: Tuple[float, ...]
    psnr_denoised: Tuple[float, ...]
    ssim_denoised: Tuple[float, ...]

@torch.no_grad()
def sample_edict_from_closest(model, data, element_idx: int, closest_element_idx: int, baseline=False, steps=50, guidance_scale=7.0):
    element = load_idx(data, element_idx, model.device)
    closest_element = load_idx(data, closest_element_idx, model.device)
    conditioner_input_keys = [e.input_key for e in model.conditioner.embedders]
    ucg_keys = conditioner_input_keys
    x = element[model.input_key]
    x = x.to(model.device)
    N = x.shape[0]
    x = x.flatten(0, 1)
    x_closest = closest_element[model.input_key]
    x_closest = x_closest.to(model.device).flatten(0, 1)
    c, uc = model.conditioner.get_unconditional_conditioning(
        element,
        force_uc_zero_embeddings=ucg_keys
        if len(model.conditioner.embedders) > 0
        else [],
        )
    z_first_stage = model.encode_first_stage(x)
    z_first_stage_closest = model.encode_first_stage(x_closest)

    latents_reverse_noised_closest = coupled_stablediffusion(
        model,
        steps=steps,
        guidance_scale=guidance_scale,
        init_image_latents=z_first_stage_closest,
        conditioning_conditional=c,
        conditioning_unconditional=uc,
        reverse=True,
        run_baseline=baseline,
        device="cuda",
        sampling_kwargs={
            "image_only_indicator": element["image_only_indicator"],
        }
    )
    latents_reversed_denoised_closest = coupled_stablediffusion(
        model,
        steps=steps,
        guidance_scale=guidance_scale,
        init_image_latents=z_first_stage_closest,
        fixed_starting_latent=latents_reverse_noised_closest,
        conditioning_conditional=c,
        conditioning_unconditional=uc,
        reverse=False,
        run_baseline=baseline,
        device="cuda",
        sampling_kwargs={
            "image_only_indicator": element["image_only_indicator"],
        }
    )

    samples_reverse_noised = [model.decode_first_stage(latent_reverse_noised.to(dtype=model.first_stage_model.decoder.conv_in.weight.dtype)) for latent_reverse_noised in latents_reverse_noised_closest]
    samples_reversed_denoised = [model.decode_first_stage(latent_reversed_denoised.to(dtype=model.first_stage_model.decoder.conv_in.weight.dtype)) for latent_reversed_denoised in latents_reversed_denoised_closest]
    psnr_ssim_noised = [compute_3d_psnr_ssim(rearrange(x, "d c w h -> 1 c w h d"), rearrange(sample_reverse_noised, "d c w h -> 1 c w h d"), (-1.0, 1.0)) for sample_reverse_noised in samples_reverse_noised]
    psnr_ssim_denoised = [compute_3d_psnr_ssim(rearrange(x, "d c w h -> 1 c w h d"), rearrange(sample_reversed_denoised, "d c w h -> 1 c w h d"), (-1.0, 1.0)) for sample_reversed_denoised in samples_reversed_denoised]
    psnr_noised, ssim_noised = zip(*psnr_ssim_noised)
    psnr_denoised, ssim_denoised = zip(*psnr_ssim_denoised)
    # Move everything to the CPU
    z_first_stage = z_first_stage.cpu()
    z_first_stage_closest = z_first_stage_closest.cpu()
    x = x.cpu()
    x_closest = x_closest.cpu()
    latents_reversed_denoised_closest = [latent_reversed_denoised_closest.cpu() for latent_reversed_denoised_closest in latents_reversed_denoised_closest]
    samples_reverse_noised = [sample_reverse_noised.cpu() for sample_reverse_noised in samples_reverse_noised]
    samples_reversed_denoised = [sample_reversed_denoised.cpu() for sample_reversed_denoised in samples_reversed_denoised]
    latents_reverse_noised_closest = [latent_reverse_noised.cpu() for latent_reverse_noised in latents_reverse_noised_closest]
    latents_reversed_denoised_closest = [latent_reversed_denoised.cpu() for latent_reversed_denoised in latents_reversed_denoised_closest]  

    return EdictClosestSample(
        samples_reverse_noised=samples_reverse_noised,
        samples_reversed_denoised=samples_reversed_denoised,
        z_first_stage=z_first_stage.cpu(),
        z_first_stage_closest=z_first_stage_closest.cpu(),
        latents_reversed_denoised_closest=latents_reversed_denoised_closest,
        x=x.cpu(),
        x_closest=x_closest.cpu(),
        psnr_noised=psnr_noised,
        ssim_noised=ssim_noised,
        psnr_denoised=psnr_denoised,
        ssim_denoised=ssim_denoised,
    )

@dataclass
class EdictSample:
    samples_reverse_noised: List[torch.Tensor]
    samples_reversed_denoised: List[torch.Tensor]
    z_first_stage: torch.Tensor
    latents_reversed_denoised: List[torch.Tensor]
    x: torch.Tensor
    psnr_noised: Tuple[float, ...]
    ssim_noised: Tuple[float, ...]
    psnr_denoised: Tuple[float, ...]
    ssim_denoised: Tuple[float, ...]

@torch.no_grad()
def sample_edict(model, data, element_idx: int, baseline=False, steps=50, guidance_scale=7.0):
    element = load_idx(data, element_idx, model.device)
    conditioner_input_keys = [e.input_key for e in model.conditioner.embedders]
    ucg_keys = conditioner_input_keys
    x = element[model.input_key]
    x = x.to(model.device)
    N = x.shape[0]
    x = x.flatten(0, 1)
    c, uc = model.conditioner.get_unconditional_conditioning(
        element,
        force_uc_zero_embeddings=ucg_keys
        if len(model.conditioner.embedders) > 0
        else [],
        )
    z_first_stage = model.encode_first_stage(x)

    latents_reverse_noised= coupled_stablediffusion(
        model,
        init_image_latents=z_first_stage,
        conditioning_conditional=c,
        conditioning_unconditional=uc,
        reverse=True,
        steps=steps,
        guidance_scale=guidance_scale,
        run_baseline=baseline,
        device="cuda",
        sampling_kwargs={
            "image_only_indicator": element["image_only_indicator"],
        }
    )
    latents_reversed_denoised = coupled_stablediffusion(
        model,
        init_image_latents=z_first_stage,
        fixed_starting_latent=latents_reverse_noised,
        conditioning_conditional=c,
        conditioning_unconditional=uc,
        reverse=False,
        steps=steps,
        guidance_scale=guidance_scale,
        run_baseline=baseline,
        device="cuda",
        sampling_kwargs={
            "image_only_indicator": element["image_only_indicator"],
        }
    )

    samples_reverse_noised = [model.decode_first_stage(latent_reverse_noised.to(dtype=model.first_stage_model.decoder.conv_in.weight.dtype)) for latent_reverse_noised in latents_reverse_noised]
    samples_reversed_denoised = [model.decode_first_stage(latent_reversed_denoised.to(dtype=model.first_stage_model.decoder.conv_in.weight.dtype)) for latent_reversed_denoised in latents_reversed_denoised]
    psnr_ssim_noised = [compute_3d_psnr_ssim(rearrange(x, "d c w h -> 1 c w h d"), rearrange(sample_reverse_noised, "d c w h -> 1 c w h d"), (-1.0, 1.0)) for sample_reverse_noised in samples_reverse_noised]
    psnr_ssim_denoised = [compute_3d_psnr_ssim(rearrange(x, "d c w h -> 1 c w h d"), rearrange(sample_reversed_denoised, "d c w h -> 1 c w h d"), (-1.0, 1.0)) for sample_reversed_denoised in samples_reversed_denoised]
    psnr_noised, ssim_noised = zip(*psnr_ssim_noised)
    psnr_denoised, ssim_denoised = zip(*psnr_ssim_denoised)

    return EdictSample(
        samples_reverse_noised=samples_reverse_noised,
        samples_reversed_denoised=samples_reversed_denoised,
        z_first_stage=z_first_stage.cpu(),
        latents_reversed_denoised=latents_reversed_denoised,
        x=x.cpu(),
        psnr_noised=psnr_noised,
        ssim_noised=ssim_noised,
        psnr_denoised=psnr_denoised,
        ssim_denoised=ssim_denoised,
    )
