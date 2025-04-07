import torch
import random
import numpy as np
from tqdm import tqdm
from diffusers import DDIMScheduler
from edict.utils import clone_batch_element

def get_alpha_and_beta(t: torch.Tensor, scheduler: DDIMScheduler):
    # want to run this for both current and previous timnestep
    if t.dtype==torch.long:
        alpha = scheduler.alphas_cumprod[t]
        return alpha, 1-alpha
    
    if t<0:
        return scheduler.final_alpha_cumprod, 1 - scheduler.final_alpha_cumprod

    
    low = t.floor().long()
    high = t.ceil().long()
    rem = t - low
    
    low_alpha = scheduler.alphas_cumprod[low]
    high_alpha = scheduler.alphas_cumprod[high]
    interpolated_alpha = low_alpha * rem + high_alpha * (1-rem)
    interpolated_beta = 1 - interpolated_alpha
    return interpolated_alpha, interpolated_beta

# A DDIM forward step function
@torch.no_grad()
def forward_step(
    self,
    model_output,
    timestep: torch.Tensor,
    sample,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    return_dict: bool = True,
    use_double=False,
) :
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps
        
    if timestep > self.timesteps.max():
        raise NotImplementedError("Need to double check what the overflow is")
  
    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)
    
    
    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev)**0.5)
    first_term =  (1./alpha_quotient) * sample
    second_term = (1./alpha_quotient) * (beta_prod_t ** 0.5) * model_output
    third_term = ((1 - alpha_prod_t_prev)**0.5) * model_output
    return first_term - second_term + third_term

# A DDIM reverse step function, the inverse of above
@torch.no_grad()
def reverse_step(
    self,
    model_output,
    timestep: int,
    sample,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    return_dict: bool = True,
    use_double=False,
) :
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps
   
    if timestep > self.timesteps.max():
        raise NotImplementedError
    else:
        alpha_prod_t = self.alphas_cumprod[timestep]
        
    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)
    
    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev)**0.5)
    
    first_term =  alpha_quotient * sample
    second_term = ((beta_prod_t)**0.5) * model_output
    third_term = alpha_quotient * ((1 - alpha_prod_t_prev)**0.5) * model_output
    return first_term + second_term - third_term  
 


@torch.no_grad()
def coupled_stablediffusion(
    model: torch.nn.Module,
    prompt_edit=None,
    guidance_scale=7.0,
    steps=50,
    seed=1,
    init_image_latents: torch.Tensor=None,
    conditioning_conditional=None,
    conditioning_unconditional=None,
    init_image_strength=1.0,
    run_baseline=False,
    use_lms=False,
    leapfrog_steps=True,
    reverse=False,
    return_latents=False,
    fixed_starting_latent=None,
    beta_schedule='scaled_linear',
    mix_weight=0.93,
    device='cuda',
    sampling_kwargs={}
):
    assert not use_lms, "Can't invert LMS the same as DDIM"
    if run_baseline:
        leapfrog_steps=False
    
    t_limit = steps - int(steps * init_image_strength)

    if reverse:
        latent = init_image_latents
    else:
        #Generate random normal noise
        noise = torch.randn(init_image_latents.shape,
                            device=device,
                           dtype=torch.float64)
        if fixed_starting_latent is None:
            latent = noise
            t_limit = 0
        else:
            if isinstance(fixed_starting_latent, list):
                latent = [l.clone() for l in fixed_starting_latent]
            else:
                latent = fixed_starting_latent.clone()
    if isinstance(latent, list): # initializing from pair of images
        latent_pair = latent
    else: # initializing from noise
        latent_pair = [latent.clone(), latent.clone()]
    
    assert steps > 0, "steps must be greater than 0"

    # Set inference timesteps to scheduler
    schedulers = []
    for i in range(2):
        # num_raw_timesteps = max(1000, steps)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule=beta_schedule,
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False
        )
        scheduler.set_timesteps(steps)
        schedulers.append(scheduler)

    #Process prompt editing (if running Prompt-to-Prompt)
    if prompt_edit is not None:
        raise NotImplementedError("Prompt-to-Prompt not implemented yet")

    # timesteps is a long tensor of t_limit to t_max - t_limit to zero
    timesteps = schedulers[0].timesteps[t_limit:]
    # if we use reversion technique, we will noise the image starting from 0
    # thus we reverse the process and start from 0 to t_max - t_limit
    if reverse: timesteps = timesteps.flip(0)
    sigmas = (((1 - schedulers[0].alphas_cumprod) / schedulers[0].alphas_cumprod)**0.5).to(device)
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False):
        if (reverse) and (not run_baseline):
            # Reverse mixing layer
            new_latents = [l.clone() for l in latent_pair]
            new_latents[1] = (new_latents[1].clone() - (1-mix_weight)*new_latents[0].clone()) / mix_weight
            new_latents[0] = (new_latents[0].clone() - (1-mix_weight)*new_latents[1].clone()) / mix_weight
            latent_pair = new_latents

        # alternate EDICT steps
        for latent_i in range(2):
            if run_baseline and latent_i==1:
                continue # just have one sequence for baseline
            # this modifies latent_pair[i] while using 
            # latent_pair[(i+1)%2]
            if reverse and (not run_baseline):
                if leapfrog_steps:
                    # what i would be from going other way
                    orig_i = len(timesteps) - (i+1) 
                    offset = (orig_i+1) % 2
                    latent_i = (latent_i + offset) % 2
                else:
                    # Do 1 then 0
                    latent_i = int(not latent_i)
            else:
                if leapfrog_steps:
                    offset = i%2
                    latent_i = (latent_i + offset) % 2

            latent_j = (int(not latent_i)) if not run_baseline else latent_i

            latent_model_input = latent_pair[latent_j]
            latent_base = latent_pair[latent_i]

            #Predict the unconditional noise residual
            noise_pred_uncond = model.denoiser(
                model.model,
                latent_model_input,
                torch.full((latent_model_input.shape[0],), sigmas[t], device=latent_model_input.device),
                clone_batch_element(conditioning_unconditional),
                **sampling_kwargs
            )

            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = model.denoiser(
                model.model,
                latent_model_input,
                torch.full((latent_model_input.shape[0],), sigmas[t], device=latent_model_input.device),
                clone_batch_element(conditioning_conditional),
                **sampling_kwargs
            )

            #Edit the Cross-Attention layer activations
            if prompt_edit is not None:
                t_scale = t / schedulers[0].num_train_timesteps
                raise NotImplementedError("Prompt-to-Prompt not implemented yet")

            #Perform guidance
            grad = (noise_pred_cond - noise_pred_uncond)
            noise_pred = noise_pred_uncond + guidance_scale * grad

            step_call = reverse_step if reverse else forward_step
            new_latent = step_call(
                schedulers[latent_i],
                noise_pred,
                t,
                latent_base
            )# .prev_sample
            new_latent = new_latent.to(latent_base.dtype)

            latent_pair[latent_i] = new_latent

        if (not reverse) and (not run_baseline):
            # Mixing layer (contraction) during generative process
            new_latents = [l.clone() for l in latent_pair]
            new_latents[0] = (mix_weight*new_latents[0] + (1-mix_weight)*new_latents[1]).clone() 
            new_latents[1] = ((1-mix_weight)*new_latents[0] + (mix_weight)*new_latents[1]).clone() 
            latent_pair = new_latents

    #scale and decode the image latents with vae, can return latents instead of images
    if reverse or return_latents:
        return latent_pair
    return latent_pair
