import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from tqdm import tqdm
import torch
import edict.sampling as e_sampling
from edict.utils import load_model_and_data_and_config, load_idx

def main():
    # Plot PSNRs and SSIMs
    steps_hist = []
    guidance_hist = []
    edict_baseline_psnrs = []
    edict_baseline_ssims = []
    edict_0_psnrs = []
    edict_0_ssims = []
    edict_1_psnrs = []
    edict_1_ssims = []
    edict_closest_0_psnrs = []
    edict_closest_0_ssims = []
    edict_closest_1_psnrs = []
    edict_closest_1_ssims = []
    edict_closest_baseline_psnrs = []
    edict_closest_baseline_ssims = []

    latent_mse_edict_0 = []
    latent_mse_edict_1 = []
    latent_mse_baseline_0 = []
    latent_mse_closest_0 = []
    latent_mse_closest_1 = []
    latent_mse_closest_baseline_0 = []

    latent_mse_to_closest_edict_0 = []
    latent_mse_to_closest_edict_1 = []
    latent_mse_to_closest_baseline_0 = []
    latent_mse_to_closest_closest_0 = []
    latent_mse_to_closest_closest_1 = []
    latent_mse_to_closest_closest_baseline_0 = []

    model, data, config = load_model_and_data_and_config()

    first_element = load_idx(data, 0, "cpu")
    closest_element = load_idx(data, 330, "cpu")

    name_to_patch = {
        "Baseline": mpatches.Patch(color='C0', label='Baseline'),
        "Latents 0 EDICT": mpatches.Patch(color='C1', label='Latents 0 EDICT'),
        "Latents 1 EDICT": mpatches.Patch(color='C2', label='Latents 1 EDICT'),
        "Avg EDICT": mpatches.Patch(color='C3', label='Avg EDICT'),
        "Baseline from closest z": mpatches.Patch(color='C4', label='Baseline from closest z'),
        "Latents 0 EDICT from closest z": mpatches.Patch(color='C5', label='Latents 0 EDICT from closest z'),
        "Latents 1 EDICT from closest z": mpatches.Patch(color='C6', label='Latents 1 EDICT from closest z'),
        "Avg EDICT from closest z": mpatches.Patch(color='C7', label='Avg EDICT from closest z'),
    }

    for guidance in tqdm([7.0], desc="Guidance"):
        for steps in tqdm([1, 2, 5, 10, 20, 50, 75, 100, 125, 150, 175, 200, 250, 275, 300, 350, 400, 450, 500], desc="Steps"):
            edict_result = e_sampling.sample_edict(
                model=model,
                data=data,
                element_idx=0,
                baseline=False,
                steps=steps,
                guidance_scale=guidance,
            )

            edict_result_closest = e_sampling.sample_edict_from_closest(
                model=model,
                data=data,
                element_idx=0,
                closest_element_idx=330,
                baseline=False,
                steps=steps,
                guidance_scale=guidance,
            )

            edict_baseline_result = e_sampling.sample_edict(
                model=model,
                data=data,
                element_idx=0,
                baseline=True,
                steps=steps,
                guidance_scale=guidance,
            )

            edict_baseline_result_closest = e_sampling.sample_edict_from_closest(
                model=model,
                data=data,
                element_idx=0,
                closest_element_idx=330,
                baseline=True,
                steps=steps,
                guidance_scale=guidance,
            )

            steps_hist.append(steps)
            guidance_hist.append(guidance)
            edict_baseline_psnrs.append(edict_baseline_result.psnr_denoised[0].cpu().item())
            edict_baseline_ssims.append(edict_baseline_result.ssim_denoised[0].cpu().item())
            edict_0_psnrs.append(edict_result.psnr_denoised[0].cpu().item())
            edict_0_ssims.append(edict_result.ssim_denoised[0].cpu().item())
            edict_1_psnrs.append(edict_result.psnr_denoised[1].cpu().item())
            edict_1_ssims.append(edict_result.ssim_denoised[1].cpu().item())
            edict_closest_0_psnrs.append(edict_result_closest.psnr_denoised[0].cpu().item())
            edict_closest_0_ssims.append(edict_result_closest.ssim_denoised[0].cpu().item())
            edict_closest_1_psnrs.append(edict_result_closest.psnr_denoised[1].cpu().item())
            edict_closest_1_ssims.append(edict_result_closest.ssim_denoised[1].cpu().item())
            edict_closest_baseline_psnrs.append(edict_baseline_result_closest.psnr_denoised[0].cpu().item())
            edict_closest_baseline_ssims.append(edict_baseline_result_closest.ssim_denoised[0].cpu().item())

            z_first_stage = edict_result.z_first_stage.cpu().flatten()
            z_first_stage_closest = edict_result_closest.z_first_stage.cpu().flatten()

            latent_mse_edict_0.append((z_first_stage - edict_result.latents_reversed_denoised[0].flatten().cpu()).pow(2).mean().item())
            latent_mse_edict_1.append((z_first_stage - edict_result.latents_reversed_denoised[1].flatten().cpu()).pow(2).mean().item())
            latent_mse_baseline_0.append((z_first_stage - edict_baseline_result.latents_reversed_denoised[0].flatten().cpu()).pow(2).mean().item())
            latent_mse_closest_0.append((z_first_stage - edict_result_closest.latents_reversed_denoised_closest[0].flatten().cpu()).pow(2).mean().item())
            latent_mse_closest_1.append((z_first_stage - edict_result_closest.latents_reversed_denoised_closest[1].flatten().cpu()).pow(2).mean().item())
            latent_mse_closest_baseline_0.append((z_first_stage - edict_baseline_result_closest.latents_reversed_denoised_closest[0].flatten().cpu()).pow(2).mean().item())

            latent_mse_to_closest_edict_0.append((z_first_stage_closest - edict_result.latents_reversed_denoised[0].flatten().cpu()).pow(2).mean().item())
            latent_mse_to_closest_edict_1.append((z_first_stage_closest - edict_result.latents_reversed_denoised[1].flatten().cpu()).pow(2).mean().item())
            latent_mse_to_closest_baseline_0.append((z_first_stage_closest - edict_baseline_result.latents_reversed_denoised[0].flatten().cpu()).pow(2).mean().item())
            latent_mse_to_closest_closest_0.append((z_first_stage_closest - edict_result_closest.latents_reversed_denoised_closest[0].flatten().cpu()).pow(2).mean().item())
            latent_mse_to_closest_closest_1.append((z_first_stage_closest - edict_result_closest.latents_reversed_denoised_closest[1].flatten().cpu()).pow(2).mean().item())
            latent_mse_to_closest_closest_baseline_0.append((z_first_stage_closest - edict_baseline_result_closest.latents_reversed_denoised_closest[0].flatten().cpu()).pow(2).mean().item())

            # Save lists to json file
            with open("edict_outputs/psnr_ssim_results.json", "w") as f:
                json.dump({
                    "steps_hist": steps_hist,
                    "guidance_hist": guidance_hist,
                    "edict_baseline_psnrs": edict_baseline_psnrs,
                    "edict_baseline_ssims": edict_baseline_ssims,
                    "edict_0_psnrs": edict_0_psnrs,
                    "edict_0_ssims": edict_0_ssims,
                    "edict_1_psnrs": edict_1_psnrs,
                    "edict_1_ssims": edict_1_ssims,
                    "edict_closest_0_psnrs": edict_closest_0_psnrs,
                    "edict_closest_0_ssims": edict_closest_0_ssims,
                    "edict_closest_1_psnrs": edict_closest_1_psnrs,
                    "edict_closest_1_ssims": edict_closest_1_ssims,
                    "edict_closest_baseline_psnrs": edict_closest_baseline_psnrs,
                    "edict_closest_baseline_ssims": edict_closest_baseline_ssims,
                    "latent_mse_edict_0": latent_mse_edict_0,
                    "latent_mse_edict_1": latent_mse_edict_1,
                    "latent_mse_baseline_0": latent_mse_baseline_0,
                    "latent_mse_closest_0": latent_mse_closest_0,
                    "latent_mse_closest_1": latent_mse_closest_1,
                    "latent_mse_closest_baseline_0": latent_mse_closest_baseline_0,
                    "latent_mse_to_closest_edict_0": latent_mse_to_closest_edict_0,
                    "latent_mse_to_closest_edict_1": latent_mse_to_closest_edict_1,
                    "latent_mse_to_closest_baseline_0": latent_mse_to_closest_baseline_0,
                    "latent_mse_to_closest_closest_0": latent_mse_to_closest_closest_0,
                    "latent_mse_to_closest_closest_1": latent_mse_to_closest_closest_1,
                    "latent_mse_to_closest_closest_baseline_0": latent_mse_to_closest_closest_baseline_0,
                }, f, indent=4)

            fig, axs = plt.subplots(3, 6, figsize=(15, 7.5))
            fig.suptitle(f"EDICT with {steps} steps and {guidance:.2f} cls. free guidance", fontsize=16)
            axs[0, 0].imshow(edict_result.x[67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[0, 1].imshow(edict_result.samples_reversed_denoised[0][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[0, 2].imshow(edict_result.samples_reversed_denoised[1][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[0, 3].imshow(edict_baseline_result.samples_reversed_denoised[0][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[0, 4].imshow(edict_result.samples_reverse_noised[0][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[0, 5].imshow(edict_result.samples_reverse_noised[1][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")

            axs[0, 0].set_title(f"Original {first_element['scan_id'][0]}")
            axs[0, 1].set_title("Latents 0 EDICT")
            axs[0, 2].set_title("Latents 1 EDICT")
            axs[0, 3].set_title("DDIM Baseline")
            axs[0, 4].set_title("Latents 0 EDICT\nnoised (50steps)")
            axs[0, 5].set_title("Latents 1 EDICT\nnoised (50steps)")
            axs[1, 0].imshow(closest_element["ct"][0][:,:,:,67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[1, 1].imshow(edict_result_closest.samples_reversed_denoised[0][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[1, 2].imshow(edict_result_closest.samples_reversed_denoised[1][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[1, 3].imshow(edict_baseline_result_closest.samples_reversed_denoised[0][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[1, 4].imshow(edict_result_closest.samples_reverse_noised[0][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[1, 5].imshow(edict_result_closest.samples_reverse_noised[1][67].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[1, 0].set_title(f"Closest {closest_element['scan_id'][0]}")
            axs[1, 1].set_title("Latents 0 EDICT from closest z")
            axs[1, 2].set_title("Latents 1 EDICT from closest z")
            axs[1, 3].set_title("DDIM Baseline from closest z")
            axs[1, 4].set_title(f"Latents 0 EDICT\nnoised ({steps} steps)\nfrom closest z")
            axs[1, 5].set_title(f"Latents 1 EDICT\nnoised ({steps} steps)\nfrom closest z")

            axs[2, 0].imshow(first_element["xrays"][0, 0].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[2, 1].imshow(first_element["xrays"][0, 1].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[2, 2].imshow(closest_element["xrays"][0, 0].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[2, 3].imshow(closest_element["xrays"][0, 1].permute(1, 2, 0).cpu().numpy(), vmin=-1, vmax=1, cmap="gray")
            axs[2, 0].set_title("Original Xray 0")
            axs[2, 1].set_title("Original Xray 1")
            axs[2, 2].set_title("Closest Xray 0")
            axs[2, 3].set_title("Closest Xray 1")
            # Set title title font size for all subplots
            for ax in axs.flat:
                ax.title.set_fontsize(8)
                ax.axis('off')
            fig.tight_layout()
            fig.show()
            fig.savefig(f"edict_outputs/edict_{steps}_steps_{guidance:.2f}_guidance.png", dpi=300)
            del fig, axs
            plt.close("all")

            fig, axs = plt.subplots(2, 2, figsize=(17, 5))
            fig.suptitle(f"EDICT img to img metrics using {guidance:.2f} classifier free guidance", fontsize=16)
            axs[0, 0].plot(steps_hist, edict_baseline_psnrs, label="Baseline", color=name_to_patch["Baseline"].get_facecolor())
            axs[0, 0].plot(steps_hist, edict_0_psnrs, label="Latents 0 EDICT", color=name_to_patch["Latents 0 EDICT"].get_facecolor())
            axs[0, 0].plot(steps_hist, edict_1_psnrs, label="Latents 1 EDICT", color=name_to_patch["Latents 1 EDICT"].get_facecolor())
            axs[0, 0].plot(steps_hist, (torch.tensor(edict_0_psnrs) + torch.tensor(edict_1_psnrs)) / 2, label="Avg EDICT", color=name_to_patch["Avg EDICT"].get_facecolor())
            axs[0, 0].plot(steps_hist, edict_closest_baseline_psnrs, label="Baseline from closest z", color=name_to_patch["Baseline from closest z"].get_facecolor())
            axs[0, 0].plot(steps_hist, edict_closest_0_psnrs, label="Latents 0 EDICT from closest z", color=name_to_patch["Latents 0 EDICT from closest z"].get_facecolor())
            axs[0, 0].plot(steps_hist, edict_closest_1_psnrs, label="Latents 1 EDICT from closest z", color=name_to_patch["Latents 1 EDICT from closest z"].get_facecolor())
            axs[0, 0].plot(steps_hist, (torch.tensor(edict_closest_1_psnrs) + torch.tensor(edict_closest_0_psnrs)) / 2, label="Avg EDICT from closest z", color=name_to_patch["Avg EDICT from closest z"].get_facecolor())
            axs[0, 0].set_title("PSNR")
            axs[0, 0].set_xlabel("Step")
            axs[0, 0].set_ylabel("PSNR")
            # axs[0, 0].legend()

            axs[0, 1].plot(steps_hist, edict_baseline_ssims, label="Baseline", color=name_to_patch["Baseline"].get_facecolor())
            axs[0, 1].plot(steps_hist, edict_0_ssims, label="Latents 0 EDICT", color=name_to_patch["Latents 0 EDICT"].get_facecolor())
            axs[0, 1].plot(steps_hist, edict_1_ssims, label="Latents 1 EDICT", color=name_to_patch["Latents 1 EDICT"].get_facecolor())
            axs[0, 1].plot(steps_hist, (torch.tensor(edict_0_ssims) + torch.tensor(edict_1_ssims)) / 2, label="Avg EDICT", color=name_to_patch["Avg EDICT"].get_facecolor())
            axs[0, 1].plot(steps_hist, edict_closest_baseline_ssims, label="Baseline from closest z", color=name_to_patch["Baseline from closest z"].get_facecolor())
            axs[0, 1].plot(steps_hist, edict_closest_0_ssims, label="Latents 0 EDICT from closest z", color=name_to_patch["Latents 0 EDICT from closest z"].get_facecolor())
            axs[0, 1].plot(steps_hist, edict_closest_1_ssims, label="Latents 1 EDICT from closest z", color=name_to_patch["Latents 1 EDICT from closest z"].get_facecolor())
            axs[0, 1].plot(steps_hist, (torch.tensor(edict_closest_1_ssims) + torch.tensor(edict_closest_0_ssims)) / 2, label="Avg EDICT from closest z", color=name_to_patch["Avg EDICT from closest z"].get_facecolor())
            axs[0, 1].set_title("SSIM")
            axs[0, 1].set_xlabel("Step")
            axs[0, 1].set_ylabel("SSIM")
            # axs[0, 1].legend()

            axs[1, 0].plot(steps_hist, latent_mse_baseline_0, label="Baseline", color=name_to_patch["Baseline"].get_facecolor())
            axs[1, 0].plot(steps_hist, latent_mse_edict_0, label="Latents 0 EDICT", color=name_to_patch["Latents 0 EDICT"].get_facecolor())
            axs[1, 0].plot(steps_hist, latent_mse_edict_1, label="Latents 1 EDICT", color=name_to_patch["Latents 1 EDICT"].get_facecolor())
            axs[1, 0].plot(steps_hist, (torch.tensor(latent_mse_edict_0) + torch.tensor(latent_mse_edict_1)) / 2, label="Avg EDICT", color=name_to_patch["Avg EDICT"].get_facecolor())
            axs[1, 0].plot(steps_hist, latent_mse_closest_baseline_0, label="Baseline from closest z", color=name_to_patch["Baseline from closest z"].get_facecolor())
            axs[1, 0].plot(steps_hist, latent_mse_closest_0, label="Latents 0 EDICT from closest z", color=name_to_patch["Latents 0 EDICT from closest z"].get_facecolor())
            axs[1, 0].plot(steps_hist, latent_mse_closest_1, label="Latents 1 EDICT from closest z", color=name_to_patch["Latents 1 EDICT from closest z"].get_facecolor())
            axs[1, 0].plot(steps_hist, (torch.tensor(latent_mse_closest_0) + torch.tensor(latent_mse_closest_1)) / 2, label="Avg EDICT from closest z", color=name_to_patch["Avg EDICT from closest z"].get_facecolor())

            axs[1, 0].set_title("Latent MSE to target element embedding vector")
            axs[1, 0].set_xlabel("Step")
            axs[1, 0].set_ylabel("Latent MSE to\ntarget element\nembedding vector")
            axs[1, 0].set_yscale("log")
            # axs[1, 0].legend()

            axs[1, 1].plot(steps_hist, latent_mse_to_closest_baseline_0, label="Baseline", color=name_to_patch["Baseline"].get_facecolor())
            axs[1, 1].plot(steps_hist, latent_mse_to_closest_edict_0, label="Latents 0 EDICT", color=name_to_patch["Latents 0 EDICT"].get_facecolor())
            axs[1, 1].plot(steps_hist, latent_mse_to_closest_edict_1, label="Latents 1 EDICT", color=name_to_patch["Latents 1 EDICT"].get_facecolor())
            axs[1, 1].plot(steps_hist, (torch.tensor(latent_mse_to_closest_edict_0) + torch.tensor(latent_mse_to_closest_edict_1)) / 2, label="Avg EDICT", color=name_to_patch["Avg EDICT"].get_facecolor())
            axs[1, 1].plot(steps_hist, latent_mse_to_closest_closest_baseline_0, label="Baseline from closest z", color=name_to_patch["Baseline from closest z"].get_facecolor())
            axs[1, 1].plot(steps_hist, latent_mse_to_closest_closest_0, label="Latents 0 EDICT from closest z", color=name_to_patch["Latents 0 EDICT from closest z"].get_facecolor())
            axs[1, 1].plot(steps_hist, latent_mse_to_closest_closest_1, label="Latents 1 EDICT from closest z", color=name_to_patch["Latents 1 EDICT from closest z"].get_facecolor())
            axs[1, 1].plot(steps_hist, (torch.tensor(latent_mse_to_closest_closest_0) + torch.tensor(latent_mse_to_closest_closest_1)) / 2, label="Avg EDICT from closest z", color=name_to_patch["Avg EDICT from closest z"].get_facecolor())

            axs[1, 1].set_title("Latent MSE to closest element embedding vector")
            axs[1, 1].set_xlabel("Step")
            axs[1, 1].set_ylabel("Latent MSE\nto closest element\nembedding vector")
            axs[1, 1].set_yscale("log")
            # axs[1, 1].legend()

            fig.legend(loc=7, handles=name_to_patch.values(), fontsize=8)

            # Set title title font size for all subplots
            for ax in axs.flat:
                ax.title.set_fontsize(8)
                ax.yaxis.label.set_fontsize(8)
                ax.xaxis.label.set_fontsize(8)

            fig.tight_layout()
            fig.subplots_adjust(right=0.870)

            fig.savefig(f"edict_outputs/psnr_ssim_results.pdf", dpi=300)
            del fig, axs
            plt.close("all")

if __name__ == "__main__":
    main()