def main():
    model, data, config = load_model_and_data_and_config()
    element_idx = 0
    element = load_idx(data, element_idx, model.device)
    batch_element = clone_batch_element(element)
    samples, z_first_stage, sampled_latent_steps, psnr, ssim = sample_default(model, data, element_idx)
    decoded_latents = decode_latents(model, sampled_latent_steps)
    decoded_latents, psnrs, ssims = decode_latents_and_metrics(model, decoded_latents, batch_element[model.input_key])
    
    sampling_figure = plot_sampling_results(psnrs, ssims)
    sampling_figure.savefig("sampling_results.png")
    plt.close(sampling_figure)
    del sampling_figure
