from einops import rearrange, repeat
import torch
import torch.nn as nn
from sgm.modules.attention import BasicTransformerBlock
from sgm.modules.diffusionmodules.util import timestep_embedding
from diffusers.models.embeddings import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

class VideoTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        concat_context_dim: int,
        context_dim: int,
        attention_dropout: float = 0.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.initial_projection = nn.Linear(input_dim, hidden_dim)
        self.concat_projection = nn.Linear(concat_context_dim, hidden_dim)
        self.final_projection = nn.Linear(hidden_dim, output_dim)
        self.embed_dim = hidden_dim
        self.transformer = nn.ModuleList([
            # Self-attn -> Cross-attn -> Gated FF
            BasicTransformerBlock(
                dim=hidden_dim,
                n_heads=num_heads,
                d_head=hidden_dim // num_heads,
                dropout= attention_dropout,
                context_dim = context_dim,
                gated_ff = True,
                checkpoint = use_checkpoint,
                disable_self_attn = False,
                attn_mode = "softmax-xformers"
            )
            for _ in range(num_layers)
        ])

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor,
            concat: torch.Tensor,
            num_frames: int
        ):
        """
        Args:
            x: (b t) c_in h w - input tensor
            timesteps: (b) - timestep per sample
            context: (b t) n c - context tensor
            concat: b c h w - concat tensor
            num_timesteps: int - number of timesteps to process
        Returns:
            x: (b t) c_out h w - output
        """
        model_dtype = self.initial_projection.weight.dtype
        x = x.to(model_dtype)
        timesteps = timesteps.to(model_dtype)
        context = context.to(model_dtype)
        concat = concat.to(model_dtype)

        assert x.dtype == timesteps.dtype == context.dtype == concat.dtype, f"Data types must match: {x.dtype}, {timesteps.dtype}, {context.dtype}, {concat.dtype}"
        assert x.device == timesteps.device == context.device == concat.device, f"Devices must match: {x.device}, {timesteps.device}, {context.device}, {concat.device}"

        b, t, c_in, h, w = x.shape[0] // num_frames, num_frames, x.shape[1], x.shape[2], x.shape[3]
        
        x = self.initial_projection(rearrange(x, "bt c h w -> bt h w c"))
        x = rearrange(x, "(b t) h w c -> b t h w c", b=b, t=t)

        if timesteps.shape[0] == b * t:
            timesteps = rearrange(timesteps, "(b t) -> b t", b=b, t=t)[:, 0]
        x_timestep_embedding = timestep_embedding(timesteps, self.embed_dim).to(x.device, dtype=x.dtype)
        x_timestep_embedding = repeat(x_timestep_embedding, "b c -> b t h w c", t=t, h=h, w=w)
        x = x + x_timestep_embedding

        positional_emb = get_3d_sincos_pos_embed(self.embed_dim, (h, w), t, device=x.device, output_type="tensor").to(x.device, dtype=x.dtype)
        positional_emb = rearrange(positional_emb, "t (h w) c -> 1 t h w c", h=h, w=w)
        positional_emb = repeat(positional_emb, "1 t h w c -> (b 1) t h w c", b=b)
        x = x + positional_emb

        x = rearrange(x, "b t h w c -> b (t h w) c")

        # Concatenate concat context
        concat_h = concat.shape[2]
        concat_w = concat.shape[3]
        positional_emb = get_2d_sincos_pos_embed(self.embed_dim, (concat_h, concat_w), device=x.device, output_type="tensor").to(x.device, dtype=x.dtype)
        positional_emb = rearrange(positional_emb, "(h w) c -> 1 h w c", h=concat_h, w=concat_w)
        positional_emb = repeat(positional_emb, "1 h w c -> b (h w) c", b=b)

        concat = self.concat_projection(rearrange(concat, "b c h w -> b (h w) c"))
        concat = concat + positional_emb

        num_concat_tokens = concat.shape[1]
        x = torch.cat((x, concat), dim=1)

        n = context.shape[1]
        context_positional_emb = get_1d_sincos_pos_embed_from_grid(self.embed_dim, torch.arange(n), output_type="tensor").to(x.device, dtype=x.dtype)
        context_positional_emb = rearrange(context_positional_emb, "n c -> 1 n c")
        context_positional_emb = repeat(context_positional_emb, "1 n c -> b n c", b=b)
        context = context + context_positional_emb

        for layer in self.transformer:
            x = layer(x, context = context)

        x = x[:, :-num_concat_tokens, ...]
        x = self.final_projection(x)
        x = rearrange(x, "b (t h w) c -> (b t) c h w", b=b, t=t, h=h, w=w)

        return x
    