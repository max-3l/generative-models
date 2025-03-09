from einops import rearrange, repeat
import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if "cond_view" in c:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                cond_view=c.get("cond_view", None),
                cond_motion=c.get("cond_motion", None),
                **kwargs,
            )
        else:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                **kwargs,
            )

class OpenAIVideoWrapper(IdentityWrapper):
    def __init__(self, diffusion_model, num_frames: int, compile_model: bool = False):
        super().__init__(diffusion_model, compile_model)
        self.num_frames = num_frames

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:

        for k in ["crossattn", "concat"]:
            c[k] = repeat(c[k], "b ... -> b t ...", t=self.num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=self.num_frames)

        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if "cond_view" in c:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                cond_view=c.get("cond_view", None),
                cond_motion=c.get("cond_motion", None),
                **kwargs,
            )
        else:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                **kwargs,
            )

class CustomTransformerVideoWrapper(IdentityWrapper):
    def __init__(self, diffusion_model, num_frames: int, compile_model: bool = False):
        super().__init__(diffusion_model, compile_model)
        self.num_frames = num_frames

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:

        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            concat=c.get("concat", torch.Tensor([]).type_as(x)),
            num_frames = self.num_frames,
        )
