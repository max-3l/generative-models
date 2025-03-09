"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn

from ..util import get_ckpt_path
from sgm.modules.autoencoding.lpips.swin_unetr.swin_unetr import UniversalUNETR


class LPIPSUNetR(nn.Module):
    # Learned perceptual metric with swin unetr encoder
    def __init__(self, swin_unetr_encoder_state_path: str, img_size=256, checkpointing=True, swin_unetr_encoder_fidelity_loss_stages=5, use_dropout=True):
        super().__init__()
        self.net = swinUNetR(swin_unetr_encoder_state_path=swin_unetr_encoder_state_path, img_size=img_size, checkpointing=checkpointing, swin_unetr_encoder_fidelity_loss_stages=swin_unetr_encoder_fidelity_loss_stages)
        self.linears = torch.nn.ModuleList([
            NetLinLayer(24, 1, use_dropout=use_dropout) for _ in range(swin_unetr_encoder_fidelity_loss_stages)
        ])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        outs0, outs1 = self.net(input), self.net(target)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(outs0)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(outs0))
        ]
        val = res[0]
        for l in range(1, len(outs0)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)
    
class swinUNetR(nn.Module):
    def __init__(self, swin_unetr_encoder_state_path: str, img_size=256, checkpointing=True, swin_unetr_encoder_fidelity_loss_stages=5):
        super().__init__()
        self.input_img_size = img_size
        self.model = UniversalUNETR(img_size=(128, 128, 128))
        state_dict = torch.load(swin_unetr_encoder_state_path)["net"]
        state_dict = { k.replace("module." ,""): e for k,e in state_dict.items() if "module." in k }
        self.model.load_state_dict(state_dict, strict=False)
        self.model.swinViT.use_checkpoint = checkpointing
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        assert 0 < swin_unetr_encoder_fidelity_loss_stages < 7, "The number of stages for the swin_unetr_encoder_fidelity_loss must be less than 7"
        self.swin_unetr_encoder_fidelity_loss_stages = swin_unetr_encoder_fidelity_loss_stages

    def forward(self, x):
        # Scale the input to the desired range
        x = torch.nn.functional.interpolate(x, size=(128, 128, 128), mode='trilinear', align_corners=False, antialias=False)
        return self.model.encode_stages(x, self.swin_unetr_encoder_fidelity_loss_stages)

def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x, keepdim=True):
    return x.mean([2, 3, 4], keepdim=keepdim)
