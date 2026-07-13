import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import build_model, load_model_weights
import core.liteanystereov2 as liteanystereov2
import core.submodule as submodule


def _context_upsample(depth_low, up_weights):
    b, c, h, w = depth_low.shape
    w3 = torch.eye(9, dtype=depth_low.dtype, device=depth_low.device).reshape(9, 1, 3, 3)
    depth_unfold = F.conv2d(depth_low.reshape(b,c,h,w), w3, padding=1).reshape(b,-1,h,w)
    depth_unfold = F.interpolate(depth_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
    depth = torch.sum(depth_unfold*up_weights, dim=1, keepdim=True)
    return depth


def _build_correlation_volume(left_feature, right_feature, max_disp):
    B, C, H, W = left_feature.shape
    left_volume = left_feature.unsqueeze(2).expand(B, C, max_disp, H, W)
    padded_right = F.pad(right_feature, (max_disp - 1, 0, 0, 0))
    unfolded_right = torch.stack([padded_right[:, :, :, i:i+W] for i in range(max_disp)], dim=3)
    right_volume = torch.flip(unfolded_right, [3]).permute(0, 1, 3, 2, 4)
    cost_volume = (left_volume * right_volume).mean(dim=1)
    return cost_volume.contiguous()


class Wrapper(nn.Module):
    def __init__(self, model, max_disp):
        super().__init__()
        self.model = model
        self.max_disp = max_disp

    def forward(self, left, right):
        return self.model(left, right, max_disp=self.max_disp, test_mode=True)


def export(version, model_size, restore_ckpt, width, height, max_disp, output_name):
    submodule.context_upsample = _context_upsample
    liteanystereov2.context_upsample = _context_upsample
    liteanystereov2.build_correlation_volume = _build_correlation_volume

    model = build_model(version, model_size=model_size, max_disp=max_disp)
    checkpoint = torch.load(restore_ckpt, map_location="cpu")
    load_model_weights(model, checkpoint, strict=True)
    model = Wrapper(model.eval(), max_disp=max_disp).eval()

    left = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32)
    right = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32)

    torch.onnx.export(
        model, (left, right), output_name,
        input_names=["left", "right"], output_names=["disparity"],
        opset_version=18,
        dynamo=False
    )
    print("Saved", output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LAS2 checkpoint to ONNX.")
    parser.add_argument("--version", default="las2")
    parser.add_argument("--model_size", default="s")
    parser.add_argument("--restore_ckpt", default="./checkpoints/LAS2_S.pth")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--output_name", default="./checkpoints/las2_s_640x384.onnx")
    args = parser.parse_args()
    export(version=args.version, model_size=args.model_size,
           restore_ckpt=args.restore_ckpt, width=args.width, height=args.height,
           max_disp=args.max_disp, output_name=args.output_name)
