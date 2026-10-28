"""Export a fixed 200-frame ERes2Net-base verification model to ONNX.

Uses the pinned, bundled 3D-Speaker implementation plus torch, onnx, onnxsim
and onnxruntime. The checkpoint is the public 200k-speaker Chinese model. The
network is evaluated as trained; this script does not fine-tune its weights.
"""
import argparse
from pathlib import Path
import sys

# Resolve from the repository, not an unrelated globally installed speakerlab.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "3d-speaker"))

import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify
import torch
from torch import nn
from speakerlab.models.eres2net.ERes2Net import ERes2Net


class FixedFeatures(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fbank):
        # Existing board Fbank emits [1,1,80,200] feature-major float32.
        model = self.model
        x = torch.relu(model.bn1(model.conv1(fbank)))
        first = model.layer1(x)
        second = model.layer2(first)
        fused12 = model.fuse_mode12(second, model.layer1_downsample(first))
        third = model.layer3(second)
        fused123 = model.fuse_mode123(third, model.layer2_downsample(fused12))
        fourth = model.layer4(third)
        return model.fuse_mode1234(fourth, model.layer3_downsample(fused123))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = ERes2Net(feat_dim=80, embedding_size=512, m_channels=32).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    wrapped = FixedFeatures(model).eval()
    torch.manual_seed(14)
    x = torch.randn(1, 1, 80, 200)
    target = args.output / "eres2net.onnx"
    torch.onnx.export(wrapped, x, str(target), input_names=["fbank"],
                      output_names=["frames"], opset_version=12)
    candidate, valid = simplify(onnx.load(target))
    if not valid:
        raise RuntimeError("ONNX simplifier rejected ERes2Net")
    onnx.save(candidate, target)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = wrapped(x).numpy()
    obtained = session.run(None, {"fbank": x.numpy()})[0]
    np.testing.assert_allclose(obtained, expected, rtol=5e-4, atol=5e-4)
    with torch.no_grad():
        pooled = model.pool(wrapped(x))
        embedded = model.seg_1(pooled).numpy()
        original = model(x[:, 0].transpose(1, 2).contiguous()).numpy()
        np.testing.assert_allclose(embedded, original, rtol=5e-4, atol=5e-4)
    frames = obtained.reshape(1, -1, obtained.shape[-1])
    pooled_np = np.concatenate([frames.mean(-1),
                                np.sqrt(frames.var(-1, ddof=1) + 1e-8)], axis=1)
    weight = model.seg_1.weight.detach().numpy()
    bias = model.seg_1.bias.detach().numpy()
    post = pooled_np @ weight.T + bias
    np.testing.assert_allclose(post, embedded, rtol=5e-4, atol=5e-4)
    np.concatenate([weight.ravel(), bias]).astype("<f4").tofile(args.output / "eres2net_tail.f32")
    print("ERes2Net backbone and CPU tail verified", x.shape, obtained.shape, embedded.shape, flush=True)


if __name__ == "__main__":
    main()
