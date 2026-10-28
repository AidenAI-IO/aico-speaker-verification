# Model provenance and redistribution

The upstream model is [iic/speech_eres2net_base_200k_sv_zh-cn_16k-common](https://modelscope.cn/models/iic/speech_eres2net_base_200k_sv_zh-cn_16k-common), revision **`v1.0.0`**. Its `UPSTREAM_MODEL_CARD.md` and `upstream-configuration.json` are preserved here; model card declares **Apache License 2.0**. The exact upstream license text from the pinned 3D-Speaker source is `../../licenses/3D-Speaker-LICENSE`.

These artifacts were prepared from the public checkpoint for **RV1106**. The upstream PyTorch weights are unmodified; all other binaries are derived artifacts:

| File | Role | SHA-256 |
| --- | --- | --- |
| `pretrained_eres2net.pt` | Original pretrained PyTorch state dict, 512-D model | `baa48f0e3b666af561661d1c08695df09d2922e62170e9c5cb2864c6acf8486c` |
| `eres2net.onnx` | Float32 fixed-window **backbone only**, `[1,1,80,200] -> [1,512,10,25]` | `8d1511338836cafaa139543eca9e3499edd686daa53d6e2fa9e4f7e6c843e6c2` |
| `eres2net_backbone_rv1106_mmse.rknn` | NPU backbone, RKNN-Toolkit2 **2.5.0**, target `rv1106`, W8A8 MMSE | `f54ea8473ceccdc92d05f7bff67186ee57c25b74193a5b6ce0b0871c2fb8ad86` |
| `eres2net_tail.f32` | CPU statistics/projection tail weights; 512×10240 row-major float32 plus 512 bias | `4462f8f9a3780745fd2926f48e54da036276c91c509bfeef8c76ecaa7329b863` |

The fixed input is **200 Kaldi Fbank frames** (80 bins) from 16 kHz mono PCM16,
roughly **2.015 seconds**. The 4-second enrollment in the web demo uses three
*overlapping* 2.015-second windows, normalizes the three resulting 512-D
embeddings, averages, then normalizes again. It does not turn the model into a
4-second-input network.

`eres2net.onnx` and `eres2net_tail.f32` were exported by
`../../speaker_verification/export_eres2net.py` using the pinned
`../../vendor/3d-speaker/speakerlab/models/eres2net` implementation. The RKNN
file was built with `../../speaker_verification/build_eres2net.py`, with a
three-file public example calibration set using `tests/fixtures/public-3d-speaker/`.
Calibration data are **not** an evaluation set; use independent labeled short
utterances for threshold evaluation.

**Redistribution:** preserve Apache-2.0 license and upstream attribution, and
identify these conversions as modifications. The upstream's suggested decision
threshold is not appropriate for this quantized, short-window board pipeline;
**no production speaker threshold has been established here**. Rockchip's RKNN
runtime and SDK have separate terms; they are not bundled with these model files.
Voice recordings made by users are not included or implicitly licensed.
