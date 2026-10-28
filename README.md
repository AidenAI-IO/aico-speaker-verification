# Aico Speaker Verification

RV1106 上的 **ERes2Net-base 中文声纹验证**：已验证的模型权重、RKNN 导出/量化脚本、C 语言前后处理、板载麦克风网页演示与回归测试。

本仓库只保留最终使用的 ERes2Net 路径，不依赖原实验仓库。它是**可复现的部署与诊断工具**，不是已标定的身份认证产品，也尚未接入 agent 的自动打断/拒绝逻辑。

## 包含什么

| 路径 | 内容 |
| --- | --- |
| `models/eres2net/` | **实际二进制文件**：原始 `.pt`、固定输入 backbone `.onnx`、已在板上跑通的 `.rknn`、CPU 输出层 `.f32`，以及 SHA-256、模型卡 |
| `speaker_verification/native/` | Fbank、RKNN zero-copy runner、512 维声纹输出层 |
| `speaker_verification/export_eres2net.py` / `build_eres2net.py` | 导出与 RV1106 MMSE INT8 量化 |
| `speaker_verification/raw_record.py` | 板载 ALSA 麦克风通道、50 Hz 去直流、1 秒预热 |
| `speaker_verification/web_demo.py` | 手动录制参考、实时相似度、参考录音诊断保存/试听 |
| `vendor/3d-speaker/` | 固定版本的 ERes2Net 实现，不必另外安装 speakerlab |
| `tests/` | 合成回归测试及上游公开示例音频；**没有用户录音** |
| `deploy/` | systemd 示例（需按安装路径修改，不会自动安装） |

模型文件合计约 **81 MiB**，直接存储在仓库，不是下载链接或 LFS 指针。核对文件完整性：

```sh
python3 -m speaker_verification.verify_models
# 或 (cd models/eres2net && sha256sum -c SHA256SUMS)
```

## 已验证的数据路径

```text
16 kHz / 单声道 / PCM16
  -> 80 维 Kaldi Fbank，200 帧，按时间去均值
  -> RV1106 NPU：ERes2Net backbone，INT8
  -> CPU：时间均值与无偏标准差 + 512 维线性输出层
  -> L2 归一化，余弦相似度
```

- 单次模型输入为 `[1,1,80,200]`，对应 **2.015 秒**，不是 4 秒模型。
- NPU 输出 `[1,512,10,25]`；CPU 输出层需按逻辑 NCHW 顺序读取。
- 网页参考录音为 4 秒：取起点 `0 / 0.9925 / 1.985` 秒的三个窗口，各自归一化后等权平均，再归一化；低音量窗口会跳过。
- 不要求长时间注册，不包含另一个“累积六秒语音”的实验分支。
- 当前是 RMS 静音过滤，**不是 VAD**；短首句、旁人交叠、环境噪声仍会影响声纹。

## 1. 构建板端程序

测试平台是 **RV1106 + Debian armhf/glibc**，Python 3.13、RKNN runtime 2.5.0、驱动 0.9.2。常见 Luckfox uClibc 镜像需使用匹配的编译器和 runtime，不能混用。

本仓库**不分发 Rockchip SDK 头文件、runtime 动态库或交叉编译器**，请从你有权使用的 SDK/固件获取。

```sh
# 在板上编译，或在主机使用对应 ARM hard-float glibc 交叉编译器。
make all CC=arm-linux-gnueabihf-gcc \
  RKNN_INCLUDE=/path/to/rknn/include \
  RKNN_LIB=/path/to/rknn/lib

# 只在主机验证前后处理，不需要 RKNN SDK：
make cpu
```

运行时把匹配的 `librknnmrt.so` 放到 `build/`，或放到系统的运行库搜索路径。网页会将 `build/` 加入 `LD_LIBRARY_PATH`。不要将你的 runtime 私自提交到仓库；它有独立的再分发许可。

关键：runner 已自行量化输入，所以必须设置 `pass_through=1`。省略会导致重复转换，产生能运行但声纹错误的结果。

## 2. 离线验证

在板上从仓库根目录运行（只读取公开 WAV，不打开麦克风）：

```sh
python3 -m speaker_verification.offline \
  --tools-dir build --model-dir models/eres2net \
  tests/fixtures/public-3d-speaker/speaker1_a_cn_16k.wav \
  tests/fixtures/public-3d-speaker/speaker1_b_cn_16k.wav \
  tests/fixtures/public-3d-speaker/speaker2_a_cn_16k.wav
```

每段只使用最前 2.015 秒；输出余弦相似度，不自动选阈值。完整原生流程也可单独运行：

```sh
build/fbank input.wav /tmp/features.f32
build/rv1106_bench models/eres2net/eres2net_backbone_rv1106_mmse.rknn \
  /tmp/features.f32 /tmp/frames.f32 1
build/eres2net_tail models/eres2net/eres2net_tail.f32 /tmp/frames.f32 /tmp/embedding.f32
```

## 3. 网页与板载麦克风

```sh
python3 -m speaker_verification.web_demo \
  --bind 192.168.42.1 --port 8768 \
  --tools-dir build --model-dir models/eres2net \
  --reference-dir /home/aico/eres2net-references
```

打开 `http://192.168.42.1:8768/`：录制参考声音 → 开始持续比较 → 停止采集。启动服务不会自动打开麦克风。

- 默认监听 `127.0.0.1`；示例仅绑定可信 USB 地址。**没有登录认证或 TLS，请勿暴露到公网。**
- 默认使用板子 `arecord` 的 `hw:CARD=rv1106acodec,DEV=0`，取左通道，不用浏览器麦克风；跳过 1 秒硬件预热。
- 实测该硬件有约 +0.19 直流偏置，使用 50 Hz 高通去直流；旁路整个系统 VQE（AEC/降噪/AGC），不是仅关 AEC。
- ALSA 直采独占麦克风；请先停止其它网页/agent 的采集。程序不会杀掉竞争进程，也不会修改系统 mixer/音频服务。
- 状态每 0.5 秒轮询；实时声纹约每秒计算一个 2.015 秒窗口，繁忙时跳过，不积压；45 秒无心跳自动停止。
- `--reference-dir` **可选**。指定后完整 4 秒 WAV/元数据保存为私有权限，可在页面试听；最多 30 段。清除会话/停服务不会删除它们，需操作者诊断后清理。不指定则没有持久录音。
- 参考声纹保存在内存里，重启服务需重新录制。监测音频不持久保存，不自动上传。
- 支持可信操作员 `--record-command '…'` 自定义采集命令，但 HTTP 不接受文件路径或命令。

如需 systemd，参考 `deploy/aico-speaker-verification.service`，核对用户、网卡、安装路径后再手动安装。不要与已有 8768 服务同时启动。

## 4. 复现导出与量化

在 Linux x86_64 主机使用 Python 3.10 的独立环境：

```sh
python3.10 -m venv .venv
.venv/bin/pip install -r speaker_verification/requirements-host.txt
.venv/bin/python -m speaker_verification.export_eres2net \
  models/eres2net/pretrained_eres2net.pt artifacts/eres2net
make cpu
.venv/bin/python -m speaker_verification.prepare_calibration artifacts/calibration \
  tests/fixtures/public-3d-speaker/*.wav
.venv/bin/python -m speaker_verification.build_eres2net \
  artifacts/eres2net artifacts/calibration/calibration.txt
```

导出会验证 PyTorch 与 ONNX backbone、拆分 CPU tail 的数值一致性。转换目标 `rv1106`，`w8a8`，MMSE，opset 12。不要将未拆分的完整图直接部署：它虽然能编译，板端会因统计池化的 `Sqrt` CPU fallback 失败。

随仓库的 RKNN 文件是实际已验证的构建物。重新构建不保证字节级一致：现存 RKNN 在 backbone 输出节点后续重命名为 `frames` 前构建；图运算不变。初始校准用参考库 Fbank，本仓库原生 Fbank有微小浮点差异；请比较数值、误差和真实性能，不要用相同 hash 作为重建成功条件。生产校准应替换成有代表性的数据，并与留出验证集严格分离。

## 性能与限制

| 项目 | 历史板端实测 |
| --- | ---: |
| NPU backbone | 约 52 ms / 2.015 秒窗口 |
| CPU 统计池化 + 流式读取线性输出层 | 约 173–232 ms |
| 文件式网页后端（含进程启动、模型加载、Fbank） | 约 0.66–1.08 秒 |
| 公开样本：同人 / 异人余弦 | 约 0.554 / 0.024 |
| 两段未标注原始板录音：六窗口量化/浮点声纹一致性 | 约 0.994–0.999 |

公开示例也用于校准，**不是独立准确率评测**；一致性不是 FAR/FRR。用户已在现场反馈此模型路径可用，但没有完成多说话人、短首句、距离/噪声的可靠性评估；不得把网页分数当作匹配概率或现成的生产阈值。CPU 输出层流式读权重为省内存设计，比较慢；低延迟 agent 应保持模型/权重常驻并重新 profile，而非每窗启动 CLI。

## 测试与许可

```sh
# 测试环境：NumPy + Kaldi 对照，另需 C 编译器、Node。
python3 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements.txt
PATH="$PWD/.venv/bin:$PATH" make test
```

32 项 Python 回归测试及独立 Node UI 检查覆盖输出层公式、非法输入、采集通道、去直流、4 秒原文/窗口一致性、固定参考、重置和超时、UI 按钮流程等；没有测试会主动打开麦克风。数值对照测试使用公开示例/合成数据。

### 本次独立仓库整理的验证范围

- 主工作区及 Linux 验证目录的四个模型 SHA-256 全部通过；实际二进制已加入 Git 索引，不依赖 LFS。
- 本地独立虚拟环境的 `make test` 通过（32 项 Python + Node UI），`make cpu` 通过；Linux 验证目录的 32 项 Python 测试也通过。
- Linux 主机从本仓库导出成功，PyTorch / ONNX / CPU tail 数值断言通过；重新导出的 ONNX、tail 与随仓库文件 SHA-256 一致。
- 使用本仓库公开样本生成校准数据并完成 RKNN 重建；三个原生工具完成 ARM hard-float/glibc 交叉编译。构建时复用了已有主机 Python 环境及外部 SDK/编译器，未验证全新机器上的依赖安装。
- **本次未完成新仓库的板端离线推理复验**：验证时板子 SSH 不可达。随仓库仍保留此前板端跑通的 RKNN，未用本次重建产物替换；上面的板端性能与分数仅是历史实测，不是本次重建产物的验收结果。未替换既有网页服务。

模型和 vendored 源码：Apache-2.0，详见 `models/eres2net/README.md`、`licenses/3D-Speaker-LICENSE`、`THIRD_PARTY_NOTICES.md`。Rockchip runtime 不在此许可范围内。用户语音/声纹属于敏感数据，许可不自动覆盖其采集与保存。仓库中只收录上游公开示例，没有板上用户录音、凭据或机器地址相关私有配置。
