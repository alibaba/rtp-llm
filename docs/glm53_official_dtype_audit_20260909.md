# GLM-5.3-Flash / Kimi K3 官方实现与数据类型复核

> 后续三项修复及验证见 [glm53_precision_fix_20260909.md](glm53_precision_fix_20260909.md)。本文保留修复前审计时点的结果。

2026-09-09。复核对象为 RTP 外源 `984fd1f020c4f94c8a527aa409ee20fc9485bc57`、内源 `c9033e9e974760cb443e8fc99131bac55d6d2af9`，以及上一轮实际运行的 runtime。

**结论：新增 KDA 融合通过了独立数值回归，但当前整模型基线尚未与官方推理精度完全对齐。** 发现三项已有差异：专家权重 FP8→FP4、router BF16 GEMM、BF16 history 下的卷积乘法舍入。上一轮新旧实现一致性、检索答案和 timeline 验收仍然有效；这些验收不能替代官方模型的精度验收。现存 timeline 应标为 **FP8 激活×FP4 专家权重的性能结果**。

本次提交的是独立数值 UT 和审计结论。生产计算路径没有改动；以下三项差异尚未修复，也没有重新生成所谓“官方 FP8”timeline。

## 来源及复现

| 来源 | 固定版本 | 用途 |
|---|---|---|
| [Z.ai GLM-5.3-Flash 发布配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json) | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 模型参数、FP8 E4M3、128×128 weight block、非量化模块名单 |
| [Moonshot K3 发布配置](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)、[模型源码](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py) | `f831ab66814297da540d832a5235f8e904f29d06` | KDA 门控、latent MoE、SiTU、MXFP4 发布格式 |
| [SGLang GLM 实现](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/models/glm5_next.py)、[KDA backend](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/layers/attention/linear/kda_backend.py) | `30e7a3072d3f1e9bd70cd5e44146ca27c80522c4` | GLM 的实际优化推理路径；FP32 卷积权重、raw beta、FP32 norm/gate |
| [Transformers GLM 实现](https://github.com/huggingface/transformers/blob/0a959de1d2dd0c981f1f732dbd0fc31192bbfa66/src/transformers/models/glm5_next/modeling_glm5_next.py) | `0a959de1d2dd0c981f1f732dbd0fc31192bbfa66` | 易读的 FP32 recurrent / norm / router 算法参考 |

Z.ai 的[模型仓库](https://github.com/zai-org/GLM-5#glm-53-flash)列出了 SGLang、Transformers 等部署实现。这里区分“厂商发布物”和“厂商推荐框架实现”，不把 RTP 的 `feat/k3_dev` 当作官方源码。Hugging Face 文件通过镜像按完整 revision 下载；GitHub 文件也按完整 revision 重取并核对，均与首次获取的文件一致。具体 URL、SHA256 和下载来源记录在本地 `sources/manifest.json`。

本地 checkpoint 的 config 与官方文件逐字节一致，SHA256 为 `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`。读取 62 个 safetensors header，未加载全模型；抽取第 0、4 层 KDA 权重和第 3、4 层专家 0 / router 做数值测试。官方仓库相同 config 不等于逐个 shard 内容已与远端校验。

测试在 azk113 GPU0 进行，每次启动前保存至少五秒低显存门禁。CUDA Runtime 确认 SM103、148 SM；没有根据 NVML 8.9 推断架构。使用上一轮最终 runtime 和 DeepGEMM wheel。

## KDA：可借鉴融合方式，保留 GLM 参数化

| 项目 | GLM-5.3-Flash | Kimi K3 | 当前 RTP 核对 |
|---|---|---|---|
| Heads / head dim | 64 / 128 | 96 / 128 | GLM H64 未被改成 K3 H96 |
| 输出门控投影 | 4096→128→8192，两级 BF16 GEMM | 7168→12288，全秩投影 | 保留 GLM G_A/G_B；没有移植 K3 全秩权重 |
| Forget projection | F_A/F_B，lower bound −5 | F_A/F_B，lower bound −5 | `−5 × sigmoid(exp(A_log) × (g + dt_bias))`；A_log、dt_bias、计算均 FP32 |
| Q/K/V 和短卷积 | checkpoint 权重 BF16；四 tap、SiLU | 四 tap、SiLU，独立 Q/K/V | 连续 Q/K/V 平面仅改变布局；GLM 页表、cache 顺序保留 |
| Beta | BF16 projection；SGLang KDA 使用 raw beta 在 FP32 做 sigmoid | 官方模型显式将 beta 转 FP32，交给 FLA 内部门控 | 新融合将 FP32 sigmoid 移入 recurrent；没有增加 BF16 gate 中间值 |
| Recurrent state | FP32、K-major、Q/K L2 norm eps 1e−6 | FP32 recurrent；具体缓存管理不同 | FP32 state、delta update 和输出 BF16 保留；低 warps 配置不改变模型参数 |
| 输出 norm/gate | per-head RMSNorm eps 1e−5，sigmoid；FP32 计算后一次输出转换 | FLA FusedRMSNormGated，sigmoid | 新 kernel 符合该算术顺序；不是普通 RMSNorm 的多次 BF16 舍入 |

Transformers 当前 eager KDA 的 beta sigmoid 在输入 dtype 执行，且 eager convolution 会先产生一个卷积输出张量；这些边界和 SGLang/FLA 优化路径不逐位相同。因此本次以 SGLang/FLA 的 FP32 gate 契约验证融合，并用独立 FP32 递推检查算法；不能把任一框架 eager 输出当作所有优化 kernel 的逐位 Golden。

**已有卷积差异。** GLM 的 SGLang loader 将短卷积权重提升为 FP32；RTP loader 保持 BF16。RTP kernel 在 BF16 history 下先做 BF16×BF16 乘法，再累加到 FP32。仅把 accumulator 声明为 FP32 不能消除前面的乘法舍入。上一轮新卷积特意保留了此旧行为，所以它与旧实现逐位一致，却未完成这一项上游数值对齐。

真实权重、B48×24576 channels、固定随机输入下，相对独立 FP32 卷积→SiLU→BF16 参考：

| 层 / history | 当前新旧路径相对 L2 | 将权重提升 FP32 后相对 L2 | 新旧卷积及 cache |
|---|---:|---:|---|
| 0 / BF16 | 0.2744% | 0.001714% | 逐位一致 |
| 4 / BF16 | 0.2762% | 0.001445% | 逐位一致 |
| 0 / FP32 | 0.001917% | 0.001917% | 逐位一致 |
| 4 / FP32 | 0.000748% | 0.000748% | 逐位一致 |

“提升 FP32”这一行是调用已有通用卷积的实验，没有修改生产 loader。正式修复应同时覆盖 Prefill 和 Decode、融合及 fallback 分支；不能只改 Decode，让两阶段采用不同算术。

## MoE：当前并不是官方 FP8 的无损实现

| 项目 | GLM 官方 | K3 官方 | 迁移边界 |
|---|---|---|---|
| 专家结构 | hidden4096，288 routed、top8，1 shared，intermediate2048 | hidden7168→latent3584，896 routed、top16，2 shared，intermediate3072；routed combine 后 norm/up projection | 不迁移 K3 latent 投影、norm 或 shared 布局 |
| 激活 | `silu(min(gate,10)) × clamp(up,−10,10)` | SiTU，beta4、linear_beta25，使用 tanh；FP32 算术后输出转换 | GLM 保留带截断 SwiGLU；不能直接换成 K3 SiTU |
| 路由 | FP32 GEMM、sigmoid、correction bias 只用于选专家；原始 scores 归一化后×2.5 | FP32 GEMM、sigmoid、相同 bias/权重区别；scale1.0 | GLM top8、scale2.5 保留；dtype 问题见下文 |
| 发布权重 | routed 和 shared 均 FP8 E4M3，128×128 block，FP32 scales | routed MXFP4、group32、U8 scales；attention/shared 不在该 MXFP4 组内 | K3 原生 FP4 不构成对 GLM FP8 重新量化的精度证明 |

**已有专家重量化。** 当前 `moe_strategy=mega_moe` 的 loader/后端把 GLM FP8 专家转 BF16，再量化成 MXFP4。前一轮迁移没有新加这个步骤，新旧 DeepGEMM 在该路径逐位一致也无法证明重量化本身无损。

本次从官方 FP8 checkpoint 解量化两个真实专家，并与其 MXFP4 重量化后再次解码的权重比较。两组使用相同 BF16 输入、相同 BF16 GEMM 和相同 GLM SwiGLU；仅改变专家权重。B1/48/257 的专家输出相对 L2 差异为 **21.61%–23.32%**。这是随机输入下的单专家隔离实验，既不是整模型准确率变化，也不是实际 MegaMoE EP combine 的误差，不能外推为任务分数下降。

**已有 router dtype 差异。** checkpoint router 权重虽然存为 BF16，但官方计算显式将输入和权重转成 FP32。当前 RTP `CudaF16Linear` 返回 BF16 logits，之后 `.float()` 无法恢复丢失的位。固定随机输入及真实 router 权重、相同 correction bias / top8 / scale2.5 的对照结果：

| 层 | B48 中专家集合不同的行 | B4096 中专家集合不同的行 |
|---|---:|---:|
| 3 | 2 / 48 | 114 / 4096（2.78%） |
| 4 | 1 / 48 | 118 / 4096（2.88%） |

这是 BF16 logits 舍入导致的路由差异诊断，输入不是实际层 hidden-state dump，不能据此估计线上路由变化率。正式修复需要固定 TF32 设置并覆盖近并列 logits、SP token 顺序和非整除 batch。

**不能只改后端名字。** `mega_moe_fp8` 当前也会通过 `requant_weight_ue8m0` 对非二次幂 scale 的原始 FP8 权重再次量化，且上一轮大 M 重复调用测试在旧库就失败。它不是直接保留发布的 FP8 bytes + FP32 scales 的已验收替代品。本轮未用它替换当前生产路径。

## 新增优化的独立数值回归

- 真实第 0、4 层短卷积权重：B48，BF16/FP32 history，当前新旧输出和 cache 逐位一致。
- 真实 norm 权重：B1/7/48/64，独立 FP32 norm→weight→sigmoid gate→一次 BF16 cast。最大相对 L2 为 `1.74e−5`；少量 BF16 最末位差异，不能写成全部逐位一致。
- 真实 A_log/dt_bias、B48/H64/D128，64 个变化输入的 Decode step，对独立 FP32 K-major 递推。第 64 步新 kernel 输出相对 L2 `3.59e−5`、max abs `6.10e−5`；state 相对 L2 `6.16e−7`、max abs `8.05e−7`。新旧都在预先设定的误差阈值内，没有发现新增低精度状态或错误递推。
- 新增 `glm53_reference_numerics_test` 两项 GPU UT：BF16/FP32 norm weight，B1/7/48/64，饱和门控、零 Q/K、随机重排页、FP32 32-step state；**2 passed**。参考公式不调用旧 recurrent 或旧 norm kernel。

这些检查提供算子级误差边界；还没有官方完整模型 logits、长序列 PPL 或质量评测结果。不能给出“整模型没有新增精度问题”的无条件保证。

## 精度修复优先级及性能结论

1. **P0：建立保留官方 FP8 专家权重及 FP32 scales 的 MoE 对照。** 校验实际 loader tensor、weight bytes/scales、SwiGLU clamp、shared expert、router 和 combine。FP4 只能作为另行量化评测的候选，不能作为官方 FP8 基线。
2. **P0：GLM router FP32 计算。** 同步修复初始化权重和执行输入 dtype；不能只改 top-k 输入 dtype。测近并列专家、BF16 checkpoint→FP32 compute、不同 batch/SP，以及任务级影响。
3. **P0：GLM 短卷积 FP32 乘加。** 同步修改 P/D、融合/fallback，并补跨页、prefix reuse、动态 batch、连续多 step；保持缓存存储契约。
4. 三项完成后，以官方对照重做完整主模型 logits/质量和双机 PD，再采 P B8×128K、D 全局 B384×128K。FP8 专家会增加显存，原 FP4 路径的容量和性能不能直接沿用。

上一轮 P7.820s、D23.312ms 的原始 JSON 保留，属于既有 FP8×FP4 路径的有效性能测量。本次没有伪造官方精度下的 TPS、TPOT、TTFT 提升百分比。

本地证据根目录：`/data2/aozhengkai.azk-home/glm53_flash_1layer_perf/artifacts/official_dtype_audit_20260909/`。主要文件为 `sources/manifest.json`、`sources/local_glm_headers.json`、`audit_precision.py`、`precision_results.json`、`audit_router.py`、`router_results.json`、`reference_ut.log` 和三次 `gate*.tsv`。
