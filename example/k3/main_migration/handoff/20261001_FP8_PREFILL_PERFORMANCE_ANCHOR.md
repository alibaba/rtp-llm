# FP8 Prefill 性能锚点：四层通算融合与 93 层正确性

代码锚点是本文件所在提交的父提交 `c8c2dd91fcfa2bf1c5897a4226cf4f68559a40bf`。四层 TP8/EP8 双机 PD 使用 65,536 token 的同一输入，SHA256 为 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`。Prefill 在 114，Decode 在 115；两版 RTP 均加载原生 MTP，至少执行十次同路径预热，关闭前缀复用，并从八个 Prefill rank 的 CUDA launch 关联回 GPU kernel。原始请求和 trace 位于个人 `/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/` 下的 `timeline-64k-feat-a9bf-fourlayer-114115-20261001-r3/` 与 `timeline-64k-fusion-c8-fourlayer-114115-20261001-r2/`。

用户批准 TP 内 FP8 AllGather。固定 `feat/k3_dev@a9bf762e` 的 AG 传 E4M3 值和 UE8M0 scale，通过 PyTorch symmetric-memory pipeline 与 DeepGEMM 重叠；输出用 DeepGEMM `fp8_gemm_rs_nt` 完成 GEMM 和 BF16 ReduceScatter。集成版在通用 `Fp8CollectiveProjection` 中接入同一类融合路径，并排除了 BF16 MTP draft。以下为实际融合 scope 中首个到末个 GPU kernel 的时间，四层合计、排除首个 profiler 请求，取八 rank × 四请求的中位数：

| 版本和正式拓扑 | FP8 AG＋输入 GEMM | FP8 输出 GEMM＋BF16 RS | target GPU 关键窗口 |
| --- | ---: | ---: | ---: |
| feat `a9bf762e`，FFN TP8 | 5.776 ms | 5.759 ms | 末三次中位 70.060 ms |
| 集成 `c8c2dd91f`，SP8 | 6.454 ms | 5.906 ms | 复测末四次中位 64.714 ms |

集成版第一层 AG＋GEMM 为 2.080 ms，feat 为 1.517 ms；两版该处 FP8 producer 约 0.056 ms，每个 rank 的 DeepGEMM 约 0.19 ms。差额主要在 symmetric-memory barrier 等待。两版正式 FFN 拓扑不同，集成版会拒绝 `ffn_sp_size=1`，因此不能把 0.56 ms 归为 GEMM 核的优劣，也不能用上述 target 窗口宣称所有模块已择优。feat 第二个 profile 请求的 target 达 356 ms；它保留在原始记录中，稳定窗口只取末三次。集成版第一次采样有 NCCL 等待离群值，改用重新预热后的 `r2`，原始 `r1` 仍保留。

固定 vLLM `3df4ae153` 的四层 PD 配置把 AG/GEMM 和 GEMM/RS 分开执行。形状 trace 确认其 KDA/MLA 输入投影实际为 BF16 `aten::linear`，AG/RS 也传 BF16，因此它不是等精度 FP8 GEMM 候选；可用来核对真实服务的分离路径。rank 1 单次形状 trace 中，第 1、2 层 KDA 输入 AG 分别约 1.433、1.323 ms，后续 BF16 GEMM 分别约 2.714、2.799 ms。该单 rank 样本不能与上表八 rank 中位直接相减。vLLM 的 MLA QKV/cache epilogue 与 MXFP4 MoE 已分别按相同形状做算子 A/B；详见个人 artifact `fp8-prefill-operator-selection-ledger-20260930.md` 中逐模块记录。比较融合段时必须把通信与 GEMM 放在一个关键路径窗口内，同时保留子 kernel 归因。

当前选择保留集成版的通用 FP8 AG＋GEMM 和 GEMM＋BF16 RS。feat 的融合算法已接入，没有发现迁移其模型专用包装层可带来新的核收益。四层 flow 为 11/11，独立审计 `checked=11, errors=[]`。随后使用上述候选重新编译并运行完整 93 层 FP8 双机 PD capped smoke：121 条实际执行请求通过，独立审计 `passed=true, checked=121, errors=[]`；原最后三条长输出 repeat 与三条已知超时请求按约定跳过，合计六条均不计通过。逐 case、审计和服务日志在个人 artifact `smoke-93layer-fp8-fusion-c8-114115-20261001/`。本次验证了回答文本、重复吐字、乱码、截断、MTP 执行及 PD 状态交接。

此提交固定性能阶段的代码与证据。它只证明四层融合候选和 93 层 smoke 正确性；还没有完成三方完整 93 层热态 64K 对照，也没有证明集成版每个算子都比两个候选快。峰值激活改动将从此锚点继续，以另一个提交保存。`skip-head-mid` 和历史 FP8 KV 默认 6 GiB 分块已在祖先提交 `4dcde8045e3f1c143cb1328179ebb1bcb151b329` 中存在，后续针对这两项做验证与显存测量，不重复迁移。
