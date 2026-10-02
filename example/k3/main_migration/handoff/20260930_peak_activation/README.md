# K3 FP8 峰值激活锚点（2026-09-30）

本锚点接在性能锚点 `af2dfb089` 之后。新增通用 FP8 线性层的 skip-head-mid 输出布局，让 MLA KV-up 跳过随后会被 RoPE key 覆盖的中间列；同时按每 rank 默认 6 GiB 的展开后 K/V 预算分片读取历史 MLA latent cache。预算计入 BF16 投影结果与 FP8 attention operand 的重叠存活量，不限制当前轮 query 的 K/V。分片结果用 FP32 LSE 合并，片内临时张量在下一片前释放。整模型 Chunk Prefill 没有迁入。

四层上用较小预算强制多片历史 prefix，做过定向算子检查和 TP8/EP8 双机 PD flow；11 条请求及独立链路复核通过。完整 93 层按同一源码在 114 Prefill、115 Decode 各自编译并用 FastSafetensors 从经索引核实的 3FS target 与 MTP 权重加载。限时 `main-text-64k-capped` smoke 发送 121 条，121 条通过独立答案、Unicode、重复输出、MTP 和 PD 交接复核；约定的六条长耗时用例没有发送，原完整 suite 不能称为全通过。原始记录在 115 的个人目录 `artifacts/k3-fp8-opt-20260927/smoke-93layer-fp8-peak-114115-20260930/`。

已在性能锚点里的 shared expert gate/up view 和按单轮 token 容量配置的 MoE workspace 保持原实现。`feat/k3_dev` 的静态 AttnRes bank、跨整模型 chunk 的 MTP hidden 清理和跨 chunk scratch 复用依赖整模型分轮执行；集成版一次 forward 只使用一个 AttnRes bank，MTP hidden 要保留到 draft 消费完毕。将这些常驻 buffer 原样引入会增加待机显存并带来并发覆盖风险，因此本锚点没有搬运。历史 KV 分片使用局部 scratch，并在每片结束后释放引用。

114 的完整 smoke 期间，NVML 观察到每 rank 最高约 269.4–272.4 GiB，其中最高 rank 比采样起点增加 8.94 GiB。这个总量包括权重、cache 和 CUDA allocator 预留，不能直接当作激活节省量。skip-head-mid 的单算子热态实测为旧路径 0.14024 ms、打包路径 0.080816 ms；分配峰值分别为 88.08 MB 和 67.67 MB。四层完整 target 热态 span 为 70.114 ms，先前同机基线为 69.712 ms，因此该算子收益尚未证明模型级提速。最终全层三方热态对照仍需单独完成。
