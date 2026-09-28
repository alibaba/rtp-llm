# AllGather 输出免零填充：四层验证

本次候选源码是 `d733278ac1f87420afacd930f1da9b5d3284fe9a`。唯一的运行时代码改动是通用 `collective_torch.all_gather` 将 NCCL 输出张量的 `torch.zeros` 改为 `torch.empty`；同步的 `all_gather_into_tensor` 随后写满输出。111 Prefill、112 Decode 分别在个人 `lhc_GPU` 中编译，再用各自已有的 3FS 服务容器启动。两端都是 TP8/EP8、四层 FP8 target、BF16 Native MTP、BF16 NCCL；Decode 开启 CUDA Graph。target 和 MTP shard 都从 3FS 读取，使用任务局部的 64 线程 pread helper；[111](111/weight-identity.txt)和[112](112/weight-identity.txt)记录了权重身份与 MTP shard 链接。

先运行四卡 NCCL collective 测试，结果为 1/1 通过；随后四层双机 PD flow 为 11/11 通过。 [独立 flow 审计](../smoke-4layer-fp8-agempty-d733-r1-111112-20260929/independent-flow-audit.json)逐条重读原始响应，确认 PD handoff、Native MTP draft、300 秒上限与无乱码，最长 74.288 秒。四层随机权重的输出不能作为语义答案正确性的证据。为消除测试 namespace 写入提示词造成的差异，又回放了一条与旧版完全相同的 65,537-token 请求：[输出逐字一致](../smoke-4layer-fp8-agempty-d733-r1-111112-20260929/exact-baseline-replay.json)，MTP 执行 15 轮，Decode 接收 65,536-token KV。

64K timeline 使用与旧版相同的输入 token SHA256 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`，独占检查见[机器选择记录](performance-selection.json)。同路径预热 10 次，最后三次完整 HTTP 耗时为 2.085/2.081/2.065 秒；正式采集 16 次，全部 HTTP 200。 [独立请求审计](independent-response-audit.json)确认每次 target Prefill reuse 为 0、Decode handoff 为 61,440 token、MTP draft 为 7 轮，且没有替换字符。HTTP 耗时不是 Prefill 耗时。

| 四层 64K 同机对照 | Prefill GPU 关键路径 | target | draft |
| --- | ---: | ---: | ---: |
| 集成版 `47b1222ff` | 137.272 ms | 73.593 ms | 63.818 ms |
| 本候选 `d733278ac` | 135.722 ms | 72.617 ms | 63.480 ms |
| 固定 `feat/k3_dev` 版本 | 135.266 ms | 71.556 ms | 62.564 ms |

以上是[八 rank 对齐审计](aligned-phase-audit.json)中 7 条完整匹配请求的「每条取最慢 rank，再取中位数」，不是把所有 kernel 时间相加。本候选 target 比 `47b1222ff` 快 0.976 ms，但仍比固定 feat target 慢 1.061 ms。 [模块归因](module-target.json)显示，旧版 AllGather 范围的 BF16 `FillFunctor` 在 7×8×4 个层/rank/请求位置均出现，最慢 rank 每请求零填充累计中位数 0.950 ms；本候选为 0，NCCL AllGather kernel 仍在。target 其他同名模块的中位数变化均不超过 0.066 ms。归因后的累计 kernel 时间可以跨 stream 重叠，不能直接当作关键路径差值。

固定 feat 的融合 AllGather/GEMM 使用 FP8 value/scale 通信，不满足这里的 BF16 NCCL 约束，不能整块迁移。vLLM 固定版本在同机四层 PD 的 target 中位数为 111.386 ms，但没有 Native MTP，且 routed MoE 精度路径不同；其独立记录在 `../vllm_3df4_r2/`。这些四层数据尚不能证明完整 93 层性能或答案正确性，本提交也不是性能锚点。

[八 rank 原始 trace](timeline-64k-agempty-d733-r1-allrank.tar.gz)、[26 条原始请求](requests-64k-agempty-d733-r1.tar.gz)和[归档哈希](timeline-archives.sha256)已保存。此次执行的脚本从上一版复制时遗留了 trace 名 `k3_64k_integrated_kda47b_r2_wr*_1.json`；[实际执行脚本](timeline-executed-script.sh)、源码提交、独立工作树和启动日志都指向 `d733278ac`。仓库中的复现脚本已改正 trace 名，**本次没有为改名重跑请求**。[构建与启动原始日志](111/raw-logs.tar.gz)及[112 对应日志](112/raw-logs.tar.gz)有逐文件哈希；[全部 rank 启动审计](independent-startup-audit.json)确认 FastSafetensors 与 target FP8/draft BF16。111 target 权重加载为 7.76–14.68 秒，112 为 40.85–48.87 秒；后者落在此前同机冷加载的 41.93–47.97 秒范围附近。加载耗时包含 CPU/GPU 处理，不能直接当作 3FS 取数时间；已有真实 FastSafetensors 路径测试表明把任务读取线程从 64 增至 128/256 没有可确认收益，见上级目录的 `3fs-owner-read-report-20260928.md`。
