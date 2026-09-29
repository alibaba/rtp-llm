# 2026-09-29 暂停点：3FS 直读与四层性能对照

这份目录保存暂停前新增的任务脚本和小型证据。产品代码仍是提交 `97b492cc62319eab283cc2802a724dd704137524`；本目录的 3FS 并发读取器、权重视图和 guard 属于实验工具，没有接入 RTP-LLM 运行时。原始全 rank timeline 太大，仍在个人 artifact 目录，**本提交未备份这些原始 trace**。

## 3FS 恢复后实际执行的读取

- 110/114/115 的 3FS 元数据读取已恢复。110 对 96 个不同 shard 各读 256 MiB，direct I/O 短测从单 worker 的 1,206 MiB/s 提高到 92 worker 的汇总 45,477 MiB/s。见 `evidence/3fs-direct-throughput-110-recovered-20260929.jsonl.gz`。短测可能受缓存影响，也不是 FastSafetensors 吞吐。
- 本轮 110/115 的四层 RTP 服务通过本地元数据视图把七个 target shard 直接指向 3FS 完整 target 的相同字节；MTP 的九个 shard 也直接指向 3FS。110 已重新计算这七个完整 target 源 shard 的 SHA256，与此前本地 MS4 来源一致，见 `evidence/3fs-full-target-selected-ms4-sha256-110-20260929.json.gz`。3FS 自带的 `kimi-k3-4layers` **不是**这组 MS4 权重，不能替代。
- 在这两台实际取数机器，FastSafetensors SHM 的任务进程用 `LD_PRELOAD` 启用每 shard 64 个并行 `pread64` worker。这是当前任务实现和实测过的最高档；没有修改共享 FUSE 或集群配置。110/115 两端 target 的各 rank 最慢 `copy_files` 分别约 3.99/3.51 秒，rank0 `load weights took` 分别为 8.22/7.81 秒；MTP 另行加载。退出服务并移除 `LD_PRELOAD`、`K3_3FS_PREAD_THREADS` 即回滚。读取器源码、视图脚本和启动脚本都在本目录；编译生成的 `.so` 未提交。
- 两端 FastSafetensors guard 与启动日志确认 target/MTP 路径、索引及实际加载方式。修正后的任务 guard 会同时检查索引中的视图文件名与 symlink 指向的物理 shard；此前按物理 basename 匹配会误报。

## 四层 PD 链路结果

110 Prefill / 115 Decode 从上述 3FS 视图启动集成版，四层 TP8/EP8、FP8 target/KV、BF16 NCCL、Native MTP。限时 300 秒的短 `flow` 为 11/11 完成，逐条 `pd_sep=true`，MTP draft rounds 均大于零；独立链路复核为 11/11、无 Unicode replacement character。记录在 `evidence/direct3fs-flow-result.json` 和 `evidence/direct3fs-flow-independent-flow-audit.json`。四层截断模型只能证明链路，不证明语义答案正确。此前本地同权重 16-token flow 的一条回复含不完整 UTF-8 token 序列；8-token 短 flow 通过，不覆盖这个问题。

## 性能数据的适用范围

110/115 本地同字节权重的热态 64K RTP all-rank timeline 已在独占 GPU 上采集，模块归因数据在 `evidence/comparison-64k-*.json.gz`。集成版有效 target GPU span 约 70–79 ms；`feat/k3_dev` 有效 target 样本约 69–75 ms，另有数百毫秒至数秒的 NCCL 等待离群值。约 3 秒的 `feat/k3_dev` trace 不能当作正常 Prefill 速度。两组的 Native MTP draft 都确实执行。模块对照里集成版的 routed experts、dense MLP 更快；`feat/k3_dev` 明显较快的 fused AG/GEMM 与 GEMM/RS 使用自定义通信，不符合本任务固定 BF16 NCCL 的约束，尚未迁移。

先前固定 vLLM `3df4ae153` 的 114/115 NIXL PD trace 虽有 FP8 E4M3 KV 和 TokenSpeed MLA，启动参数**没有** `--quantization fp8`，因此不能作为三方普通 E4M3 GEMM 的性能基线。原 trace 的分类汇总仅作为排查资料保存在 `evidence/vllm-r5-fourlayer-kernel-family-summary-20260929.json.gz`。暂停前已准备新的 `--quantization fp8 --moe-backend deep_gemm`、3FS 直读启动脚本；114/115 容器在模型初始化时按用户要求停止，没有生成 FP8 vLLM timeline。恢复时应先验证该固定版本实际采用 E4M3 GEMM/MoE、BF16 NCCL 和 NIXL，再重新预热、采集。

## 恢复顺序

1. 检查迁移后的个人目录、110–115 登录、GPU 独占、3FS 挂载和权重 SHA；重建本地元数据视图，继续直接从 3FS 读取。
2. 补齐真正的 vLLM FP8 四层 PD timeline 与数值检查，重新做同口径逐模块择优；不要把旧 vLLM trace 计入 FP8 结论。
3. 性能候选完成后才建立性能锚点；峰值激活优化及第二个锚点、最终 93 层三方对照和最终 FP8 双机 smoke 都还没有完成。原长输出 repeat 与已知超时 case 继续跳过。

暂停时 110/115 RTP 服务和 114/115 vLLM 容器已停止，GPU 已释放。没有调整共享 3FS 配置，也没有提交性能或峰值激活锚点。
