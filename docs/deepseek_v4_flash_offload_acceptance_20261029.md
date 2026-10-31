# DeepSeek V4 Flash offload 验收（2026-10-29）

完整 DeepSeek-V4-Flash-0731 的 CSA sparse KV offload 已跑通，保留默认 CSA/HCA megakernel、DSpark 和 D 端 CUDA Graph。主要收益是限制 CSA 的 HBM backing 容量；本轮每张 D 卡峰值减少约 2.54 GiB，每轮解码耗时增加约 3%～5%。

## 模型与部署

- 模型：本地 NAS 权重副本 `/home/renyuanming.rym/models/DeepSeek-V4-Flash-0731`，43 层；48 个 safetensors 分片均已核对文件大小，config/index 的 SHA256 与 NAS 一致（沿用上一轮记录）。专家 FP4，其余权重 FP8，activation BF16，KV FP8。原 `/mnt/hf3fs/3fs/models/` 在容器内未挂载。
- 4 张 NVIDIA L20D，P 使用 GPU 0/1（CP2/EP2），D 使用 GPU 2/3（TP1/DP2/EP2），MegaMoE SE；TCP P/D。
- DSpark 两端 `gen_num_per_cycle=3`；D capture buckets 为 `1,2,4,8`，P eager。双方均保留默认 V4 CSA/HCA 融合路径。
- 不使用 prefix reuse、memory/disk/remote connector；Host KV 为 pinned DRAM。
- 物理 block 256 个原 token，kernel page 128 个原 token；CSA 每页 32 个压缩条目。两侧开关对照均固定 4096 个逻辑 block，约覆盖 1M 个原 token 的总缓存容量/每个 D rank。此容量与上一轮自动按 16 GiB 预算分配的容量不同，性能与显存均以本次 resident 组为基线。
- offload 只在 D 启用：`--enable_sparse_kv_offload 1 --sparse_kv_offload_host_mb 4096 --sparse_kv_offload_test_hbm_blocks 2`。最后一个是测试强制路由参数，包含 sentinel 0；生产应保持默认 0。
- 仅 `csa_kv` offload。Indexer、128x HCA、SWA、compressor state 和 DSpark draft 缓存留在 HBM。默认 request 路由逻辑复用现有实现。

## 性能

每个请求固定输出 256 token，greedy、固定 seed、关闭 prefix reuse。输入是重复技术文档组成的精确 4096 / 32768 / 131072 token 提示。每组 44 个正式请求，另外做形状 warmup 和 graph probe。吞吐为整个并发批次的输出 token / 客户端墙钟时间。`每轮` 使用引擎耗时减去首 token 耗时，再除以实际 decode 轮数；DSpark 每轮可能接受不同数量的 token。

| 输入 | 并发 | 请求数/组 | 关闭 offload tok/s | 开启 offload tok/s | 吞吐变化 | 每轮 ms：关 → 开 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 1 | 3 | 101.7 | 101.5 | -0.2% | 22.15 → 23.00 |
| 4K | 4 | 8 | 300.9 | 321.7 | +6.9% | 23.25 → 24.09 |
| 4K | 8 | 16 | 521.3 | 506.8 | -2.8% | 24.21 → 24.91 |
| 32K | 1 | 3 | 92.8 | 94.2 | +1.5% | 22.12 → 22.95 |
| 32K | 4 | 8 | 256.6 | 288.0 | +12.3% | 23.15 → 24.17 |
| 128K | 1 | 2 | 58.8 | 55.7 | -5.3% | 22.34 → 23.36 |
| 128K | 4 | 4 | 98.6 | 100.1 | +1.5% | 22.67 → 23.77 |

这是共享机器上的短合成测试，每个场景为 2～16 个请求。MTP 接受率、请求分配和 P/D 排队会影响端到端吞吐；本轮开关两组输出文本会随多卡 FP8 数值和批次发生变化，不宣称完整模型输出 ID 相等，也不能把部分场景的吞吐上升解释为 offload 加速。按每轮耗时观察，7 个场景的额外开销均约 3%～5%。GPU 4～7 上在 resident 组期间另做了关闭 DSpark 的功能 smoke，详见原始 telemetry；这些请求未计入吞吐。

## 显存与 Host 证据

5 秒采样，取正式请求时间窗内的设备总使用量峰值，不含启动阶段。

| GPU / 角色 | 关闭 offload GiB | 开启 offload GiB | 减少 GiB |
|---|---:|---:|---:|
| 0 / P | 102.67 | 101.65 | 1.03 |
| 1 / P | 101.64 | 101.64 | 0.00 |
| 2 / D | 92.49 | 89.95 | 2.54 |
| 3 / D | 92.48 | 89.94 | 2.54 |

每张 D 卡的 CSA backing：H=2、Host block=4094、N=4096，实际 pinned Host 分配 3.044 GiB，resident KV 分配约 0.382 GiB。resident 上限为 32768 个压缩行，即最坏情况下的 batch 16 × verify 4 query × TopK 512；上下文继续增长时不会按完整 CSA 历史增加 HBM backing，但 index cache / HCA 仍占 HBM。

DEBUG graph probe 的 `PD_CACHE_KEY_READ_BLOCK` 记录了 336 次真实 CSA block 导入，其 ID 均为 2～17，全部满足 block_id >= H，地址属于 pinned Host backing。随后请求完成且存在实际 graph replay 日志。详见 `host-routing-evidence.json` 与 `dspark-offload_decode_logs/process.log`。

V4 本身已采用 4x / 128x 压缩，本轮 128K、4 并发也未接近 HBM 容量。因此本负载下 offload 的容量收益不大；它更适合需要更大总上下文容量且愿意承担搬运开销的部署。

## 实现和验证

- CSA 的 FP8 page 使用 striped 布局：全部 576B payload 后接全部 8B scale。native entry layout 暴露两个 plane，避免按 584B 连续行搬运造成错误。
- 普通 compressor 与默认 CSA megakernel 共用固定地址的 native staging page；写回正式 Host/HBM backing 后，TopK 压缩行地址进入现有 sparse runtime，FlashMLA 使用重映射后的 resident 索引。未关闭 attention 融合。
- 支持 V4 DSpark 的 speculative 入口；TP1 D 可导入 CP2 P 的缓存；版本发布和 graph reset 复用已有机制，多 wrapper 初始化保留 runtime/staging 地址。
- 14 个 Bazel 回归目标通过，包含 GLM MTP、cache manager、native page layout、engine 配置和原 sparse runtime；最后单独重新通过 4 个 V4 GPU 测试。
- 新 GPU 测试验证：Host fetch 后 FlashMLA 输出逐位相同；Host 重新导入后的 graph 版本失效；普通 compressor 的多 query 写回；完整 fused CSA 的 2×4 query verify 及 graph replay，与 resident 路径输出逐位相同。
- 原 CSA adapter 的 26 个单测、原 fused CSA GPU 的 6 个测试通过。其 Bazel 目标在当前 host platform 被标记 incompatible，改用同一构建 runfiles 直接运行并保留日志。
- 两组正式请求共 88 个，全部完整生成 256 token、P/D 成功且 DSpark 确实执行；另有开启 DSpark / 关闭 DSpark 各 4K、32K 两个完整模型 smoke。所有实验服务和 telemetry 均已停止。
- 本轮仅验证 FP8 KV 的 V4 Flash、4 GPU 拓扑和最高 128K 输入；不覆盖 V4.1、BF16 KV、1M 单请求或所有 TP 配置。逐位相同仅指上述算子级数值对照，不是全模型多卡输出一致性结论。

## 复现与原始结果

目录：`/home/renyuanming.rym/v4-flash-offload-20261029/`。

```bash
/opt/conda310/bin/python3 /home/renyuanming.rym/v4-flash-offload-20261029/run_comparison.py
/opt/conda310/bin/python3 /home/renyuanming.rym/v4-flash-offload-20261029/report.py
```

`bench.py` 保存精确 CLI、env、prompt 与原始返回；`dspark-{resident,offload}-config.json` 为两组部署配置。`comparison.json` 保存每轮延迟、engine TPOT、TTFT 与吞吐，`verification.json` 列出全部正式请求，`gpu-telemetry.csv` 保存显存采样。`before.patch` / `before-sha256.json` 保留开始前的工作区状态，`v4-only.patch` 为本轮独立改动，`provenance.json` 记录最终源码与 native library hash。测试日志为 `test.log`、`final-checks/test.log`、`final-checks/mega-csa-{adapter,eager}.log`。首次未接通 fused 写回的失败启动单独保留在 `smoke/`，不计入正式结果。
