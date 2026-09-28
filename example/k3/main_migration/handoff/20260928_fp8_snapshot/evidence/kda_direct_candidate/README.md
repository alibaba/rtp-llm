# KDA 直接输出四层 PD 候选

此目录记录源码 `47b1222ff64a5f206df9daa7eb2dedbaafdd560b` 的双机验证。它在单序列 packed cuLA KDA 路径直接返回输出，省去另一份整张量清零和拷贝。上一版 `928ad6a6d` 的双机 flow 与 timeline 保留在 `../kda_packed_candidate/`；当前改动的 65,539 token GPU 数值对照和单卡热态 A/B 也记录在那个目录的 `direct_output/`，不能由此推断四层 PD 的性能已达标。

110 Prefill、112 Decode 分别在个人容器 `lhc_GPU_k3_3fs_20260927` 中以 `luohaocheng.lhc` 构建；两边都使用本机 ext4 源码和 Bazel 输出目录，命令包含 `--config=cuda13 --config=sm10x`。`host_110112/build-summary-*.txt` 的 Bazel 退出码均为 0，每端完成 23,714 个 action。两端源码处于独立工作树，只有为内部 RDMA 依赖设置的本机 `internal_source` 链接未提交。

启动配置见 `host_110112/kda47b-launch-config-*.json`：四层 target 为普通 E4M3 FP8 GEMM 与 KV、BF16 activation，原生 MTP draft 为 BF16，显式 `LOAD_METHOD=fastsafetensors`。target 直接从 3FS `/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers` 加载；MTP 的个人数据盘视图引用已核实的 3FS shard。两端 `target/mtp-guard-*.txt` 均为 PASS，分别核对了 7/9 个 shard、16,402/5,404 个张量及 Safetensors header；目标配置与索引 SHA256 为 `8754ec8b…` / `eb064b56…`，MTP 为 `6e5457c4…` / `52603395…`。这是用户指定的 3FS 直读路径，任务进程沿用 64 线程并发读取层，不修改共享挂载或集群配置。

`host_110112/pair-selection-launch.json` 记录了启动前两端八卡均无外部计算进程、每卡至少约 268.6 GiB 空闲。双向 ping 两包均无丢失，平均约 0.24 ms；`mlx5_bond_0` 至 `mlx5_bond_7` 的 RDMA link 均为 ACTIVE，27100/27200 端口空闲。服务启动、逐 rank loader 日志、四层 flow 和 64K timeline 的结果尚需另行归档与复核；本目录当前只证明构建和启动前条件。
