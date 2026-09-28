# KDA 直接输出四层 PD 候选

此目录记录源码 `47b1222ff64a5f206df9daa7eb2dedbaafdd560b` 的双机验证。它在单序列 packed cuLA KDA 路径直接返回输出，省去另一份整张量清零和拷贝。上一版 `928ad6a6d` 的双机 flow 与 timeline 保留在 `../kda_packed_candidate/`；当前改动的 65,539 token GPU 数值对照和单卡热态 A/B 也记录在那个目录的 `direct_output/`，不能由此推断四层 PD 的性能已达标。

110 Prefill、112 Decode 分别在个人容器 `lhc_GPU_k3_3fs_20260927` 中以 `luohaocheng.lhc` 构建；两边都使用本机 ext4 源码和 Bazel 输出目录，命令包含 `--config=cuda13 --config=sm10x`。`host_110112/build-summary-*.txt` 的 Bazel 退出码均为 0，每端完成 23,714 个 action。两端源码处于独立工作树，只有为内部 RDMA 依赖设置的本机 `internal_source` 链接未提交。

启动配置见 `host_110112/kda47b-launch-config-*.json`：四层 target 为普通 E4M3 FP8 GEMM 与 KV、BF16 activation，原生 MTP draft 为 BF16，显式 `LOAD_METHOD=fastsafetensors`。target 直接从 3FS `/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers` 加载；MTP 的个人数据盘视图引用已核实的 3FS shard。两端 `target/mtp-guard-*.txt` 均为 PASS，分别核对了 7/9 个 shard、16,402/5,404 个张量及 Safetensors header；目标配置与索引 SHA256 为 `8754ec8b…` / `eb064b56…`，MTP 为 `6e5457c4…` / `52603395…`。这是用户指定的 3FS 直读路径，任务进程沿用 64 线程并发读取层，不修改共享挂载或集群配置。

`host_110112/pair-selection-launch.json` 记录了启动前两端八卡均无外部计算进程、每卡至少约 268.6 GiB 空闲。双向 ping 两包均无丢失，平均约 0.24 ms；`mlx5_bond_0` 至 `mlx5_bond_7` 的 RDMA link 均为 ACTIVE，27100/27200 端口空闲。两端服务随后都返回 HTTP 200 健康响应；`host_110112/kda47b-loader-verify-*.txt` 保存了各八个 rank 在服务健康后的 guard 复核，全部 PASS，明确选中 FastSafetensors 且未发现 fallback。原始启动记录保存在两端 `startup-kda47b-r1-*.tar.gz`。

四层 flow 的 11 条请求全部在 300 秒内返回，runner 退出码为 0。`kda47b-fourlayer-flow-audit.json` 独立重读每条原始响应，检查了 Decode 路由、整页 KV 交接、实际 MTP draft 轮次、输出长度和 UTF-8，结果 **11/11 PASS**；最长耗时 83.223 秒，没有观察到替代字符。`flow-kda47b-r1-110112.tar.gz` 保留逐 case 请求和审计。四层 checkpoint 的随机输出不用于判断答案语义。

同一组服务做了 10 次不同前缀、同长度 64K 的完整 PD 请求预热，末三次 HTTP 耗时 2.054、2.025、2.029 秒，偏离中位数最多约 1.21%。`pair-selection-before/after-timeline.json` 均确认两端只有本次服务的各八个 rank，没有观察到其他 GPU 计算进程。接着以固定 token SHA256 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee` 采集 16 次请求。`independent-request-audit.json` 逐个重算原始响应 SHA 和模型实际输入 token SHA：16 次均为 HTTP 200、Prefill 复用 0、Decode 接收 61,440 token KV，MTP draft 均执行，未观察到替代字符。原始请求与预热数据在 `requests-kda47b-r1.tar.gz`；HTTP 耗时只用于预热收敛，不当作 Prefill GPU 时间。

八 rank trace 在 `timeline-kda47b-r1-allrank.tar.gz`，`kda47b-aligned-phase-audit.json` 匹配了 7 个完整请求。每请求取最慢 rank GPU span 再取中位，Prefill **138.338 ms**，target **73.925 ms**，draft **64.489 ms**。`kda47b-module-target.json` 显示 L0/L1/L2 KDA core 累计 GPU 时间分别为 **2.778/2.779/2.781 ms**；相同 110/112 机器上上一版 `928ad6a6d` 为 **2.832/2.834/2.834 ms**。单卡交错 A/B 的更严格局部证据见上一候选目录，KDA 直接输出在该形状下少约 2.92%。整段 Prefill GPU span 相比上一版下降约 16.8 ms，远大于 KDA 三层累计节省，不能把整段变化归因于此改动。固定 `feat/k3_dev` 的 KDA core 约 2.777 ms，测于 111/112；当前只可说该模块时间接近，仍缺同机三方完整路径对照。`kda47b-module-draft.json` 同时保留 draft 的模块归因，累计模块时间可能跨 stream 重叠。

原始档案 SHA256：flow `ebb0abfb50cdda197aab6a29811a1165d8c9e41bd7bd88feac24ab6666d4629c`，请求 `26c07de75533b970c7b2b93370f098f514ba5fa0896fc916217d893c6c78ca72`，八 rank trace `521bc8260ef6b0355d229f038cc08a18bc9421240dda0fea11a4ecfaa9e10337`，110/112 启动档案分别为 `14632a2d7fd66c6be5ff7a87a6caabcd5db98b9f595aa20b806f7b5f6f75c0f8` 和 `0ef008f2144194e8f590a8cbd667add738354a2886d7d50ec8165ccdd39d0a24`。这仍是四层筛选结果；完整 93 层答案、三方全层性能和两次正式锚点均未完成。
