# 四层 `feat/k3_dev` FP8 + MTP PD 热态对照 r7

固定源码为 `a9bf762e878fc54ee9176da5c34ffbe6babc8d45`，其父链包含固定的 `feat/k3_dev` 基线 `55641e09bc09cdafcf8f31b28aa55b18bc66d24b`。后续两次提交只增加可选的模型范围标记和可复现构建脚本。111 Prefill、112 Decode，均为 TP8/EP8、四层 FP8 target 加原生 MTP；Decode 启用 CUDA Graph。两端从已核实的 3FS 四层 target 用 FastSafetensors 直接加载，任务进程设置 64 线程预读；MTP checkpoint 视图在个人数据盘，shard 指向 3FS。`feat-profile-r7-*-evidence.tar.gz` 保留启动、RDMA、FP8 和逐 case 证据。GPU 筛选快照在 `fleet-selection.json`；测量时两端只有本次服务的八个 rank，利用率为 0%，没有其他活动计算进程。

四层 flow 共 10 条请求。`independent-flow-audit.json` 独立解析原始 HTTP 响应，检查每条为 200、PD 路由指向 112、输出 16 token、耗时不超过 300 秒，结果均通过。这份旧 `feat/k3_dev` flow 没有要求 MTP 的专门 case，aux 也不暴露 draft 轮次或可靠的 Decode KV 交接长度，因此 flow 单独不证明这两项。四层权重输出不能用于答案语义判断。

64K 诊断使用固定输入 token SHA256 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`，禁用前缀复用。先做 10 次同路径、8 token 请求预热；末三次首 token 时间为 138.680、137.774、139.176 ms，最大偏差 0.65%。随后 16 次请求均返回 HTTP 200 且复用长度为 0。八 rank 原始 trace 在 `timeline-64k-feat-profile-a9bf-r7-allrank.tar.gz`，请求在 `requests-64k-feat-profile-a9bf-r7.tar.gz`。`aligned-phase-audit.json` 匹配了 7 个共同请求，均可见 target 与 draft Prefill 范围，因此证实 MTP draft 实际运行。按每请求最慢 rank 的 GPU span 取中位：Prefill **135.266 ms**、target **71.556 ms**、draft **62.564 ms**。

同机器、同输入的集成版 r45 是 138.230 / 74.344 / 63.907 ms；这组 `feat/k3_dev` 整体快约 2.965 ms，但不能直接归因于某个模块。`module-target.json`、`module-draft.json` 用 CUDA launch correlation 将 kernel 归属到模型范围；累计时间可能跨 stream 重叠，不能相加成关键路径。相同范围下，`feat/k3_dev` 三层 KDA core 各约 2.777 ms，集成版约 2.975–2.979 ms；三层 routed experts 在这次测量中均比集成版慢。MLA、dense 与通信范围因融合边界不同，只能做候选定位，须以等价路径重新测量。`feat/k3_dev` 的 target FP8 通信融合不满足最终 BF16 NCCL 契约，不能仅凭整体时间迁入。

这份证据只涵盖四层 Prefill。尚未据此证明全层性能、三方最优、完整模型答案正确性或最终 FP8 smoke 通过。
