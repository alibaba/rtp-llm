# cuLA KDA packed checkpoint 候选

四层 64K 热态 trace 中，`feat/k3_dev` 的三层 KDA core 各约 2.777 ms，集成版各约 2.975–2.979 ms。两边都使用 cuLA，但 `feat/k3_dev` 把一条长序列的 checkpoint 一次提交；集成版按最多四个 page 分组。此候选只在单请求、page 对齐且 checkpoint scratch 不超过每 rank 32 MiB 时合并 cuLA 调用；多请求、page 内复用前缀和更大的 scratch 继续走已有分组逻辑。没有修改 BF16 NCCL 或 FP8 格式。

`test_native_kda_packed.py` 在改动前因 7 个 page 被拆为两次调用而失败；改动后两个 CPU 契约测试通过，其中一个检查 page 内复用前缀仍走分组路径。110 的单卡实际 cuLA 测试使用 20,483 token、12 个 head、4,096 token/page，并比较改动前后输出和所有 FP32 checkpoint。`numerical-result-110.json` 显示两者最大绝对差均为 0，测试退出码为 0。`gpu-selection-fleet.json` 与 `gpu-selection-110.json` 保存测试前的资源检查。

基线源码为集成版 `c7479de2ae9c1577f6ddbb9b75dcb1ec04d5f9b2` 的 `native_kda.py`，SHA256 `a096c0029ee6924d8871b5689609796664d14208adf993796b5f704be3781319`；候选源码 SHA256 `5d7bbd5bc973efd527e68449c5999190254102551d0ac129994cde7269917277`。这只是算子数值检查，尚未证明四层 PD 功能、热态性能提升或完整模型效果；须先重编译并通过这些门槛，才能纳入性能锚点。

`host_110112/` 保存提交 `928ad6a6d23e1205113334efaeeb7e6e18f9f1b1` 在 110 和 112 各自的个人容器中，用 `--config=cuda13 --config=sm10x` 构建的完成摘要。两端构建退出码均为 0，耗时分别为 851.748 秒和 697.575 秒。目录也保存了两端四层 target、MTP 的 FastSafetensors 结构预检，启动配置和启动前全机 GPU 筛选；当时选中同一集群的 110/112。四层 target 经用户指定的 3FS 直读，MTP 的本地视图引用 3FS shard，两端均启用任务进程内的 64 线程读取层。启动后两端各八个 rank 的日志均通过 FastSafetensors 校验；具体请求与性能结论见下文。

四层 flow 后来跑完 11 条，全部在 300 秒内返回。独立审计核对了每条 HTTP 200、Decode 路由、KV 交接长度、MTP draft 轮次和 UTF-8；`flow-kda928-r1-110112.tar.gz` 留有原始请求和审计，11 条均通过，最长的 65,537 token 请求耗时 84.787 秒，没有观察到替代字符。四层权重只证明链路，不用于判断回答语义。原审计脚本的 Decode 端口固定为 25200；这次将其改成可传参数，并以实际端口 27200 复核，改动前的 11 条审计因端口不符全部失败。

同一组服务完成 10 次 64K、8 token 的完整请求预热，末三次 HTTP 耗时偏离中位数最多 0.396%。随后 16 次固定输入请求均为 HTTP 200、Prefill 无缓存复用、Decode 接收 61,440 token KV、MTP draft 7 轮。`requests-kda928-r1.tar.gz` 留有原始响应，`request-audit.json` 是独立检查。输入 token SHA256 为 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`。

八 rank 原始 trace 在 `timeline-kda928-r1-allrank.tar.gz`，其中 6 个请求能在全 rank 对齐。每请求取最慢 rank 的 GPU span 中位数：Prefill **155.152 ms**，target **74.469 ms**，draft **66.130 ms**。三个 KDA core 的累计 GPU 时间分别为 **2.832/2.834/2.834 ms**，每层从原集成版的 47 个 kernel 减至 14 个；原集成版 r45 在另一台 111 上分别为 **2.975/2.979/2.977 ms**。`module-target.json`、`module-draft.json` 保存 CUDA launch correlation 归因，累计时间可能跨 stream 重叠。候选的整体 Prefill 比 r45 的 138.230 ms 更慢，而且机器从 111 换成了 110；这组数据只支持 KDA 局部候选继续筛选，尚不能证明完整 Prefill 提速。固定 feat 版同口径 KDA core 约 2.777 ms，仍更快。

原始压缩档案 SHA256：flow `7a69216b75cefc7623795eca5c919a30eee03b4a6000f891398ffab57083ffb2`，请求 `b270e54b9c157039aa79a2b0129346bfb2e1283d9783a0a89bcd5a563cc80ac8`，全 rank trace `42f643b4d8cfb31ebffd2dd8e70520ae1c110785b5dc63ac6792bbaba5f760a1`，110/112 启动日志分别为 `56170c0721fcb9c6c4d89293cdf900e2ec66f2f2619cbc8699a9b1b963afbf4a` 和 `84b60e9e87ef877a54cf6c21f87e58186ba3dc1abddb69c2f359a8b5346c27fb`。

后续提交 `4daf2b9b6` 让单序列 packed 路径直接返回 cuLA 的输出，省掉整块输出的清零和拷贝；其他路径仍按需分配并拷贝。`direct_output/numerical-near-64k.json` 是 112 上实际 cuLA GPU 对照：65,539 token、12 heads、4,096 token/page，输出和全部 FP32 checkpoint 与提交 `928ad6a6d` 的原 packed 实现最大绝对差均为 **0**。两份受测源码 SHA256 分别为 `5d7bbd5bc973efd527e68449c5999190254102551d0ac129994cde7269917277` 和 `c995a1e32ba14c401483064847f17c1239fae96a5ae657042291dbeb7eb4885c`；`direct_output/gpu-selection-112.json` 保存该次 GPU 检查。

`direct_output/benchmark-near-64k-110.json` 是同一对源码在 110 单卡独占运行的热态算子计时。全机筛选选中 110，启动前单机复查没有外部 GPU 进程；两边各完成一次 JIT 预运行和 10 次收敛预热，再交错采集各 30 次 CUDA event。每轮在计时外清零同形状 checkpoint cache。原 packed 实现中位数 **2.6217 ms**，直接输出 **2.5451 ms**，本地算子时间降低 **2.92%**。筛选记录在 `direct_output/gpu-selection-performance-*.json`。这只证明上述输入形状的局部改动；提交 `4daf2b9b6` 还没有重编译成双机服务，也没有四层 PD flow、完整模型 smoke 或 Prefill 关键路径验证，不能据此作为性能锚点。
