# cuLA KDA packed checkpoint 候选

四层 64K 热态 trace 中，`feat/k3_dev` 的三层 KDA core 各约 2.777 ms，集成版各约 2.975–2.979 ms。两边都使用 cuLA，但 `feat/k3_dev` 把一条长序列的 checkpoint 一次提交；集成版按最多四个 page 分组。此候选只在单请求、page 对齐且 checkpoint scratch 不超过每 rank 32 MiB 时合并 cuLA 调用；多请求、page 内复用前缀和更大的 scratch 继续走已有分组逻辑。没有修改 BF16 NCCL 或 FP8 格式。

`test_native_kda_packed.py` 在改动前因 7 个 page 被拆为两次调用而失败；改动后两个 CPU 契约测试通过，其中一个检查 page 内复用前缀仍走分组路径。110 的单卡实际 cuLA 测试使用 20,483 token、12 个 head、4,096 token/page，并比较改动前后输出和所有 FP32 checkpoint。`numerical-result-110.json` 显示两者最大绝对差均为 0，测试退出码为 0。`gpu-selection-fleet.json` 与 `gpu-selection-110.json` 保存测试前的资源检查。

基线源码为集成版 `c7479de2ae9c1577f6ddbb9b75dcb1ec04d5f9b2` 的 `native_kda.py`，SHA256 `a096c0029ee6924d8871b5689609796664d14208adf993796b5f704be3781319`；候选源码 SHA256 `5d7bbd5bc973efd527e68449c5999190254102551d0ac129994cde7269917277`。这只是算子数值检查，尚未证明四层 PD 功能、热态性能提升或完整模型效果；须先重编译并通过这些门槛，才能纳入性能锚点。
