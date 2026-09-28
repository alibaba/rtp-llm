# MLA K 合并核的 12-head 候选

固定 `feat/k3_dev` 四层 FP8 trace 的 MLA pipeline 中，`Tensor.copy` 累计约 0.097 ms；集成版 `d733278ac` 的 `attention.mla.core` 中约 0.307 ms。两个范围的融合边界不同，这些累计值只用于定位候选，不能直接证明整体耗时差异。源码显示集成版的 `concat_and_cast_mha_k_triton` 要求 head 数为 2 的幂；四层 K3 的 12 个 head 因而走两次 PyTorch 切片拷贝。本候选让原核按 2 的幂向上取整 head 范围，并对越界 head 的读写加掩码；K/V 数值布局和通信路径不变。

在 110 的个人 `lhc_GPU` 中、GPU 1 上运行 Bazel `--config=cuda13 --config=sm10x` 用例，源码和构建缓存都在个人 ext4 数据盘。测试输入为 257 token、12 head、128 维 NoPE、64 维 RoPE，NoPE 是从 256 维 KV 投影切出的非连续 BF16 view；与独立的 PyTorch 拼接结果做逐元素精确比较。

- 旧核（文件 SHA256 `1a53157ce9df809f404a5e91ce868331817d2b77a5d61576ee2e273dae6a50de`）的有效红灯在归档内的 `mla-kmerge-red-bazel-20260929-r2.log`：用例实际运行，Triton 在 `tl.arange(0, 12)` 报 `arange's range must be a power of 2`。构建 21,904 个动作，测试 1/1 失败，符合预期。
- 新核（文件 SHA256 `b4f9c4818e6c9e130b148d0aae658291d417a0f0d494cf8163eecc2b753e3f11`）的绿灯在归档内的 `mla-kmerge-green-bazel-20260929-r2.log`：相同用例 1/1 通过，Bazel 退出码 0。新增 `CC=/usr/bin/gcc` 只供 Triton 运行时编译使用；此前一次因容器中不存在 Bazel 传入的 GCC 路径而失败，不能当作代码红灯。

两个原始日志保存在 `raw-logs.tar.gz`，归档校验值见 `raw-logs.sha256`。

用相同的 65,536-token、12-head、BF16 非连续 KV 投影 view，预先分配输出，在 110 的 GPU 1 上做两次拷贝与单次 Triton 合并核的 A/B/B/A 交错测量。两种输出先逐元素精确比对。每组先完成一次惰性编译、至少 10 次同路径预热，末三次 CUDA event 时间落在中位数 ±5% 内；每组再采 30 次。原始样本在 `benchmark-64k.json`，运行日志在 `benchmark-64k-raw.tar.gz`，归档哈希在 `benchmark-64k-raw.sha256`。

| 64K 局部算子 | 第一组中位数 | 第二组中位数 |
|---|---:|---:|
| 两次 PyTorch 拷贝 | 0.313296 ms | 0.313344 ms |
| 单次 Triton 合并 | 0.096768 ms | 0.096400 ms |

这个形状的局部 GPU 时间缩短约 0.217 ms。完整 110–115 探测记录在 `host-selection-benchmark.json`：110 和 113 都无外部 GPU 计算，数值排序选了 113；因为候选已在 110 独立编译，随后对 110 的 GPU 1 单独复查为独占可用，见 `host-selection-benchmark-110.json`。本次只测算子，没有加载模型权重或改动 3FS 并发。

这仍不是四层 FP8+MTP PD flow 或 64K timeline 的结果，不能据此将候选并入性能锚点。下一步需在双机服务里核对新核实际被调用、答案链路正确，再比较全 rank target Prefill 关键路径。
