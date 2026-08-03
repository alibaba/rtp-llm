# TopK v3 整理与验证报告

日期：2026-08-03

## 分支与远端核对

- RTP-LLM `azk/glm5_indexer_topk_dsv4`：本地和远端均为
  `a59c09151f5f86132bcdcbfa0c32d00bd682f673`。
- RTP-LLM `azk/mqa_topk_fuse`：本地和远端均为
  `bf80c11104f7ea21bd9ca62434cf57f5f1b5e6f9`。
- DeepGEMM 旧 `azk/mqa_topk_fuse`：本地和远端均为
  `3ce7f833340baf99384ceca64f7854aff8d086ff`。
- DeepGEMM 最新 `origin/dev`：
  `031f883aad614f24455f838889ef3f0914732c86`。

核对时目标分支均没有未整合的远端提交。RTP-LLM 的四个 TopK 提交已压缩到
`azk/topk_v3` 的一个提交。DeepGEMM 没有沿用旧分支中的 partial TopK、RoPE
等历史，而是从最新 `origin/dev` 重新移植 paged-MQA 直方图功能。

## 实现整理

- 对外算子统一命名为 `topk_v3` 和 `topk_v3_from_histogram`。
- CUDA 文件统一命名为 `topk_v3.cu`、`topk_v3.cuh`、`topk_v3.h` 和
  `topk_v3_compat.cuh`。
- 保留原始 paged MQA + TopK 路径；仅在实测更有利的长序列 shape 上生成
  1024-bin ordered-FP16 直方图并调用 histogram finalize。
- DeepGEMM 的直方图逻辑移入最新 unified SM100 paged-MQA 内核，不再带入旧
  Q/RoPE 或 partial TopK 代码。

本轮额外修复了以下安全和精度问题：

- scheduler 返回任务前会推进到下一请求，直方图最初可能读取下一请求的长度；
  现在像 row/page 元数据一样暂存当前任务的精确 context length。
- TopK 的负长度和超长长度在设备端限制到 `[0, max_seq_len]`，避免越界读取。
- 空 batch 不再发起零 grid CUDA kernel。
- 所有输入增加同设备检查和 CUDA device guard，使用对应设备的当前 stream。
- 对 row 数、`max_seq_len`、block-table stride/capacity 增加整数范围检查。
- 正负 NaN 统一映射到最高 TopK 优先级和最高直方图 bin，使 direct 和 fused
  路径与 `torch.topk` 的 NaN 语义一致。

## 正确性与安全测试

运行设备由 CUDA Runtime 确认为 compute capability 10.3、148 SM；构建使用
`sm_103`，没有采用 `nvidia-smi` 报告的冲突架构信息。

RTP-LLM：

- Bazel 真实目标 `//rtp_llm/models_py/bindings/cuda/kernels:topk_v3` 编译通过，
  生成 `sm_103` cubin。
- 完整 `//rtp_llm/models_py/bindings/cuda:cuda_bindings` 编译和链接通过。
- 从同一份 RTP `.cu/.cuh` 构建 standalone pybind 后，完整 UT 通过。
- 覆盖 K=512/1024/2048、长度和 dispatcher 临界点、ragged/空行、非对齐
  stride、CUDA Graph、跨设备 guard、1/2/4/8-CTA、候选缓冲溢出、负
  midpoint、FP32 subnormal、`±inf`、`±NaN` 和直方图 continuation。
- Compute Sanitizer memcheck、synccheck 均为 0 错误；非 cluster 路径
  racecheck 为 0 hazard。racecheck 对 CTA-cluster 的 DSMEM 跨 CTA 访问报告
  “目标 block 可能尚未进入”，但同一路径通过 synccheck、memcheck、100 次
  重放和精度校验，这是工具对 cluster DSMEM 的限制，不作为真实竞态处理。

DeepGEMM：

- 新增 shape 矩阵覆盖 0/1 长度、非整页、32K shared-histogram 阈值、
  4K 到 1M、page size 32/64/128、双 stream、跨设备和非法参数。
- 每个有效 paged-MQA logit 与非直方图路径 bitwise 一致；设备直方图与由
  logits 重建的参考直方图完全一致，bin 总数精确等于 context length。
- 原有 paged-MQA 随机回归 12/12 通过，覆盖 FP8/MXFP4/MXFP8、
  FP32/BF16、varlen 和多组 head/page/dim。
- Compute Sanitizer memcheck、synccheck、racecheck 均为 0 错误或 hazard。

完整 Bazel Python test target 还会拉取与本改动无关的 `nanopb` 外部归档；该
镜像返回 403，因此使用真实 RTP 源码的 standalone 模块执行了同一完整测试文件。

## 性能结果

基线为最新 DeepGEMM paged MQA 加原始 vLLM persistent TopK；新路径为 paged
MQA 加 `topk_v3`，长序列按生产规则切换到 MQA histogram 加
`topk_v3_from_histogram`。测试 K=2048，T 为 1/2/4/8/16/32/64/128，KV 为
4K/8K/16K/32K/64K/128K/256K/512K/1M。每点先校验精度，再使用 CUDA Event、
交替 provider 顺序和三轮中位数计时。

| 长度分布 | shape 数 | 初测更快点 | 路由后几何平均提升 | 最小值 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 72 | 72/72 | 1.318x | 1.020x | 1.868x |
| linear | 72 | 72/72 | 1.403x | 1.006x | 2.078x |
| bimodal | 72 | 71/72 | 1.387x | 0.992x | 1.867x |
| geometric | 72 | 72/72 | 1.443x | 1.021x | 2.572x |

四种分布合计 288 个 shape，几何平均提升 1.387x。bimodal 唯一初测回退点是
T=1、KV=1M；T=1 时四种分布输入相同，另外两轮测得 1.020x 和 1.021x。该点
进一步使用 500 次迭代、9 轮中位数复测为 1.006x，确认没有稳定性能回退。

可复现 benchmark 位于 DeepGEMM：
`benchmarks/mqa_topk_fusion/bench_topk_v3.py`。
