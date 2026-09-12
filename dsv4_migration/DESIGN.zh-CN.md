# DSV4 CSA KV Offload 设计与显存口径

基于 main `1f57ca1b5b73467575702fbeebbd23ddebf18a83`，开发分支为
`rym/feat/dsv4_dsa_kvoffload`。实验结果见 [RESULTS.md](RESULTS.md)，
启动和测试命令见 [REPRODUCE.md](REPRODUCE.md)。

## 适用范围

本方案利用 Indexer 驱动的稀疏 attention：每层先选择需要访问的 KV，
再搬运 GPU 缺失的数据。当前代码适配 DeepSeek V4 Pro 的 CSA 路径，
不改变模型的 TopK 选择或 KV 量化方式。

当前支持 FP8 KV、CP1、融合 P/D、无 MTP、无前缀复用、单 token decode。
实现按请求隔离热缓存；本轮性能实验使用逐条真实 prefill 后的固定 decode batch，
尚未验证线上动态请求流的端到端 goodput。

## 四类存储

| 区域 | 内容 | 生命周期和容量 |
| --- | --- | --- |
| GPU 必驻留区 | CSA Indexer Key、HCA KV、SWA KV、compressor states | Indexer Key 和 HCA 随上下文增长；SWA/state 按池容量预留 |
| GPU 私有热缓存 | 每请求、每个 CSA 层 2048 个压缩 KV 条目 | 容量固定，条目随访问更新；请求之间不互相驱逐 |
| GPU 额外驻留区 | 预算内的 CSA KV 副本 | 当前实现驻留物理地址前缀，分配器优先复用小编号空闲页 |
| CPU 完整副本 | 全部 CSA KV | pinned 主存，保存可 offload 数据的完整副本 |

“2K 热缓存”指 2048 个压缩条目，不是 2048 个原始 token，不是所有层共用一份。
Pro 每 4 个原始 token 生成一个 CSA 条目，每层每步选择 1024 个条目。
当前实现有 30 个 CSA 层，因此每请求有 30 份独立的热缓存数据。

GPU 额外驻留区当前不是滑动窗口，也不是按命中率动态调整的全局 LRU。
CPU 副本从一开始就维护；GPU 放不下的数据可以只保留 CPU 副本，
不需要等显存不足时再同步搬走旧 KV。GPU 已驻留的条目可直接被 attention 读取，
不必再复制到私有热缓存。

## 每层 Decode 流程

该流程位于每个 CSA 层内部。后续层的 Indexer 依赖前面层的隐藏状态，
不能在整个模型 forward 开始前一次性完成所有层的数据准备。

1. 更新当前层 Indexer Key，计算本次 TopK 压缩条目的局部索引。
2. 用框架的请求页表转换为 CPU 完整池的物理条目索引。
3. 检查 GPU 驻留区和当前请求私有热缓存，保护本次仍要使用的命中项。
4. 在私有热缓存内按环形扫描选择未受保护的位置，为缺失项分配槽位。
5. 在副流从 CPU 回填缺失条目，同时主流运行当前层主 compressor。
6. 等回填和当前压缩边界条目写入完成，更新必要的 GPU 副本及物理索引，执行 attention。
7. 热缓存内容保留到后续 step；未被淘汰的更早历史条目也能命中。

例如 TopK 的 1024 个条目中，600 个命中驻留区、300 个命中私有热缓存，
只需要从 CPU 回填 124 个条目。attention 从驻留区和热缓存共同读取完整 TopK。

热缓存驱逐是保护当前 TopK 后的环形扫描，不是严格 LRU。页表首个物理页和
prefill 生成号共同标识请求，防止请求结束后复用物理页时误命中旧数据。
热缓存槽位也记录标签和生成号，旧映射必须经过验证才能命中。

## CUDA Graph 与一致性

GPU 缓冲、请求映射和回填队列在初始化时分配，decode 不读取 GPU 计数到 CPU。
动态命中数和回填任务由 GPU kernel 处理。主流和副流的事件依赖进入 Graph，
每次重放仍按相同依赖关系执行，实际条目和缺失数量可以变化。

本步刚生成的压缩边界条目必须等 compressor 写好，不能在预取时读取旧内容。
实现对该条目延迟读取，并在写入后更新对应 GPU 槽位。已有 KV 搬运保持
MODEL1 的 payload、scale 和分页布局，不重新量化。

`DSV4_CSA_VALIDATE_BYTES=1` 可在 attention 前逐字节比对所选 GPU 条目与
CPU 完整副本。它支持 Graph 重放，但会增加开销，必须与性能实验分开运行。

## 显存与主存的三种口径

以下以每张 GPU/rank 为单位，GiB 为 2^30 字节，MiB 为 2^20 字节。

| 数字 | 含义 |
| --- | --- |
| 12 GiB | 本轮 GPU KV 总预算，包含必驻留池、CSA 驻留区、热缓存和缓存元数据 |
| 68.6 GiB | 提前分配的 CPU CSA 池总容量，供全部请求共享，不是单请求用量 |
| 548.44 MiB，约 0.536 GiB | 单条 128K 请求在全部 30 个 CSA 层上的 KV 用量 |
| 698.34 MiB | 单条 128K 请求的 CSA、Indexer Key 和 HCA 分页 KV 合计，不含固定池 |
| 20.2 GiB | 首次真实 128K prefill 的 GPU allocated 峰值相对执行前的增量 |

CPU 池配置为每层 65537 个物理块，一个块对应 256 个原始 token、64 个
压缩条目。每层每块含对齐共 37440 字节。因此：

```text
单请求 CSA KV = 512 blocks × 37440 bytes × 30 layers = 548.4375 MiB
CPU 完整池   = 65537 blocks × 37440 bytes × 30 layers = 68.556 GiB
```

B16 的 128K 输入需要约 8.57 GiB/rank 的有效 CSA 数据，B32 约 17.14 GiB/rank。
其余 CPU 池空间尚未被有效请求 KV 使用；输出增长会增加所需页数。

20.2 GiB 不是 prefill 结束后永久增加的 KV，也不能乘以 batch 累加。
测量公式是 `max_memory_allocated_during_prefill - memory_allocated_before_prefill`，
包括计算中间结果，也可能包括首次初始化的持久缓冲。顺序执行请求时可以复用
临时工作空间；PyTorch reserved 内存和 NVML 总占用又是另外的统计口径。

这条代码路径中的大张量包括：

| 张量 | 形状、精度 | 容量 |
| --- | --- | --- |
| mHC 隐藏状态 | `[131072, 4, 7168]`，BF16 | 7 GiB |
| 本地 Q 有效数据 | `[131072, 32, 512]`，BF16 | 4 GiB |
| Q 工作区实际预留 | 按大于 128K 的 max_seq_len 预留，向上对齐到整 GiB | 本配置 5 GiB |
| 普通隐藏状态或投影输出，单份 | `[131072, 7168]`，BF16 | 1.75 GiB |
| Indexer 一块打分结果 | `[8192, 32768]`，FP32 | 1 GiB |

这些是根据代码形状算出的容量，存活时间不同，不能直接相加当成精确峰值明细。
虽然 attention/MoE 按块计算，当前仍保留全长 mHC 隐藏状态和 Q 工作区。
逐项解释 20.2 GiB 的峰值还需要显存分配追踪。

## 容量和性能结论的边界

主方案按 CSA GPU 区 6 GiB、其他必驻留池 6 GiB 分配；补测按 8 GiB + 4 GiB。
两者的 2K 私有热缓存均已计入总预算，没有额外加算 GPU 显存。

Vanilla B24/B32 是固定整批准入时的 KV 配额不足错误，不是物理 CUDA OOM，
也不是把 TPOT 超过 SLO 算作失败。普通服务可以排队或缩小 batch。
这轮没有逐步扩大总 KV 预算测到硬件边界，不能宣称整机绝对容量上限。

CUDA Graph 和跨步命中可以减少额外开销，但不会消除 CPU 带宽需求。
必驻留数据、私有热缓存和计算量仍随 batch 增长。性能结果不代表完整质量等价：
多请求自由生成出现过输出分叉，vanilla 重跑也有类似现象，原因尚未隔离。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `rtp_llm/models_py/modules/dsv4/fp8/csa_cache.py` | 驻留区、私有缓存、命中、驱逐、回填、请求生命周期 |
| `rtp_llm/models_py/modules/dsv4/fp8/attention.py` | Indexer 后启动回填，接入 attention |
| `rtp_llm/models_py/modules/dsv4/fp8/compressor.py` | 新 KV 写入和 GPU 副本更新 |
| `rtp_llm/models_py/modules/dsv4/fp8/kv_offload.py` | 字节搬运参考实现及诊断校验器 |
| `rtp_llm/models/dsv4_kv_cache.py` | CPU CSA 池的框架描述 |
| `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc` | GPU KV 预算扣除与分配 |
| `rtp_llm/models_py/modules/dsv4/offload_config.py` | 显式启用与环境变量 |
| `rtp_llm/models_py/modules/dsv4/flash_mla_compat.py` | 两种方案共用的 TP4 FlashMLA 兼容路径 |
