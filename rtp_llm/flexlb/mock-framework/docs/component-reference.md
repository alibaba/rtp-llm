# 组件配置参考

[返回使用说明](../README.md)

本页记录 Mock 组件的配置、默认值和接线约束。Case 的目录和 YAML 约定见 [Case 指南](case-guide.md)，执行流程见 [架构说明](architecture.md)，指标名称、单位和采样边界见 [指标覆盖](../views/coverage.md)。

## Master 配置

`environment.master.config` 使用原生环境变量字符串。`FLEXLB_CONFIG` 是业务 JSON，`MODEL_SERVICE_CONFIG` 是路由 JSON。本地支持单 endpoint 组、P/D/V 分离角色和 gRPC。Runner 使用实验内的发现文件替换路由目标，保留服务 ID、组和角色信息。

P 使用 `mock.scheduler.mode: fifo` 时，Master 使用 NON_BATCH 交付。Master 通过 EnqueueBatch 或 EnqueueGroup 交付时，P 必须开启 `group` 模式。原生批量入口支持 DP rank 0。

## Engine 配置规则

Case 的 `component.config` 可内联 YAML，或引用相对 Case 目录的配置文件。Python 只展开点号和组织实例，不维护 Engine 参数默认值。Java 的 `EngineConfig` 负责输入转换，`RtpConfig`、`VitConfig` 负责默认值、范围和语义；没有配置源基类或覆盖层。

已实现且语义对齐的 real 参数沿用 real 命名。未实现的 real 参数直接忽略，值不参与校验，也不进入 effective-config。例如磁盘/远端缓存、独立加载/重试超时、异步传输队列、模型路径、设备与尚未模拟的并行通信参数；出现这些参数不代表 Mock 实现了对应能力。原生参数拼写错误也会被忽略，因此复用 real 配置后应核对 effective-config 中真正生效的字段。

Mock 专用参数使用小写点号名称，例如 `mock.compute.forward.expression`，也可写成嵌套 YAML。未知 Mock 参数、大小写错误、空对象、null、路径冲突和重复输入均报错。缓存淘汰固定使用 LRU。`device_blocks`、`reserve_blocks` 只是 effective-config 的派生输出，不能用于配置覆盖。不要在 Mock Case 放入不生效的 `memory_cache_size_mb`、`enable_disk_cache`、`block_tree_*_queue_max_size` 或独立加载/重试超时。`enable_memory_cache=true` 和 `enable_remote_cache=true` 仍拒绝启动。扩大统一池请设置 `mock.cache.total.blocks`。

### 已实现的 real 参数

| 参数                                                         | 默认值/约束                                                 | 行为                                                                                                                                                                    |
| ------------------------------------------------------------ | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| seq_size_per_block / max_seq_len / test_block_num            | 64 / 8192 / 必填；前两者为 0 时用默认值，test_block_num ≥ 2 | 基础哈希/物理块 token 数、单请求序列上限、每 rank 的 Device 物理块数（含 null block）。P 的逻辑分配粒度为基础块大小乘 prefill_cp_size。                                 |
| concurrency_limit                                            | 32，正整数                                                  | P/D 每轮请求数上限；real 的 ConcurrencyConfig 输入经 engine_config.py 写入内部 max_generate_batch_size。                                                                |
| reserve_block_ratio                                          | 5，0–100                                                    | 首次分配预留百分比。                                                                                                                                                    |
| reuse_cache / enable_device_cache                            | false / true                                                | 前缀复用和 Device 索引。                                                                                                                                                |
| prefill_cp_size                                              | 1（0 视为默认）                                             | Mock P 每组包含的基础缓存块数。大于 1 时只取每组末尾的键，逻辑块大小为基础块大小乘该值，未命中 token 的准入预算按 2 × CP size 对齐。D 的本地 FULL KV 始终按基础块管理。 |
| cp_force_single_prefill                                      | true                                                        | 仅在 P 的 CP size 大于 1 时生效。设为 false 可在同轮接纳多个 Prefill。                                                                                                  |
| max_batch_tokens_size                                        | 0，启动时解析为 max_context_batch_size × max_seq_len        | P 的 batch token/矩形准入；合法 singleton 可超过预算。D 不使用该预算选批，仍通过 WorkerStatus 报告。                                                                    |
| max_context_batch_size                                       | 1，正整数                                                   | 只在 max_batch_tokens_size 缺省或为 0 时参与预算推导，不直接限制 batch 请求数。                                                                                         |
| max_batch_tokens_without_cache / max_inited_kv_cache_streams | 0 / 0                                                       | 仅 P 生效；0 不启用对应限制。                                                                                                                                           |
| disable_decode_first_malloc_device_reuse                     | false                                                       | real 内部 CacheConfig 属性，文件输入，仅支持文件配置。                                                                                                                  |
| gpu_max_batch_size / gpu_batch_wait_ms                       | 1 / 10 ms                                                   | 仅 V；batch=1 为逐请求执行且等待为 0；>1 同时限制 Forward 工作块数与媒体数，较大图片组分轮执行。                                                                        |
| vit_concurrency / vit_max_queue_size                         | 64 / 64                                                     | 异步工作线程上限 / 额外等待额度；总准入和 GPU 就绪队列容量为两者之和。                                                                                                  |
| mm_cache_gpu_max_bytes / mm_cache_cpu_max_bytes              | 各 2 GiB                                                    | V embedding 缓存逻辑预算，单位 byte。                                                                                                                                   |
| mm_hash_key_cache_max_bytes                                  | 256 MiB                                                     | V hash 缓存逻辑预算，单位 byte。                                                                                                                                        |

real 默认值和映射以 `rtp_llm/server/server_args`、`rtp_llm/config/engine_config.py`、`rtp_llm/config/py_config_modules.py` 和 `rtp_llm/cpp/config/ConfigModules.h` 为依据。Mock 不用真实 GPU 分配这些预算，也不按实例伪造物理资源消耗。

### Mock 参数

| 参数                                                                          | 含义                                                                                                                                                                       |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| mock.cache.total.blocks                                                       | 统一池总块数，不含 null block；默认 test_block_num - 1，不得小于 device 可用总块数。ENV：MOCK_CACHE_TOTAL_BLOCKS。                                                         |
| mock.scheduler.mode                                                           | P 调度方式，默认 `fifo` 按单请求入队、自主组 Forward batch，拒绝 `EnqueueBatch`/`EnqueueGroup`；`group` 显式启用组语义。ENV 为 `MOCK_SCHEDULER_MODE`；D 始终使用普通调度。 |
| mock.vit.skip.download                                                        | 默认 false；true 时只跳过实际图片读取，使用输入宽高，其余异步调度不变。                                                                                                    |
| mock.compute.forward.expression                                               | 必填，Forward 公式，结果单位 ms。                                                                                                                                          |
| mock.compute.forward.noise.stddev.ms / stddev.ratio / max.deviation.ms / seed | 有界噪声，详见 Compute；数值后缀均相对 mock.compute.forward.noise。                                                                                                        |
| mock.compute.decode.tokens.per.step                                           | D 平均每轮进度，与真实投机机制不同。                                                                                                                                       |
| mock.compute.eos.probability / seed                                           | P/D 的概率停止模型。                                                                                                                                                       |

### ENV 输入

real 公开参数按其 server_args 声明读取 ENV。P/D 上表的公开项对应大写下划线名，例如 `CONCURRENCY_LIMIT`、`SEQ_SIZE_PER_BLOCK`、`PREFILL_CP_SIZE`、`CP_FORCE_SINGLE_PREFILL`。V 的 batch 参数分别使用 `VIT_GPU_MAX_BATCH_SIZE`、`VIT_GPU_BATCH_WAIT_MS`，队列参数为 `VIT_CONCURRENCY`、`VIT_MAX_QUEUE_SIZE`；缓存参数为 `MM_CACHE_GPU_MAX_BYTES`、`MM_CACHE_CPU_MAX_BYTES`、`MM_HASH_KEY_CACHE_MAX_BYTES`。

Mock 名称的点号转换为下划线并大写，例如 `mock.compute.decode.tokens.per.step` → `MOCK_COMPUTE_DECODE_TOKENS_PER_STEP`。`MOCK_COMPUTE_*`、`MOCK_CACHE_*`、`MOCK_VIT_*` 是严格校验的业务命名空间；框架启动变量另行处理。只读取已声明的业务字段，其他进程 ENV 不写入配置产物。

设置 `MOCK_CONFIG_FILE` 时以实例文件为业务配置源，其他方式使用 ENV 和 `ROLE_TYPE=PREFILL|DECODE|VIT`。Whale 启动配置见 [Whale 部署指南](whale-engine.md)。

## KV 容量和 Prefill CP

`test_block_num - 1` 是每实例可用的 device 块数，`mock.cache.total.blocks` 是统一索引可保留的总块数。首次分配的预留比例按 device 计算。WorkerStatus 和 CacheStatus 的容量字段按 device 报告，统一索引的有效 key 数可以超过 device 块数。启用本地前缀复用时，Engine 和请求都须启用 `reuse_cache`，Engine 的 `enable_device_cache` 须为 true。

### Prefill CP

本地回放时，Client 与 P 对齐基础块 `seq_size_per_block`、`prefill_cp_size` 和有符号 int64 `cache_key_hash_seed`。Client 缺省基础块大小时不生成路由键。Whale 使用真实 Frontend 时按部署指南对齐同一组值。

基础块为 128、CP 为 4 时，每四个基础键取末尾的 K4、K8，路由跨度和 P 的逻辑块大小均为 512 token。媒体身份按基础块边界参与哈希。D 的 FULL KV 使用自己的基础块大小，P/D 在原生请求中交接语义 token 和媒体缓存区间。KVCM 注册使用基础块大小。配置的块数按块计，token 容量按对应角色的逻辑块大小换算。

## ViT 配置

`environment.vit` 无需填写 `engine`，显式提供五个几何参数。下例使用 DSV4.1 的配置写法。

```yaml
vit:
  deployment:
    replicas: 1
  config:
    patch_size: 14
    downsample_ratio: 3
    hidden_size: 4096
    min_pixels: 295936
    max_image_tokens: 1024
    mock.vit.skip.download: true # 本地假 URL 回放；Whale 使用默认 false
    gpu_max_batch_size: 8
    gpu_batch_wait_ms: 10
    vit_concurrency: 64
    vit_max_queue_size: 64
    mm_cache_gpu_max_bytes: 67108864
    mm_cache_cpu_max_bytes: 268435456
    mm_hash_key_cache_max_bytes: 16777216
    mock.compute.forward.expression: "2 + 0.00001 * totalImagePixels"
```

本地匿名图片回放设置 `mock.vit.skip.download: true`，使用数据集记录的宽高。真实 URL 使用默认 false，从 HTTP(S) 图片读取原始尺寸。支持 ImageIO 可读取的格式及 WebP 头尺寸。下载使用 `download_headers`（JSON 字符串）、`mm_image_max_file_size_kb` 和 `mm_timeout_ms`，同时受 DSV4.1 的 10 MiB 文件上限及请求剩余期限约束。

Client 和 P/D 对齐 `image_token_id`。真实 Frontend 接入使用其生效标记。Mock Client 和 Prefill 从 ViT 接收每张图的 seed 和 split size，按需展开行 hash。真实 Frontend 仍通过原生 RPC 获取逐行 hash。

## Compute、Forward、Decode 和 EOS

以 Decode 的 `config` 为例。

```yaml
mock.compute.forward.expression: "19.5 + 0.175 * batchSize"
mock.compute.decode.tokens.per.step: 2.6
mock.compute.eos.probability: 0.0025
mock.compute.eos.seed: 20260922
```

每个 P/D/V 必填 `mock.compute.forward.expression`，结果为每个 batch 的毫秒数。公式支持 `param(name, value)` 和 `sum(...)`。逐请求变量只在 `sum` 中使用，包括 `inputTokens`、`hitCacheTokens`、`computeTokens`、`hasHitCache`、`outputTokens`、`generatedTokens`、`contextTokens`、`stepTokens`、`priority`、`imageCount` 和 `imagePixels`。Batch 公式可用 `batchSize` 和 `total*`、`max*` 派生量，RTP 还提供 `cpSize` 和 `cpEnabled`。公式倍率直接写在表达式中。

`forward.noise` 默认关闭，启用时指定 seed 和正的 `max.deviation.ms`。Decode 的 `tokens.per.step` 默认 1，设为 2.6 时每轮每请求产出 2 或 3 个 token。P/D 的 `eos.probability` 默认 0，概率介于 0 和 1 时需要 seed。实际输出受 EOS、`min_new_tokens`、`ignore_eos`、`max_new_tokens` 和剩余上下文约束。

## DSV4.1 Engine 与 KVCM 接线

Java Master 与 Engine RPC 的源码对照基线为 `ba26ddf1dd86c02bcee3cc6eee966ca59e4ae7a6`（`feature/dsv41-dev-main-202161013`）。直接调用 P 的多 token 请求需开启 `can_use_pd_separation`，并提供一个有效的 Decode 地址。`prefill_kv_handoff.computed_kv_end` 表示已计算 prompt 的排他边界，不含首个采样 token。

Master 在 KVCM 模式下查询 MetaService，不再同步 P/D 完整缓存键。查询失败时使用有容量和 TTL 限制的 Local Standby 预测，按 Engine 代次隔离，健康恢复后回到 KVCM。LOCAL_SYNC 继续使用 Engine 同步索引。ViT 缓存同步独立于 KVCM 模式。

WorkerStatusPB 的缓存 traits 使用字段 23–27，ViT 的 `mm_inflight_inputs` 使用字段 28。Master 与 Engine 需使用同一份协议构建。当前 KVCM 对齐范围为显式 namespace、Full Attention、单 Engine/端点和 CP=1。

KVCM 接入使用以下配置。

```yaml
kv_cache_event_publisher_type: kvcm
kv_cache_event_manager_endpoint: http://<meta-http-host>:<port>
kv_cache_event_instance_group: mock-experiment
kv_cache_event_instance_id: deployment_128
kv_cache_event_snapshot_interval_ms: 600000
```

显式 `kv_cache_event_manager_endpoint` 优先使用。省略时通过 `kvcm_enable_vipserver`、`kvcm_vipserver_domain` 或 `kvcm_server_address` 发现 Meta HTTP leader。实例 ID 显式指定共享查询空间，模型名和实例组需与消费端匹配，快照间隔必须为正。

Reporter 注册一个 Full spec，发布统一池的有效块事件、快照和心跳。HBM 命名空间表示本实例可恢复的逻辑 key 集合。KVCM 能力标记为 `payload_io=false`，原生 READ/WRITE 返回 `FAILED_BUFFER_SAFE`。
