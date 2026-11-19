# Mock 配置参考

本页按角色列出 Mock 专用参数和常用 RTP 参数。本地 Case 在 `environment.<role>.config` 内联 YAML，或引用相对于 Case 目录的配置文件。独立 Engine 用 `ROLE_TYPE` 选角色，经 ENV 读业务参数。设置 `MOCK_CONFIG_FILE` 时，以实例文件为准。Mock 点分名使用大写下划线对应 ENV，例如 `mock.cache.total.blocks` 对应 `MOCK_CACHE_TOTAL_BLOCKS`。

复用 RTP 配置时，到本地 `effective-config.json` 或独立 Engine 的 `engine_started.effective_config` 核对生效值。未识别的原生字段会被忽略，未知 Mock 字段会报错。[Case 写法](case-guide.md) · [Whale 取值与换算](whale-engine.md#部署配置)

## Master

- `FLEXLB_CONFIG` 是调度配置，`MODEL_SERVICE_CONFIG` 是路由配置，两者都填原生 JSON 字符串。本地支持单 endpoint 组、P/D/V 分离和 gRPC，Runner 写入发现地址和选举配置。

## Frontend

- `seq_size_per_block`、`prefill_cp_size` 用于生成缓存路由键，与 Prefill 对齐。省略基础块大小时跳过路由键生成。
- `cache_key_hash_seed` 是有符号 int64，须与 Prefill 一致。
- `image_token_id` 是带图请求的占位 token，须与 P/D 一致。

## Prefill

- Case 声明 `engine: rtp`。`mock.scheduler.mode` 默认 `fifo`，对应 Master 的 `NON_BATCH`。通过 `EnqueueBatch` 或 `EnqueueGroup` 提交时设 `group`，原生批量入口支持 DP rank 0。
- `mock.compute.forward.expression` 必填，返回每个 Forward batch 的模拟耗时，单位 ms。`mock.compute.forward.noise.*` 可模拟耗时波动，`mock.engine.prefill.overhead.ms` 控制轮间额外等待。[性能模拟](whale-engine.md#性能模拟)
- `seq_size_per_block` 是基础块大小，默认 64 token。`prefill_cp_size` 默认 1，填 0 按 1 处理。逻辑块大小和路由跨度均为基础块大小 × CP，带图请求按基础块边界混入媒体身份。`cp_force_single_prefill` 默认开启，仅在 CP 大于 1 时生效。
- `test_block_num` 必填，至少 2，含 1 个保留的 null block。Device 可用块数为 `test_block_num - 1`，按逻辑块换算 token 容量。
- `mock.cache.total.blocks` 控制统一池的保留量，单位为逻辑块，不含 null block。默认等于 Device 可用块数，最小值也是该数。WorkerStatus 和 CacheStatus 按 Device 容量上报。
- `reserve_block_ratio` 默认 5%，按 Device 可用块数向下取整。`reuse_cache` 默认 false，`enable_device_cache` 默认 true。前缀复用要求 Engine 和请求都启用 `reuse_cache`，Engine 同时启用 Device 索引。
- `max_seq_len` 限制单请求序列长度。`concurrency_limit` 限制每轮请求数，`max_batch_tokens_size` 控制组批 token 预算。
- `cache_key_hash_seed` 和 `image_token_id` 按 Frontend 生效值配置。[线上 CP 配置](whale-engine.md#cp-与-block-size) · [Device 块数换算](whale-engine.md#kv-容量)

## Decode

- Case 声明 `engine: rtp`。`mock.compute.forward.expression` 必填，返回每个 Forward batch 的模拟耗时，单位 ms。`mock.compute.forward.noise.*` 可模拟耗时波动。
- `mock.compute.decode.tokens.per.step` 控制每请求每轮的平均 token 产出。`mock.compute.output.mean.tokens`、`mock.compute.output.stddev.percent`、`mock.compute.output.seed` 控制输出长度分布。[性能模拟](whale-engine.md#性能模拟)
- `seq_size_per_block` 按目标 Decode 配置。Decode 固定 CP1，以自身基础块大小换算 KV token 容量。
- `test_block_num` 必填并含 1 个 null block。`mock.cache.total.blocks`、`reserve_block_ratio`、`reuse_cache` 和 `enable_device_cache` 的含义同 [Prefill](#prefill)。
- `max_seq_len` 限制单请求序列长度，`concurrency_limit` 限制每轮请求数。`cache_key_hash_seed` 和 `image_token_id` 按 Frontend 生效值配置。

## ViT

- Case 省略 `engine`。必填 `patch_size`、`downsample_ratio`、`hidden_size`、`min_pixels`、`max_image_tokens`，取目标模型的媒体配置。
- `mock.compute.forward.expression` 必填，返回每个 Forward batch 的模拟耗时，单位 ms。`mock.compute.forward.noise.*` 可模拟耗时波动。
- `gpu_max_batch_size`、`gpu_batch_wait_ms` 控制合批规模和等待，用 ENV 时加 `VIT_` 前缀。`vit_concurrency`、`vit_max_queue_size` 控制异步准备并发和额外等待额度。
- `mm_cache_gpu_max_bytes`、`mm_cache_cpu_max_bytes` 分别设置 Device 和 Host 的 embedding 缓存预算。`mm_hash_key_cache_max_bytes` 设置 hash 缓存预算。三者均以字节计。
- `mock.vit.skip.download` 默认为 false，V 从 HTTP(S) 图片读取原始尺寸。匿名回放设 true，直接使用记录的宽高。实际下载受 10 MiB 硬上限、`mm_image_max_file_size_kb` 和请求剩余期限约束。[Whale 图片下载](whale-engine.md#whale-mock-vit-下载)
