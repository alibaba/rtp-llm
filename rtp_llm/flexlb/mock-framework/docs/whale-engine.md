---
name: mock-engine-whale-deploy
description: 在 Whale 上构建并部署 FlexLB Mock Prefill、Decode、ViT，配置 CP、KV 容量和性能模拟。
---

# 在 Whale 上部署 Mock Engine

P/D/V 共用一个 Mock Engine 镜像，每个 Pod 启动一个 JVM。Master 使用 FlexLB 镜像，Frontend 使用与节点架构匹配的完整 RTP 镜像。

## 部署流程

### 1. 构建 CI 镜像

使用官方 `a1` CLI。将 `TARGET_ARCHITECTURE` 设为 `amd64` 或 `arm64`，将 `SOURCE_SHA` 设为已推送到开源仓的 40 位源码 SHA，`RECIPE_REF` 设为内部仓中承载构建配方的分支。在 Code 查看该分支的 `.aoneci/image.yaml`，确认含有 `image-mock-engine` 和 `hangzhou-convert-dadi-image-mock-engine`，构建参数 `target-architecture` 选择目标架构：`amd64`（默认）或 `arm64`。配方统一选择构建平台、固定版本的 Java 基础镜像和镜像 tag；两种架构使用同一份源码及 Dockerfile，不需要切换提交。以下使用 `foundation_models/RTP-LLM` 的镜像流水线 `18410`。

```bash
a1 ci pipeline run 18410 \
  --repo foundation_models/RTP-LLM --branch "${RECIPE_REF:?}" \
  --param 'pre-jobs=["mock-engine"]' --param "github_commit=${SOURCE_SHA:?}" \
  --param "target-architecture=${TARGET_ARCHITECTURE:?}" \
  --param 'github_source_repo=https://github.com/alibaba/rtp-llm.git' \
  --param 'cancel-in-progress=false' --param 'trigger_beta_downstream_sm9x=false' -f json

# RUN_ID 取触发返回的 Run ID。
a1 ci run get "$RUN_ID" --repo foundation_models/RTP-LLM -f json
a1 ci job list --run "$RUN_ID" --repo foundation_models/RTP-LLM -f json
# DADI_JOB_ID 取 hangzhou-convert-dadi-image-mock-engine 的 Job ID。
a1 ci job summary "$DADI_JOB_ID" --run "$RUN_ID" --repo foundation_models/RTP-LLM -f json
```

等整次运行结束且无失败，再从成功的 DADI Job 摘要取完整镜像地址。摘要可能同时列出 `_accelerated` 和 `_containerd_accelerated`，按目标模板的容器运行时选用对应产物。打开 `image-mock-engine` Job 的 **Package Mock Engine** 步骤日志，核对 `source_sha`、`recipe_sha`、`internal_discovery=true`、`metrics_backend=kmonitor` 和 `service_startup_verified=true`。构建日志还会列出目标架构与基础镜像；ARM 产物 tag 含 `_arm64`，x86 保持原有 tag 格式。

### 2. 创建模板

在 Whale 控制台的模板管理中创建模板和首版。以兼容的 RTP/FlexLB 模板为底座，固定源版本，保留框架、模型类型和角色启动契约。以 DSV4.1 V1 为例，框架类别为 `FT`，P/D 共用 `inference_zone_plan`，V 用 `vit_zone_plan`，Master 和 Frontend 分别用 `master_zone_plan` 和 `front_app_zone_plan`。

P/D/V 的资源候选绑定同一个 CI 镜像，按实际节点架构选择资源。启动命令用 `/bin/bash /home/admin/whale_start.sh`，脚本会直接启动 Engine JAR。Master 和 Frontend 沿用各自镜像的启动命令。模板保存共用入口和资源候选，本部署的容量、公式和并发写在 zone。

#### 创建无 GPU 的 Mock P/D/V

1. **模板先提供 GPU=0 的候选。** 在推理角色的资源候选中添加一项供 P/D 共用，在 ViT 的资源候选中另加一项。两项都绑定 Mock Engine 镜像，GPU 数量（`resource_tag.gpu_amount`）填 `0`，CPU、内存、磁盘和节点架构按目标资源池配置。
2. **创建部署时选中该候选。** 在 Prefill、Decode、ViT 三个 zone 的资源规格中分别选匹配的 GPU=0 项，再填副本数。资源列表没有该项时，先给模板补候选并让服务关联该模板版本。
3. **提交后看 Carbon 实际申请。** 确认 P/D/V 均无正数 GPU 资源申请。资源池若要求填写 GPU 型号，可保留匹配的型号标签。例如 `L20D` 可以是节点选择条件，GPU 数量仍为 0。

### 3. 创建服务和部署

1. **创建服务。** 在 Whale 选择仅创建服务。填写服务名、团队、属主、模型类型、模型和 tokenizer 资源，关联上一步的模板版本。启用 P/D 分离、独立 ViT、Master 和 Frontend。记下服务配置版本，创建部署时固定到该版本。
2. **创建部署。** 在该服务下新建多组部署。填写部署名、服务配置版本、地域/代理、配额组和各 zone 副本。按 [Zone 类型、端口与发现](#zone-类型端口与发现) 设置角色和服务发现。

### 4. 配置 zone 并发布

1. 用正式字段设置 zone 类型、端口、资源和副本。平台根据角色及端口配置提供 `ROLE_TYPE`、`START_PORT`，以 Carbon 启动计划为准。P/D/V 的 JVM 堆用 `JAVA_TOOL_OPTIONS` 设置。
2. 按 [部署配置](#部署配置) 取值和换算，在 `zone_process_setting.process_info.envs` 填业务参数。
3. 在 Whale 的部署页保存目标 zone 配置，检查启动命令、镜像、资源覆盖和 ENV 的差异。修改 zone 配置后提交当前部署。更换服务配置版本时，在发布页面选中对应版本和目标部署。
4. 回读 Carbon 启动计划、镜像、健康探针和 VIP 注册端口。确认各角色就绪、`engine_started` 日志中的 `effective_config` 与预期一致，再从 Frontend 发一条真实请求。滚动更新后观察成功率、TTFT 和 Prefill 复用率回稳。

## 部署配置

### Zone 类型、端口与发现

| Zone                   | `zone_type`                           | 参考 HTTP 端口与探针     | 发现与调用                                                                |
| ---------------------- | ------------------------------------- | ------------------------ | ------------------------------------------------------------------------- |
| Prefill / Decode / ViT | `PY_PREFILL` / `PY_DECODE` / `PY_VIT` | `8088 /health`           | VIP 注册 HTTP 端口，Master 按 HTTP 发现将端口加 1 得到 Engine gRPC 端口。 |
| Master                 | `PY_MASTER`                           | `7001 /health`           | VIP 注册 HTTP 端口，原生 Master gRPC 端口为 HTTP + 2。                    |
| Frontend               | `PY_FRONT_APP`                        | `20550 /frontend_health` | 入口 VIP 注册 Frontend HTTP 端口。                                        |

为每个 zone 注册本部署独立的 VIP 域名，在 zone 的服务发现配置中绑定。Master 的 `MODEL_SERVICE_CONFIG.role_endpoints` 指向本部署的 P/D/V，使用 `vipserver` 发现和 `protocol: http`。`FLEXLB_CONFIG.cacheMatching.type` 设为 `LOCAL_SYNC`。Frontend 的 `MODEL_SERVICE_CONFIG.master_endpoint` 指向本部署的 Master VIP。独立 ViT 使用 `VIT_SEPARATION=2`、`MM_TRANSPORT_MODE=grpc`，流式返回启用 `FETCH_OUTPUT_STREAM=1`。

### CP 与 block size

1. 从真实 Frontend 的生效配置取 `SEQ_SIZE_PER_BLOCK`、`PREFILL_CP_SIZE` 和 `CACHE_KEY_HASH_SEED`。显式 CP 场景使用支持该能力的 Frontend，并设置 `PREFILL_CP_KV_CACHE_SHARDED=1`。
2. Mock P 填相同的基础块和 CP，Mock P/D 使用相同的 seed。Mock D 的基础块取目标 Decode 的生效值。带图时，P/D 的 `IMAGE_TOKEN_ID` 取 Frontend 的生效值。
3. 读回 Mock P 的 `effective_config.cache_key_block_size` 和 `CacheStatus.block_size`，确认与 Frontend 的路由跨度一致。用同一 prompt 回放两次，按实验目标检查第二次路由和 Prefill 复用。

纯文本且 Master 使用 `LOCAL_SYNC` 时，只对齐缓存路由和容量可以用等跨度配置。例如 Frontend 维持 `64 × CP8`，Mock P 可以设为 `512 × CP1`。相同 token 和 seed 对应相同路由 key，Mock WorkerStatus 的 `tp_size` 会变为 1，CP 相关调度和耗时公式也随之变化。带图场景按真实 Frontend 的基础块与 CP 分别对齐。[Prefill 参数](configuration.md#prefill)

### KV 容量

#### Device 容量与 Forward 门禁

Device 容量决定在途请求能占用多少 KV。验证 Forward 吞吐或耗时时，先对齐 Device 容量，再观察目标负载下有多少请求进入 Forward。已命中的请求也要取得 Device 驻留额度。

1. 分别选一个 Real Prefill 和 Decode 实例，读取 `/worker_status` 的 `total_kv_cache`。确认取的是单实例的 **Device token 容量**。该口径已计入生效 CP，直接用作目标值。
2. 按 [Prefill](configuration.md#prefill) 或 [Decode](configuration.md#decode) 的逻辑块定义计算 **Mock 对应角色**的每块 token 数。目标 token 数除以每块 token 数，向上取整后加 1，填入对应 zone 的 `TEST_BLOCK_NUM`。修改 Mock 的基础块大小或 CP 后重新换算。
3. 发布后回读 `effective_config.device_blocks` 和 WorkerStatus 的 KV token 容量。预期 token 容量是可用 Device 块数 × 每块 token 数。跑 Forward 门禁时，同时检查请求是否因 KV 不足排队或失败。

例如目标 Real P 的 Device 容量为 **524,288 token**，Mock P 的块大小是 `128 × 4 = 512` token。可用块数为 `524,288 ÷ 512 = 1024`，因此填 `TEST_BLOCK_NUM=1025`。[Device 与统一池的口径](configuration.md#prefill)

#### 总容量与前缀保留量

按实验计划保留的前缀量设置 `MOCK_CACHE_TOTAL_BLOCKS`。沿用上面的 Device 配置，如果希望统一池保留 4096 个逻辑块（2,097,152 token），就填 `4096`。发布后读回 `effective_config` 中的 `mock.cache.total.blocks` 和 WorkerStatus 的 Device 容量，分别核对两项。

#### Prefill 缓存命中率

开启 `REUSE_CACHE` 和 Device 索引，并让请求允许复用。预热后按目标流量回放，在同一测量窗口统计 Prefill 实际命中 token 数 ÷ 查询 token 数。Case 可用 `engine/query/cache_hit_rate` 写门禁。短回放可以重复同一前缀验证接线，性能验收使用能代表目标前缀分布的流量。[命中率门禁示例](../cases/performance_regression/case.yaml)

### 性能模拟

P/D 的 `MAX_SEQ_LEN` 取目标序列长度上限。各角色 zone 设置 `MOCK_COMPUTE_FORWARD_EXPRESSION`，本地 Case 的对应项是 `mock.compute.forward.expression`。按目标模型、负载和角色校准。以下公式用来展示写法。

| 角色    | Forward 公式示例                         | 配置和回读                                                                                                                                                             |
| ------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prefill | `110 + 0.0032 * sum(computeTokens)`      | 用 `CONCURRENCY_LIMIT` 对齐每轮请求数，按需设置 `MAX_BATCH_TOKENS_SIZE` 和 `MAX_BATCH_TOKENS_WITHOUT_CACHE`。用相近输入长度和缓存命中率对比 TTFT、每轮批次和计算吞吐。 |
| Decode  | `15 + 0.1 * batchSize`                   | 用 `CONCURRENCY_LIMIT` 控制每轮请求数。对齐输出长度和每轮产出后，对比 TPOT 和生成吞吐。                                                                                |
| ViT     | `43` 或 `2 + 0.00001 * totalImagePixels` | 用 `VIT_GPU_MAX_BATCH_SIZE`、`VIT_GPU_BATCH_WAIT_MS` 控制合批，用 `VIT_CONCURRENCY`、`VIT_MAX_QUEUE_SIZE` 控制准备并发和排队。校准时分别看 Forward、下载和排队。       |

用 `param(name, value)` 定义常量。Batch 可用 `batchSize`、`total*` 和 `max*`，RTP 另有 `cpSize`、`cpEnabled`。逐请求的 `inputTokens`、`hitCacheTokens`、`computeTokens`、`hasHitCache`、`outputTokens`、`generatedTokens`、`contextTokens`、`stepTokens`、`priority`、`imageCount` 和 `imagePixels` 放在 `sum(...)` 内。ViT 的像素量取 resize、补边后的尺寸。

#### Forward 抖动和 Decode 产出

- **Forward 抖动。** 需要模拟耗时波动时，用 `MOCK_COMPUTE_FORWARD_NOISE_STDDEV_MS` 或 `MOCK_COMPUTE_FORWARD_NOISE_STDDEV_RATIO` 设置幅度，同时配置同前缀的 `MAX_DEVIATION_MS` 和 `SEED`。先对齐基础公式耗时，再用相同负载核对耗时分布。
- **每轮产出。** 在 Decode 设 `MOCK_COMPUTE_DECODE_TOKENS_PER_STEP`。例如设为 `2.6`，每请求每轮产出 2 或 3 个 token，长期均值为 2.6。该值用于模拟接受后的产出量。
- **输出长度。** 在 Decode 用 `MOCK_COMPUTE_OUTPUT_MEAN_TOKENS`、`MOCK_COMPUTE_OUTPUT_STDDEV_PERCENT` 和 `MOCK_COMPUTE_OUTPUT_SEED` 对齐目标长度分布。均值 1000、百分比 10 对应标准差 100 token。Decode 每请求采样一次，计入 Prefill 首 token，再按 `min_new_tokens`、`max_new_tokens` 和上下文容量处理。省略均值或请求启用 `ignore_eos` 时，生成到有效上限。

#### Prefill 轮间开销

用于对齐一轮输出提交到下一轮组批的间隔。在 Prefill zone 配置 `MOCK_ENGINE_PREFILL_OVERHEAD_MS`，Case YAML 写 `mock.engine.prefill.overhead.ms`。取非负整数，单位 ms。先在相近的单 P 负载下比较批次、轮次间隔、Wall TPS 和 TTFT，再调整间隔并重复观测。等待期间新到的请求可以进入下一批。

### ViT 部署与图片输入

Mock V zone 的 `PATCH_SIZE`、`DOWNSAMPLE_RATIO`、`HIDDEN_SIZE`、`MIN_PIXELS`、`MAX_IMAGE_TOKENS` 取目标模型的媒体配置。

#### Whale Mock ViT 下载

- 使用 `OSS_INTERNAL_REGION_ALLOWLIST` 配置可保留 OSS 内网地址的地域（JSON 数组）。空数组会移除所有地域 OSS 地址的 `-internal` 后缀。未配置时，已知的公网 OSS 地址按 Real 默认规则转成集团内网地址。设 `ENABLE_OSS_INTERNAL_REWRITE=0` 可关闭改写。
- 改写只用于实际下载，图片缓存仍使用原始输入。本地使用 `mock.vit.skip.download: true` 时直接使用输入宽高。

#### Mock Prefill 连接 Real ViT

请求携带 `multimodal_token_layout` 时，Mock Prefill 校验媒体区间，并直接使用 Frontend 已展开的 `token_ids` 计算缓存 key。媒体行 hash 已包含在这些 token 中，Prefill 无需再调用 ViT。

让请求中的 VIT role 地址指向 Real ViT，并确认 Real ViT 能下载原图。Mock Prefill 会自动使用原生 `GetMultimodalHashes` 获取元数据，Real ViT 可保留已有的 RDMA 配置。若 Frontend 也要直接查询该 Real ViT，使用支持原生 ViT RPC 的 Frontend。

发一条带图请求，确认 Prefill 获得视觉结果且请求成功结束。记录 TTFT 时计入等待视觉结果的时间。
