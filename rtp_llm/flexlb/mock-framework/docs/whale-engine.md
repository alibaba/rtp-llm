---
name: mock-engine-whale-deploy
description: 在 Whale 上构建并部署 FlexLB Mock Prefill、Decode、ViT，配置 CP、KV 容量和性能模拟。
---

# 在 Whale 上部署 Mock Engine

P/D/V 共用一个 Mock Engine 镜像，每个 Pod 启动一个 JVM。Master 使用 FlexLB 镜像，Frontend 使用与节点架构匹配的完整 RTP 镜像。

## 1. 构建 CI 镜像

使用官方 `a1` CLI。将 `SOURCE_SHA` 设为已推送到开源仓的 40 位源码 SHA，`RECIPE_REF` 设为内部仓中承载构建配方的分支。在 Code 查看该分支的 `.aoneci/image.yaml`，确认含有 `mock-image` 和 `hangzhou-convert-dadi-image-mock-engine`，并确认 amd64 配方与目标节点架构一致。以下使用 `foundation_models/RTP-LLM` 的镜像流水线 `18410`。

```bash
a1 ci pipeline run 18410 \
  --repo foundation_models/RTP-LLM --branch "${RECIPE_REF:?}" \
  --param 'pre-jobs=["mock-image"]' --param "github_commit=${SOURCE_SHA:?}" \
  --param 'github_source_repo=https://github.com/alibaba/rtp-llm.git' \
  --param 'cancel-in-progress=false' --param 'trigger_beta_downstream_sm9x=false' -f json

# RUN_ID 取触发返回的 Run ID。
a1 ci run get "$RUN_ID" --repo foundation_models/RTP-LLM -f json
a1 ci job list --run "$RUN_ID" --repo foundation_models/RTP-LLM -f json
# DADI_JOB_ID 取 hangzhou-convert-dadi-image-mock-engine 的 Job ID。
a1 ci job summary "$DADI_JOB_ID" --run "$RUN_ID" --repo foundation_models/RTP-LLM -f json
```

等整次运行结束且无失败，再从成功的 DADI Job 摘要取完整镜像地址。摘要可能同时列出 `_accelerated` 和 `_containerd_accelerated`，按目标模板的容器运行时选用对应产物。打开 `mock-image` Job 的 **Package Mock Engine** 步骤日志，核对 `source_sha`、`recipe_sha`、`internal_discovery=true`、`metrics_backend=kmonitor` 和 `service_startup_verified=true`。

## 2. 创建模板

在 Whale 控制台的模板管理中创建模板和首版。以兼容的 RTP/FlexLB 模板为底座，固定源版本，保留框架、模型类型和角色启动契约。以 DSV4.1 V1 为例，框架类别为 `FT`，P/D 共用 `inference_zone_plan`，V 用 `vit_zone_plan`，Master 和 Frontend 分别用 `master_zone_plan` 和 `front_app_zone_plan`。

P/D/V 的资源候选绑定同一个 CI 镜像，按实际节点架构选择资源。启动命令用 `/bin/bash /home/admin/whale_start.sh`，脚本会直接启动 Engine JAR。Master 和 Frontend 沿用各自镜像的启动命令。模板保存共用入口和资源候选，本部署的容量、公式和并发写在 zone。

### 创建无 GPU 的 Mock P/D/V

1. **模板先提供 GPU=0 的候选。** 在推理角色的资源候选中添加一项供 P/D 共用，在 ViT 的资源候选中另加一项。两项都绑定 Mock Engine 镜像，GPU 数量（`resource_tag.gpu_amount`）填 `0`，CPU、内存、磁盘和节点架构按目标资源池配置。
2. **创建部署时选中该候选。** 在 Prefill、Decode、ViT 三个 zone 的资源规格中分别选匹配的 GPU=0 项，再填副本数。资源列表没有该项时，先给模板补候选并让服务关联该模板版本。
3. **提交后看 Carbon 实际申请。** 确认 P/D/V 均无正数 GPU 资源申请。资源池若要求填写 GPU 型号，可保留匹配的型号标签。例如 `L20D` 可以是节点选择条件，GPU 数量仍为 0。

## 3. 创建服务和部署

1. **创建服务。** 在 Whale 选择仅创建服务。填写服务名、团队、属主、模型类型、模型和 tokenizer 资源，关联上一步的模板版本。启用 P/D 分离、独立 ViT、Master 和 Frontend。记下服务配置版本，创建部署时固定到该版本。
2. **创建部署。** 在该服务下新建多组部署。填写部署名、服务配置版本、地域/代理、配额组和各 zone 副本。按下表选择正式角色类型和 HTTP 端口。
3. **接好发现。** 为每个 zone 注册本部署独立的 VIP 域名，在 zone 的服务发现配置中绑定。Master 的 `MODEL_SERVICE_CONFIG.role_endpoints` 指向本部署的 P/D/V，使用 `vipserver` 发现和 `protocol: http`。`FLEXLB_CONFIG.cacheMatching.type` 设为 `LOCAL_SYNC`。Frontend 的 `MODEL_SERVICE_CONFIG.master_endpoint` 指向本部署的 Master VIP。独立 ViT 使用 `VIT_SEPARATION=2`、`MM_TRANSPORT_MODE=grpc`，流式返回启用 `FETCH_OUTPUT_STREAM=1`。

| Zone | `zone_type` | 参考 HTTP 端口与探针 | 发现与调用 |
| --- | --- | --- | --- |
| Prefill / Decode / ViT | `PY_PREFILL` / `PY_DECODE` / `PY_VIT` | `8088 /health` | VIP 注册 HTTP 端口，Master 按 HTTP 发现将端口加 1 得到 Engine gRPC 端口。 |
| Master | `PY_MASTER` | `7001 /health` | VIP 注册 HTTP 端口，原生 Master gRPC 端口为 HTTP + 2。 |
| Frontend | `PY_FRONT_APP` | `20550 /frontend_health` | 入口 VIP 注册 Frontend HTTP 端口。 |

## 4. 配置 zone 并发布

1. 用正式字段设置 zone 类型、端口、资源和副本。平台根据角色及端口配置提供 `ROLE_TYPE`、`START_PORT`，以 Carbon 启动计划为准。P/D/V 的 JVM 堆用 `JAVA_TOOL_OPTIONS` 设置。
2. 在 `zone_process_setting.process_info.envs` 写业务参数。P/D 配 `MAX_SEQ_LEN`、`CONCURRENCY_LIMIT` 及下文的 CP、容量、性能参数。多模态 P/D 的 `IMAGE_TOKEN_ID` 取目标模型的图像 token ID。V 配 `PATCH_SIZE`、`DOWNSAMPLE_RATIO`、`HIDDEN_SIZE`、`MIN_PIXELS`、`MAX_IMAGE_TOKENS`，取实际模型的媒体配置。
3. 在 Whale 的部署页保存目标 zone 配置，检查启动命令、镜像、资源覆盖和 ENV 的差异。修改 zone 配置后提交当前部署。更换服务配置版本时，在发布页面选中对应版本和目标部署。
4. 回读 Carbon 启动计划、镜像、健康探针和 VIP 注册端口。确认各角色就绪、`engine_started` 日志中的 `effective_config` 与预期一致，再从 Frontend 发一条真实请求。滚动更新后观察成功率、TTFT 和 Prefill 复用率回稳。

## CP 与 block size

| 位置 | 配置 | 生效规则 |
| --- | --- | --- |
| Mock Prefill | `PREFILL_CP_SIZE` | 默认 1。设为 4 时，缓存键每 4 个取最后一个，未命中 token 的调度预算按 8 对齐。 |
| Mock Prefill | `CP_FORCE_SINGLE_PREFILL` | 仅 CP 大于 1 时生效，默认 true。允许同一轮接纳多个 Prefill 时显式设为 `0`。 |
| Mock P/D | `SEQ_SIZE_PER_BLOCK` | 基础哈希块的 token 数，缺省为 64。按真实 Frontend 的基础块大小显式设置。 |
| 真实 Frontend | `SEQ_SIZE_PER_BLOCK`、`PREFILL_CP_SIZE`、`PREFILL_CP_KV_CACHE_SHARDED` | 与 Mock P 使用相同基础块和 CP size。CP4 时显式启用分片 `PREFILL_CP_KV_CACHE_SHARDED=1`，发给 Master 的键为 K4、K8… |

例如基础块为 128、Prefill CP4，P/D 的 `SEQ_SIZE_PER_BLOCK` 都填 `128`。Mock P 的分配和路由逻辑块为 `128 × 4 = 512` token，D 固定 CP1，每块 128 token。P/D 的 `CACHE_KEY_HASH_SEED` 与真实 Frontend 生效的模型哈希 seed 对齐。

**TP 自动计算。** Mock 在 WorkerStatus 中让 P 的 TP 等于 `PREFILL_CP_SIZE`，D 的 TP 为 1，DP 固定为 1。Whale 资源候选的 TP/DP 仍按单 JVM 的实际规格选。Mock P/D 不读取 `TP_SIZE`、`DP_SIZE`、`CP_ROTATE_METHOD`、`PREFILL_CP_KV_CACHE_SHARDED`，这些键无需写入 Mock zone ENV。

## KV 容量

P/D 按每实例的**逻辑 block**配置。先用该角色的每块 token 数换算目标 token 容量，再填块数。CP 只改变每块代表的 token 数，填写的块数保持原值。

| 参数 | 单位与含义 |
| --- | --- |
| `TEST_BLOCK_NUM` | 必填，至少 2。Device block 数，含一个 null block。可用 Device 容量为 `TEST_BLOCK_NUM - 1`。 |
| `MOCK_CACHE_TOTAL_BLOCKS` | 统一缓存能保留的 block 数，不含 null block。默认等于可用 Device 容量，至少为该值。 |
| `RESERVE_BLOCK_RATIO` | 首次分配预留的 Device 百分比，默认 5。按可用 Device block 数向下取整。 |
| `REUSE_CACHE` | 前缀复用开关，默认 false。需要模拟缓存命中时对目标角色显式开启。 |

例如 P 为 CP4、基础块为 128，配置 `TEST_BLOCK_NUM=1025`、`MOCK_CACHE_TOTAL_BLOCKS=4096`。Device 可用 1024 块，即 524,288 token，统一缓存最多保留 4096 块，即 2,097,152 token。缓存命中也要取得 Device 驻留额度才可进入计算。

## 性能模拟

每个 P/D/V zone 必填 `MOCK_COMPUTE_FORWARD_EXPRESSION`。结果为**每个 Forward batch 的毫秒数**。按目标模型、负载和角色校准。下列公式仅展示写法。

| 角色 | Forward 公式示例 | 同时配置 |
| --- | --- | --- |
| Prefill | `110 + 0.0032 * sum(computeTokens)` | `computeTokens` 是实际重算 token。用 `CONCURRENCY_LIMIT` 控制每轮请求数，按需要设置 `MAX_BATCH_TOKENS_SIZE` 和 `MAX_BATCH_TOKENS_WITHOUT_CACHE`。 |
| Decode | `15 + 0.1 * batchSize` | 用 `CONCURRENCY_LIMIT` 控制每轮请求数。每轮耗时与每轮产出量共同决定 TPOT。 |
| ViT | `43` 或 `2 + 0.00001 * totalImagePixels` | 用 `VIT_GPU_MAX_BATCH_SIZE`、`VIT_GPU_BATCH_WAIT_MS` 控制合批。用 `VIT_CONCURRENCY`、`VIT_MAX_QUEUE_SIZE` 控制准备并发和等待额度。公式只模拟 Forward，图片下载与排队另计。 |

- **投机解码。** 只在 Decode 配 `MOCK_COMPUTE_DECODE_TOKENS_PER_STEP`，默认 1。设为 `2.6` 时，每请求每轮产出 2 或 3 个 token，长期均值为 2.6。Mock 用这个值模拟接受后的产出量。
- **EOS。** P/D 可配 `MOCK_COMPUTE_EOS_PROBABILITY`，默认 0，按 token 判断。概率在 0 和 1 之间时必填整数 `MOCK_COMPUTE_EOS_SEED`。请求的 `min_new_tokens` 和 `ignore_eos` 控制 EOS 是否生效，`max_new_tokens` 和剩余上下文限制最终长度。

完整 ENV、默认值和公式变量见 [组件配置参考](component-reference.md#engine-配置规则)。
