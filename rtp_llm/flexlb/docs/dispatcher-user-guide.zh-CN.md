# Dispatcher 拆批配置与接入手册

本文默认使用 Whale 主干最新模板，Master 和推理服务已配置完成。以下只说明启用拆批需要增加的配置。

Dispatcher 是 Master 上的批量请求入口。客户端把 N 条请求放进一个 body，Dispatcher 按配置拆成多个子批（chunk），并行发给 Frontend（FE），合并后返回。文本生成、chat、embedding 按原序合并；reranker 按请求要求返回全局排名。

## 客户端集成

已有 biz 切换到拆批接口，只需将地址指向 Master 7001，原 URL 路径前加 `/dispatcher`。业务请求字段沿用 FE 接口，客户端不需要设置 `force_batch` 或内部选址参数。

需要额外处理三种情况：成功响应可能带 `_partial_failure`；全部子批失败时返回 Dispatcher 的错误 body；Master 选址不可用时返回 503。

```sh
curl -X POST http://<master>:7001/dispatcher/batch_infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt_batch":["你好","介绍一下杭州"],"generate_config":{"max_new_tokens":16}}'
```

## 拆批配置

以下参数在 Whale 的 **Master → 高级设置** 中配置。

### 1. 打开拆批入口

在 Master 高级设置中增加环境变量：

```text
DISPATCH_ENABLED=true
```

保留 Whale 自动注入的 `FLEXLB_CONFIG`，无需修改或重新填写。这个开关只覆盖其中的 `httpDispatcher.enabled`，其他 Master 配置保持原样。

设为 `false` 关闭拆批入口；不配置时沿用平台配置中的值，代码默认关闭。只接受 `true` 或 `false`，不区分大小写，允许首尾空格；空值或拼写错误会导致启动失败。

### 2. 按部署形态选择 BE 预分配

单 role PDFUSION、FE/BE 同机部署保持默认 `DISPATCH_PRE_ASSIGN_BE=true`，每个子批使用一个同机 FE/BE 目标，无需额外配置。

独立 FE、PD 分离，或希望由 FE 按原流程选择 BE 时，设置：

```text
DISPATCH_PRE_ASSIGN_BE=false
```

适用条件见「BE 预分配」。

### 3. 确定 FE 入口

| 部署形态 | 需要配置 |
|---|---|
| 单 role，worker 的 HTTP 端口就是 FE 入口 | `DISPATCH_FE_POOL_SERVICE_ID` 留空，自动复用已有服务发现 |
| 独立 FE、PD 分离或 FE 入口不唯一 | 填写 `DISPATCH_FE_POOL_SERVICE_ID`，并保持 `DISPATCH_PRE_ASSIGN_BE=false` |

FE 服务名在接收请求的 role 的「服务发现」中查看，例如：

```text
DISPATCH_FE_POOL_SERVICE_ID=com.aicheng.whale.prod.master_dispatcher_test.publish
```

填写的是 FE HTTP 服务名。

### 4. 按需调整粒度和超时

不修改时默认拆成最多 5 个子批。需要每批最多 50 条、子批响应超时 60 秒时，设置：

```text
DISPATCH_SUB_BATCH=size:50
DISPATCH_BATCH_TIMEOUT_MS=60000
```

FE 健康接口默认 `/frontend_health`；使用其他探针路径时配置 `DISPATCH_PROBE_PATH`。

### 5. 发布并验证

保存配置并在 Whale 发布生效后，调用 `_dryrun` 检查拆批，再发送真实推理请求。调用示例见「客户端集成」和「诊断端点」。

**最小配置**：单 role PDFUSION、FE/BE 同机场景，在 Master 高级设置中增加 `DISPATCH_ENABLED=true`，默认开启 BE 预分配。独立 FE 或 PD 分离还需设置 `DISPATCH_PRE_ASSIGN_BE=false` 和 FE 服务发现名。

## 配置参考

### Dispatcher 配置

| 配置 | 默认值 | 说明 |
|---|---|---|
| `DISPATCH_ENABLED` | 未设置：沿用平台配置，代码默认关闭 | 拆批启用开关；只覆盖 `httpDispatcher.enabled`，不替换整份配置 |
| `DISPATCH_FE_POOL_SERVICE_ID` | 空 | 默认复用唯一非 VIT worker role 的 FE HTTP 入口；显式填写时必须关闭预分配 |
| `DISPATCH_SUB_BATCH` | `count:5` | 拆批策略，见下表 |
| `DISPATCH_BATCH_TIMEOUT_MS` | `30000` | 子批响应读取空闲超时；非流式请求要覆盖生成期间不返回数据的时间 |
| `DISPATCH_PROBE_PATH` | `/frontend_health` | FE 健康探针路径；vLLM 通常使用 `/health` |
| `DISPATCH_PRE_ASSIGN_BE` | `true` | 为适用的文本生成子批预分配同机 FE/BE 目标 |

以上参数都可在 Master 高级设置中用环境变量配置。

### 拆批策略

| 写法 | 含义 | 例：500 条输入 |
|---|---|---|
| `count:N` | 最多 N 个非空子批，条目尽量平分 | `count:5` → 5 个子批，各 100 条 |
| `size:N` | 每个子批最多 N 条 | `size:50` → 10 个子批 |
| 裸数字 `N` | 等价于 `size:N` | `100` → 5 个子批 |

默认 `count:5`。输入不足 5 条时不生成空批。N 必须为正整数，非法写法会导致启动失败。

### 超时与大小限制

| 项目 | 限制 |
|---|---|
| 已注册批量端点的输入 body | 默认 5 MB，通过 `spring.codec.max-in-memory-size` 调整 |
| 单个成功 FE 子批响应 | 16 MiB |
| 聚合请求 / 聚合响应 | 各 128 MiB；请求预算包含每个 chunk 重复的公共字段 |
| 单请求并行子批数 | 最多 64 个，其余排队 |
| 单个 FE 子调用整体期限 | `DISPATCH_BATCH_TIMEOUT_MS + 30000` 毫秒 |
| FE 服务发现返回空列表 | 保留最近非空池最多 5 分钟，期间继续探活 |

除输入 body 上限和子批响应超时外，上述值为固定内部限制。子批排队会增加整批耗时，客户端总超时应相应设置。流式透传不合并响应，其响应流空闲超时为 10 分钟。

## 访问接口

业务端口默认 7001，管理端口默认 7002。

| 接口 | 行为 |
|---|---|
| `POST /dispatcher`、`POST /dispatcher/` | 文本生成批量入口 |
| `POST /dispatcher/<注册的批量 path>` | 可拆分请求执行拆批、fanout、合并 |
| `/dispatcher/<其它业务 path>` | 保留 HTTP 方法，剥掉前缀后整包转给一台 FE |
| `POST /dispatcher/_dryrun/<注册的批量 path>` | 只预览拆批，不选址、不推理 |

新版不提供 `/dispatcher/_snapshot`。`POST /dispatcher/_dryrun` 对应根路径的文本生成批量预览。

## 请求

请求体是 JSON 对象，各端点的数组字段如下：

| 路径，拼在 `/dispatcher` 后 | 请求数组 | 响应数组 | 失败占位 |
|---|---|---|---|
| `/`、`/batch_infer` | `prompt_batch` | `response_batch` | `null` |
| `/v1/batch/chat/completions` | `requests` | `responses` | `{"index":N,"error":{"code":"dispatcher_sub_batch_failed","message":"..."}}` |
| `/v1/embeddings` | `input` | `data` | `{"index":N,"embedding":null,"error":"..."}` |
| `/v1/reranker` | `documents` | `results` | 不返回部分排名；任一子批失败则整体失败 |

文本生成示例：

```json
{
  "prompt_batch": ["你好", "介绍一下杭州"],
  "generate_config": {"max_new_tokens": 16}
}
```

其他普通字段随 chunk 复制。预分配会写入 `generate_config.role_addrs`；reranker 会先向各子批请求完整分数，再统一处理 `sorted` 和 `top_k`。

约束与特殊处理：

- 客户端不要提供内部字段 `role_addrs`；在受校验的批量请求中提供该字段会返回 400。
- 空的可拆分数组直接返回空结果，不选址、不推进轮询游标。
- 无效 JSON、非对象 body、非法生成配置 → 400；输入或拆批预算超限 → 413。
- 流式、多模态、adapter 等不适合拆批的 raw 请求整包透传；标量或非文本 embedding 输入也可能透传，最终由 FE 校验。因此数组缺失或不是数组，并非一律由 Dispatcher 返回 400。
- `/batch_infer` 只支持非流式推理。可拆分文本批次从 `/` 或 `/batch_infer` 进入，都会发送到 FE `/batch_infer`。
- Embeddings 合并时将 index 改为全局索引，并累加成功子批的 `usage.prompt_tokens`、`usage.total_tokens`；reranker 累加 `total_tokens`。

## 响应

Raw、chat、embedding 至少一个子批成功就返回 200；全部失败返回 500。Reranker 任一子批失败即整体失败。

**全部成功 — HTTP 200**

结果按原序合并，不带 `_partial_failure`：

```json
{
  "response_batch": [
    {"response":"你好，有什么可以帮你？","finished":true,"aux_info":{}},
    {"response":"杭州是浙江省省会。","finished":true,"aux_info":{}}
  ]
}
```

**部分失败 — HTTP 200**

失败项在原位置填占位符，顶层增加 `_partial_failure`：

```json
{
  "response_batch": [
    {"response":"你好，有什么可以帮你？","finished":true,"aux_info":{}},
    null
  ],
  "_partial_failure": {
    "failed_count": 1,
    "total_count": 2,
    "failed_indices": [1]
  }
}
```

`failed_count` 按输入条数计数，`failed_indices` 是从 0 开始的全局索引。一个子批失败会影响其中所有条目。HTTP 仍为 200，`raise_for_status()` 不会抛错，客户端必须主动检查标记。

**全部失败 — HTTP 500**

```json
{
  "error": "all_sub_batches_failed",
  "failed_count": 500,
  "total_count": 500,
  "total_chunks": 5,
  "failed_reasons": ["fe_unavailable", "fe_server_error"]
}
```

`failed_reasons` 是去重后的原因列表：

| 原因 | 含义 |
|---|---|
| `fe_client_error` | FE 返回 4xx |
| `fe_server_error` | FE 返回 5xx |
| `fe_unavailable` | 子批连接、读取失败或超时等 |
| `malformed_sub_batch` | FE 返回 2xx，但 JSON、响应数组或条数不符合协议 |

特例：**所有子批**均返回同一个 4xx 时，保留该状态码；混有连接失败时不适用。

错误码汇总：

| 场景 | HTTP |
|---|---|
| 请求格式或选址参数不合法 | 400 |
| 输入、单 FE 成功响应、聚合预算或子批数量超限 | 413 |
| Raw/chat/embedding 部分失败 | 200，带 `_partial_failure` |
| Reranker 部分失败 | 500，`error:sub_batch_failed` |
| 全部子批失败 | 500，或所有子批一致的 4xx |
| Master 选址不可用、超时或转发失败 | 503，`error:batch_schedule_failed` |
| 整包透传连接失败，且尚未提交响应 | 502 |
| Pre-stop 后到达的新推理请求 | 503，body 可以为空 |
| 其他内部处理异常 | 500 |

选址错误发生在 fanout 前，不发送 FE 推理。响应过大引发的 413 可能发生在部分 FE 已执行之后。Dispatcher 不自动重放失败子批。

## 工作流程

Dispatcher 与 Master 共用 JVM 和 7001 端口，分别使用 `/dispatcher/**` 和 `/rtp_llm/**` 路径。

1. **拆批**：读取请求数组，按 `DISPATCH_SUB_BATCH` 生成 chunk。
2. **选址与 fanout**：向 Master 批量选址一次，再并行发送给目标 FE。预分配生效时选同机 FE/BE worker；否则只选 FE，由 FE 按原流程选 BE。
3. **合并**：收集子批响应，按原序合并或生成全局排名，返回结果及失败信息。

单条请求调度仍走 FE 或 `/rtp_llm/schedule`。流式请求整包转给一台 FE，不做 SSE 合并。

## 诊断端点

`POST /dispatcher/_dryrun/<path>` 只运行请求校验和拆批，返回 chunk 内容：

```sh
curl -X POST http://<master>:7001/dispatcher/_dryrun/batch_infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt_batch":["a","b","c"]}' | jq
```

默认 `count:5`：

```json
{
  "mode": "split",
  "chunk_count": 3,
  "chunks": [
    {"prompt_batch":["a"]},
    {"prompt_batch":["b"]},
    {"prompt_batch":["c"]}
  ]
}
```

不调用 Master 选址或 FE，不推进游标，不写 `role_addrs`；加 `?pre_assign=true` 也不会选址。空批返回零个 chunk；透传请求返回 `mode:passthrough` 和一个原始 JSON chunk。未知预览路径或不支持的方法返回 400。

`_dryrun` 成功只代表拆批规则通过，不代表服务发现、FE 健康或推理正常。

## BE 预分配

这是拆批的可选功能。单 role PDFUSION、FE/BE 同机部署需要开启时：

1. 在 Master 高级设置中，将 `DISPATCH_PRE_ASSIGN_BE` 设为 `true`，`DISPATCH_FE_POOL_SERVICE_ID` 留空。
2. 发布配置后，用真实批量请求验证。

`DISPATCH_PRE_ASSIGN_BE=true` 时，对适用的 `/`、`/batch_infer` 子批，每个 chunk 轮询选一个 FE/PDFUSION BE 同机 worker：HTTP 请求发给该 worker 的 FE，BE 地址写入 `generate_config.role_addrs`，FE 直接使用该地址。

例如 500 条输入拆成 5 批，分配的是 5 个目标，每批 100 条使用同一目标。目标不足 5 台时轮询复用。

前提与约束：

- 仅适用于单阶段 PDFUSION 同机部署，不支持独立 FE 或 PD 分离预分配。
- 显式填写 FE 池服务名与开启预分配不能同时使用，否则启动失败。
- 不做容量准入、任务预留或记账。需要这些能力时关闭预分配，让 FE 走普通 Master 调度。
- 分组路由或 `EMBEDDING` 引擎自动跳过 BE 预分配；其他端点只分配 FE。
- 选址超时为 3 秒；失败返回错误，不再自动降级为 FE 自选 BE。

关闭方式：在 Master 高级设置中设 `DISPATCH_PRE_ASSIGN_BE=false`。拆批、FE 分发和合并仍然生效。PD 分离时还需填写 FE HTTP 服务名；不同 chunk 不保证落在不同的 P/D worker。

## 排障

| 现象 | 排查 |
|---|---|
| `/dispatcher/**` 未启用或 404 | 检查 `DISPATCH_ENABLED=true` 是否已发布生效；只填 FE 服务名不会启用 |
| 开启后配置未生效 | 检查 Master role 的实际环境变量和 Whale 发布状态 |
| 配 FE 服务名后启动失败 | 显式 FE 池必须关闭预分配 |
| 503 `batch_schedule_failed` | 核对 Leader、FE 池及 BE 可用性；此阶段还未发送推理 |
| 400 `batch_schedule_failed` | 核对 role、分组路由及参数；PD 分离关闭预分配 |
| 全失败，原因含 `fe_unavailable` | 核对 FE HTTP 地址、连通性及子批超时 |
| FE 池为空 | 查看 `app.dispatcher.fepool.size` 和发现日志，核对服务名 |
| FE 探针失败 | 查看 `app.dispatcher.fepool.alive`；从 Master 访问 `<fe_url><probe_path>` 复现 |
| FE 拒绝预分配请求 | 核对 FE 版本、PDFUSION 部署形态和预分配地址 |
| 非流式大批频繁超时 | 调整子批大小和 `DISPATCH_BATCH_TIMEOUT_MS`，检查客户端总超时 |
| 返回 413 | 区分输入、单 FE 响应、聚合预算和子批数量限制 |
| `_dryrun` 成功但推理失败 | 继续检查选址、FE 健康和真实推理日志 |

请求指标看 `app.dispatcher.all.qps`、`app.dispatcher.all.rt`；子批指标看 `app.dispatcher.chunk.detail.qps`、`app.dispatcher.chunk.rt`。FE 池非空但全部探针失败时，仍可能尝试向已发现实例发送请求，日志为 `FE pool all-dead fallback`。

Pre-stop 后拒绝新推理和批量选址请求，等待已接收请求完成；最终强制退出时限由平台控制。Dispatcher 和 Master 共用 JVM 与堆，普通请求异常通常返回 HTTP 错误，进程退出或严重内存故障会同时影响两者。
