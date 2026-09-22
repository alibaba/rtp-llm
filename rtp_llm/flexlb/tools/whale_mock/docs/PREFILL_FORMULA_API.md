# Prefill 执行时间公式热更新

## 这个端点解决什么问题

Mock 控制 HTTP 端口提供 `GET /prefill_formula` 和 `POST /prefill_formula`，用于在不重启引擎的前提下替换 mock prefill 引擎的执行时间估算公式。

它只影响 mock 侧的 prefill 计时，不改 master 的路由估算，也不改 master 自己那份公式（后者在 `FLEXLB_CONFIG` 的 `executionTimeEstimator.expression`）。要调 master 侧公式，走配置，不走这里。

## 调用流程

先 `GET` 读回当前公式并保存，作为回滚基线，再 `POST` 新公式：

```json
{"expression":"200 + sum(computeTokens / 1024.)"}
```

上面的表达式只是 API 示例，不是标定过的生产公式。

作用范围由 `engine` 字段决定：省略 `engine` 应用到当前所有 P 引擎，写 `"engine":"prefill-0"` 定向单个引擎。回滚即用保存的表达式再 `POST` 一次，scale 仍为 1.0。

## 公式语法

表达式由既有的数学公式解析器求值。可用元素如下。

### 请求级变量

在 `sum(expr)` 内使用，对批次内每个请求各求值一次：

| 变量 | 含义 |
|---|---|
| `inputTokens` | 请求输入 token 数 |
| `hitCacheTokens` | 缓存命中 token 数 |
| `computeTokens` | `inputTokens - hitCacheTokens` |
| `hasHitCache` | 有缓存命中为 1，否则 0 |

### 批次级变量

在 `sum(expr)` 外使用，反映整批聚合值：

| 变量 | 含义 |
|---|---|
| `batchSize` | 批次中的请求数 |
| `totalInputTokens` | 批次输入 token 合计 |
| `totalHitCacheTokens` | 批次缓存命中 token 合计 |
| `totalComputeTokens` | `totalInputTokens - totalHitCacheTokens` |
| `maxInputTokens` | 批次中最大输入长度 |
| `maxComputeTokens` | 批次中最大计算长度 |

### 运算符与函数

运算符 `+ - * / ^`（`^` 为幂，右结合）；函数 `sqrt log exp abs max min pow`。

`totalComputeTokens^2` 是批次总量的平方，`sum(computeTokens^2)` 是每请求平方后求和，两者语义不同，不可互换。

## 生效时机与语义边界

- **前置校验**：更新前先解析并校验表达式。语法错误、目标引擎无效返回 400；空 / 超长表达式，以及在校验样本上返回负值或非有限值的公式，一律拒绝。
- **校验不覆盖全域**：校验样本只能证明在样本上合法，不保证对任意 workload 成立，调用方需自行验证自己的取值域。
- **逐引擎原子替换**：每个引擎各自原子替换公式引用；同时更新多个引擎不构成全局屏障，切换期间被重建的请求可能使用新公式。
- **不影响已排期批次**：已经排下去的 batch 延迟不变；其后的求值改用新公式，prefill scale 固定为 1.0。
- **优先级高于固定 ms**：该 override 优先于 fixed-ms 设置。
- **不持久化**：运行时改动重启即失效。要长期生效，需把表达式单独落到部署配置里。`GET` 读回会列出每个引擎当前的表达式与 scale。
- **不清缓存**：本 API 不清除 KV cache。

## 关联 metric

- `rtp_llm_context_batch_size`：每个启动的 prefill 执行 batch 上报一个样本。空闲轮询不产生 0 样本，要看空闲占用请用 running-stream 指标。
- `rtp_llm_device_reuse_length`：与 `rtp_llm_stream_cache_device_reuse_length` 是同一 device-only token 值的别名，不含 memory reuse。
- `rtp_llm_input_token_length`：沿用既有上报点，不新增重复事件。
