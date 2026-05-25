# Dispatcher Stage 2 实现总结（2026-05-25）

**日期**：2026-05-25
**分支**：`feature/master-batch-schedule`
**HEAD**：`0a4db15fb`
**提交范围**：`11908110f..0a4db15fb`（V1–V12 + R1–R2，共 14 个 commit）
**对应计划**：`docs/superpowers/plans/2026-05-20-flexlb-dispatcher.md`（Stage 2 节；v1 Tasks 14–17 已在 V12 标 SUPERSEDED）
**源真理文档**：`docs/dispatcher.md`（架构与决策的常驻文档；此文是这一轮实现的"做了什么"）

---

## TL;DR

1. **从"一个 endpoint 写死"提取到"endpoint registry + SPI"**：新增 `BatchEndpointSpec`（路径 / 请求字段 / 响应字段 / 失败构造器 / 后合并器），`/batch_infer`、`/v1/chat/completions/batch`、`/v1/embeddings` 现在共用同一个 `GenericBatchHandler`。新接入第 4 个批量端点的成本 = 一条 `BatchEndpointSpec`。
2. **部分失败合约统一**：`PartialFailureMerger` 用第一个成功子批的 envelope 作为模板替换响应数组；失败子批由 `FailedItemFactory` 逐条占位（NULL / OPENAI_ERROR / EMBEDDING_NULL）；任何部分失败追加 `_partial_failure: {failed_count, total_count, failed_indices}` 元数据。embedding 端点叠加 `EmbeddingPostMerger`，重排绝对 index 并累加 `usage`。
3. **修复了一个隐藏的生产 bug**（V10 E2E 暴露）：`WebClientPassthroughClient` 用 `.exchangeToMono()` 会在 lambda Mono complete 时立即归还 FE 连接，但 `ServerResponse.body(BodyInserter)` 是同步完成、body Flux 直到 `writeTo` 才被订阅，于是 passthrough 响应体被截断/挂起。改用 deprecated `.exchange()`（Spring Cloud Gateway 模式）延迟释放，直到 body 被消费。
4. **超时策略一分为二**：批量 fanout 用 `responseTimeout`（hard ceiling）保护 fail-fast，passthrough 只用 `CONNECT_TIMEOUT`、不装 `ReadTimeoutHandler`，避免把 SSE/长响应在中途 silence 时误杀。两个 `HttpClient` 实例共享同一个 `ConnectionProvider("dispatcher-fe")`，连接池统一管理。
5. **测试到生产相位（V10 `DispatcherE2ETest`）**：绑真实 Reactor Netty server（不是 `bindToRouterFunction`，那个搬不动 lazy `Publisher<DataBuffer>` body），4 个 case 覆盖 split→fanout→merge 全链路，包含人为注入的中间 chunk 失败。
6. **合并后做了 3 处 review 修复**：empty batch 现在返回 shaped `{response_batch: []}` 而非 `{}`（之前是悄无声息的 wire 违约）；老 `BatchSplitter.split(List<String>, int)` dead code 删了；3 个新超时字段补了 parse 测试。

---

## 1. 实现范围

Stage 2 把 v1 留下的"硬编码 `/batch_infer`、字段名散在各处、partial-failure 只在 GenericBatchHandler 里 inline"这套结构改成 spec-driven：

| 维度 | v1（已存在） | Stage 2（这一轮） |
|------|--------------|-------------------|
| 端点定义 | 路径 `/batch_infer`、字段名都散在 handler 里 | `BatchEndpointSpec` record + `BatchEndpointRegistry` bean |
| 失败占位 | inline `null` | `FailedItemFactory` SPI（3 种实现） |
| 后合并 | 无 | `PostMerger` SPI（embedding usage 累加） |
| 端点数量 | 1 (`/batch_infer`) | 3 (`/batch_infer` + OpenAI batch chat + embeddings) |
| 失败元数据 | 无 | 统一 `_partial_failure: {failed_count, total_count, failed_indices}` |
| 超时策略 | 整个 HttpClient 一个 `responseTimeout` | fanout 与 passthrough 分两个 HttpClient，共享 ConnectionProvider |
| 端到端测试 | 单元测试 + mock | 真 Netty server bind + induced 失败的 4 case E2E |

不在范围内（计划里被 SUPERSEDED 或推迟）：

- v1 plan Tasks 14（路由 pre-assign）、15（slot 拓展）、15b、16、17（流式 v1）。Stage 2 §"What is NOT changing" 明确推迟。
- 16b（perf 调优）保留为未来工作。
- 流式 `/batch_infer` —— 仍走 passthrough，dispatcher 不参与子批拆分。

---

## 2. 与 `docs/dispatcher.md` 的关系

`docs/dispatcher.md` 是 dispatcher 的常驻设计文档，描述拓扑、决策、配置项、端到端协议等不易过时的内容。本文是"这一轮 commit 做了什么、文件落在哪、为什么这么改"的实现快照——读完应该能复现 14 个 commit 的因果链，但日常运行/排障应该回到 `docs/dispatcher.md`。

---

## 3. 文件清单

新增（`rtp_llm/flexlb/flexlb-api/src/main/java/org/flexlb/dispatcher/`）：

```
BatchEndpointSpec.java           # 端点 spec：path + reqField + respField + failedFactory + postMerger
FailedItemFactory.java           # SPI：NULL / OPENAI_ERROR / EMBEDDING_NULL
PostMerger.java                  # SPI：embedding renumber + usage sum
EmbeddingPostMerger.java         # PostMerger 实现
BatchEndpointRegistry.java       # Spring config：3 个 spec bean + duplicate-path 保护
PartialFailureMerger.java        # 通用合并器（envelope 模板 + 失败占位 + _partial_failure）
MergedResponse.java              # record：body + counts + failedIndices
SubBatchResult.java              # record：单 chunk 的 ok/failed
GenericBatchHandler.java         # spec-driven handler（替代 v1 BatchInferHandler）
```

新增测试：

```
flexlb-api/src/test/java/org/flexlb/dispatcher/
├── BatchEndpointSpecTest.java        # SPI 等价/边界
├── FailedItemFactoryTest.java        # 3 种失败项 shape
├── PartialFailureMergerTest.java     # envelope 选择 + 失败占位 + 元数据
├── EmbeddingPostMergerTest.java      # index 重排 + usage 求和
├── BatchEndpointRegistryTest.java    # 3 个 bean + duplicate path 抛错
├── GenericBatchHandlerTest.java      # 单 chunk / 多 chunk / 全失败 / 错误体 / 空 batch
├── FanoutServiceTest.java            # 顺序保持 / 单点失败软化 / 空 pool
├── DispatchRouterTest.java           # 前缀路由 + 非 dispatcher 路径不匹配
├── StreamingPassthroughTest.java     # production-mirror 配置 + 反例（带 responseTimeout 会切流）
└── DispatcherE2ETest.java            # 真 Netty server：4 个端点全链路 + induced 部分失败
```

修改（关键改动）：

- `DispatchRouter.java`：从写死 `/batch_infer` 改成"按 spec 列表注册"。
- `DispatcherConfiguration.java`：split 出两个 `HttpClient`（fanout 用 `responseTimeout`，passthrough 不用），共享 ConnectionProvider；连线 `BatchEndpointRegistry` → `GenericBatchHandler` → `DispatchRouter`。
- `WebClientFeClient.java`：构造签名 `(WebClient.Builder, int maxResponseBytes)`，去掉 per-Mono `.timeout()`，时序统一由 HttpClient 层负责。
- `WebClientPassthroughClient.java`：`.exchange()`（V10 修复）；body Flux 上保留 `.timeout(maxStreamDurationMs)` 防止彻底卡死。
- `DispatchConfig.java`：新增 `feConnectTimeoutMs=2000`、`feResponseTimeoutMs=5000`、`feMaxStreamDurationMs=600_000`；保留 `feRequestTimeoutMs=3000`（只喂 `ConnectionProvider.pendingAcquireTimeout`）。
- `BatchSplitter.java`：留 `splitArray`，删 dead `split(List<String>, int)`（R2）。
- `rtp_llm/flexlb/CLAUDE.md`：加 `### DISPATCH_CONFIG` 子节、端点注册表、partial-failure 合约。
- `rtp_llm/flexlb/README.md`：+2 行 Features/Configuration 指针。

不动（Stage 2 严格守约）：

- `org.flexlb.httpserver.*`（Master 代码）。
- `stub_source` 软链。
- 任何 `rtp_llm/flexlb/` 之外的文件。

---

## 4. 关键设计决策

### 4.1 为什么是 spec record + registry，而不是抽象基类 + 子类

每个批量端点的差异只在 4 个数据（路径、请求字段、响应字段、失败 shape、可选 post-merge）。继承会引入"虚函数表 + 类层级"，调试时跳来跳去；record + lambda 把差异收进**值**，handler 是**算法**——遵循"data + algorithm > inheritance"。

### 4.2 为什么 partial-failure 元数据键的顺序固定

`_partial_failure` 的字段顺序固定为 `{failed_count, total_count, failed_indices}`，**因为 Jackson `ObjectNode` 用 LinkedHashMap 保插入顺序**，下游做字段对比的客户端（哪怕只是日志 diff）会受益。CLAUDE.md 里写死了这个顺序，测试也按这个顺序断言。

### 4.3 为什么 passthrough 不用 `responseTimeout`

`HttpClient.responseTimeout(d)` 装的是 Netty `ReadTimeoutHandler`，**对中途 silence 触发**——不是只在初次响应延迟时触发。SSE / 流式生成长时间无输出是合法状态，装了就会被误杀。fanout 是请求-响应模型不存在这个问题；passthrough 必须只配 `CONNECT_TIMEOUT` + body-level `Flux.timeout(maxStreamDurationMs)`（一个保底上限）。

`StreamingPassthroughTest.addingResponseTimeoutWouldKillTheStream` 是个反例 case，重新引入 `responseTimeout(2s)` 后断言 `ReadTimeoutException`——锁死这个分歧不被无意"修复"。

### 4.4 为什么 `.exchange()`（deprecated）而不是 `.exchangeToMono()`

`.exchangeToMono(lambda)` 在 lambda 返回的 Mono complete 时立刻释放上游连接。`ServerResponse.body(BodyInserters.fromDataBuffers(flux))` **是同步完成的**——它只是把 BodyInserter 装进 ServerResponse 对象。body Flux 的实际订阅发生在 `ServerResponse.writeTo` 阶段，那时上游连接已经被收掉了，body 出来全是空 / 挂起。

`.exchange()` 是 deprecated（Spring 建议 .retrieve / .exchangeToMono）但**延迟释放语义是唯一对的**：连接在 body Flux 真正被消费完才归还。这正是 Spring Cloud Gateway 内部的做法，已经验证过若干年。

### 4.5 为什么 empty batch 短路在 handler 而非 merger

`{prompt_batch: []}` 走完 splitArray→fanout→merger 是合法的（empty list → empty subs → null envelope），但 merger 在没有 envelope 模板时返回 `{}`。修复有两种位置：

- 修 merger：在 `subs.isEmpty()` 时返回 `{spec.responseArrayField: []}`。
- 修 handler：在 `arr.isEmpty()` 时短路，根本不走 fanout。

选了后者：fanout 是 no-op，不应该被调用；merger 的"envelope 模板来自第一个成功子批"不变性更干净。

---

## 5. 配置（`DISPATCH_CONFIG`，opt-in）

完整字段见 `rtp_llm/flexlb/CLAUDE.md` 的 `### DISPATCH_CONFIG` 一节。Stage 2 新增：

| 字段 | 默认值 | 用途 |
|------|-------|------|
| `feConnectTimeoutMs` | 2000 | TCP connect ceiling（两个 HttpClient 都装） |
| `feResponseTimeoutMs` | 5000 | fanout HttpClient 的 `responseTimeout`；passthrough **不装** |
| `feMaxStreamDurationMs` | 600_000 | passthrough body Flux 的 `.timeout()` 兜底上限 |

保留的旧字段：

- `feRequestTimeoutMs=3000`：只喂 `ConnectionProvider.pendingAcquireTimeout`，**不再控请求时序**。CLAUDE.md 标了 legacy。

---

## 6. 测试覆盖

模块测试统计（`./mvnw -P-internal -pl flexlb-api -am test -DfailIfNoTests=false`）：

- flexlb-api: **74 tests, 0 failures, 0 errors, 0 skipped** （Stage 2 前 75，删了 4 个 dead 测试 + 加了 3 个新测试，净 -1）
- BUILD SUCCESS，total ~50s

测试金字塔覆盖：

- **单元**：每个 SPI 一个测试类，每个 record 的边界条件单测。
- **集成**（同 JVM，mock FE）：`GenericBatchHandlerTest`、`FanoutServiceTest`、`DispatchRouterTest`。
- **E2E**（真 Netty server + MockWebServer 模拟 FE）：`DispatcherE2ETest` 4 case + `StreamingPassthroughTest` 2 case。

E2E case 列表（V10）：

1. `batchInferNineSplitsThreeWithMiddleChunkFailure` — 9 prompt / K=3 / fe2 = 500；assert 响应数组 size==9、[3..5] 为 null、`_partial_failure={failed_count:3,total_count:9,failed_indices:[3,4,5]}`。
2. `openAiBatchChatFourSplitsTwoWithSecondChunkFailure` — 4 请求 / K=2 / fe2 = 500；OPENAI_ERROR shape `{index, error:{code:ERROR_CODE_SUB_BATCH_FAILED, message}}`。
3. `embeddingsSixSplitsThreeWithMiddleChunkFailureRenumbersAndSumsUsage` — 6 input / K=2 / fe2 = 500；index 重排 0..5、`usage.prompt_tokens=8`（4+4 累加）。
4. `nonBatchPathFallsThroughPassthroughVerbatim` — POST `/dispatcher/v1/chat/completions`；fe1 收到 `/v1/chat/completions`（前缀已剥）。

---

## 7. Commit 地图

```
0a4db15fb  chore(dispatcher): drop dead BatchSplitter.split + cover new timeout fields  ← R2+R3 (post-review)
40308733b  fix(dispatcher): empty batch returns shaped envelope, not {}                  ← R1 (post-review)
58f24cc24  chore(dispatcher): supersede v1 Tasks 14-17 in plan                            ← V12
933e3c20e  docs(dispatcher): DISPATCH_CONFIG, endpoint registry, partial-failure contract ← V11
51a921ed4  test(dispatcher): E2E split/fanout/merge for all 4 batch endpoints             ← V10 (新增 E2E + 暴露 V10 prod fix)
933869efa  fix(dispatcher): defer FE connection release until passthrough body consumed  ← V10 prod fix
59c81f0ab  fix(dispatcher): stream-friendly passthrough timeouts                          ← V9 (两 HttpClient + ConnectionProvider 共享)
4451869ac  feat(dispatcher): wire batch routes per BatchEndpointSpec via registry         ← V7
a32538dae  feat(dispatcher): GenericBatchHandler dispatches any batch endpoint via spec   ← V6
7585eb21f  feat(dispatcher): BatchEndpointRegistry for batch_infer + OpenAI batch + emb   ← V5
7669d4f81  feat(dispatcher): EmbeddingPostMerger renumbers indices + sums usage           ← V4
5b5e40d51  feat(dispatcher): PartialFailureMerger with per-spec failure shaping           ← V3
96fe47bfc  feat(dispatcher): BatchSplitter.splitArray for generic JSON-array batches      ← V2
7132334a1  feat(dispatcher): BatchEndpointSpec + FailedItemFactory + PostMerger SPI       ← V1
```

V8 在 review 后被折叠进 V7 amend（router 装配同一改动）。R1/R2/R3 是 review 后的 fix-forward。

---

## 8. 已知的 Nit（推迟）

代码 review 找到、但没在这一轮处理的 Nit（按优先级排）：

1. `BatchSplitter.splitArray:33` 用 `assert chunkSize >= 1`——production JVM 默认关 assertion，等于无防御。换成 `IllegalArgumentException`。
2. `DispatchProtocol.java` 5 个常量里 4 个 unreferenced，剩下的 `ERROR_CODE_SUB_BATCH_FAILED` 可以折进 `FailedItemFactory` 然后删类。
3. `BatchEndpointRegistry.batchSpecsByPath` bean 无消费者，duplicate-path 检查只靠 Spring eager init 触发——要么 inline 进 `batchSpecs()`，要么标 `@SuppressWarnings("unused")` 说明意图。
4. `WebClientPassthroughClient` forward header 时透传了 `Host`——FE 当前不 vhost，但万一以后 vhost 会路由错。Strip `Host`/`Content-Length`。
5. `FanoutService.dispatchChunks` 用 `Flux.mergeSequential` 默认并发 256，3000-prompt batch @ K=5 → 600 chunks 会排队 ~344；要么显式传并发参数要么在 CLAUDE.md 标 batch size 上限。
6. `BatchEndpointSpec` 是 Lombok `@Value`，`SubBatchResult`/`MergedResponse` 是 Java 21 `record`——SPI 风格不一致，挑一个统一。

这些都不影响正确性或性能（在当前使用强度下），留给下一次 dispatcher 改动顺手清理。

---

## 9. 后续工作（不在这一轮里）

- v1 Plan 16b：fanout 的 perf 调优（jackson 序列化、ConnectionProvider tune）。
- 流式 `/batch_infer`：目前走 passthrough；若要在 dispatcher 层做子批拆分 + SSE 合流，需要重新设计 merge 协议。
- pre-assign（v1 Tasks 14/15）：选 worker 然后把分配下推给 dispatcher 子请求，跳一跳 Master → FE → Master 的回程。Stage 2 §"What is NOT changing" 推迟。
- ZGC 选型在 Dispatcher 与 Master 都用同一 JVM 下的 GC 影响（参见 `docs/dispatcher-remeasure-handover-2026-05-19.md` Track A 结论）。

---

## 10. 验证清单

落地后任何人接手只需要 4 步确认：

```bash
# 1. 验证当前在正确 HEAD
git log --oneline -1   # 应是 0a4db15fb

# 2. 跑 flexlb-api -am 测试套
cd rtp_llm/flexlb
export JAVA_HOME=/opt/taobao/install/ajdk21_21.0.6.0.6
export no_proxy="localhost,127.0.0.1,.aliyuncs.com,.alibaba-inc.com"
./mvnw -P-internal -pl flexlb-api -am test -DfailIfNoTests=false
# 期望：BUILD SUCCESS，flexlb-api Tests run: 74, Failures: 0, Errors: 0, Skipped: 0

# 3. 翻 docs/dispatcher.md（架构与决策）

# 4. 翻 rtp_llm/flexlb/CLAUDE.md ### DISPATCH_CONFIG（配置参考）
```
