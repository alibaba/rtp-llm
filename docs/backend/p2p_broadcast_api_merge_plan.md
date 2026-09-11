# 合并 broadcast / broadcastPerRank

## 现状

`P2PBroadcastClient` 两个广播入口实际只差"是否逐 rank 下发传输计划"：

| | `broadcast` (.cc:29) | `broadcastPerRank` (.cc:58) |
|---|---|---|
| KV 负载 | 单份，N 个 worker 复制 | `RankLayerCacheBuffers`，须等长 workerNum |
| 传输计划 | 无，routes 恒空、digest 恒 0 | `rank_routes` + `plan_digest` |
| 形状校验 | 无 | 不等长返回 nullptr |

下游 `broadcastRequests`（deadline 校验、gRPC 超时、Result 构造）完全共用。调用方仅三处：prefill `no_transfer` 用前者（SchedulerPrefill.cc:111），prefill 传输与 decode 用后者（:118 / SchedulerDecode.cc:423）。

## 关键发现：顶层 layer_blocks 已是死负载

两个 `broadcastPerRank` 调用点传的都是等长**空壳**（:110、:289 注释已言明）。全仓只有 .cc:164 写顶层 `layer_blocks`，无任何读方。worker 侧真正消费：prefill 用 `peer_workers`+`routes`+`plan_digest`，块视图取自身投影；decode 的块视图来自 route 内嵌 `layer_blocks`，且 `routes_size()==0` 就是权威的"本 worker 无任务"。

所以 "per-rank layer_blocks" 维度已废弃，合并只是删冗余。

## 方案：单一 `broadcast(BroadcastParams)`

```cpp
struct BroadcastParams {
    int64_t   request_id;
    std::string unique_key;
    int64_t   deadline_ms;          // 物理传输 deadline，兼作 gRPC deadline
    int64_t   request_deadline_ms;  // 原始请求 deadline
    P2PConnectorBroadcastType type;
    std::vector<std::pair<std::string, uint32_t>> peer_workers;
    RankRoutes routes;      // 空=统一广播；size==workerNum=逐 rank 下发
    uint64_t   plan_digest{0};
};
std::shared_ptr<Result> broadcast(const BroadcastParams&);  // 删除 broadcastPerRank
```

一条规则：`routes.empty()` → N 份相同请求（不带 plan）；否则要求 `routes.size()==workerNum()` 逐 rank 取。不引入新语义。9 个位置参数收敛成 struct，也顺带消掉调用点那个难读的三元表达式。

同时删除 `RankLayerCacheBuffers`/`LayerCacheBuffers` 别名与 `layer_blocks` 写路径，解掉对 `LayerCacheBuffer.h` 的依赖。

## 改动清单

1. `P2PBroadcastClient.h/.cc`：加 `BroadcastParams`，合并实现，`genBroadcastRequest` 改收 params + routes 指针。
2. `P2PConnectorSchedulerPrefill.cc:108-126`：塌缩为一次构造一次调用，`no_transfer` 只体现为 type 不同、routes 留空。
3. `P2PConnectorSchedulerDecode.h/.cc`：`startAsyncReadCalls` 去掉 buffer 形参及 lambda 捕获。
4. `test/P2PBroadcastClientTest.cc`：两个断言顶层 `layer_blocks` 往返的用例改为断言 routes/plan_digest 逐 rank 下发；`RejectsMismatchedWorkerCount` 改为 route 数不匹配；其余 6 处位置参数改 params。
5. `docs/backend/*.md` 里 4 处 `broadcastPerRank` 提法**不改**：它们是 route 化改造当时的设计记录
   （如 §3.1 的「`broadcast` → `broadcastPerRank`」描述的是那次改动），不是当前 API 文档。

## 风险

**wire 兼容**：停写 proto field 1 对本仓零影响，但 PD 分离跨进程、可独立发布，灰度期旧 worker 若仍依赖顶层 `layer_blocks` 会读到空。故分两步：本次只合并接口 + 停写该字段（保留 proto 定义），全量升级后再 `reserved 1`。

**失败模式**：无新增。prefill `no_transfer` 走统一实现但 routes 为空不触发校验；deadline 相关的 nullptr 判定本就共用。

## 语法校验记录（本机，不编译）

- `clang-format` 能完整 tokenize 这 6 个文件，无诊断。
- `git-clang-format --style=file`（v19，仓库 pin 版本，仅作用于本次改动行）已跑并落地；再跑报
  「did not modify any files」。
- 注意：本机 v19/v23 对**未改动的 HEAD 原文**也给出 58 行重排建议，且 v19 的
  `--output-replacements-xml` 与自身 stdout 互相矛盾，故未用全文件 `clang-format -i`，
  以免污染无关代码。

验证（待有 GPU 环境时跑）：`p2p_connector_test`、`components_test`。
