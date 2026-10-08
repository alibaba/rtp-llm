# PD 反转：FlexLB 缺口记录

2026-10-08 静态核对；Cancel 修复尚未提交和运行验证。代码基准 `595845f426`，后续修复尚未提交，旧路径基准 `397a27cfac`。未编译、未运行测试；本文记录缺口及修复状态。

| 能力 | 当前情况 | 旧路径对照 |
|---|---|---|
| 引擎 Cancel | 已补 D.Cancel→P.Cancel；FlexLB 向选定 D 取消，携带 P 地址覆盖连接前竞态 | 旧 PrefillRpcServer::Cancel 有实现 |
| 优先级抢占闭环 | 已迁移到 D 终态确认；P 本地终态不再结算 D 预留，待远端验证 | 旧 DeferredPrefillContextMap 会取消下游并发布进度 |
| 取消原因 | 已传递真实错误码及消息；普通取消、超时、故障保留原始来源，只有抢占记 8429 | 普通取消使用 RPC 清理链，专用 Cancel 用于抢占 |
| 取消先于入队 | 已补 TOMBSTONED 和 10 分钟保留记录，晚到入队返回 8429 | 旧路径保存取消记录，拒绝竞态晚到的入队 |
| 入队后无人连接的期限 | 已补独立连接期限：从入队成功到首次 StartLoad；默认 10 分钟，并受整体请求期限约束，待远端验证 | 旧 publishSlot 按该字段设置 TTL，并受整体期限约束 |
| Master 入队后的跨 P/D 取消 | Master 取消与 Frontend 断开已接入 D→P 清理链，等待两端清理确认；待远端验证 | 旧路径有 DeferredPrefillContext 与 P→D 取消链 |
| EnqueueGroup / FetchResponse | 新服务返回 UNIMPLEMENTED；结果改从 Decode.GenerateStreamCall 获取 | 旧服务支持，需要迁移旧工具 |
| 多 DP EnqueueBatch | 新实现要求 dp_size=1 | 旧 EnqueueBatch 同样限制单 DP，不能算新增退化；旧 EnqueueGroup 可按本地 rank 接收 |

基本 GetWorkerStatus 未缺失：运行/完成列表、阶段、batch_id、版本、KV 容量都有序列化；缺的是取消与抢占的控制闭环。

定位（路径均相对仓库根目录）：
- `rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h`、`RemoteRpcServiceImpl.h`：接口分派。
- `rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.cc:266`：EnqueueBatch；`:104`：完成 context 回收。
- `rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.cc:292`：Master 入队不创建 Prefill caller。
- `rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h:98`：抢占状态接口。
- 旧基准 `PrefillBatchRpcServer.cc:1437`：连接 TTL；`PrefillRpcServer.cc:941`：Cancel。

Cancel、取消屏障、终态确认和独立连接期限已补实现及测试源码，待远端验证。新路径以首次 StartLoad 为连接标志。多 DP 属于另行设计范围。
