# Routing and Balancing

`RequestWorkerSelector` 负责角色选择。Generation 选点在同一个 `RequestRoute` 中冻结角色元数据、
执行时间与代际 pin；成功后原对象完成构造，失败时关闭已取得的全部 pin。提交容量时创建新的
回调身份，共享已冻结的分配事实与 pin；对外 DTO 仍返回独立副本。Encoder 阶段使用
`EncoderSelection` 持有单个 pin；混合角色选点可将该 pin 转交给 RequestRoute。
选择器不向引擎发布请求，资源提交和失败清理由
`AbstractRequestScheduler` 的 DIRECT / QUEUE 实现负责。

## 角色和 group

Schedule 可通过 `schedule_roles` 选择 Encoder、Prefill、Decode、PDFusion。
显式角色去重后按 `ModelMetaConfig.requiredRoles()` 顺序决策；未配置或非法角色返回
`INVALID_REQUEST`，未传则使用模型配置角色。选中角色的 group 约束后续角色。
Encoder 阶段与 Generation 阶段分别调用和跟踪。

VIT-only 请求经 `VitCacheSelector` 选择，结合多模态 hash、GPU/CPU embedding 命中和
worker 代际；其余 VIT 路由使用 `VitWorkerSelector`。客户端携带的 selected VIT 必须仍
指向有效实例。VIT-only 请求不需要 Generation 的批量 payload。

## Prefill / PDFusion

`CostBasedPrefillStrategy` 从已发布的逻辑 endpoint 读取候选，按请求配置和优先级检查
可用性。`RouteTimelineProjector` 根据不可变工作快照、GroupingPolicy、执行时间模型和
容量边界推演本次请求的 TTFT，保留所有硬约束内的候选。

默认选择最小 TTFT，并通过 `EndpointRoundRobin` 轮换完全并列的候选。
配置 cache affinity 后，以最短 TTFT 候选中最高命中数为基线；满足最低命中率且在
`maxExtraTtftMs` 范围内的缓存优势候选可优先。已知时间与未知时间分开比较，未知时间
不会仅因饱和排序值相等而成为最优并列项。

缓存匹配支持 LOCAL_SYNC、KVCM、LOCAL_STANDBY，按逻辑 worker 身份查找。
KVCM 单引擎旧物理 key 可以兼容回退；多引擎必须精确匹配 index。
选点后捕获 generation pin，并校验与预测时的 endpoint 一致。预测缓存同时校验资源进度
和当前收集窗口；配置更新可改变后续预测，不修改已经提交的决策组。

## Decode

`DecodeSelector` 使用每个 endpoint 的 `DecodeRoutingView` 检查并发、硬 KV 和预期 KV
容量。容量不足但满足请求抢占规则的候选可以进入抢占规划，不能只因低 cost 而选择一个
无法释放足够容量的 worker。

Decode cost 使用请求配置快照中的表达式和容量参数；最小 cost 完全并列时轮换，分别记录
`DECODE_MIN_COST` 与 `DECODE_MIN_COST_ROUND_ROBIN_TIEBREAK`。选中后再次捕获精确代际，
由调度器提交 typed reservation 和引擎派发许可。

## Encoder

`EncoderStrategy` 要求 endpoint 已发布、存活且满足 group。请求带 `encoder_cache_hit_len`
时优先最少未命中编码工作量，并列比较 `running + waiting + 本地待观察请求`；未带时仅
比较该并发数，不用 KV 剩余量决胜。QUEUE 另检查 `maxInflightPerEncoderWorker`。
首次状态上报前按 `seq_len - encoder_cache_hit_len` 估算工作；上报后用实际合成输入长度，
不再减 prefix。Finished-only 的缓存命中任务不计为编码负载。

## 失败与响应

任一角色选择失败时关闭之前捕获的 pin。已提交的资源通过精确预留和请求状态统一清理，
不按地址重新查找或回滚另一个代际。临时容量不足可进入 QUEUE 等待，并记录 blocker 的
角色、group、endpoint 和容量版本；唤醒后重新判断。

响应保留物理 IP、HTTP/gRPC port、group 和实际角色。N=1 省略 `engine_index`；N>1
设置所选 index，包括显式 0。转发透传字段 presence。角色错误为 PREFILL 8402、DECODE
8403、PDFUSION 8404、VIT 8405、ENCODER 8408。
