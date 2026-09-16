# Decode 活动阶段回退：红测与修复

## 本地复现

当前工作区的 `DecodeState.doCalibrate` 仍继承了旧 DecodeEndpoint 的问题：用只包含 KV_ALLOCATED/RUNNING 的 confirmedNow 判断记录是否消失。已经 confirmed 的请求回到 RECEIVED/PENDING 时，会被误删并留下 token=0 的 settled 记录，后续 finished 无法通知原 RequestSlot。

先添加 UT，未改生产实现时执行，7 个用例全部产生断言失败（非编译/环境失败）：

- RequestResourceAccountingTest：真实请求注册、派发成功、Prefill 完成后，组合 RUNNING/KV_ALLOCATED → RECEIVED/PENDING → finished，共 4 个用例均报 `request inflight: expected 0, actual 1`。断言发生在清理 fixture 之前，不调用 inactivity/sweep、不等待 60 秒。
- DecodeStateTest：RECEIVED/PENDING 连续上报和恢复 RUNNING，共 2 个用例发现准确身份已丢失。
- DecodeStateTest：两个正在跟踪的请求中一个阶段回退，发现本地并发占位从 2 错误下降到 1。

红测日志：`/tmp/flexlb-decode-phase-regression-red.log`。

## 修复

1. 增加本轮局部集合 `presentNow`，记录完整活动快照中的有效请求身份。它用于 absent pruning，独立于当前 KV_ALLOCATED/RUNNING 集合。
2. 已 confirmed 的记录在活动阶段回退时，保留原 reservation token 和最后一次确认的资源阶段，刷新最近观测时间。ACTIVE 持续投影到原 Slot，后续 finished 仍带同一个准确身份。
3. 保守保留其本地 Engine 所有权并发占位直到结算，不将 RECEIVED 当作终止或已释放的证明。该计数不是 GPU 物理执行并发；物理 KV 样本仍按 Worker 上报发布。这样既避免误删，也避免后续 expiry/terminal 清理误减其他请求的并发占位。
4. 存在抢占 claim 时仍由原有 synthetic hold 分支计数，不再加普通 retainedConfirmed，防止重复计数。普通回退记录不额外创建 KV hold。
5. 同一观测中 finished 优先于 active 的规则保持不变；真正不在活动快照中的既有清理策略属于另一条路径，本次没有把该策略推广成“缺席即完成”的新请求层通知。

没有新增持久状态表或状态机；presentNow/retainedConfirmed 仅为本次校准的局部变量。

## 回归

上述 7 个红测在修复后全部通过，连同资源账本、阶段视图、过期、抢占容量、inactivity、gRPC 状态同步共 81 项通过：`/tmp/flexlb-decode-phase-regression-green.log`。

另补 3 个抢占回归，分别覆盖 CANCEL_REQUESTED、NOT_FOUND_STALE、CANCEL_UNKNOWN：连续 RECEIVED 保留且只保留一份并发/KV hold；同一快照同时含 active 和 finished 时，只产生准确 TERMINAL，incoming 可以提交，容量完全可回收。

完整回归日志：`/tmp/flexlb-decode-phase-regression-full.log`。线上样本与字节码核验来自用户提供的报告；本轮验证为本地代码和 UT，未部署线上。

提交前将 DecodeState 改动与工作区另一组 RequestSlot 重构分离，在隔离目录检出暂存内容重新运行 `./mvnw -B -P 'opensource,!internal' -pl flexlb-sync -am test`：flexlb-sync 930 项全部通过，日志 `/tmp/flexlb-staged-commit-test.log`。此结果不依赖尚未提交的 RequestSlot 重构。
