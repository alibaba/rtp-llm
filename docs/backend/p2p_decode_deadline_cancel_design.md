# P2P Decode 超时取消与 Lease 保护修改方案

## 目标与范围

- 只保护 Decode（D）目标 KV，不处理 Prefill 源 KV。
- 到 `D = transfer_deadline_ms` 才停止新传输；D+10 秒断连接，D+20 秒仍无法确认则由 rank 0 进程 `std::abort()`。
- 目标 KV 只能在物理传输全部停止后释放。

## 1. D Scheduler 与 AsyncContext

- `asyncRead` 计算 `D = min(request_deadline, now + p2p_max_transfer_deadline_ms)`；AsyncContext 按绝对时间推进。
- 到 D 向 Stream 返回 `GENERATE_TIMEOUT`，并向所有 rank 发 `CANCEL_READ`；context 持有 KV，后台等待 cancel-done。
- worker 的 `TRANSFER_NOT_DONE` 仅表示资源尚不可释放，不向 Stream 透传。

## 2. D Worker 与 WriteLease

- 删除 `D-250ms` steal 和 `D-100ms` return，RecvTask deadline 统一设为 D。
- D 前完成则正常返回；到 D 未完成时一次执行：
  1. steal 全部任务，阻止新的 sender 匹配；
  2. `seal()`，冻结 `started_ops`；
  3. cancel 未完成 task；
  4. 向 scheduler 返回内部状态 `TRANSFER_NOT_DONE`。
- PENDING task 立即 done；已 `startTransfer()` 的 task 只记录取消并等待 completion。
- completion 在 worker 本地推进 `finished_ops`；注册中的 READ 对查询返回未停止，注册态与 lease map 切换无空窗。
- 仅当 `sealed && started_ops == finished_ops` 时为 stopped。

## 3. RDMA TransferService

- `ActiveTransfer` 记录 connection、`D+10s` 和 `disconnect_triggered`。
- D 到期的 watchdog 只结束 RPC，不伪造物理完成，也不删除运行中的 ActiveTransfer。
- 正常、失败和 QP flush completion 都调用 `notifyTransferDone()`。
- 各 rank 本地线程在 D+10 秒检查；未完成则幂等 close 相关 connection，不依赖 rank 0 查询。

## 4. RdmaClient、Connection 与连接池

- 新增 `IRdmaConnection::close()`，复用 Barex 的 `XConnection::Close()`。
- `close()` 标记 FAILED、使 QP 进入 ERR，并失败处理 `pending_requests_`；连接池后续创建新连接。
- channel 共享，close 可能影响同 channel 请求，作为兜底接受。

## 5. Lease 查询、释放与 Fail-stop

- all-rank cancel-done 后，rank 0 才查询 lease；任一 rank 无响应，本次结果无效。
- D 到 D+20 秒持续退避查询；获得有效 `allStopped()` 后立即释放目标 KV。
- D+10 秒各 rank 本地 close，不释放 KV；等待 completion 并继续查询。
- rank 1…N 在本地 D+10 close RDMA connection；除 worker gRPC 不可达外，应在 completion 后向 rank 0 返回 stopped。
- D+20 秒仍无 `allStopped()`，不得释放 KV，rank 0 执行 `std::abort()`。

## 6. 配置与测试

- 删除两个提前量配置；原 10 秒配置改为断连接等待时间，增加 D+20 查询期限，且后者必须更大。
- 单测覆盖正常完成、D 取消、迟到 completion、D+10 单次 close、查询恢复、D+20 整实例 fail-stop、未 stopped 不释放。
- TCP 不执行 channel close；D 时刻的取消语义一致。
