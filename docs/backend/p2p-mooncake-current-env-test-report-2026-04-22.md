# P2P Mooncake 当前环境测试报告

日期：2026-04-22
环境：`connector` / `qiongshi_rtp_dev`
代码目录：`/data0/qiongshi.gb/RTP-LLM/github-opensource`
分支：`feature/p2p_connector-3`

## 1. 测试目标

本轮目标不是验证真实 RDMA one-sided WRITE 性能，而是把当前无 RDMA 环境下能够完成的验证做满，确认以下几点：

1. Mooncake 后端在 `IKVCacheSender.h` / `IKVCacheReceiver.h` 既有接口语义下可工作。
2. 薄控制面 `prepare(unique_key)` / `finish(unique_key, status)` 的 sender/receiver 协作语义正确。
3. Mooncake sender 在 descriptor 拉取、请求构建、batch 提交、finish 收敛等关键状态机上有充分单测覆盖。
4. Mooncake receiver 在 task store、descriptor 索引、timeout/cancel/finish 语义上有充分单测覆盖。
5. 非 Mooncake 既有 TCP 路径没有被新增测试改坏。

## 2. 当前环境事实

### 2.1 已确认能力

- 可运行 Bazel 单元测试与部分集成测试。
- 可运行 Mooncake 控制面真实 TCP round-trip 测试。
- 可运行 classic TransferEngine 的非 RDMA/TCP transport 测试。

### 2.2 已确认限制

- 当前 host 没有可用 HCA。
- `/sys/class/infiniband` 为空。
- `/dev/infiniband` 为空。
- `rdma link show` / `rdma dev show` 无设备。

结论：当前环境**不能**验证真实 RDMA one-sided WRITE 数据面，也不能做跨机 RDMA E2E。

## 3. 本轮新增测试

本轮在 `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeBackendStubTest.cc` 增加以下测试：

### 3.1 Sender 侧新增

1. `SendReturnsBuildRequestFailedWhenBlockInfoEmpty`
2. `SendReturnsBuildRequestFailedWhenDescriptorSegmentEmpty`
3. `SendReturnsBufferMismatchWhenDescriptorCountDiffers`

### 3.2 Receiver 侧新增

1. `RegMemDelegatesToAdapter`
2. `InitSucceedsWhenControlPlaneListenPortDisabled`
3. `PrepareDescriptorFallsBackToConfiguredLocalServerName`
4. `PrepareDescriptorFallsBackToClassicHostAndPort`

说明：加上此前已经补过的用例，`MooncakeBackendStubTest.cc` 当前总计 40 个测试，覆盖 sender/receiver/control-plane/classic TE 的主要状态分支。

## 4. 实际执行结果

### 4.1 已通过

以下命令在 `qiongshi_rtp_dev` 中执行通过：

1. `bazelisk test //rtp_llm/cpp/cache/connector/p2p/transfer/mooncake:mooncake_backend_stub_test ...`
   - 结果：`PASSED`
   - 说明：Mooncake stub/control-plane/classic TE 相关 40 个测试均通过

2. `bazelisk test //rtp_llm/cpp/cache/connector/p2p/transfer/test:transfer_backend_config_test ...`
   - 结果：`PASSED`
   - 说明：backend/config 透传与选择逻辑回归通过

3. `bazelisk test //rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test:tcp_sender_receiver_test ...`
   - 结果：`PASSED`
   - 说明：既有 TCP sender/receiver 路径回归通过

### 4.2 已尝试但被环境阻塞

以下 target 在当前容器中未能进入测试执行阶段，而是被 RDMA 相关链接问题阻塞：

1. `//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_worker_test`
2. `//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_test`

共同错误特征：

- `/usr/local/lib64/librdmacm.so.1` 链接时缺少多个 `IBVERBS_*` 符号
- 典型报错包括：
  - `ibv_get_pkey_index@IBVERBS_1.5`
  - `ibv_get_device_index@IBVERBS_1.9`
  - `_ibv_query_gid_ex@IBVERBS_1.11`
  - `ibv_reg_dmabuf_mr@IBVERBS_1.12`

结论：这是**当前容器 RDMA verbs / rdmacm 链接环境问题**，不是 Mooncake 新增测试逻辑问题。

## 5. 覆盖结论

### 5.1 当前环境下已经充分覆盖的内容

#### A. Mooncake sender 语义

- `regMem()` 委托 adapter
- 控制面端口解析
  - request port 优先
  - config fallback
  - port 缺失失败
- `prepare()` 失败传播
  - timeout / control-plane failure
- `openSegment()` 失败后的 finish 收敛
- `buildWriteRequests()` 主要错误分支
  - `block_info` 为空
  - descriptor `segment_name` 为空
  - descriptor block 缺失
  - descriptor length mismatch
  - target address 为空
  - descriptor count mismatch
  - 全部 block 为空
- batch 生命周期
  - `allocateBatchID()` 失败
  - `submitTransfer()` 失败
  - `freeBatchID()` 调用时机
- 轮询状态与 finish 语义
  - success
  - timeout
  - transfer failure
  - finish failure 覆盖 success 的情况
  - transfer failure 不被 finish failure 覆盖的情况

#### B. Mooncake receiver 语义

- `recv()` 注册 task
- duplicate `unique_key` 拒绝
- `regMem()` 委托 adapter
- descriptor 构建
  - cache_key 排序稳定
  - 跳过 empty block
  - segment name 回退到 configured `local_server_name`
  - segment name 回退到 `ip_or_host:rpc_port`
- `prepareDescriptor()` 主要分支
  - 正常 start transfer
  - deadline 已过
  - task 被 steal / not found
  - task 在 prepare 前取消
- `finishTransfer()` 主要分支
  - success
  - explicit failure
  - cancel during transfer
  - finish after deadline -> timeout
  - empty unique key
  - task not found
  - 重复 finish 的幂等语义
- control-plane server
  - 可正常启动并完成 prepare/finish round-trip
  - listen port 关闭时可正常初始化

#### C. 控制面与经典传输

- 真实 TCP 控制面 round-trip 已验证
- classic TransferEngine 非 RDMA/TCP transport payload copy 已验证
- classic TransferEngine invalid transport / registerLocalMemory 幂等 已验证

#### D. 非 Mooncake 回归

- `transfer_backend_config_test` 通过
- `tcp_sender_receiver_test` 通过

### 5.2 当前环境下仍无法完成的验证

以下内容必须等带 RDMA 的环境：

1. 真实 RDMA one-sided WRITE 数据面
2. GPU memory registration 与 RDMA MR 相关行为
3. 跨机 RDMA 网络异常、重试、链路抖动
4. Mooncake one-sided WRITE 的真实 E2E
5. Mooncake 模式下的 smoke 与性能测试

## 6. 作为测试 owner 的结论

在**当前无 RDMA 环境**下，我认为 Mooncake P2P 后端已经做到了足够充分的单元测试和可执行集成验证，达到可以申请下一阶段 RDMA 环境验证的程度，理由如下：

1. Mooncake sender/receiver/control-plane 的核心状态机已经被单测收口。
2. 薄控制面真实 TCP round-trip 已验证，不是纯 mock。
3. classic TE 的非 RDMA/TCP transport 路径已验证，说明“控制面 + 数据面适配层”整体形状是可工作的。
4. 现阶段剩余的验证缺口，已经集中在**真实 RDMA 设备与网络环境**，不再是单机无 RDMA 环境能有效补充的范围。

## 7. 下一阶段建议

拿到 RDMA 环境后，建议按下面顺序继续：

1. 先跑 Mooncake one-sided WRITE happy path E2E
2. 再跑 timeout / cancel / finish failure 的 RDMA 端到端场景
3. 再跑 smoke
4. 最后补性能与稳定性

