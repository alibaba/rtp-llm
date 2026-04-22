# Mooncake P2P Transfer 测试补强与本地 RDMA one-sided 实施计划

最后更新：2026-04-22
适用分支：feature/p2p_connector-3

## 目标

先把当前 Mooncake backend 的可用性通过单元测试、功能测试、e2e/端到端测试测稳，再扩展到 connector 单机本地 RDMA one-sided。

## 阶段 1：当前可用性补强

### 1. 单元测试

1. sender
   - regMem 幂等：同地址同长度重复注册成功，不重复产生副作用
   - prepare 失败传播：prepare 返回 TIMEOUT/CANCELLED 时，send 直接失败
   - openSegment 失败：finish 被调用且错误码正确
   - submitTransfer 失败：finish 被调用且 batch_id 被释放
   - transfer status 失败/timeout：finish 被调用且错误码正确
2. receiver
   - descriptor 构建顺序稳定
   - prepare 在 task 不存在、descriptor 不存在、task 已 done、deadline 已过时的返回值
   - finish 对 task 不存在、重复 finish、error_code 透传的行为
3. adapter
   - classic adapter openSegment 成功判定
   - batch_id 生命周期 freeBatchID

### 2. 功能测试 / 集成测试

1. fake adapter + 真 arpc 控制面
   - happy path
   - timeout
   - cancel
   - finish error override
2. 真 classic TE + tcp transport
   - sender/receiver 本地 round-trip
   - descriptor -> openSegment -> WRITE -> poll -> finish 全链路

### 3. E2E / 端到端测试

1. P2P worker 层 backend 选择与初始化
   - kTcp
   - kMooncake
2. Mooncake backend 在 worker 层的基本收发 smoke
   - 先复用现有 mock sender/receiver 模式
   - 后续条件允许时接真实 sender/receiver

## 阶段 2：本地 RDMA one-sided

### 1. 实现

1. 扩展 Mooncake classic adapter transport 选择，不再固定 tcp
2. 基于 connector 本地 RDMA 环境选择 rdma transport
3. 保持控制面 prepare/finish 不变，仅替换大 payload 数据面为 one-sided WRITE

### 2. 测试

1. 单元测试
   - transport 选择逻辑
   - RDMA 配置校验与错误传播
2. 功能测试
   - 真 classic TE + 本地 RDMA happy path
   - timeout
   - cancel
3. E2E
   - connector 单机 sender/receiver 双端本地 RDMA smoke

## 完成判定

1. TCP-only Bazel 路径不回归
2. enable_mooncake_te=true 构建与测试通过
3. 每一层测试都有至少一条 happy path 和一条失败路径
4. 完成后回填每一项完成状态
