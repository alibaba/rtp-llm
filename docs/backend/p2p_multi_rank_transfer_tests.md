# 多 rank 跨机取消、超时与故障用例

## 范围

目标：验证新 P2P 路径的多 rank 广播、StartLoad 登记等待、加载期限、取消及 D 目标内存保护。

两台远端 GPU 机器分别运行 P、D；每个进程包含两个逻辑 TP rank，各自拥有独立 connector、gRPC/TCP 服务和 GPU block 池，共用该机器的可见 GPU 0。使用真实 TCP 传输和 GPU 字节校验，固定 MHA、TP=2、CP=1。故障由明确的 RPC/传输状态触发。

这是 connector 集成测试：直接登记 P 的 resource/首 token，直接调用 D scheduler，不加载模型、不经过 GenerateStream 或 NCCL。测试用聚合引用模拟两 rank 的资源持有；不验证真实 engine 的跨进程分配同步。P 源资源由夹具保留，不覆盖此前暂缓的 P 源保护。

## 用例

代码：`rtp_llm/cpp/model_rpc/test/P2PMultiRankTransferTest.cc`。

| 用例名（测试名后缀） | 是否通过 | 用例内容 |
|---|---|---|
| BothRanksTransferRealGpuBytes | 待远端运行 | 两个 P rank 向对应 D rank 传输两层、两个 block；校验所有字节和首 token；再次请求仍成功。 |
| CancelRetainsTargetsWhileNonzeroRankH2dIsBlocked | 待远端运行 | rank 0 已完成、rank 1 H2D 未结束时主动取消；校验两 rank 收到取消，D 保留目标资源直至物理完成。 |
| LoadTimeoutRetainsTargetsWhileNonzeroRankH2dIsBlocked | 待远端运行 | 相同在途状态下不主动取消；3 秒加载期限触发失败、取消与资源保护，不能等到 120 秒请求期限。 |
| NonzeroRankRpcFailureCancelsAlreadyStartedWrites | 待远端运行 | rank 1 READ 已接受且 H2D 在途时返回 UNAVAILABLE；保留故障原因，并取消、排空其他工作。 |
| StartLoadWaitsForMatchingRegistration | 待远端运行 | StartLoad 先到，登记前不发送 HANDLE_READ；登记同一 key 并发布数据后成功。 |
| MissingRegistrationExpiresWithinLoadBudget | 待远端运行 | 不登记请求；1 秒加载预算内到期，未发 HANDLE_READ，随后新 key 可成功。 |
| CancelInterruptsRegistrationWaitBeforeLoadDeadline | 待远端运行 | StartLoad 等登记时取消；60 秒加载期限前结束等待，随后新请求可成功。 |
| ConcurrentRequestsKeepEveryRankPayloadIsolated | 待远端运行 | 同一组 connector 依次承载 8/16/32 并发请求，混合 1/4 block、乱序发布；逐请求校验两 rank 字节、首 token 和回收。 |
| ConcurrentSuccessCancelAndTimeoutRecoverTogether | 待远端运行 | 每批 10 请求，6 成功、2 在首层传输完成后取消、2 不发布数据直至超时；三轮混合批次后分别执行健康批次。 |
| RepeatedOverloadDrainsAndRecoversWithoutRestart | 待远端运行 | 测试夹具将 D 共享池缩为单线程、队列容量 8；屏障确定性制造饱和，连续 20 轮健康、过载、恢复，验证拒绝与资源回收。 |

另有默认禁用的两小时长测 `DISABLED_SustainedMixedLoadHasBoundedResources`，运行方法见 [并发与长测说明](p2p_concurrency_queue_soak_tests.md)。

三条 H2D 故障用例用 CUDA event 阻塞 rank 1 的真实复制流。失败后丢弃调用方资源和 context 引用，只让生产 checker 持有资源；再次观察 lease 查询后耗尽剩余池，确认旧 block 不可分配。放行 H2D 后检查新分配数据未被覆盖、旧 block 可复用、池回到基线，再执行成功请求。

RPC 故障仅模拟“工作已接受但响应失败”，取消和 lease RPC 仍可达。不覆盖进程被杀、长期网络隔离、RDMA/DMA 完成或真实四进程推理服务；这些仍需对应环境验证。

## 远端运行

仅在指定远端环境通过 `test-execution` 流程构建目标：
`//rtp_llm/cpp/model_rpc/test:p2p_multi_rank_transfer_test`。
目标带 `manual` 标签，不自动加入通配测试。两端使用同一版本、架构的二进制及完整 runfiles；保留构建环境的动态库配置。

两端在各自外源仓库根目录设置（仅示例，当前未运行）：

```bash
export TEST_BIN="$PWD/bazel-bin/rtp_llm/cpp/model_rpc/test/p2p_multi_rank_transfer_test"
export TEST_SRCDIR="${TEST_BIN}.runfiles"
export TEST_WORKSPACE=rtp_llm
export TEST_BINARY=rtp_llm/cpp/model_rpc/test/p2p_multi_rank_transfer_test
export TEST_USING_DEVICE=CUDA
export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=INFO
```

先在 P 启动，替换 `P_IP` 为可被 D 访问的 IPv4：

```bash
P2P_MULTI_HOST=P_IP P2P_MULTI_LISTEN=P_IP:19090 P2P_MULTI_SESSIONS=10 \
  timeout 900s "$TEST_BIN" \
  --gtest_filter=P2PMultiRankPeer.DISABLED_Serve --gtest_also_run_disabled_tests \
  --gtest_output=xml:multi_rank_prefill.xml
```

再在 D 启动，替换 `D_IP` 和 `P_IP`：

```bash
P2P_MULTI_HOST=D_IP P2P_MULTI_PEER=P_IP:19090 \
  timeout 900s "$TEST_BIN" --gtest_filter='P2PMultiRankTransferTest.*' \
  --gtest_output=xml:multi_rank_decode.xml
```

两端需互通控制端口及动态选择的 gRPC/TCP 端口。缺少参数或对端不可达会失败，不静默跳过。H2D gate 用例要求 INFO 日志；DEBUG 会触发生产复制代码的全设备同步，使另一 rank 也被阻塞。P 每接受一个测试创建一组 rank，最多等连接 180 秒；D 清理后通知 P 销毁该组。两端都应退出 0，并检查 D 的 10 条用例和 P 服务均通过。筛选单条时将 P 的 sessions 改为 1；不要启用测试分片、并发运行或无配套 sessions 的重复运行。

当前状态：仅实现并静态检查，未编译、未运行。
