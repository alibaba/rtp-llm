# P2P 并发、队列饱和与持续运行测试

已实现，待远端编译和运行。没有增加生产 API 或配置项。

## UT

`P2PQueueSaturationTest.cc` 加入 `//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_scheduler_test`；最后一条位于同目录的 `P2PConnectorWorkerTest.cc`，目标为 `p2p_connector_worker_test`。

| 用例名 | 是否通过 | 用例内容 |
|---|---|---|
| FullKickoffQueueRejectsWithoutRpcOrRetainedResource | 待运行 | 屏障占住 D 工作线程并填满共享队列；入队在放行前返回错误，无 RPC/context/资源遗留，恢复后新请求成功。 |
| QueuedCancelNeverStartsRpcAfterCapacityReturns | 待运行 | 排队期间取消；放行后不再启动旧 key 的 RPC，资源释放、新请求成功。 |
| QueuedDeadlineNeverStartsRpcAfterCapacityReturns | 待运行 | 排队期间请求到期；在线程仍被占用时判定超时，放行后不再启动旧 RPC。 |
| SaturatedControlPoolStillExpiresAndEventuallyDrainsEveryKey | 待运行 | 8 个 context 竞争已饱和的控制池；均及时超时且保留 target，放行后每个 key 都完成取消、lease 查询与回收。 |
| SharedSenderQueueIsolatesRejectedCancelledAndSuccessfulRequests | 待运行 | P 的 A 请求占用 sender，B 排队，C 遇满队列被拒；取消 B、放行 A，校验独立终态、B/C 未发送及新请求恢复。 |

UT 使用真实小线程池及屏障，不靠随机调度制造饱和。控制池用例检查任务进展与资源持有，不把它当作 CPU 忙轮询性能证明。

## 跨机集成

沿用 [跨机夹具与环境配置](p2p_multi_rank_transfer_tests.md)，目标为 `//rtp_llm/cpp/model_rpc/test:p2p_multi_rank_transfer_test`。两台机器各运行两个逻辑 rank，真实 TCP/GPU 传输；不加载模型、不经过 GenerateStream/NCCL，也不覆盖 RDMA。

| 用例名（后缀） | 是否通过 | 用例内容 |
|---|---|---|
| ConcurrentRequestsKeepEveryRankPayloadIsolated | 待运行 | 并发 8/16/32，混合 1/4 block，固定种子乱序发布；逐请求检查两 rank 的全部字节、首 token、资源回收。 |
| ConcurrentSuccessCancelAndTimeoutRecoverTogether | 待运行 | 三轮 6 成功、2 首层完成后取消、2 超时；每轮随后成功一批，确认失败不污染后续请求。 |
| RepeatedOverloadDrainsAndRecoversWithoutRestart | 待运行 | D 小池容量 8，20 轮健康 8→屏障过载 16→健康 8；要求确有队列拒绝，每轮排空回基线，始终复用实例。 |
| DISABLED_SustainedMixedLoadHasBoundedResources | 待运行 | 默认 2 小时，预热后持续并发批次；每 10 分钟插入混合失败和过载，每分钟检查资源、过期记录及进程指标。 |

每批结束检查 D 任务/lease/checker、P resource/计算结果和两端 GPU block 池。终态登记与取消标记允许按自身 TTL 保留；长测检查超过清理宽限的过期条目，默认时长跨过 1 小时取消标记 TTL。P 源资源由夹具保留到 sender 屏障通过，不覆盖此前暂缓的 P 源保护；退出清理问题不作为本组验收目标。

## 单独运行长测

先按跨机说明设置两端 `TEST_BIN`、runfiles、CUDA 和 INFO 日志环境。以下仅为远端运行命令，当前未执行；构建须经 `test-execution` 流程。

P 端先启动：

```bash
P2P_MULTI_HOST=P_IP P2P_MULTI_LISTEN=P_IP:19090 P2P_MULTI_SESSIONS=1 \
  timeout 10000s "$TEST_BIN" --gtest_also_run_disabled_tests \
  --gtest_filter=P2PMultiRankPeer.DISABLED_Serve \
  --gtest_output=xml:soak_prefill.xml
```

D 端随后启动：

```bash
P2P_MULTI_HOST=D_IP P2P_MULTI_PEER=P_IP:19090 \
  timeout 10000s "$TEST_BIN" --gtest_also_run_disabled_tests \
  --gtest_filter=P2PMultiRankTransferTest.DISABLED_SustainedMixedLoadHasBoundedResources \
  --gtest_output=xml:soak_decode.xml
```

两端都须退出 0，并保留 XML 与控制台 `[P2P-BATCH]`、`[P2P-SOAK]` 记录。

| D 端环境变量 | 默认值 | 用途 |
|---|---|---|
| P2P_SOAK_SECONDS | 7200 | 运行时长；缩短仅供调试，不能作为两小时验证。 |
| P2P_SOAK_CONCURRENCY | 8 | 每批并发，范围 5–16；过载为两倍。 |
| P2P_SOAK_PAUSE_MS | 100 | 批次间暂停毫秒数。 |
| P2P_SOAK_SEED | 20260917 | 可复现的发布顺序种子。 |
| P2P_SOAK_RSS_GROWTH_MB | 256 | 每端相对预热基线的 RSS 增长预算。 |
| P2P_SOAK_CHECK_PERF | 0 | 开启后，健康窗口 p99 恶化超过 20% 或吞吐下降超过 10%，连续三窗口失败。 |

FD/线程数各允许比预热基线多 16；CPU 和 D GPU 已用内存只记录。资源预算是测试默认值，需远端基线确认。吞吐与延迟包含夹具控制开销，不能当作模型服务 SLO。

长测采用“并发一批、排空、下一批”，复用连接与内部表；尚不等价于按服务稳定容量 70% 的连续到达压测。真实服务持续在途流量和 RDMA 仍需部署环境验证。
