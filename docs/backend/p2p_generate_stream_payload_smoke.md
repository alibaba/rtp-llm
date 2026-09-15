# GenerateStreamCall P2P payload smoke

参考 [PR #1221](https://github.com/alibaba/rtp-llm/pull/1221) 的真实缓存字节校验方式，入口上移到 D 的 `GenerateStreamCall`。

## 实际执行链路

客户端 gRPC → DecodeRpcServerNew2 → GetPeerInfo / P 的 GenerateStreamCall → 两端真实 FIFOScheduler、NormalGenerateStream、KVCacheManager → StartLoad / ExecuteFunction → P2PConnector TCP 或 RDMA → D 恢复首 token → D 执行前逐字节检查 → 返回剩余 token。

默认由 D 测试进程使用 `posix_spawn + exec` 启动独立 P 进程。两端分别初始化 CUDA、缓存池、P2PConnector 和网络端点；不会继承已初始化的 CUDA/RDMA 状态。默认可共享一张物理 GPU，但拥有独立 CUDA context 和地址空间。

仅模型计算和采样由 `PayloadEngine` 替换：P 写入确定性数据，经生产 `writeP2PLayer` 发布；D 加载完成后读取实际 GPU 缓存并检查。控制通道只交换配置、端口、统计和退出指令，不传缓存 payload。D 检查两端统计及释放状态，P 清理后确认退出；自动启动的进程还会检查退出码。

跨机模式由用户分别启动 P、D；缓存传输仍使用相同的生产链路。两端须使用同一版本、同一架构的测试二进制。

## 覆盖

- FP16 KV，以及 INT8 KV 和 scale buffers；3 层、多 block。
- 8 token 完整块、11 token 非完整尾块；连续请求使用不同数据，检查缓存池复用后的残留数据。
- D 缓存预填 `0xff`；预期值取决于 request、layer、逻辑 block、buffer、字节偏移，与物理 block ID 无关。
- 校验真实 RPC 次数、逐层发布次数、非零且一致的总字节数、完整输出 `[100, 101, 102]` 和两端空闲 block 恢复。
- P 故意损坏一个字节：D 报出 request/layer/tag/block/buffer/byte，不能成功完成生成。
- P 不发布最后一层：真实加载失败，D 不执行，资源最终释放。
- query 绕过 PD：单 token、固定 beam、变宽 beam、多返回序列、显式关闭 PD；分别使用空 key 和业务 key，检查 P 本地完成、P2P 注册/传输不发生、缓存释放。

## 运行入口

目标：`//rtp_llm/cpp/model_rpc/test:p2p_generate_stream_smoke_test`，标记 `manual`，不自动加入现有 CI 测试集合。编译及运行须遵循仓库 `test-execution` skill，在指定远端环境执行。

默认使用 TCP，仍需要 CUDA 设备和可用本机监听端口，不需要模型权重。

RDMA 使用同一目标及同一套断言，向测试传入：

- `P2P_SMOKE_TRANSPORT=rdma`
- `P2P_SMOKE_HOST=<本机 RDMA 网络接口 IPv4 地址>`

RDMA 需要启用内源 backend 的构建、RDMA 网卡和对应驱动。后端不支持或初始化失败会直接失败，不回退 TCP。同机 RDMA 测试还要求后端支持同机端点互传。

多网卡环境可在两端设置 Barex 现有的 `ACCL_USE_NICS=<网卡名>`，例如 `mlx5_2`，限定本轮验证使用的网卡；通过单网卡测试不代表其他网卡或跨机链路已通过。

这是传输完整性测试；生成 token 是确定值，不校验模型推理精度。


## 跨机运行

先编译上述 target；相同架构和运行环境可将二进制及其运行依赖从构建机器 SCP 到对端，无需两端分别构建。按 `test-execution` 流程预检、运行，平台配置使用 `sm9x`、`cuda12_9` 并按实际 CUDA 版本调整。下面列出两端各自需要的测试参数（追加到 `bazelisk test <target>`）；选择同一 transport，使用各自机器的可达网卡 IP。

**P 机器先启动：**

```bash
--test_filter=P2PPayloadWorker.DISABLED_PrefillProcess \
--test_arg=--gtest_also_run_disabled_tests \
--test_env=P2P_SMOKE_TRANSPORT=rdma \
--test_env=P2P_SMOKE_HOST=<P_IP> \
--test_env=P2P_SMOKE_CONTROL_LISTEN=<P_IP>:29800 \
--test_output=streamed --test_timeout=600 --nocache_test_results
```

看到 `P2P control listening` 后，在 120 秒内启动 D。P 每次服务一个测试用例，该用例内部可以发送多次 GenerateStreamCall。

**D 机器运行 FP16 用例：**

```bash
--test_filter=P2PGenerateStreamSmokeTest.GenerateStreamTransfersEveryFp16CacheByte \
--test_env=P2P_SMOKE_TRANSPORT=rdma \
--test_env=P2P_SMOKE_HOST=<D_IP> \
--test_env=P2P_SMOKE_CONTROL_ADDR=<P_IP>:29800 \
--test_output=streamed --test_timeout=600 --nocache_test_results
```

D 完成后，两端测试均应 PASSED，P 自动清理退出。换用 INT8 或反例时，重新启动一次 P，并在 D 选择对应的单个测试名称；不要在跨机模式下使用全套用例过滤器。TCP 跨机验证将两端 transport 都改成 `tcp`。

控制端口 29800、两端动态分配的 gRPC 和传输端口需互通。GPU 选择遵循各自远端运行环境的 `CUDA_VISIBLE_DEVICES`；测试内部使用可见设备 0。

跨机默认请求预算 120 秒、加载预算 90 秒；同机仍为 10 秒、3 秒。可在 D 设置 `P2P_SMOKE_REQUEST_TIMEOUT_MS` 和 `P2P_SMOKE_LOAD_TIMEOUT_MS`，加载预算必须小于请求预算，均不超过 600000 毫秒；D 通过控制通道告知 P。外层 gRPC watchdog 比请求预算多 2 秒，控制通道和清理等待使用单调时钟。

用例不要求修改宿主机时间。生产 P2P 仍传递并检查绝对 deadline，因此加载预算需覆盖 P 比 D 快的时差和实际传输耗时；这只能容纳预算范围内的时差，不是时钟无关的生产协议。缺层反例仍等待真实加载错误，不能仅凭外层 watchdog 超时通过。

## PD 取消传播测试

同一 target 新增以下用例，从客户端 `GenerateStreamCall` 的 `TryCancel()` 进入：

- `CancelAtEachPDStageDrainsBothSides`：P 发布缓存前、发布第一层后、D 已输出且请求未完成时取消。
- `RandomCancelRepeatedlyDrainsBothSides`：默认 32 轮 INT8 KV/scale 请求，随机延迟 0～80 毫秒取消。`P2P_CANCEL_SEED` 默认 `20260915`；`P2P_CANCEL_ROUNDS` 可设为 1～1000。日志记录 seed、轮次、延迟及取消/正常完成数量。

阶段暂停仅作用于模拟计算和层发布，调度器继续处理取消。每轮检查已启动的 P/D 生成、StartLoad 和传输派发 RPC 退出，两侧调度器清空、非保留 block 引用归零、空闲块恢复；随后发送正常请求并逐字节校验缓存。固定阶段必须返回取消；随机用例允许请求先正常完成，并单独计数，且必须实际发生过取消。

取消用例的加载预算至少 90 秒，清理检查最多等待 30 秒，避免依靠普通请求超时通过。测试同步与轮次控制使用单调时钟。新增控制协议要求 P/D 使用同一版二进制。

远端执行时选择对应 `--test_filter`，可追加 `--test_env=P2P_CANCEL_ROUNDS=100 --test_env=P2P_CANCEL_SEED=20260915`。跨机模式每个用例单独启动 P。以上新增用例尚未运行，不属于下方历史验证结果。

## 2026-09-15 验证记录

代码版本 `2663f60457`，111 的 `yzh` 容器编译，产物由 111 SCP 到 112。CUDA 13.2 环境在 `sm9x`、`cuda12_9` 后追加 `cuda13`、`sm10x`，并使用远端独立 `arch-config-rdma` 覆盖配置启用真实内源 RDMA backend。

| 验证项 | 环境 | 结果 |
| --- | --- | --- |
| 缓存层 non-PD / PD / allocator 回归 | 111，`yzh` | 4 项通过 |
| Python gRPC 错误处理 | 111，`yzh` | 6 项通过 |
| non-PD GenerateStreamCall 入口回归 | 111，同机双进程 | 10 种请求组合 × 3 轮通过 |
| TCP 完整 payload smoke | 112，同机双进程，GPU 0 | 5 用例 × 3 轮通过 |
| RDMA 完整 payload smoke | 112，同机双进程，GPU 1，`ACCL_USE_NICS=mlx5_2` | 5 用例 × 3 轮通过 |

每轮正例逐字节检查 FP16 的 7680 字节、INT8 KV/scale 的 4800 字节；同时验证 token、故障检测和缓存释放。这里的 GPU 编号指 `CUDA_VISIBLE_DEVICES`，进程内设备编号均为 0。

归档目录：111 的 `/home/yanzhan.yzh/p2p-payload-20260915/build_logs/`。`p2p_portfix_build.log` 为构建日志，`payload_112/{tcp,rdma}_stability112.log` 为三轮记录，对应 `.signal` 均为 0。测试运行副本位于两台机器的 `/dev/shm/yzh-p2p-payload-20260915/runtime-rdma-2663f60457/`。

跨机验证尚未完成：112 时钟比 111 慢约 46 秒，导致绝对 deadline 在 P 端已过期；尚未校准宿主机时钟。默认多网卡的 RDMA 首轮也出现超时，单网卡通过不能证明其他网卡可用。112 根盘写满后，重跑日志改存 `/dev/shm` 并归档回 111。
