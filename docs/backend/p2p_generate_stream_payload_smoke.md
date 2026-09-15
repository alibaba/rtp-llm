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

## 运行入口

目标：`//rtp_llm/cpp/model_rpc/test:p2p_generate_stream_smoke_test`，标记 `manual`，不自动加入现有 CI 测试集合。编译及运行须遵循仓库 `test-execution` skill，在指定远端环境执行。

默认使用 TCP，仍需要 CUDA 设备和可用本机监听端口，不需要模型权重。

RDMA 使用同一目标及同一套断言，向测试传入：

- `P2P_SMOKE_TRANSPORT=rdma`
- `P2P_SMOKE_HOST=<本机 RDMA 网络接口 IPv4 地址>`

RDMA 需要启用内源 backend 的构建、RDMA 网卡和对应驱动。后端不支持或初始化失败会直接失败，不回退 TCP。同机 RDMA 测试还要求后端支持同机端点互传。

这是传输完整性测试；生成 token 是确定值，不校验模型推理精度。


## 跨机运行

先在两台远端环境编译上述同一 target。按 `test-execution` 流程预检、运行，平台配置使用 `sm9x`、`cuda12_9`。下面列出两端各自需要的测试参数（追加到 `bazelisk test <target>`）；选择同一 transport，使用各自机器的可达网卡 IP。

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

控制端口 29800、两端动态分配的 gRPC 和传输端口需互通。两台机器时钟需同步，生产 P2P 使用绝对 deadline。GPU 选择遵循各自远端运行环境的 `CUDA_VISIBLE_DEVICES`；测试内部使用可见设备 0。
