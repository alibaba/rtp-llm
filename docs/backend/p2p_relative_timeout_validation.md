# P2P 相对 timeout 验证（2026-09-15）

## 版本与环境

- 外源：`75155a03d2`，原分支 `codex/dsv4-block-tree-p2p-region`。
- 内源发布：`1f731af9eb`，原分支 `feature/rdma-deadline-guard-a1a2`。
- 111 验证使用原验证基线上的内源 backport `34fc2d3686`，并保留已有 CUDA 13 构建配置；不是完整内源发布分支的重新构建。
- 111/112 均为 `yzh` 容器、`yanzhan.yzh` 用户。111 构建后 SCP 产物到 112，未在 Mac 编译。
- CUDA 13.2；配置 `sm9x cuda12_9 cuda13 sm10x`，compute capability `10.0`；独立 `arch-config-rdma` 启用内源 backend。
- 二进制 SHA256：`4c46c262203b05e182df9c68aaff79a7149db902a946abc2905776cad2a217a5`。

## 结果

| 项目 | 结果 |
| --- | --- |
| 11 个相关目标构建 | 通过 |
| 10 个 UT 目标 | 4 通过，6 失败，详见下文 |
| TCP 跨机，P111 / D112 | 原五个 payload 场景全部通过 |
| TCP 同机双进程，112 GPU 1 | 五场景 × 三轮全部通过 |
| RDMA 跨机，P111 / D112 | 原五个 payload 场景全部通过 |
| RDMA 同机双进程，112 GPU 1 | 五场景 × 三轮全部通过 |

五场景：FP16；INT8 KV/scale；query 绕过 PD；损坏一个字节；缺失一层。正例检查真实缓存字节，反例检查错误和资源释放。新增两个专门取消 payload 用例不包含在这五场景内。

两机时钟未调整，111 比 112 快约 47 秒。跨机默认请求 10 秒/加载 3 秒的首例因 P 端 CUDA 拷贝与同步约 3.9 秒而失败；随后使用请求 20 秒/加载 10 秒，均小于时差。TCP/RDMA 各五场景通过，证明本次单 rank P/D 路径不再要求两机时间对齐。112 同机两种传输各三轮使用默认 10 秒/3 秒。跨机 RDMA 选择 `ACCL_USE_NICS=mlx5_bond_0`，同机 RDMA 选择 `mlx5_2`；两者均设置 `ACCL_CONTEXT_CQE=1024`，未设置 `ACCL_FORCE_IPV6`，没有回退 TCP。本次未验证跨机 TP 组内 deadline 广播。

## UT 尚未通过的项目

通过：`components_test`、`tcp_task_context_test`、`tcp_sender_receiver_test`、`rdma_sender_receiver_test`。包含完整时长发送、保留 D 本地截止时间、首次到达建立 deadline、终态不续期、非法时长拒绝。

失败：

- `p2p_connector_test`：两个旧用例把过去的绝对 deadline 转成负时长，实际返回参数/资源错误，断言仍要求生成超时。
- `tcp_transfer_service_test`、`rdma_transfer_service_test`：B3 首因/超时覆盖断言不匹配。
- `p2p_model_rpc_test`：block pool 配置、GetPeerInfo 预期及两个 batch 取消等待断言失败。
- `p2p_connector_scheduler_test`：17 个失败，涉及路由启动、取消广播及已释放 block 的引用检查。
- `p2p_connector_worker_test`：缺层错误文本及异步派发数量断言失败。

这些失败尚未全部定位或修复，不能宣称整套回归通过。`75155a03d2` 仅修复了阻断 payload 的测试引用读取：在 pool 锁内读取引用快照，避免对未分配 block 调用 `refCount()`，也避免分开检查分配状态产生竞争。

## 复现与日志

111 工作区 `/home/yanzhan.yzh/p2p-payload-20260915`。所有 Bazel 调用经 `test-execution` 预检和包装器；不使用会干扰其他任务的 GPU 锁脚本。

```bash
# 在 111 执行，包装器包含平台、远端缓存和独立 cache 配置
bash /home/yanzhan.yzh/p2p-payload-20260915/rebuild-validation.sh \
  relative_payload_build_r7 build \
  //rtp_llm/cpp/model_rpc/test:p2p_generate_stream_smoke_test \
  --override_repository=arch_config=/home/yanzhan.yzh/p2p-payload-20260915/arch-config-rdma
```

跨机实际启动参数在 111 的 `cross_payload_relative_75155a03d2_20s.py`；同机三轮在 `samehost_relative_75155a03d2.py`。前者通过 SSH 启动 D112，每个场景重新启动 P111。运行目录两端均为 `/dev/shm/yzh-p2p-payload-20260915/runtime-relative-75155a03d2`。

111 日志：

- `build_logs/relative_timeout_build_r6.log`：11 目标构建，signal 0。
- `build_logs/relative_payload_build_r7.log`：payload 读取修复后的构建，signal 0。
- `build_logs/relative_timeout_ut.log`：完整 UT 汇总，signal 3。
- `/dev/shm/yzh-p2p-payload-20260915/cross-relative-75155a03d2-20s/`：每场景 P/D 日志及结果 JSON。
- `/dev/shm/yzh-p2p-payload-20260915/samehost-relative-75155a03d2/`：同机三轮日志及 signal。

Cache：`/home/yanzhan.yzh/.cache/bazel_cuda13_p2p_payload_20260915_cache`。
