# P2P 失败用例复验（2026-09-15）

## 环境

外源修复：`b20bdc9d50`、`215b99b6e1`、`1ec8b53ec4`，已推送原分支 `codex/dsv4-block-tree-p2p-region`。内源使用原验证基线回移版本 `8546cf5069`，非完整最新分支。

111/112 的 `yzh` 容器，用户 `yanzhan.yzh`，GPU 0；UT 在 111，双机 P111 / D112。沿用 CUDA 13.2、`sm9x cuda12_9 cuda13 sm10x`、`arch-config-rdma`、原远端缓存配置，经 test-execution 预检与原包装器运行。

## 修复

- 取消 RPC：结果析构会取消尚未送达的取消 RPC；保留至全部 Finish，仍受原 timeout 约束。检查送达与对象释放。
- 缺层错误补回 route 诊断。
- Fixture：修正路由、worker 重建、RPC 计数、宏及 model RPC 测试前提。
- Lease：在锁下读取回收后引用；检查最终完成计数及 map 清理，保留迟到写入、重复回调和逐字节复用保护。
- Timeout：正值实际等待超时，负值检查参数错误；B3 检查传输首因。
- RDMA 随机取消曾在第 94 轮读到错误 scale 字节。测试的 pageable H2D 拷贝可能在 GPU DMA 完成前返回，现补默认流同步后再发布 layer-ready。[CUDA 同步语义](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

## UT：353 项通过

A=b20bdc9d50，B=215b99b6e1；首轮通过目标未重跑。

| 目标 | 通过数 | 版本 |
| --- | ---: | --- |
| components_test | 95 | A |
| p2p_connector_test | 27 | A |
| p2p_connector_scheduler_test | 44 | B |
| p2p_connector_worker_test | 52 | B |
| p2p_connector_worker_decode_lease_test | 23 | B |
| p2p_model_rpc_test | 78 | A |
| tcp_transfer_service_test | 13 | A |
| rdma_transfer_service_test | 21 | B |

## 双机

全部通过：1ec8b53ec4，TCP 100 / RDMA 300 轮随机取消，seed=20260915，另各跑三个阶段取消。检查 RPC 退出、引用释放及后续请求缓存字节。

111→112 SCP，两端 SHA256：
`5dac4568e6a396d6a583929eda03d3140d46b18324c1c5e659c4a61b7f0d8cbc`。

## NIC / GID

两端设置 `ACCL_USE_NICS=mlx5_bond_0`。Barex 扫描 GID，RoCE 选 v2、过滤 fe80，保留同类最后一个候选；按控制连接地址族优先，缺少则回退。本次 IPv4 控制连接回退 IPv6 GID，未设置 `ACCL_FORCE_IPV6`。

| 机器 | port / index | RoCE v2 GID |
| --- | --- | --- |
| 111 | 1 / 3 | fd03:4516:200:7f40::1 |
| 112 | 1 / 3 | fd03:4516:200:3480::1 |

本地 sgid_index 独立选择，对端 dgid 从握手取得；index 无需相同。

## 证据

111 工作区 `/home/yanzhan.yzh/p2p-payload-20260915`，`build_logs/` 下完整命令及结果：

- `repair_failed_ut_b20bdc9d50.log`：首轮。
- `repair_remaining_ut_215b99b6e1.log`：剩余目标通过。
- `payload_h2d_sync_1ec8b53ec4.log`：重建通过。
- `cross_cancel_1ec8b53ec4.log`：双机汇总；工作区同名 `.py` 为驱动，参数 `tcp rdma`。

历史验证见[原报告](p2p_relative_timeout_validation.md)，不计入本轮。
