# 静态组启动 RPC 验证，2026-10-30

已验证：真实跨 host/跨容器 mTLS 交换 + 现有类型的 TCPStore + 生产启动适配器 + Fabric 分片映射 + CUDA Graph。不是完整模型 chat 或 decode 性能验收。

## 测试结果

| 测试 | 结果 | 原始记录 |
|---|---|---|
| 核心 metadata/storage/bootstrap/lifecycle CPU 回归 | 204/204 | [core.txt](core.txt) |
| framework CPU 回归 | 225/225，无跳过 | [framework.txt](framework.txt) |
| 新 RPC 契约 + 分布式计划测试 | 28/28（14 RPC + 14 plan/storage） | [exchange-and-plan.txt](exchange-and-plan.txt) |
| 两台开发 host、两个容器、N=2/P=2、每实例两 worker | 4/4 worker 通过 | [source.txt](source.txt)、[a0006.txt](a0006.txt) |
| 关闭 RPC 监听后 Graph replay | 每 worker 100 次，结果与参考一致 | 同上 |
| CUDA Driver 资源释放 | 每 worker imports/releases=4/4，maps/unmaps=4/4，reserves/frees=2/2 | 同上 |
| 分片持有进程退出 | 两端 CLOSED，native 资源计数恢复基线 | 同上 |

framework 的 225 项包含上述 14 项 plan/storage，不能再重复相加。新 RPC UT 使用真实 localhost mTLS；三个实例的用例验证恰好两条连接，每条复用 exchange + ACK。覆盖缺成员、缺 ACK、慢/失败 builder、错误指纹/冒充实例、混 generation、重复冲突提交、错误 P、实例内配置不一致、帧大小限制及本地缓存权限。

核心、framework 和新增测试在真实 Linux/CUDA13 Python 环境通过 `unittest` 执行。已增加 Bazel CPU target，但本轮没有运行 Bazel/CI，也没有构建镜像。

## 实际环境

| 实例 | SN | 容器 | 实测 hostname |
|---|---|---|---|
| i0 | e01-cn-fhh4jm45401-a0003 | serina.wzq_engram_rdma | tre-2-a1-l20a-ipv6-019-003 |
| i1 | e01-cn-fhh4jm45401-a0006 | serina.wzq_engram_handoff_20261029 | tre-2-a1-l20a-ipv6-019-006 |

复用已存在的个人开发容器，使用 GPU 0、NUMA 0。没有修改正在服务的 runtime，没有重启 P/D。独立实验目录、generation、端口见 [run.json](run.json)；被测代码摘要见 [source-sha256.json](source-sha256.json)。

每台持有 69,206,016 字节（66 MiB）的 weight/scale 合计分片；整表 132 MiB。四个 worker 映射的是同一组两个 host 的分片，dataset_id 一致。小表使用确定性字节模式；协调者从合成 safetensors checkpoint 生成预期样本，不把 GPU 读回的数据当作真值。

每个 worker：

1. 连接本实例 TCPStore；只有 rank 0 参加跨实例 mTLS RPC。
2. 自动得到本地私有 bundle，校验布局/完整分片和 hash ABI。
3. 确认临时 RPC 端口已关闭，再导入/映射各 host backing。
4. GPU 读取每片首尾行和全局中间行，与 checkpoint 样本摘要一致。
5. capture weight/scale gather，replay 100 次，核对结果。
6. drain、销毁图和视图、unmap/release/free；由真实 Driver 调用计数检查配对。
7. 两端全部 worker 都退出后，控制脚本才写测试专用 STOP，让 holder 释放 backing。

`startup_ms` 包含等待其他实例、Store、RPC、checkpoint 样本及写缓存，受进程启动先后影响；它不是单次 RPC 基准，更不是 decode 的微秒延迟。本轮不据此给出“<100 μs”结论。

## 复现入口

普通 CPU 测试：

```bash
python3 -m unittest \
  rtp_llm.models_py.modules.dsv41.test.test_engram_exchange \
  tools.engram_sidecar.framework.test_distributed_storage
```

CUDA 探针使用 `native/probe_static_exchange_cuda.py`。给两端准备 [STATIC_RPC.md](../STATIC_RPC.md) 所述 startup config，固定 N=2、P=2、每实例 world_size=2、publisher_rank=0、同一 generation 和证书指纹表；checkpoint 指向本次实验的**新空目录**，不要指向真实模型。两端 config 只包含本地 shard descriptor 路径。证书和 native 扩展按文档准备。

分别执行：

```bash
# host i0；PYTHONPATH 需包含当前仓库和预编译 native .so 目录
python3 -m tools.engram_sidecar.native.probe_static_exchange_cuda \
  --config /private/probe/config.json --shard 0 --store-port 29530

# host i1
python3 -m tools.engram_sidecar.native.probe_static_exchange_cuda \
  --config /private/probe/config.json --shard 1 --store-port 29531
```

必须等两端均打印 `PASS` / `consumers_exited=true` 后，才分别创建 config 所在目录下的 `STOP`。测试进程将打印 `CLOSED`。这只是探针清理协议，不是生产在线回收协议。不要将真实 capability、私钥或 bootstrap 缓存提交到 Git。

## 尚未证明的部分

- 新 RPC 钩子已接 BackendManager，但尚未通过这个钩子重启完整 P/D 模型并发 chat。
- 未跑真实完整表的端到端 decode/prefetch/TTFT 性能回归，不能证明额外访存 <100 μs。
- 本轮真实跨机为 N=2/P=2。N=3 的连接数及可变 P/N 有 CPU 契约测试，不等于大规模 GPU 实测。
- 不包括生产调度、网络域自动发现、holder 健康探测、动态扩缩容或节点故障恢复。
