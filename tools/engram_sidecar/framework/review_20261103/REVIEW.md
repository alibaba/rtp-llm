# 分布式 Engram 代码清理与验证（2026-11-03）

基于 `c18577c4e`（包含 Harness 最新 `91f7fda48`），继续用户要求的 worker 自持分片方案。
保留两个现有启用入口：decode 的 `ENGRAM_SHARD_COUNT`、跨实例的 `RTP_ENGRAM_COHORT`。

## 已修正

| 原问题 | 修改及影响 |
|---|---|
| 发现 `/etc/ganginfo` 就可能启用共享 | 只有上述两个 env 能启用；辅助配置/文件不会成为开关 |
| gang 已有成员表，却用 hostname 猜地址、重新经 Store 汇集 | 直接使用 `DistributedServer.WorldInfo`；删除重复发现协议与固定 topology keys |
| 显式 IP 列表会丢失多 host gang 的局部 rank 关系 | 成员归属和本 gang 的 rank 偏移分别计算，覆盖两个实例各跨两个 host |
| 单 gang 多 host 仍开临时 RPC | 直接在已有 Store 汇集/广播分片；不读 TLS 文件、不绑定额外端口 |
| 用户可配置 generation，却不能可靠代表启动轮次 | 删除该 env 的读取；Store 内部按 rank 计数隔离重复启动，平台不用知道计数/UUID |
| prefll 仅配 COHORT，未完成 exchange 时可能走本地加载 | startup 和 model attach 共用启用判断；配置共享但没有 package 时显式失败 |
| 服务仍能通过旧 descriptor 文件跳过自发布 | 移除服务路径的文件输入模式；旧 env 报错；离线探针通过显式测试注入提供 synthetic descriptor |
| 旧 `require_eager_engram` 别名暗示与实际 Graph 行为不符 | 删除别名，调用方统一 `require_engram_runtime` |
| 部分 CPU Bazel 测试缺少真实模块依赖 | 增加窄 bootstrap contract library，补齐 publish/model attach/exchange 依赖 |
| 对接文档混杂旧 owner、holder、UUID 和 publication_root 示例 | 重写 `STATIC_RPC.md`，区分启用开关、拓扑来源、真实连接数和未验收项 |

成功后只有 O(world_size) 个内部启动计数留在 Store；package/descriptor/ACK 清理。
该计数只解决本 gang Store 的旧启动数据隔离，不是跨服务认证，也不是运行时故障恢复。
单 rank 重启有界失败；完整恢复需重建共享组及其 gang Store。

## 验证

1. 本机 Python 3.10：160 项通过，覆盖 exchange / publish / descriptors / capacity /
   manifest / shard / shard config / shard storage / storage / model attach。
2. 启动及 HTTP 性能工具回归：52 项通过（`test_full_model_http`、`test_sharded_pd`、
   `test_http_decode_benchmark`）。这些是工具单测，不是实际吞吐测试。
3. GB200 a0004 现有个人容器 `serina.wzq_engram_rdma` 的隔离目录
   `/tmp/engram-review-20261102`：36 项通过（33 exchange + 3 real TCPStore）。
   实际使用独立 TCPStore 客户端连接验证：不同 key 的 wait 不要求 collective；
   150ms 等待按调用超时；旧 package 不被新启动使用；单 gang 汇集无需 RPC。
4. `backend_manager.py` 与 synthetic CUDA probe 的 Python 编译检查通过；`git diff --check` 通过。

远端命令：

```bash
python -m unittest tools.engram_sidecar.framework.test_startup_store \
  rtp_llm.models_py.modules.dsv41.test.test_engram_exchange -v
```

本轮没有重启现有完整模型或修改 Whale 部署。远端测试使用真实 TCPStore、模拟 allocation，
不据此声称重新通过了跨 host CUDA 映射、全模型 chat、Graph 或 decode <100us 验收。
Bazel BUILD 依赖已修正，但本轮没有运行完整 Bazel 镜像编译。

## 仍有的限制，不能用删校验掩盖

- checkpoint 索引哈希只描述元数据，不是完整权重内容身份；部署必须提供同一不可变 checkpoint。
- 外部存储 async prefetch 仍关闭；其异步生命周期及性能应单独验证，不能仅为减少 env 而打开。
- Graph 保留现有 process-lifetime pin 与 decode 范围约束，未放宽未经验证的模型执行路径。
- 跨 gang 当前每存储 host 一条启动连接；单 host 实例为 N−1 条，多个 host 的实例为 P−1 条。
  后者尚未合并成每 gang 一条；在线 lookup 都没有 RPC/Store 操作。
- 不提供运行时成员补片、自动缩组或故障节点无缝替换。

实际部署字段和拓扑约束见 [STATIC_RPC.md](../STATIC_RPC.md)。
