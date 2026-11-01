# Engram 分布式存储审查（2026-10-30）

审查基线：`engram-sharding-wip`，`f1b723d5f`；release 基线为 `feature/dsv41-release-20261019`。本文中的 P 是**存储物理 host 数量**，不是 Prefill，也不是模型实例数、GPU 数、TP/DP/EP。

> 后续澄清：采用 [N 个实例静态共享组](STATIC_INSTANCE_GROUP.md)，全部实例共同启动并持有分片，纯消费者扩展不属于当前部署方案。下文的低层多消费者映射测试仍用于验证独立进程资源生命周期。

## 结论

原实现已有跨 host 分片及多进程映射，但不能据此认定新要求已经完整验收：P 环境变量未接入运行路径，启动样例固定四节点，异常清理和分片身份校验存在缺口。当前修改补齐这些控制与生命周期问题，并移除取消的单点部署入口。生产发现、嵌入业务启动和故障恢复仍未实现。

| 要求 | 审查结果与本次处理 |
|---|---|
| 单实例、多 host 分布式存储 | 每个 worker 导入同一完整分片集合；补 P=2/3/5 的 CPU 接入和清理测试。此前实机 D 为两 host 单实例 TP1/DP8/EP8。 |
| 多实例、多 host 共享 | 各实例引用同一 bundle/generation/allocation 集合；关闭一个 consumer 不释放其他实例的映射。新增多实例、多 worker UT。此前两独立 P 实例共享一个 P=2 分片组。 |
| P 可配置 | `ENGRAM_SHARD_COUNT` 接入模型 bootstrap；不指定时从完整 manifest 推导，指定时严格一致。新通用启动计划接受独立的 `storage_host_count` 和 `instances`。 |
| 取消 sidecar 部署 | 删除无调用的 session registry、已取消的整表 owner CLI，以及内源旧 sidecar 专用镜像 CI/Dockerfile；不创建 sidecar 容器、不选主、不下发 RPC endpoint。 |
| 取消单点存储 | 模型外部存储入口拒绝 P=1；分片 holder/bundle CLI 只接受 P≥2。本机 pinned A/B 路径与单块 VMM 原语测试保留。 |
| P 的边界 | 当前 manifest ABI 支持 2–64；且每层每个 shard 必须有非空真实数据。P 不要求 2 的幂。P 调整需新 generation、重新加载/映射和重建 Graph。 |

注意：此前 P 组和 D 组分别持有自己的两分片集合，不能称为 P/D 全部共用唯一 backing。新的通用计划支持多个实例指定同一组；是否实际使用同一组，应核验 generation 和 allocation ID，不能只看路径或内容相同。

## 已发现并修正的问题

1. **导入引用泄漏**：`ShardedLayerMapping.close()` 丢弃 handle 列表但未 `cuMemRelease`。现在逐个释放；对中途失败保留准确剩余资源，允许重试。
2. **授权失败遗留映射**：`cuMemMap` 成功、`cuMemSetAccess` 失败时，旧代码未记录已映射区间。现在分开记账，授权失败也会排空、unmap、release、free。
3. **悬空 tensor / GPU 访问**：旧分片路径没有 storage lease 检查及 GPU drain。新增 CUDA-array-interface exporter lease，关闭前拒绝活跃 view，并同步 consumer device。Graph 仍遵守原来的 process-lifetime pin；不支持运行中强制 detach。
4. **P 配置无效**：旧配置只有独立 UT，未被模型调用，且把分片数等同于实例数。已移除 `MODE/INDEX/ENDPOINTS` 假接口，旧环境变量出现时报错，避免静默忽略。
5. **空分片和非法几何**：对齐后最后一些 shard 可出现 begin>end；block=0 会除零；负 index 任意值都被当纯消费者。现在提前校验并拒绝，覆盖非 2 次幂 P。
6. **分片混用**：旧 bundle 不校验 descriptor 的 model revision、物理 host、index 与行范围对应关系，也把列宽写死 256/8。v2 descriptor 包含 shape 和 host_id，严格校验完整性、布局、版本、hash ABI、唯一 allocation/handle。
7. **抽样遗漏 host**：只抽整表头、中、尾可能完全不读取部分 shard。新增所有分片边界抽样。
8. **预算对象错误**：旧 holder 用整表大小检查单 host 预算。现在 checkpoint 元数据检查与本 host 的 padded shard 预算分离，分配前核对所有本地 allocation 总量。
9. **只读源文件状态漏检**：实机回归发现权限修改可能未被粗粒度文件时间戳捕获。读取映射现在直接复核文件权限、UID 和 hardlink 数。
10. **构建遗漏**：旧内源专用镜像任务还引用已不存在的 sidecar/registry 脚本，已删除该任务及 Dockerfile，通用镜像 job 保持不变。删除 registry 的 Bazel target；将分片模块和 UT 补进 CPU runfiles/test target。

## 当前启动方式

`tools/engram_sidecar` 是历史目录名称，不表示继续部署 sidecar。分布式 holder 是每个存储 host 上持有**本 host 分片**的进程；没有独占完整 Engram 的中心 owner。

通用入口：

```bash
python3 -m tools.engram_sidecar.framework.distributed_plan --config topology.json > launch-plan.json
```

示例配置：

```json
{
  "instance_count": 2,
  "storage_host_count": 3,
  "generation": "84f39bf1-73da-4ea1-a667-9c6959341824",
  "checkpoint": "/models/dsv41",
  "model_revision": "operator-verified-revision",
  "publication_root": "/run/engram",
  "storage_hosts": [
    {"host_id": "SN-A", "device": 0, "numa": 0},
    {"host_id": "SN-B", "device": 0, "numa": 0},
    {"host_id": "SN-C", "device": 0, "numa": 0}
  ],
  "instances": [
    {"instance_id": "instance-1", "hosts": ["SN-A", "SN-B"]},
    {"instance_id": "instance-2", "hosts": ["SN-C"]}
  ]
}
```

输出包括 P 个 holder 命令、一条 bundle 合并命令、每个实例所有 worker 要注入的环境变量。先在声明的 host 运行 holder，通过受信传输收齐私有 descriptor；再合并并将**同一份** bundle/manifest/capabilities 传到所有 consumer，最后按既有 RTP 启动配置启动模型。该计划生成器不 SSH、不创建容器、不自动起服务。

同一 host 上多个实例不需要多加载一份 shard。存储 host 集合可以与消费 host 集合重合，也可以不同；必须实际位于兼容的 NVLink/IMEX 共享域。host_id 是受信启动器提供的物理 SN，代码不能单靠字符串证明物理拓扑。

停机顺序：所有消费者停止接流、排空 GPU、销毁 Graph/退出，再允许对各 holder 创建 STOP。不能因某一个实例退出或控制连接中断回收共享分片。

## 升级注意事项

- v2 descriptor 增加 `host_id` 和 tensor `shape`，旧 v1 descriptor 不再接受。新启动需让 P 个 holder 用同一个新 generation 重新发布，并重新生成 bundle；不能只换 consumer 代码后继续使用旧 bundle。
- 新 bundle 抽样包含所有 shard 边界，旧的仅三行抽样文件可能不符合要求。本次测试在独立目录完成，没有替换或重启既有 P/D。
- A/B 切回实例本地 pinned 时同时取消 `RTP_ENGRAM_EXTERNAL_BUNDLE` 和 `ENGRAM_SHARD_COUNT`。设置 P 却漏配 bundle 会失败，不再静默本地加载。
- 显式传入的环境配置会原样传至 bootstrap，P 和容量策略使用同一份启动配置。

## 尚不能宣称完成的事项

- P>2 的真实跨 host 多实例模型、并发吞吐、长序列、CUDA Graph 全量回归尚需实机验收；CPU fake-driver UT 不能证明 NVLink、IMEX、驱动映射或真实性能。
- 当前 bootstrap 仍通过可信启动器分发私有文件；无自动发现、RPC、嵌入 RTP worker 的分片持有服务、平台自动保活。`sharded_pd.py` 只保留为历史四节点 PD 实验工具，不是通用生产启动器。
- 无副本/重建/在线 reshard：任何分片所在 host 故障都可能影响整个共享组。不要将分布式容量均摊等同于容错。
- 256 GiB 整组/128 GiB 单 allocation 的当前开发上限仍存在；checkpoint 检查和 consumer budget 均受它约束。不是无限规模存储。
- 当前 revision/generation 由受信启动器声明；每片记录完整 SHA256，但消费者只校验边界样本，不能证明所有中间行与预期 checkpoint 完全一致。若各 host 使用了不同权重却误标同一 revision，需要可信的分片 hash 清单或完整校验来兜底。
- 物理拓扑、容器设备权限、每个 HOST_NUMA 实际容量需要平台/启动器验证；配置 P 个不同字符串不能替代这些检查。
- GPU 硬件 READ 映射保护共享权重，但 Torch 接口仍呈现可写 tensor；所有调用方必须禁止原地修改。显式 close 与 enqueue 必须通过 storage 生命周期协议串行化。
- 本次清理保留历史实验结果和低层单块内存测试；旧文档中的中心 owner 命令仅是历史记录，不再是当前部署方法。

## 验证记录

本次在 GB200 开发容器的独立 checkout 中验证，Torch `2.11.0+cu130`，复用已有 native bridge，无运行时重新编译，也未覆盖 serving 库。

| 检查 | 结果 |
|---|---|
| 核心 CPU 回归（bootstrap、capacity、checkpoint、manifest、model attach、storage、shard/config/rollback、external checkpoint/warmup） | **204/204 通过，0 skip** |
| framework 工具回归（含新增 distributed storage/launch plan） | **219/219 通过，0 skip** |
| 实际 CUDA 生命周期回归 | **1/1 通过**；内部覆盖 P=2、P=3、两独立映射、真实 Tensor alias、禁止提前 close、三次 Graph replay、原指针不复制、import/release 平衡、native counters 回到初值 |
| 本地 portable build-path UT | **5/5 通过** |
| whitespace / Python syntax | 通过 |
| 内源镜像 CI YAML | 解析通过；确认仅移除 `image-engram_arm`，其余 job 定义完全一致；未触发镜像构建 |
| Bazel | 补齐 targets/runfiles，但未执行 bazel test，不宣称已通过 |

原始结果：[核心 CPU](review_20261030/cpu.txt)、[framework](review_20261030/framework.txt)、[CUDA](review_20261030/cuda.txt)、[机器可读摘要](review_20261030/summary.json)。CUDA 测试只证明同机真实映射与生命周期，不证明跨机性能或完整模型正确性。

可复跑命令：

```bash
python3 -m unittest discover -s tools/engram_sidecar/framework -p 'test_*.py'
python3 -m unittest rtp_llm.models_py.modules.dsv41.test.test_engram_shard \
  rtp_llm.models_py.modules.dsv41.test.test_engram_shard_config \
  rtp_llm.models_py.modules.dsv41.test.test_engram_shard_storage \
  rtp_llm.models_py.modules.dsv41.test.test_engram_storage
# prebuilt native module must be on PYTHONPATH; explicitly opt in on a dev GPU.
ENGRAM_RUN_CUDA_TESTS=1 python3 -m unittest tools.engram_sidecar.native.test_sharded_lifecycle_cuda -v
```

下一轮实机验收应使用 P=3/5，分别跑单实例多 host 与两个实例共享同一组：实际 chat 结果、本地 pinned 对照、并发 Graph replay、关闭一个实例后另一个继续工作、分片 host 故障后的摘流/恢复行为。未完成前不将其标记为生产验收完成。
