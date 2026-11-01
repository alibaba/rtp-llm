# N 个实例静态共享 Engram

2026-10-30 澄清：配置 N 个 RTP 实例，它们一起启动，每个实例都参与持有分片，共享一份完整的 Engram。运行中不加入新实例，不改变成员，也不在线重新分片。

- **N**：共享组的模型实例数。
- **P**：共享组的存储物理 host 总数；不是实例数，也不是 TP/DP/EP。
- 每个实例持有分配给自己的分片，同时 GPU 映射共享组全部分片。
- 一次启动绑定同一成员表、模型版本和 generation；改变成员或 P 必须重新启动一组。

例如 N=2，每实例两个 host，P=4：

```text
实例 0: host A 持有 shard 0，host B 持有 shard 1
实例 1: host C 持有 shard 2，host D 持有 shard 3

实例 0 的所有 worker ─┬─ 映射 shard 0/1/2/3
实例 1 的所有 worker ─┘
```

总计一份 Engram，分布在四个 host。实例 1 也持有自己的一部分，不是后加入的纯消费者。

## 一组 IP 是否足够

**可以把地址列表作为用户主要输入，但需要保留实例分组。**

单 host 实例可以写成 `[IP_A, IP_B]`，对应 N=2；多 host 实例应写成 `[[IP_A, IP_B], [IP_C, IP_D]]`，对应 N=2、四个 host。若端口统一，可作为一次公共配置；否则地址使用 IP:port。

只给扁平的四个 IP，无法判断它们属于一个四 host 实例、两个两 host 实例，还是四个单 host 实例。也不能只凭 IP 判断它们是否真的同属可共享的 NVLink/IMEX 域。

程序需要在启动时交换成员身份、启动 generation、模型/布局信息、分片范围和 Fabric handle。模型版本和资源权限来自既有部署配置，generation 由本次组启动统一生成，不需要用户手填 handle。地址用于建立控制连接；在线查表依然由 GPU 读取映射内存。

## 当前代码已实现什么

`distributed_plan.py` 新增 `instance_count`，并检查：

1. N 与完整实例列表一致，实例 ID 唯一。
2. P 个存储 host 各自有明确所属实例。
3. N 个实例每个都至少持有一个 shard；拒绝纯消费者及组外 holder。
4. 输出每个 holder 的 instance_id 和各实例的 held_shards。

现有实现是一 host 一 holder。两个实例声明同一存储 host 时，需要用该 storage_host 的 instance_id 明确归属；不会自行重复加载。每个实例都须有分片，所以在这个实现限制下需要 P≥N。同机更多实例各持独立分片的布局尚未实现，不能把它混称为已支持的物理 host 分片。

当前已实现[一次性 RPC 交换](STATIC_RPC.md)：实例内复用现有 Store，实例间仅 rank 0 建立 N−1 条临时 TLS 连接，自动交换 handle，全部代表 ACK 后关闭监听。设置 `RTP_ENGRAM_COHORT` 等环境变量接入；离线 bundle 路径已移除，服务不写任何 bundle 文件。没有新 sidecar 容器，也没有完整表的中心存储服务。

## 启动与退出约束

```text
启动 N 个实例对应的 P 个分片持有进程
  → 各自加载本地分片
  → 收齐同一 generation 的 P 片并检查完整性
  → 每个 RTP worker import/map、校验 GPU 读、warmup/Graph
  → 各实例达到模型就绪条件后接流量
```

“同时启动”指同一批固定成员共同启动，不要求每个进程同一时刻执行。慢成员可以等待；超时或身份不一致必须失败，不能偷偷缩减 N/P。

退出必须先停止所有使用该组的模型 worker、排空 GPU 和 Graph，再释放任一持有分片的 backing。一个实例持有的分片退出，会影响依赖它的其他实例；当前没有自动补片或缩组继续服务。

初次静态成员配置变更的历史验证（本轮新增 RPC 验证见 STATIC_RPC.md）：`test_distributed_storage` 12/12 通过；framework 本地回归共 223 项，其中 209 通过、14 项因本机缺少 Torch 跳过。本次没有新增跨机模型或性能测试结果。
