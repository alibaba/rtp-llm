# Engram 常驻与容器内恢复

目标：Prefill 业务进程退出后，在原容器内重启业务；Owner 和已分配的 Engram 权重保持不变，其他实例继续使用原权重。此模式需要显式开启，不改变默认启动模式。

## 进程关系

```text
maga_start.sh
└─ python -m rtp_llm.start_server  [Supervisor，平台等待的入口]
   ├─ resident --discover         [Owner，独立进程组，持有一个权重分片]
   └─ rtp_llm.start_server        [Worker，独立进程组]
      ├─ backend
      └─ frontend / ranks
```

Worker 退出时，Supervisor 清除就绪标记，清理并回收该 Worker 的残留子进程，再按 1–60 秒退避重新启动。新 Worker 通过各 Owner 的只读网络接口重新获取同一代描述并导入原 Fabric allocation。启动及真实 warmup 完成前 health 不通过。

Owner 由 Supervisor 启动，与 Worker 是兄弟进程。杀 Worker 不会沿父子关系杀 Owner；Owner 不参加 Worker 的进程组清理。平台停止 Supervisor 时，按 Worker → Owner 的顺序退出。

## 不需要共享目录

开启容器内恢复后默认使用 TCP，每个 Owner 直接提供自己的描述，默认端口 29484；没有中心协调进程，不写共享盘，不复制 ready.json，也没有文件心跳。权重数据仍通过 CUDA VMM/Fabric 访问，TCP 只用于启动连接和身份校验。

Owner 读取 C2 下发的 `/etc/podinfo/annotations` 和 `labels`，并在读取请求前按 TCP 源 IP 拒绝组外连接。描述在 C2 隔离的网络内传输，平台须保留真实 Pod 源 IP 并限制组外访问 Owner 端口。Worker 校验请求挑战、模型版本、分片和本地分配摘要。C2 元数据缺失、成员不匹配或同组配置不一致时，实例保持不就绪。

Supervisor 在 `/dev/shm` 下建立私有状态目录，保存描述摘要供 Worker 重启后复核，Fabric 句柄只留在进程内。若 `/dev/shm` 不可用，回退到系统临时目录并打印告警，此时 `TMPDIR` 必须是容器本地目录。

## 配置

已有共享 Engram Prefill 配置上，只新增一个业务开关，**仅配置在 Prefill role**：

```bash
RTP_LLM_IN_CONTAINER_RESTART=1
```

现有 checkpoint、gang 成员、`ENGRAM_SHARD_COUNT` 和存储模式继续复用。
未显式指定存储模式、且已有分片数大于 1 时，本开关内自动选 `host_shared`。
Decode 和不开启该开关的部署不执行这些自动配置。

| 自动配置 | 行为 |
|---|---|
| Owner | 自动启动 `--discover --device 0` |
| 发现 | TCP，复用 gang 成员，默认端口 29484 |
| generation | 从 C2 gang 版本和有序成员表推导 |
| 模型标识 | 从 checkpoint 的 config 和索引生成稳定摘要；这是元数据标识，不是全量权重校验和 |
| 状态 | Supervisor 私有本地目录（默认 `/dev/shm`）；摘要跨 Worker 重启保留 |
| 挂载 | Supervisor 先解析启动参数，再挂载 checkpoint、tokenizer 和 draft 模型，向子进程传本地路径 |

同组必须可达 Owner 端口并使用相同的模型版本。旧消费者在线期间禁止就地修改权重。
设置空 `RTP_LLM_RESIDENT_OWNER_ARGS` 可以保留纯消费者模式。

## 故障与责任边界

| 故障 | RTP-LLM 行为 | ASI / C2 责任 |
|---|---|---|
| 内层 Worker / rank SIGKILL、SIGTERM、崩溃 | 进程管理检测退出；清理本实例进程树，原地重启；Owner 不动 | health 失败照常摘本实例；Supervisor 存活时不得因此回收容器或重建 gang |
| Worker 仍存活但业务无响应 | 保留现场，记录故障；本版本仅在 Worker 退出后自愈 | health 失败照常摘本实例；按故障原因处置 |
| Worker 专用 cgroup OOM | Worker 被杀后恢复；Owner 在该 cgroup 外 | 需委派独立 worker leaf、memory.max、memory.oom.group=1；必须实际联调 |
| 整容器/节点 OOM、Owner 被杀、GPU/Fabric 故障 | 不宣称单实例可恢复；本地 Owner 死亡后停止本地 Worker，入口存活并报 recovery_blocked | 按共享权重依赖范围摘流、换代；不能把此类故障归为普通 Worker 重启 |
| Owner 网络暂时不可达 | 已导入权重的实例不依赖 TCP 持续服务；新 Worker 在启动期限内重试并保持不健康 | 保持容器；定位网络；不要仅因启动重试重建健康 Owner |
| generation/成员/描述不一致 | 拒绝导入；保留旧 Owner，报告验证失败 | 修复部署配置；需要换代时统一安排依赖组升级 |
| Supervisor / 容器退出、节点丢失 | 无法保证该容器 Owner 存活 | 作为共享依赖故障处理；入口存活策略不能覆盖 SIGKILL Supervisor 或节点故障 |
| 计划升级/整组停止 | 收到平台停止信号后停止 Worker，再释放 Owner | 新组完整预热并接流后摘旧组；最后退出旧组。禁止逐个销毁仍被旧组使用的 Owner |

重启失败会退避并持续不就绪。单次 Worker 启动期限默认 7200 秒，可用 `RTP_LLM_PREFILL_STARTUP_TIMEOUT_S` 调整。新 Worker 完成 warmup 且每个本地前端开始监听后才重新就绪。模型挂载或 Owner 启动阶段失败时记录 bootstrap_failed 并保持入口进程，等待显式平台停止。Supervisor 在创建 Owner 前校验 full-world；model_type、warmup 和 worker cgroup 由 Worker 按最终生效配置复核。入口参数格式校验失败会在创建 Owner 前退出，需在发布前检查配置。

## 必须留的证据

日志包含时间、PID、generation，以及 Owner 分片/地址或 Worker 退出码；连接日志记录耗时、重试和失败类别，描述只记录 SHA256。禁止记录 Fabric 句柄。

验收每次记录：容器 ID/重启计数、Supervisor/Owner/Worker PID 与启动时间、generation、各分片摘要、health 变化、重启耗时、故障前后真实推理结果、其他实例连续请求成功率。对实际注入的信号单独记录目标 PID、身份校验和时间。

模拟测试只能验证控制流程；真实权重、跨容器推理、cgroup OOM、平台不回收容器和新旧组切换必须分别给出实测结果，不能互相代替。
