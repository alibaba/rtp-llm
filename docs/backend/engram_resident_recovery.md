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

配置 `RTP_ENGRAM_RESIDENT_TRANSPORT=tcp` 后，每个 Owner 直接提供自己的描述，默认端口 29484；没有中心协调进程，不写共享盘，不复制 ready.json，也没有文件心跳。权重数据仍通过 CUDA VMM/Fabric 访问，TCP 只用于启动连接和身份校验。

每个容器仅保留自己的私有本地文件：

- 每代独立的认证密钥文件，权限 0600；通过部署凭据机制分别下发，同组内容一致。环境变量只传路径，不传密钥。
- Supervisor 创建的临时状态目录，保存 generation 占位和描述摘要，供本容器重启后的 Worker 校验 allocation 未被替换。目录不与其他容器共享，不保存 Fabric 访问句柄。

请求和响应均认证加密；随机 challenge 防止旧响应冒充当前存活 Owner。generation、模型版本、分片、描述和本地摘要均校验；错误不降级为重新加载或覆盖原权重。网络请求有大小、并发和时间限制；错误日志限频，避免错误请求刷屏。

## 配置

| 项目 | 配置/约束 |
|---|---|
| 容器内恢复 | `RTP_LLM_IN_CONTAINER_RESTART=1` |
| 启动 Owner | `RTP_LLM_RESIDENT_OWNER_ARGS=--discover --device 0` |
| 网络发现 | `RTP_ENGRAM_RESIDENT_TRANSPORT=tcp` |
| 权重存储 | `RTP_ENGRAM_STORAGE=host_shared`；GPU shared 需对应配置 |
| 分片数 | `ENGRAM_SHARD_COUNT` 与同组 Owner 数一致 |
| 本代标识 | `RTP_ENGRAM_RESIDENT_GENERATION`：同组相同 UUID，换代必须更换 |
| 模型版本 | `RTP_ENGRAM_MODEL_REVISION`：同组一致的不可变版本 |
| 认证 | `RTP_ENGRAM_RESIDENT_KEY_FILE`：本地绝对路径；每次换代换密钥 |
| 地址 | 默认复用 gang 成员发现；可显式设置 `RTP_ENGRAM_RESIDENT_ENDPOINTS=host:port,...`，同组顺序一致 |
| 网络 | 同组可达 Owner 端口；容器时钟需同步以验证短时效令牌 |
| 模型 | 复用现有 checkpoint；Supervisor 负责远端模型挂载并向子进程传本地路径 |

TCP 模式不得同时设置旧的 `RTP_ENGRAM_RESIDENT_DIR`。`RTP_ENGRAM_RESIDENT_STATE_DIR` 由 Supervisor 自动生成，无需 ASI 配共享盘。

## 故障与责任边界

| 故障 | RTP-LLM 行为 | ASI / C2 责任 |
|---|---|---|
| 内层 Worker / rank SIGKILL、SIGTERM、崩溃 | Watchdog/进程管理检测退出；清理本实例进程树，原地重启；Owner 不动 | health 失败照常摘本实例；Supervisor 存活时不得因此回收容器或重建 gang |
| 业务无响应，Watchdog 能判定 | 终止本实例 Worker 后重启；无法清理的进程不重叠启动 | 同上；不能将单实例不健康升级为整组替换 |
| Worker 专用 cgroup OOM | Worker 被杀后恢复；Owner 在该 cgroup 外 | 需委派独立 worker leaf、memory.max、memory.oom.group=1；必须实际联调 |
| 整容器/节点 OOM、Owner 被杀、GPU/Fabric 故障 | 不宣称单实例可恢复；本地 Owner 死亡后停止本地 Worker，入口存活并报 recovery_blocked | 按共享权重依赖范围摘流、换代；不能把此类故障归为普通 Worker 重启 |
| Owner 网络暂时不可达 | 已导入权重的实例不依赖 TCP 持续服务；新 Worker 在启动期限内重试并保持不健康 | 保持容器；定位网络；不要仅因启动重试重建健康 Owner |
| generation/认证/描述不一致 | 拒绝导入；保留旧 Owner，报告验证失败 | 修复部署配置；需要换代时统一安排依赖组升级 |
| Supervisor / 容器退出、节点丢失 | 无法保证该容器 Owner 存活 | 作为共享依赖故障处理；入口存活策略不能覆盖 SIGKILL Supervisor 或节点故障 |
| 计划升级/整组停止 | 收到平台停止信号后停止 Worker，再释放 Owner | 新组完整预热并接流后摘旧组；最后退出旧组。禁止逐个销毁仍被旧组使用的 Owner |

重启失败会退避并持续不就绪。Owner 丢失时不在旧 generation 下重建 allocation。模型挂载或 Owner 启动阶段失败时记录 bootstrap_failed 并保持入口进程，等待显式平台停止。入口参数格式校验失败会在创建 Owner 前退出，需在发布前检查配置。

## 必须留的证据

日志包含时间、PID、generation，以及 Owner 分片/地址或 Worker 退出码；连接日志记录耗时、重试和失败类别，描述只记录 SHA256。禁止记录认证密钥或 Fabric 句柄。

验收每次记录：容器 ID/重启计数、Supervisor/Owner/Worker PID 与启动时间、generation、各分片摘要、health 变化、重启耗时、故障前后真实推理结果、其他实例连续请求成功率。对实际注入的信号单独记录目标 PID、身份校验和时间。

模拟测试只能验证控制流程；真实权重、跨容器推理、cgroup OOM、平台不回收容器和新旧组切换必须分别给出实测结果，不能互相代替。
