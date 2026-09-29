# KVCM/PACE P1 smoke

测试复用现有 Bazel、Python smoke、KVCM SDK RPM 和服务器包，不增加测试框架、模型、容器编排或服务部署依赖。新增两个 CPU 辅助程序：一个只链接 SDK，用于真实 PACE 字节和接口校验；另一个复用 RTP 的 Publisher、日志和 HTTP 组件，验证真实事件链路。GPU 用例复用已有模型及结果文件。

## 版本输入

`deps/kvcm.bzl` 的三个 commit 是唯一来源锁。通过 `--repo_env=KVCM_ARTIFACT_MANIFEST=/absolute/path/MANIFEST.json` 选择已经构建好的配套制品，`KVCM_CLIENT_VARIANT=cpu` 或 `cuda` 选择客户端。测试入口不会构建 SDK、发布制品或启动 PACE 服务。

清单格式兼容测试环境交付的 `MANIFEST.json`。顶层 `source_id` 和每个制品的 `source_id` 都必须等于 `internal_commit:opensource_commit:pace_commit`；所选客户端和服务器各只能有一条记录，必须带真实 URL 和 SHA256。例如：

```json
{
  "source_id": "<internal commit>:<opensource commit>:<pace commit>",
  "artifacts": [
    {"variant": "cpu", "source_id": "<same source_id>", "url": "<CPU RPM URL>", "sha256": "<SHA256>"},
    {"variant": "cuda", "source_id": "<same source_id>", "url": "<CUDA RPM URL>", "sha256": "<SHA256>"},
    {"variant": "server", "source_id": "<same source_id>", "url": "<server tar.gz URL>", "sha256": "<SHA256>"}
  ]
}
```

Bazel 校验下载哈希和服务器包内的 `KVCM_SOURCE_ID`。运行 smoke 时再校验两个仓库的来源标识和客户端类型，拒绝旧 RPM、新头文件与旧库混搭，以及将 CPU RPM 用于 GPU 用例。

## 外部 PACE fixture

通过 `KVCM_PACE_FIXTURE=/absolute/path/pace-fixture.json` 提供环境。fixture 中的 `source_id` 声明外部 Meta/Provider/Consumer 使用同一版本组合；此声明需要对应的部署记录，Provider 状态接口本身不能证明远端服务的 Git commit。

```json
{
  "source_id": "<same locked source_id>",
  "domain": "http://<PACE Meta>:<port>",
  "provider_status_urls": ["http://<Provider>:<port>/api/status"],
  "ssd_enabled": false,
  "client_env": {
    "TAIR_MEMPOOL_ENABLE_TENT": "1",
    "MC_TENT_CONF": "{\"transports\":{\"tcp\":{\"enable\":true},\"aft\":{\"enable\":false},\"rdma\":{\"enable\":false},\"barex\":{\"enable\":false},\"shm\":{\"enable\":false}},\"policy\":[{\"name\":\"tcp_default\",\"segment_type\":\"memory\",\"transports\":[\"tcp\"]}]}"
  }
}
```

调用方需提供可用的 PACE Consumer，并保证测试进程能访问其 IPC/共享内存及 Meta/Provider。只有 CPU RPM 和 CPU/TCP Consumer 的契约 smoke 不需要 GPU；裸 `dev` 上仍需具备这些依赖。SSD 用例还要求 Provider 已启用 SSD、fixture 显式设置 `ssd_enabled=true`。

后续执行仅限另行授权的 Linux 开发/验收环境。按用户指定的执行约束，该环境须预先提供可写的 `/User/gray/tmp`，使用 `bazel --output_user_root=/User/gray/tmp/rtp-smoke ...` 将编译产物及制品放在允许的目录；PACE smoke 会在启动前核对服务器和辅助程序的实际路径。这里的 `/User` 是用户指定的 Linux 执行目录，不是本机 macOS 的 `/Users`；这些 Linux RPM/二进制不在 Mac 执行。CPU 目标使用 `--config=cpu`；SDK helper 单独采用配套 CPU RPM 的 C++11 string ABI，Publisher 继续跟随 RTP 构建配置。

smoke 从锁定的服务器包启动自己的 KVCM 进程，使用独立工作目录、仅内存协调、PACE 启动配置，以及独立 Instance Group 和事件 storage。每组远端配额上限为 1 GiB，这是写入限额，不是新增的 1 GiB 内存预留。清理只停止该 KVCM 进程，不操作外部 PACE 服务。缺 fixture、配置失败或故障注入未安装都会使 PACE 用例失败，不回退 NFS。

## 入口与覆盖

| 入口 | 用途 |
|---|---|
| `//rtp_llm/test/smoke:smoke_kvcm_p1_cpu` | CPU SDK + PACE DRAM 字节、元数据和事件协议 |
| `//rtp_llm/test/smoke:smoke_kvcm_p1_cpu_ssd` | 相同契约，使用 PACE SSD、媒体类型 5 |
| `//rtp_llm/test/smoke:smoke_kvcm_p1_gpu` | RTP 模型 PACE 路径和已有 RTP 组件回归 |
| `//rtp_llm/test/smoke:smoke_kvcm_p1_gpu_ssd` | RTP 经 SSD backend 查询并读取缓存 |

CPU 和 GPU 入口分别选择对应 RPM，分两次执行并记录相同的 `source_id`。通用模型 smoke 保留原入口；上述专用入口需要在 P1 验收任务中显式调用，不因添加了测试目标就视为 CI 已验收。

GPU 入口还需要选择与 CUDA RPM 兼容的 RTP/CUDA 构建配置，并在平台配置之后启用 `--config=remote_kv_cache`。CUDA/驱动/ABI 兼容性以制品构建记录为准；现有测试环境的 CUDA RPM 尚无 GPU 运行验收，不能凭来源锁认为所有 CUDA 平台都可用。

| P1 能力 | 校验位置与断言 |
|---|---|
| 配套 commit、真实 PACE SDK、DRAM/SSD | 仓库制品门禁 + fixture 校验 + CPU 字节校验；每条 URI 必须是 `pace://` |
| 多 IOV、多 spec、不同 pool 大小 | CPU helper 同时写 FULL 256 KiB 和 STATE 64 KiB，每块分成两个非等长 IOV，逐字节读回 |
| batch/prefix/SWA/Mamba | CPU helper 校验长度、SWA 窗口、最后完整 Mamba checkpoint；RTP 组件正例检查 SWA 参数传递及 FULL+SWA 的完整窗口复用；GPU batch 和 hybrid TP2 用例 |
| Instance 默认 query type | CPU 注册 1/2/3/4 并检查服务端记录、未显式指定 query 的结果及缺失中间 key 后的不同语义；GPU batch 使用 RTP 默认配置 |
| MatchLocationLen/MatchMeta/RemoveCache | CPU 正例、删除后位置保留；RTP 组件正例检查命中格式化、token/detail/spec 传递、bool/offset mask；GPU 经 ExecuteFunction 检查 miss 和非法 query 的错误 |
| GetCacheLocationsByBackend/GetHostCacheState | CPU 后端身份、miss/mask 对齐及事件 host 正例；RTP 组件正例检查 backend/spec 大小和 host 全部字段；GPU adapter 的 miss 对齐 |
| min_replica_count | CPU 校验 1 副本跳过、要求 2 副本追加并再跳过；GPU backend 配置路径 |
| mask、未完成写入、部分失败、实际 URI | CPU 检查 Finish 前不可见、abort、成功 bool mask、按实际 URI 发布及重新查询读回 |
| RTP 故障恢复、同布局 TP、P/D | PACE match/start/finish fault、kill、TP2、P/D、边界请求复用已有结果断言 |
| ReportEvent storage_type/snapshot_required/retry_after_ms | CPU 对真实 Manager 检查注册、快照、限流重试、host 查询、删除，每个 fixture 一次；真实 Publisher 辅助程序验证初始快照和变更后的 host 可见性；组件单测检查反馈处理与心跳；`remote_cache_pace_publisher` 使用真实模型 key 检查模型接线 |
| RTP hybrid/cache 层及 buffer/错误路径 | `kvcm_mock_only_full_test`、`kvcm_mock_full_linear_test`、`kvcm_independent_pool_test`、`client_wrapper_test` 与配置/策略回归纳入 GPU 入口 |

CPU helper 使用 HOST IOV 仅验证 SDK 数据契约，不表示 RTP 已实现 CPU 源 remote 写入。事件协议和 Publisher helper 的事件由测试生成；新增模型事件用例复用已有 Qwen2.5 缓存复用数据，保留 DEVICE 缓存，启用 Publisher，通过 GetCacheStatus 取得真实模型 key，再轮询 KVCM HBM host state。此用例验证模型事件接线，不宣称远端采用量；强制远端读取的模型用例继续单独保留。SDK 队列饱和、超时后底层 I/O drain 的确定性慢 I/O 验证仍需 SDK 自身的契约测试；本入口的字节/参数拒绝检查不证明这些时序。

四种默认 query 的注册和 gap 语义保留原流程，不新增 helper 快速模式。事件协议从四次循环移到循环外，DRAM/SSD 合计从八次降至两次。Publisher helper 的周期快照设在用例窗口之外，避免周期性全量同步掩盖变更上报遗漏；限流协议仍由 HTTP 契约用例验证，确定性退避由组件单测验证。

非对称 TP/CP、GDR、zero-copy、单批混合后端/实例、CPU 源写入和全量写入不属于这次 smoke 补齐范围。PP/CP 事件扩展及 CUDA/ARM 平台扩展仍单独处理。

## 当前交付状态

本次只修改源码并做静态审查，未编译、运行测试或启动服务。CPU 运行会将 `source_id`、backend、默认 query 和字节/事件结果写入 `pace_contract_result.json`，供后续验收留证；GPU 结果由现有 smoke runner 记录。制品下载成功和已有外部环境的 CPU/TCP 字节记录不等于这些新增 smoke 已通过。

Pi（`glm-5.3-prime`，session `30e49365-f792-41ca-8c8a-53682d46aa4b`）独立静态审查发现的 gRPC 标签缺失已修正；默认 query 用例、无效快照清理和来源环境变量错误提示也已调整。reviewer 关于改成 Mac `/Users` 路径的建议未采纳：用户明确指定的是目标 Linux 环境的 `/User/gray/tmp`，已在前置条件中说明。reviewer 曾尝试对本机 `/User` 做可写探测，操作被只读文件系统拒绝，未产生文件；未进行项目编译或测试。

2026-09-29 按覆盖/冗余 review 优化后，Pi 全新会话 `10a995b7-e4ed-46f0-a8dc-f61d91231d18` 静态复审未发现 P0/P1 阻塞问题；提出的模型 key 等待与 Manager 上报等待共用期限问题已改为分别等待 15 秒。元数据正例、SWA 参数/窗口用例和模型事件接线源码已补，运行结果仍未验收；SDK 队列饱和及 drain=true 的受控测试不在本仓库中补写。
