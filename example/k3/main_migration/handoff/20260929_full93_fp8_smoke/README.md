# 93 层 FP8 双机 PD 限时 smoke（2026-09-29）

这轮验证使用集成分支 `f67f903e09e7db113820325f775ac21d783b0037`。该提交相对 `86d93fbfe46df6697fca28efe4020b994ed10d7e` 只增加诊断归档，模型代码相同。114 Prefill 使用在本机编译的 `86d93fb` 产物，115 Decode 使用在本机编译的相同模型代码；两端均在个人任务容器内，以 `luohaocheng.lhc` 启动。

114 的 3FS 预检和加载曾卡住，因此本轮改用 `/data0/luohaocheng.lhc/models/kimi-k3-ms-fc0873d9-20260929` 和 `/data0/luohaocheng.lhc/models/kimi-k3-mtp-verified-20260929`。本目录的 target 审计核对了 96 个 shard 的 SHA256、配置和索引，均与此前核实的 3FS manifest 一致；MTP 的 9 个 shard 也全部匹配 3FS 记录。115 仍从 3FS 直接读取 target 和 MTP，使用任务进程内的 32 线程预读辅助库；40、48、64 线程在这一轮整模加载中都未通过，32 是已验证可用的最高并发，不是集群级配置。两端的 FastSafetensors 预检及启动日志校验均通过，各 8 个 rank 的 target 和 MTP loader 都实际选择了 `fastsafetensors`。

运行参数为完整 93 层、target `FP8_PER_BLOCK`、FP8 KV、draft BF16 Native MTP、TP8/EP8、BF16 NCCL、Prefill TokenSpeed MLA、Decode CUDA Graph，以及双机 PD。114/115 的健康检查、gRPC 监听和互 ping 在发请求前通过。114 target 权重加载日志为 239.76 秒、MTP 权重加载为 4.31 秒，整个服务启动为 539.00 秒；这些阶段不能当成纯磁盘吞吐。115 的 3FS 加载分阶段记录在上一份 [诊断归档](../20260929_full93_load_diag/README.md)。

`main-text-64k-capped` runner 退出码为 0，121 条正式请求全部返回。独立审计从原始响应逐条复算答案，并检查 HTTP 状态、UTF-8、替换字符、连续重复字符、PD 路由和 MTP 执行：**121/121 通过，0 条审计错误**。121 条均有 PD 交接、MTP draft 轮次和被接受的 draft token。单请求最长 62.27 秒，未触发 300 秒上限。约定的 6 条长输出/已知超时用例没有发送，也没有计入通过；`full_original_suite_passed=false`。这说明当前代码能通过本轮限定的完整模型 FP8 正确性检查，尚不构成性能锚点、峰值激活锚点或原完整 suite 全通过的结论。

文件：

- `smoke-raw-114115-20260929.tar.gz`：121 条原始请求与响应、token fixture、runner 结果和独立审计。
- `prefill-startup-114-20260929.tar.gz`、`decode-startup-115-20260929.tar.gz`：两端启动配置、RDMA/权重预检和八个 rank 的日志。
- `local-full-target-3fs-hash-audit-114-20260929.json`、`mtp-local-vs-3fs-114-20260929.txt`：114 本地权重身份核对。
- 启动、smoke、独立审计脚本及机器选择快照：复现命令和当时条件。所有归档的 SHA256 见 `SHA256SUMS`。

服务在归档时仍运行于 114:27100 和 115:27200。再次使用这些脚本前，需要重查机器、端口、GPU、RDMA、权重和服务现状，避免重复加载或发请求。
