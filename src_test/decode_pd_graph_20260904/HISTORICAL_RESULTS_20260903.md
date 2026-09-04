# sCR + RTP-LLM CUDA Graph 实验记录（2026-09-03）

## 当前结论（截至 17:01，Asia/Shanghai）

- 本轮服务保持 `enable_cuda_graph=1`，没有关闭 CUDA Graph。
- 两卡 DSV4 Flash 0731 DECODE 服务已完成启动和 warmup；两个 rank 都报告 `rank_ready`，HTTP `/health` 和 `/v1/models` 返回 200。
- 使用 `bypass_dump_restore=false` 后，sCR scheduler 确实执行了 GPU/NCCL/metadata dump，目标目录已经写入约 347G 的 4 个文件。
- `scr_controller dump` 最终返回 rc=1，报错：`Failed to find the original rund-cr-scheduler in pkg dir`。scheduler 日志显示 GPU dump 本身返回 `Ok(())`，随后在 `persist_scheduler` 阶段找不到 CR wrapper。
- 因此目前只能确认“CUDA Graph 开启时 GPU checkpoint 已落盘”；还不能宣称完整的 CRIU/sCR dump-return 或 restore roundtrip 成功。两个 hook status 仍停在 `checkpoint_call`，没有 `checkpoint_return`。
- 还没有执行 restore；当前 dump 产物先保留，避免在根因未确认前误删。

## 不变量与环境

- 仓库：`/home/serina.wzq/RTP-LLM/github-opensource`
- 模型：`/tmp/models/DeepSeek-V4-Flash-0731`
- 两卡：`CUDA_VISIBLE_DEVICES=0,1`，`world_size=2`，`tp=1, dp=2, ep=2`
- 角色：`DECODE`（这是远程 RPC 角色；普通 HTTP 生成请求预期会路由失败，不代表 CUDA/sCR 失败）
- CUDA Graph：`--enable_cuda_graph 1`
- Graph capture：`--decode_capture_config 1,2,4,8,16`
- DeepEP：`--use_deepep_moe 1 --use_deepep_low_latency 1`
- 显存策略：`--reserver_runtime_mem_mb 10240`、`--memory_cache_size_mb 1024`、`--fp8_kv_cache 1`
- 为隔离此前的 VMM invalid-handle 问题，当前进程使用：
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,large_segment_size_mb:1024`
  （这只控制 allocator，不会关闭 CUDA Graph。）
- NCCL shadow 只通过进程内 preload 注入：
  `LD_PRELOAD=/etc/scr/shadow/libnccl.so:/usr/lib64/librt.so.1`
  没有把 `/etc/scr/shadow` 加进 `LD_LIBRARY_PATH`。
- 不要运行 `nvidia-smi`：本机 NVML/journald interposer 会 panic 或创建额外 Aion client，干扰 scheduler quorum。

## Scheduler

当前唯一活跃 scheduler（快照时 PID 1412789）：

```bash
/home/yuziqu.yzq/scr_scheduler --fork false
```

当前 `/run/scr/config.json`（本轮实际使用）：

```json
{
  "bypass_dump_restore": false,
  "log_level": "info",
  "crc_check": true,
  "log_exporter": "uds_file",
  "bounce_buffer": true,
  "cache_fs_sock_path": "/run/cachefs/cachefs-api.sock"
}
```

准备好的副本：
`/home/serina.wzq/RTP-LLM/github-opensource/.tmp/scr_checkpoint_hook/scheduler_config_bypass_off.json`

用户源配置 `/home/yuziqu.yzq/scheduler_config.json` 仍是 `bypass_dump_restore=true`，没有被改写；启动真实实验时只把上述 bypass-off 副本复制到了 `/run/scr/config.json`。

Scheduler 日志：`/run/scr/log/scheduler.log`

## 服务和 hook 文件

真实 roundtrip 脚本：
`/home/serina.wzq/RTP-LLM/github-opensource/.tmp/scr_checkpoint_hook/run_20260903_decode_graph_scr_real/launch.sh`

hook 配置：
`/home/serina.wzq/RTP-LLM/github-opensource/.tmp/scr_checkpoint_hook/run_20260903_decode_graph_scr_real/config.json`

关键 hook 设置：

- `worker_num=2`
- `role_offsets={"DECODE": 0}`
- `idle_grace_seconds=20`
- target after-hook：`rtp_llm.server.backend_manager.BackendManager.start`
- trigger：`.../run_20260903_decode_graph_scr_real/trigger`
- status：`.../run_20260903_decode_graph_scr_real/status/`

服务日志目录：`/tmp/rtp_llm_decode0731_graph_scr_real_20260903/logs`

启动方式（sudo 需要保留 conda/gcc 的库路径）：

```bash
RUN=.tmp/scr_checkpoint_hook/run_20260903_decode_graph_scr_real
/usr/bin/sudo -n env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PATH="$PATH" \
  setsid -f bash "$RUN/launch.sh" \
  >/tmp/rtp_llm_decode0731_graph_scr_real_20260903/launcher_root.out 2>&1 < /dev/null
```

快照时服务 PID/端口：

- launcher PGID：1414333
- main：1414373
- backend：1414404
- rank 0：1414536
- rank 1：1414538
- frontend：`18630`
- rank frontend：`18639`
- backend RPC：`18638`

## 时间线和证据

1. 旧实验配置为 `bypass_dump_restore=true`。两 rank 曾进入 `checkpoint_call`，但 scheduler 一直 `SteadyStateSyncing`，目录为空；这不是 dump 成功。
2. 已停止旧服务和旧 scheduler，检查无 listener 后清理了 `/run/scr/socket` 下的 stale sockets；当前只保留一个新 scheduler。
3. 旧 dump 文件已按用户要求删除（见“清理记录”）。
4. 使用 bypass-off 配置重启服务；`main_0.log`/`main_1.log` 均有：
   - `enable_cuda_graph: 1`
   - `Using decode capture batch sizes from comma-separated list: 5 items`
   - `Backend server initialized successfully`
   - `BackendManager entering serve_forever loop`
5. 短 HTTP 请求（DECODE-only 角色的预期语义结果）：

```text
HTTP 500
8500_ROUTE_ERROR ... no backend role addresses found after routing
```

`GET /health` 和 `GET /v1/models` 均 HTTP 200。DECODE 的 `RemoteRpcServiceImpl` 没有普通 `GenerateStreamCall`，所以不能用该入口证明数值生成正确。

6. 触发稳态：

```bash
touch .tmp/scr_checkpoint_hook/run_20260903_decode_graph_scr_real/trigger
```

两个 status 文件均记录：`rank_ready -> trigger_seen -> checkpoint_call`。随后：

```text
scr_controller check
{"errno":0,"msg":"","disk_usage_mb":357346,"memory_usage_mb":8272,
 "nccl_launched":2,"checkpoint_ready":true}
```

这证明两个 rank 已到稳态并被 scheduler quorum 接管。

7. 实际 dump 命令：

```bash
D=/tmp/test/rtp_decode_graph_scr_real_20260903_1700
/usr/bin/sudo -n mkdir -p "$D"
/usr/bin/sudo -n /usr/local/scr/aion/cuda/scr_controller dump \
  --path "$D" --bypass-cr-path "$D" --block-timeout-ms 120000
```

controller 输出：

```text
RpcStatus(Status { code: INTERNAL,
  message: "Failed to find the original rund-cr-scheduler in pkg dir", ... })
```

但 scheduler 日志（约 17:01）明确显示：

- 两个 CUDA client 的 `Start GPU checkpointing`
- virtual memory checkpoint 完成
- managed memory checkpoint 完成
- `do SchedulerApi::Dump success`
- `SchedulerApi::Dump get ret Ok(())`
- 随后进入 `persist_scheduler`，才触发 `rund-cr-scheduler` 缺失错误

目标目录当前 4 个文件：

```text
1414536.0-4   56,623,104 bytes
1414536.1-4   186,151,600,128 bytes
1414538.0-5   56,623,104 bytes
1414538.1-5   186,082,394,112 bytes
```

总计约 347G；`/tmp/test` 是 768G tmpfs，dump 后约剩 422G。不要在确认 restore 方案前再创建第二份同规模 dump。

## `bypass_dump_restore` 的含义

- `true`：绕过真实 dump/restore 持久化；不会产生可用 checkpoint，容易停在 `SteadyStateSyncing`，只能验证 hook/quorum 的部分路径。
- `false`：允许 scheduler 执行真实 GPU/CUDA/NCCL checkpoint 和 restore，并把状态写入 `--path`；也会继续尝试 CR/进程级持久化，因此需要可用的 `rund-cr-scheduler`/CRIU 环境。
- 该开关与 `enable_cuda_graph` 独立；本轮 Graph 一直是 1。

## 当前阻塞点与下一步

1. 先确认本机是否有可用的 `rund-cr-scheduler` 包/服务路径；`command -v criu`, `rund-cr-scheduler`, `runc`, `crun` 当时均为空。
2. 不要把 controller 的 rc=1 忽略为完整成功：当前只确认 GPU dump 成功，hook 尚未 `checkpoint_return`。
3. 可在同一 dump 目录尝试 `scr_controller restore --path "$D" --bypass-cr-path "$D"`，但需记录其是否在 CR wrapper 阶段失败；不要关闭 CUDA Graph，也不要再生成第二份 dump。
4. restore 后必须检查：两个 hook 是否出现 `checkpoint_return`、scheduler `Restore` 是否 `Ok(())`、`check` 状态是否恢复；DECODE HTTP 仍只能作为 health/route 语义检查。
5. 若要验证“前后生成 token 一致”，需另起显式 `PDFUSION`/LocalRpc graph smoke；不要把 DECODE-only 的 8500 路由错误当成模型或 SCR 错误。

## 清理记录

按用户要求、在确认无进程打开后删除了旧 dump blobs：

- `/tmp/test/0.1`, `1.0`, `4.3`, `5.2`, `6.0`, `7.1`
- `/tmp/test/rtp_decode_pd_scr_20260903_1441/`
- `/tmp/test/rtp_prefill_tp2_scrfix_vmmoff_20260903_1339/`
- `/tmp/test/rtp_prefill_tp2_scrfix_retry_20260903_1327/`
- `/tmp/test/rtp_prefill_tp2_scrfix_20260903/`

共释放约 768G。删除是 `unlink`/`rmdir`，没有回收站，原文件不可恢复；模型、源码、服务日志和本轮 dump 未删除。

## 相关日志/状态路径速查

- 服务日志：`/tmp/rtp_llm_decode0731_graph_scr_real_20260903/logs/`
- controller 输出：`.tmp/scr_checkpoint_hook/run_20260903_decode_graph_scr_real/controller_dump.out`
- dump 元信息：`.tmp/scr_checkpoint_hook/run_20260903_decode_graph_scr_real/dump_meta.txt`
- hook status：`.tmp/scr_checkpoint_hook/run_20260903_decode_graph_scr_real/status/`
- scheduler：`/run/scr/log/scheduler.log`
- 当前 GPU dump：`/tmp/test/rtp_decode_graph_scr_real_20260903_1700/`
