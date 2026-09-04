# RTP-LLM sCR `expandable_segments` A/B Test

日期：2026-09-04
目标：在相同 RTP-LLM、Epsilon、sCR scheduler 和 GPU-only bypass dump/restore 条件下，只切换 `PYTORCH_CUDA_ALLOC_CONF` 中的 `expandable_segments=True/False`。

## 1. 测试条件

- RTP-LLM：工作区 `/home/serina.wzq/RTP-LLM/github-opensource`
- 运行包：`/opt/conda310/bin/python3` / `/opt/conda310` site-packages
- 模型：`/tmp/models/DeepSeek-V4-Flash-0731`
- Epsilon：运行日志确认实际实现为 `/etc/scr/epsilon/__init__.py`
- SCR 环境：

  ```text
  SCR_ENABLE=1
  RTPLLM_ENABLE_SCR=1
  SCR_PHASE=checkpoint
  ```

- scheduler 配置：

  ```json
  {
    "bypass_dump_restore": true,
    "bounce_buffer": true,
    "crc_check": true
  }
  ```

- scheduler 启动参数：`--fork false --safe-block-backoff 5`
- 只执行 GPU dump/restore，没有 CPU dump，也没有 CRIU CPU checkpoint。
- 两组启动脚本除工作目录、注释和 allocator 配置外一致：

  - True：[prefill_unified_scr_expandable_true_20260904.sh](/home/serina.wzq/test/prefill_unified_scr_expandable_true_20260904.sh)
  - False：[prefill_unified_scr_expandable_false_bypass_true_20260904.sh](/home/serina.wzq/test/prefill_unified_scr_expandable_false_bypass_true_20260904.sh)

## 2. 变量

True 组：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:1024
```

False 组：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,large_segment_size_mb:1024
```

两组运行时的 `env.txt` 均确认了对应值，没有只修改启动脚本而未传入子进程的问题。

## 3. 统一 sCR barrier

本次对照使用 prefill-only 的 6 个参与者：

```text
start_server:0
backend_manager:0
backend_rank:0
backend_rank:1
frontend:0:0
dash_sc:0:0
```

False 组日志确认：

- `worker_num=6`
- `worker_id=0..5` 全部执行 `snapstart_checkpoint`
- 6 个 worker 均返回 success
- frontend 和 dash-sc 都到达统一 barrier

因此本轮不是 backend-only checkpoint，也不是 CPU/GPU 分裂 barrier。

## 4. dump/restore 结果

### 4.1 结果汇总

| 项目 | `expandable_segments=True` | `expandable_segments=False` |
|---|---:|---:|
| dump CLI 返回码 | 0 | 0 |
| dump 输出文件 | 2 个 | 4 个 |
| dump 文件大小 | `71,303,168`、`71,303,168` bytes | `390,070,272`、`390,070,272`、`85,781,905,408`、`87,170,220,032` bytes |
| scheduler dump | 失败路径出现 CUDA VMM 导入错误 | `Block=Ok`，`Dump Done` |
| dump 后 `wait-cr-done` | `InternalError` | 手工启动 scheduler 的 pkg 路径错误 |
| restore CLI 返回码 | 0 | 0 |
| restore 后 `wait-cr-done` | `InternalError` | `"Success"` |
| restore 后服务 | 异常 | frontend health 返回 HTTP 200 `"ok"` |

False 组完整 GPU dump 文件位于：

[/tmp/test/rtp_prefill_unified_scr_expandable_false_bypass_true_20260904_rerun](/tmp/test/rtp_prefill_unified_scr_expandable_false_bypass_true_20260904_rerun)

### 4.2 True 组现象

对应 scheduler PID 为 `1874384`。dump 过程中出现：

```text
cuMemImportFromShareableHandle get cudaError_enum(400)
alloc virtual memory
cuMemImportFromShareableHandle block failed: cudaError_enum(400)
```

后续 `wait-cr-done` 记录了：

```text
dump restore get handle return error alloc virtual memory
dump restore get handle return error Open file
```

因此 True 组虽然 dump CLI 很快返回 0，但这是异步 dump；实际 GPU dump 任务失败，最终只有两个小文件，没有完整模型/运行时状态。

### 4.2.1 sCR controller 日志线索

True 组的 controller 输出文件：

- [check_before.out](/home/serina.wzq/test/prefill_unified_scr_expandable_true_20260904/check_before.out)：`checkpoint_ready=true`，`disk_usage_mb=188868`，`memory_usage_mb=8272`，`nccl_launched=2`
- [dump.out](/home/serina.wzq/test/prefill_unified_scr_expandable_true_20260904/dump.out)：CLI 返回 0，但只有 UDS 连接输出；真正的失败在异步 scheduler 任务中
- [wait_dump.out](/home/serina.wzq/test/prefill_unified_scr_expandable_true_20260904/wait_dump.out)：返回 `"InternalError"`
- [wait_restore.out](/home/serina.wzq/test/prefill_unified_scr_expandable_true_20260904/wait_restore.out)：返回 `"InternalError"`

对应的 scheduler 原始日志在 [scheduler.log.1](/run/scr/log/scheduler.log.1) 中，关键链路如下：

| 阶段 | 日志证据 | 含义 |
|---|---|---|
| check 后开始 dump | [43025](/run/scr/log/scheduler.log.1:43025) | controller 已收到 dump 请求，路径同时传给 `path` 和 `bypass_cr_path` |
| dump 前准备 | [43027](/run/scr/log/scheduler.log.1:43027)、[43028](/run/scr/log/scheduler.log.1:43028)、[43029](/run/scr/log/scheduler.log.1:43029) | 清理 rund-cr state 目录时报 NotFound；随后发现 4 个 Aion socket 并设置 GPU VRAM dump 目录。这是旁路线索，不是首个致命错误 |
| Block | [43131](/run/scr/log/scheduler.log.1:43131) | 4 个 GPU/Aion client 的 Block 成功，说明 barrier/block 阶段没有失败 |
| GPU dump 开始 | [43298](/run/scr/log/scheduler.log.1:43298) | Aion client 开始 GPU checkpoint |
| CUDA VMM 导入失败 | [43548](/run/scr/log/scheduler.log.1:43548)、[43550](/run/scr/log/scheduler.log.1:43550)、[43584](/run/scr/log/scheduler.log.1:43584) | `cuMemImportFromShareableHandle` 返回 `cudaError_enum(400)`，上层包装成 `alloc virtual memory` |
| Dump RPC 收尾 | [43618](/run/scr/log/scheduler.log.1:43618) | scheduler 仍记录 `SchedulerApi::Dump Done`；这只代表 dump orchestration 返回，不代表异步 bypass copy 全部成功 |
| wait-cr-done 发现失败 | [43765](/run/scr/log/scheduler.log.1:43765)、[43766](/run/scr/log/scheduler.log.1:43766) | `finish count ... handles count 4` 后返回 `dump restore get handle return error alloc virtual memory` |
| 后续句柄/文件失败 | [45032](/run/scr/log/scheduler.log.1:45032) | 后续 wait 再次报告 `Open file` / `No such file or directory`，与前面未完成的异步 dump 结果一致 |

关键因果链可以压缩为：

```text
DumpRequest
  -> Block OK
  -> GPU checkpoint starts
  -> cuMemImportFromShareableHandle(cudaError 400)
  -> alloc virtual memory
  -> asynchronous handle failure
  -> wait-cr-done InternalError
  -> incomplete dump files
```

### 4.3 False 组现象

对应 scheduler PID 为 `1902539`。scheduler 日志确认：

```text
SchedulerApi::Block get ret Ok(())
SchedulerApi::Dump Done
SchedulerApi::Restore Done
barrier after_wait_cr_done: single-node scenario, skip
```

False 组没有出现 `cuMemImportFromShareableHandle ... cudaError_enum(400)`。4 个 GPU dump 文件完整生成，restore 后异步等待返回 `"Success"`，prefill frontend 仍可用。

## 5. 结论

在当前 ARM/CUDA13、sCR bypass dump/restore 环境中，A/B 结果支持以下结论：

本节结论对应的历史有效实验使用以下 scheduler 启动命令；`--safe-block-backoff 5` 是该实验的固定启动条件，需在复现实验时一并记录：

```bash
/home/yuziqu.yzq/scr_scheduler --fork false --safe-block-backoff 5
```

后续去掉该参数的严格重启复测属于单独的启动/bootstrap 复测，因 scheduler 从空状态返回 `checkpoint not ready`、尚未进入模型和 dump 阶段，不能覆盖或否定下面的 allocator A/B 结论。

1. `expandable_segments=True` 会触发 sCR bypass GPU 路径中的 CUDA VMM shareable-handle 导入失败。
2. 失败点在 scheduler 的 `cuMemImportFromShareableHandle` / `alloc virtual memory`，不是 Epsilon barrier 本身。
3. `expandable_segments=False` 时同一套 RTP-LLM、Epsilon、scheduler 和 GPU dump/restore 流程能够生成完整快照并成功恢复。
4. `bypass_dump_restore=true` 已生效；本次观察到的是 bypass GPU 内存路径，不是 CPU/CRIU 路径。

更具体的待修复方向是：检查 PyTorch expandable/VMM allocator 导出的 allocation handle、映射生命周期和 sGPU scheduler 的 `cuMemImportFromShareableHandle` 兼容性。当前证据还不能进一步断言是某个单独 CUDA API 参数或驱动缺陷。

## 6. 复现和解释注意事项

### `wait-cr-done` 参数

当前安装的 `scr_controller` 版本为 1.6.0，`wait-cr-done` 只接受 `--timeout`，不接受 `--path` 或 `--bypass-cr-path`。实际使用的是：

```bash
scr_controller wait-cr-done --timeout 600
```

### dump wait 的环境错误

False 组 dump 后的 wait 报：

```text
Failed to find the original rund-cr-scheduler in pkg dir
```

这是因为 scheduler 是手工用 `/home/yuziqu.yzq/scr_scheduler --fork false` 启动的，controller 的 persist 逻辑找不到原始 `rund-cr-scheduler` 包路径；它没有影响 GPU dump 文件生成，且 restore 后 wait 已返回成功。

### `check_before` 差异

True 组的 `check_before` 是已有 CR 状态，返回 `checkpoint_ready=true`；False 组是在 scheduler 重启后的新状态，返回 `checkpoint not ready`，但 Epsilon 6-worker barrier 已成功，随后 dump 仍完成。

因此本报告是强证据的 allocator A/B，但若工单要求完全同状态的重复实验，还应在同一个 scheduler 生命周期中分别重新启动两组，并让两组都从相同的 `check_before` 状态开始。

## 7. 证据文件

- [True 组 controller 输出](/home/serina.wzq/test/prefill_unified_scr_expandable_true_20260904)
- [False 组 controller 输出](/home/serina.wzq/test/prefill_unified_scr_expandable_false_bypass_true_20260904)
- [False 组 GPU dump](/tmp/test/rtp_prefill_unified_scr_expandable_false_bypass_true_20260904_rerun)
- [scheduler.log.1](/run/scr/log/scheduler.log.1)
- [scheduler.log](/run/scr/log/scheduler.log)

## 2026-09-04 严格重启复测：去掉 `--safe-block-backoff 5`

本轮按要求重启 scheduler，并使用不带该 CLI 参数的命令：

```bash
/home/yuziqu.yzq/scr_scheduler --fork false
```

两份 scheduler 配置没有改动，复测前后 `/home/yuziqu.yzq/scheduler_config.json` 与 `/run/scr/config.json` 的 SHA256 都是
`dc76f1b87aae037884b6c98e0cbb609275725de31bd79bf3efa3feba4e37c2d1`。

### 观察到的启动行为

直接从空的 scheduler 状态启动时，True 和 False 都没有进入模型加载、backend/frontend 或统一 Epsilon barrier，因此没有执行 dump/restore：

| 组别 | no-arg 主进程 | 现象 |
|---|---:|---|
| True | 1960170 | `set_parallelism_config` 后每秒重复 `get_scheduler_state`；`main_0.log` 没有 backend/frontend 日志 |
| False | 1961434 | 完全相同；没有生成 GPU dump 文件 |

对应工作目录分别是 [True strict logs](/tmp/rtp_llm_prefill_unified_scr_expandable_true_nosafebackoff_fresh_20260904) 和 [False strict logs](/tmp/rtp_llm_prefill_unified_scr_expandable_false_nosafebackoff_20260904)，controller `check` 返回：

```json
{"errno":0,"msg":"checkpoint not ready","disk_usage_mb":0,"memory_usage_mb":0,"nccl_launched":0,"checkpoint_ready":false}
```

### scheduler 参数的实际含义

无参数 scheduler 的启动日志仍打印：

```text
safe_block_backoff: Some(5)
```

见 [scheduler.log:4706](/run/scr/log/scheduler.log:4706)。这说明 `5` 是该二进制的默认值；本轮已确认命令行没有传 `--safe-block-backoff 5`，但删除 CLI 参数并不会把有效值变成“未设置”。

为了验证是否只是首次状态初始化问题，本轮临时启动过一次 `--safe-block-backoff 900`，并调用了一次 `scr_controller fallback`；随后按要求重启为无参数 scheduler。该 bootstrap 仍未形成可供新进程使用的 checkpoint-ready 状态，最终 True/False 两组都在同一启动前阶段停住。

### 严格复测结论

1. 本轮严格 no-arg A/B 的控制变量没有触达 allocator 或 GPU dump 阶段；因此不能把这轮的“未 dump”归因于 `expandable_segments=True/False`。
2. 去掉 CLI 参数后，scheduler 的有效默认仍是 `5`；变化是命令行表达，不是运行时 backoff 数值。
3. 之前报告中使用同一配置、同一 Epsilon/统一进程 manifest、GPU-only bypass dump/restore 得到的 allocator 结果仍然是有效证据：`True` 在 scheduler 的 `cuMemImportFromShareableHandle -> cudaError_enum(400)` 失败，`False` 完成 dump/restore。此次 no-arg 复测补充了一个独立限制：scheduler 从空状态重启时必须先解决 `checkpoint not ready` 的 bootstrap，否则两组都无法开始。

工作区源码本次未新增修改；`git diff --check` 已通过。
