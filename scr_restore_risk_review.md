# Epsilon arrival 与 CRIU dump/restore 风险评审

本评审覆盖当前链路：`start_server -> backend rank 初始化 -> Epsilon
arrival -> 外部 controller dump -> 新容器 CRIU/sGPU restore -> 重新提供服务`。
依据为仓库外层的 `src_plan.md`、`doc0.md`、`doc1.md`、`doc2.md`，以及当前
RTP-LLM、Epsilon wheel 和 `/etc/scr/epsilon` shim 的实现。

## 结论

当前代码可以作为“控制面负责 dump/restore、GPU rank 只注册并 arrival”的
实验链路，但不能直接把模板标记为“可跨容器/跨节点生产模板”。最大问题不
在 `snapstart_checkpoint` 的调用本身，而在 arrival 时进程仍可能包含业务、
网络和设备绑定状态；CRIU restore 也不会重新执行 Python 启动代码或自动更新
已恢复进程的环境变量。

本分支已经修复了三个可以在 RTP-LLM 侧安全闭环的问题：

1. 支持显式 scheduler-scope 的 `worker_id/worker_num`，避免 prefill/decode
   共用 scheduler 时 rank ID 冲突；默认仍为 Pod 内 `local_rank` 和
   `LOCAL_WORLD_SIZE`。
2. 注册主模型之外的 speculative draft Python model KV cache，并按指针去重。
3. 对非法 mapping、arrival 返回非零、Epsilon 不可用等情况增加日志；这些
   情况仍然 fail-open，但控制面必须将其视为 quorum 缺失并走 fallback。

以下问题需要控制面、RunD/sCR 或 C++ 两阶段 listener 配合，不能用 Python
worker 内再调用 controller 的方式“修复”。

## P0：在真实 dump 前必须关闭

### 1. arrival 时机早于完整服务就绪，且没有全进程 pre-bind 门闩

`start_backend_server.py` 在 `BackendManager.start()` 后注册 KV 并启动
arrival；父进程随后才创建 frontend 和 DashSc，最终还会运行 health check
及 DSV4 real warmup。见 `rtp_llm/start_backend_server.py:376-387`、
`rtp_llm/start_server.py:498-528`。

因此 GPU quorum ready 不代表以下进程已进入同一个稳定快照状态：backend
manager、frontend、DashSc、proxy、health worker 和辅助线程。`src_plan.md`
要求实际 CRIU target 覆盖完整 process tree，并要求 CPU/frontend/DashSc
参加独立的 AppPreBindBarrier；它们不应伪装成 Epsilon GPU worker。当前分支
没有这个全局 barrier。

控制面在 dump 前必须核对实际 cgroup、pstree、`/proc/<pid>/cmdline`、Aion
socket owner 和 CRIU inventory；只看到 GPU Epsilon quorum 不能继续 dump。

### 2. arrival 不是 quiesce：业务请求、scheduler 和后台线程仍可能在运行

arrival 线程是在 C++ engine 已完成初始化、内部 gRPC/HTTP 已监听、NormalEngine
loop 已启动后创建的。`register_before_checkpoint_func` 目前只做当前 CUDA
设备的 `torch.cuda.synchronize`，不会停止 request admission、等待 scheduler
队列归零、结束 streaming response、暂停 HostService heartbeat、关闭
AccessLogger/kmonitor/GrammarValidator/JIT helper 或释放临时锁。

frontend 自带的 `FrontendShutdownManager` 只在 SIGUSR1/SIGTERM 的停机路径中
drain，未与 SCR dump 关联。控制面必须先 block/drain，并证明没有 in-flight
请求、CUDA/NCCL collective、异步 copy、graph capture、动态编译和持锁线程，
再观察 Epsilon quorum；不能把 Epsilon return 当作 dump 完成。

### 3. restore 不会重新执行启动逻辑，旧 phase/IP/FD 会被原样带入

CRIU 恢复的是 checkpoint 时的地址空间、线程寄存器、文件描述符和 Python
`os.environ`。新容器启动命令中的 `SCR_PHASE=restore`、`POD_IP`、hostname、
rank 或端口不会自动改写已恢复进程。当前 `/etc/scr/epsilon` 的
`register_after_restore_func` 明确是 no-op。

所以 restore 必须由 RunD/平台 restore hook 或目标外 supervisor 注入带
generation 的 phase/fix-up 通道，在确认 phase 后依次重算本机地址、关闭并
重建 outbound gRPC/HostService/route cache、刷新服务发现，最后才允许 bind
和 health announce。不能依赖一个新启动的 launcher 或业务 HTTP 请求来唤醒
恢复进程中的旧 waiter。

### 4. host/device-bound 资源不能作为模板隐含前提

`DistributedServer` 会根据 hostname/IP 建立 TCPStore/NCCL bootstrap；PD/KVT
路径可能打开 RDMA uverbs、创建 QP/MR/CQ；HostService 会启动 heartbeat/probe
线程；C++ `RtpLLMOp` 在 Python 返回前已经 BuildAndStart 内部 gRPC，之后又
启动 HTTP。普通 CRIU 不能把这些硬件或真实地址状态迁移到另一台机器。

当前模板最多是同一网络 namespace、端口、GPU/driver/ABI 的 minimal 模式。
跨机器生产模板需要在 checkpoint 前释放/停止 RDMA、PD connector、旧 channel
和 discovery，restore 后按当前节点重建；或者由 sCR 明确证明相关资源可恢复。

## P1：代码/配置风险

### 5. Epsilon barrier 的 worker scope 必须固定

`wait_mode=1` 要求同一个 scheduler scope 内每个 GPU worker 使用唯一 ID，
且所有调用传同一个 `worker_num`。默认映射是 Pod 内
`worker_id=local_rank`、`worker_num=LOCAL_WORLD_SIZE`。若 prefill/decode 共用
一个 scheduler，需要配置：

```text
RTP_LLM_SCR_WORKER_OFFSET=<该角色在 scheduler scope 中的固定起点>
RTP_LLM_SCR_WORKER_NUM=<该 scope 的 GPU worker 总数>
```

也可给每个进程注入显式 `RTP_LLM_SCR_WORKER_ID`。CPU/frontend/DashSc/outer
manager 不得计入这个 GPU quorum。mapping 非法时当前实现拒绝 arrival，控制面
应 abort/fallback，而不是让其它 rank 永久等待。

### 6. Epsilon return、scheduler check 和 controller wait-cr-done 是三件事

当前 arrival thread 不 join，符合 `src_plan.md` 对 wait_mode=1 可能持续阻塞的
要求。controller 必须并行观察 scheduler 的连续 `errno==0 && checkpoint_ready`
和 Aion/PID inventory，再调用唯一一次 block/dump，并等待 `wait-cr-done`。
arrival 返回 0 也不能证明 CRIU dump 或 GPU snapshot 已经落盘。

### 7. KV registration 是优化提示，不等于完整状态清单

当前注册会收集主模型、draft model 的 base/region/scale KV tensor，并去重
真实 data pointer；`register_model` 在当前 shim 中仍是 no-op。权重、CUDA
graph、allocator metadata、非 torch CUDA allocation、scheduler block table
和 PD cache-store 状态是否被 sGPU 捕获，必须用实际 dump/restore 做 inventory
和数值校验，不能只依据 registration 返回 0。

### 8. C++ listener 仍早于 Python arrival

若验收标准要求 checkpoint 时“所有业务 listener 都不存在”，必须新增严格模式
的 C++ two-phase `prepared/quiesced/serving_ready` 状态，让
`RtpLLMOp::initRPCServer`、HTTP、cache-store、TCPStore、embedding/VIT 等在
FINAL_RELEASE 后再 bind。仅在 Python 侧移动 arrival 位置无法撤销已经创建的
listener；当前代码只能按 minimal 兼容矩阵使用。

## 推荐的生产控制顺序

```text
建立 service cgroup/manifest
  -> 所有 backend/frontend/DashSc/helper 完成静态初始化并 quiesce
  -> GPU rank 注册 KV，异步 arrival（不 join）
  -> 校验 cgroup/PID/Aion/FD/listener/in-flight inventory
  -> controller check（连续 ready）
  -> 按版本契约 block -> dump（唯一 coordinator）
  -> wait-cr-done
  -> source release；模板外 supervisor 保持可重连

restore:
  prepare-restore -> restore -> wait-cr-done
  -> 注入 phase/generation
  -> 无 listener 状态做 IP/channel/RDMA/route fix-up
  -> 全部 participant 报 RESTORE_FIXUP_READY
  -> FINAL_RELEASE
  -> bind/listen、服务发现、health announce、最小 token/checksum 验证
```

任一 participant 缺失、mapping 冲突、phase 未确认、旧地址/FD 未清理、RDMA
无法重建、dump/restore/wait-cr-done 非零，都应停止通告并走普通冷启动。

