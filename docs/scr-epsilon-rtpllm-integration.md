# SCR / Epsilon / RTP-LLM 集成调研与端到端 dump/restore 方案

> 调研对象：e01-cn-xp54kwggb06-a0002 上的 serina.wzq.dev.new.worker0 与 serina.wzq.dev.new.scr-scheduler。  
> 调研时间：2026-09-06（Asia/Shanghai）。  
> 第三方 SCR/Epsilon 没有源代码，本文把可执行文件帮助、日志、Python shim、挂载、历史命令和 RTP-LLM 工作树结合起来，区分实测事实和基于行为的推断。

## 1. 结论先行

当前 RTP-LLM 接入方向基本正确：

- RTP-LLM 负责发现并注册 GPU KV cache、安装 CUDA 同步 callback、在参与进程中调用 Epsilon arrival/barrier。
- scr_controller/scr_scheduler 负责控制面生命周期、GPU memory persist、CPU/process CRIU、block/release 和 wait-cr-done。
- RTP-LLM 不应该直接执行 dump、restore，也不应该自己驱动 SCR_PHASE 状态机。
- 正常服务启动必须是 SCR opt-out；启用 SCR 只能出现在明确的 checkpoint/restore 测试或生产恢复 Pod。

还需要补齐四件事：

1. 固化 scheduler scope、participant 清单、worker_id 和 worker_num，避免一个 participant 缺席导致其他 rank barrier 超时。
2. 把 restore 后 KV cache 重新发现/重新注册做成正式协议。当前 external shim 的 register_after_restore_func 是 no-op，返回 0 不代表 callback 真正执行。
3. SCR_PHASE 只能是兼容输入和诊断字段，真实生命周期应由 controller/scheduler 和 generation 拥有。
4. 使用 sidecar 中真实的 controller 路径和真实 checkpoint 路径。现有脚本默认 /usr/local/scr/aion/cuda/scr_controller，并强制 tmpfs；本环境真实路径是 /run/scr/scr_controller 和 ext4 的 SCR_CR_PATH。

## 2. 实际组件和位置

### 2.1 RTP-LLM

worker0 工作树：

~~~text
/home/serina.wzq/RTP-LLM/github-opensource
~~~

当前有未提交 SCR 接入改动的文件包括：

~~~text
rtp_llm/utils/scr_template_utils.py
rtp_llm/start_backend_server.py
rtp_llm/start_server.py
rtp_llm/start_frontend_server.py
rtp_llm/start_dash_sc_server.py
rtp_llm/server/vit_rpc_server.py
rtp_llm/test/scr_scheduler_e2e_test.py
rtp_llm/utils/test/scr_template_utils_test.py
~~~

### 2.2 Epsilon wheel

worker0 的 /home/serina.wzq 下有两个架构 wheel：

~~~text
epsilon-0.2.7+95d46bb4-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
epsilon-0.2.7+95d46bb4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
~~~

SHA256 实测值：

~~~text
632d82c8ca31496e6136af4bbd12615dacacbe7b53f6c08f3cce1b83bacf0ace  epsilon-0.2.7+95d46bb4-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
6eccbc16dacae75979e6879d78e048408a492d5dcc960e4abee24a2abc4037e0  epsilon-0.2.7+95d46bb4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
~~~

wheel 的主要内容是：

~~~text
epsilon/__init__.py
epsilon/_native/lib_native.so
epsilon-0.2.7+95d46bb4.dist-info/*
~~~

wheel 不包含 /etc/scr/epsilon。因此 /etc/scr 不是普通 pip install 直接解压出来的，而是 SCR runtime image、sidecar 或容器挂载提供的 external compatibility shim。

worker0 的用户空间安装位置：

~~~text
/home/serina.wzq/.local/lib/python3.10/site-packages/epsilon
/home/serina.wzq/.local/lib/python3.10/site-packages/epsilon-0.2.7+95d46bb4.dist-info
~~~

其他容器曾经在 /opt/conda310/lib/python3.10/site-packages/epsilon 安装过同版本。排障时应同时记录 Python 的 sys.path、epsilon.__file__ 和 pip 所属解释器。

### 2.3 worker 中的 SCR runtime shim

worker0 实测存在：

~~~text
/etc/scr/epsilon/__init__.py
/etc/scr/shadow/libaion.so
/etc/scr/shadow/libcuda.so
/etc/scr/shadow/libnccl.so
/etc/scr/shadow/libnvidia-ml.so
/etc/scr/libcuda.so.580.159.03
/etc/scr/compat/libnvidia-ptxjitcompiler.so.580.159.03
/etc/scr/compat/libnvidia-nvvm.so.580.159.03
/etc/scr/compat/libnvidia-gpucomp.so.580.159.03
/etc/scr/hook-flags
~~~

external shim 的事实：

- 只有 /etc/scr/epsilon 存在、SCR_ENABLE=1、kernel release 不含 kangaroo 时，wheel 才加载 external module。
- 它检测 /run/scr/socket 或 /run/rund-cr/socket；当前选择 SCR，加载 /etc/scr/shadow/libaion.so。
- is_snapstart_enable() 只判断 SCR_PHASE 是否为 checkpoint 或 restore。
- register_kv_caches() 展平 Tensor，取 data_ptr 和 nbytes，传入 libaion.so。
- snapstart_checkpoint() 先执行 before-checkpoint callback，再调用 native barrier。
- register_after_restore_func() 目前只打印 waiting for implementation，并没有保存或执行 callback。

### 2.4 scheduler sidecar

容器名：

~~~text
serina.wzq.dev.new.scr-scheduler
~~~

runtime 根目录：

~~~text
/run/scr/scr_scheduler
/run/scr/scr_controller
/run/scr/scr-config
/run/scr/config.json
/run/scr/record.json
/run/scr/state/blocking
/run/scr/uuid
/run/scr/socket/ttrpc
/run/scr/socket/log.sock
/run/scr/libcuda.so.595.45.04
/run/scr/log/scheduler.log
/run/scr/log/scr-config.log
~~~

二进制大小和 SHA256：

~~~text
/run/scr/scr_scheduler   9,892,752 bytes
317a8ecbfc8d357a305b0540273708150776f056944e5d4540bc7f34b91523ba

/run/scr/scr_controller   2,595,304 bytes
ce18366079fbfe83b90184628699f8b822a468774ee307002275f19cade3

/run/scr/scr-config        1,636,304 bytes
48b0ce2019bd74469ac0e4023179595c89e66c70ac7f0427a4c8d2b51e0202d
~~~

版本（来自 --help）：

~~~text
scr_scheduler / scr_controller
  Release Version: 1.6.0
  Git Commit Hash: e9982731
  Git Commit Branch: runc-release-1.6
  UTC Build Time: 2026-08-28 08:20:44
  Rust: rustc 1.81.0
  Features: runc, scr_epsilon

scr-config
  Release Version: 0.1.0
  Git Commit Hash: 367737a
  Git Commit Branch: runc-release-1.5
  UTC Build Time: 2026-09-02 03:47:37
  Rust: rustc 1.81.0
~~~

常驻进程：

~~~text
/run/scr/scr_scheduler --fork false
~~~

scr_controller 不是常驻 daemon，而是向 /run/scr/socket/ttrpc 发 RPC 的控制面 CLI。子命令包括 bind、block、check、check-steady-state、dump、fallback、health、prepare-restore、reset-restore、restore、unbind、unblock、wait-cr-done。

sidecar 中没有发现 /usr/local/scr/aion/cuda/scr_controller 或同路径下的 scheduler；该路径只是旧脚本 fallback。

### 2.5 sidecar 启动脚本和配置

/home/start_scheduler.sh 当前逻辑：

~~~text
SCR_FALLBACK=true                         -> sleep infinity
SCR_PHASE=normal                          -> sleep infinity
SCR_PHASE 非 checkpoint/restore           -> 退出
否则                                      -> /run/scr/scr-config
                                             /run/scr/scr_scheduler --fork false
~~~

/home/epsilon/start_epsilon.sh 当前逻辑：

~~~text
ECP_UDS_ADDR 默认 /scr-share/epsilon/daemon.sock
SCR_FALLBACK=true 或 SCR_PHASE=normal -> sleep infinity
只接受 SCR_PHASE=checkpoint/restore
执行 /home/epsilon/epsilon_daemon SCR_PHASE "{{}}"
日志写入 /run/scr/log/epsilon_daemon.log
~~~

sidecar 当前关键环境变量：

~~~text
SCR_SCHEDULER_SIDECAR=1
SCR_CHECKPOINT_POD_NAME=serina.wzq.dev.new.worker0
SCR_CR_PATH=/run/app_template_source/serina.wzq.dev.new.manual
SCR_ENABLE=1
SCR_PHASE=checkpoint
~~~

scheduler_config.json：

~~~json
{
  "bypass_dump_restore": true,
  "log_level": "info",
  "crc_check": true,
  "log_exporter": "uds_file",
  "bounce_buffer": true,
  "cache_fs_sock_path": "/run/cachefs/cachefs-api.sock"
}
~~~

scheduler help 对相关参数的描述：

- bypass_dump_restore：由 scheduler 执行/接管 dump-restore 路径，不等价于完全跳过 CRIU。
- persist-type：GPU memory store backend，可选 main、pmem、file；main 使用 user container 主内存，pmem 使用 RunD/FC 提供的 PMEM，file 使用容器文件系统。
- persist-size：pmem 后端主机内存上限，单位 MB。
- crc-check：是否对 device memory 做 CRC32 校验。
- pre-cuinit-bypass-cr：bypass 模式下 restore 时是否在 scheduler 启动阶段调用 cuinit。

## 3. 历史命令和两种测试场景

sidecar 的 /root/.bash_history 保存了之前的测试命令。已找到的 controller 流程：

~~~text
/run/scr/scr_controller dump \
  --path /run/app_template_source/serina.wzq.dev.new.manual/ \
  --bypass-cr-path /run/app_template_source/serina.wzq.dev.new.manual/

/run/scr/scr_controller wait-cr-done --timeout 1800

/run/scr/scr_controller restore \
  --path /run/app_template_source/serina.wzq.dev.new.manual/ \
  --bypass-cr-path /run/app_template_source/serina.wzq.dev.new.manual/

/run/scr/scr_controller wait-cr-done --timeout 1800
~~~

启动/观察命令也曾使用：

~~~text
bash -c /home/start_scheduler.sh > start_scheduler.log 2>&1 &
/run/scr/scr_controller check
/run/scr/scr_controller unblock
tail -f /run/scr/log/scheduler.log
~~~

### 3.1 GPU-only 场景

scheduler.log 中反复出现：

~~~text
persist_type: GuestFile
bypass_dump_restore: true
bypass_restore_v2: false
bounce_buffer: true
epsilon_mode: Disable
crc_check: true
~~~

同时出现 Run bypass dump restore、copy with bounce buffer、get finish count 等日志。/run/app_template_source/gpu-images 目录存在但当前为空，说明 GPU-only 测试使用过专用目录但 image 当前没有保留。

GPU-only 的数据流：

~~~text
RTP-LLM/Epsilon 注册显存地址
 -> scheduler 保存/恢复 GPU memory
 -> 不要求 cpu-images、rootfs.tar 或进程树 CRIU image
~~~

### 3.2 GPU + CPU CRIU 场景

当前 SCR_CR_PATH 是 ext4 目录：

~~~text
/run/app_template_source/serina.wzq.dev.new.manual
~~~

它包含：

~~~text
cpu-images/
state/blocking
record.json
rootfs.tar                         # 约 178 GB
~~~

cpu-images 中能看到 pages、core、mm、pstree、mountpoints、IPC namespace、tmpfs image 等 CRIU 文件。这是 CPU/process CRIU 已经发生过的直接证据，不是单纯的 GPU-only image。

scheduler.log 中曾出现：

~~~text
SchedulerApi::Dump
SchedulerApi::Dump Done
SchedulerApi::WaitCRDone
Run bypass dump restore
persist scheduler blocking file to .../state/blocking
SchedulerApi::WaitCRDone Done
~~~

因此两条链路应分开理解：

| 场景 | GPU memory | CPU/process state | 典型证据 |
|---|---|---|---|
| GPU-only | scheduler bypass/GuestFile/copy with bounce buffer | 不做 CRIU | gpu-images、bypass 日志、无 cpu-images |
| GPU + CPU CRIU | 同时由 scheduler 保存显存 | controller/scheduler 触发 CRIU | cpu-images、rootfs.tar、pages/core/pstree、WaitCRDone |

bypass_dump_restore=true 仍然可以和 CPU CRIU 同时存在：它描述的是 scheduler 的 GPU memory 后端，不是“整个 checkpoint 流程跳过 CRIU”。

## 4. 组件交互和数据流

### 4.1 进程与 IPC 拓扑

~~~mermaid
flowchart LR
  subgraph WC[worker0 / worker1 业务容器]
    R[RTP-LLM rank]
    W[epsilon wheel]
    X[/etc/scr/epsilon shim]
    A[/etc/scr/shadow/libaion.so]
    K[GPU KV cache]
    R -->|import| W
    W -->|SCR_ENABLE + external dir| X
    X -->|ctypes load| A
    R -->|register_kv_caches| X
    K -->|address + bytes| X
  end
  subgraph SC[scr scheduler sidecar]
    C[scr_controller CLI]
    S[scr_scheduler]
    U[/run/scr/socket/ttrpc]
    CR[CRIU / RunD]
    G[GPU memory store]
    C -->|RPC| U
    U --> S
    S --> CR
    S --> G
  end
  A -->|native IPC| U
  R -->|snapstart_checkpoint arrival| A
  S -->|release/result| A
~~~

worker0、worker1 的 ttrpc/log.sock inode 实测一致，说明它们在同一个 scheduler socket scope 下；这意味着 worker_num 必须覆盖该 scope 内全部 participant。

### 4.2 Python import 和 native library 选择

~~~mermaid
sequenceDiagram
  participant P as RTP-LLM
  participant W as epsilon wheel
  participant X as external shim
  participant L as libaion.so
  participant S as scheduler

  P->>P: configure_scr_environment
  P->>W: import epsilon
  W->>W: /etc/scr/epsilon + SCR_ENABLE=1 + kernel
  W->>X: load external shim
  X->>X: detect /run/scr/socket
  X->>L: load /etc/scr/shadow/libaion.so
  P->>X: register_kv_caches
  X->>L: data_ptr[] + nbytes[]
  P->>X: snapstart_checkpoint(wait_mode=1)
  X->>L: arrival/barrier
  L->>S: native IPC
~~~

### 4.3 checkpoint/restore 时序

~~~mermaid
sequenceDiagram
  participant Ctrl as scr_controller
  participant S as scr_scheduler
  participant R as RTP-LLM ranks
  participant E as Epsilon/libaion
  participant CR as CRIU + GPU store

  R->>E: register_kv_caches
  R->>E: before_checkpoint(cuda_synchronize)
  R->>E: snapstart_checkpoint(wait_mode=1, id, num, timeout)
  E-->>S: participant arrival
  Ctrl->>S: check
  S-->>Ctrl: checkpoint_ready
  Ctrl->>S: block
  Ctrl->>S: dump(path, bypass-cr-path)
  S->>CR: CPU/process CRIU dump
  S->>CR: GPU memory persist
  CR-->>S: dump complete
  Ctrl->>S: wait-cr-done
  Ctrl->>S: restore(path, bypass-cr-path)
  S->>CR: CPU/process restore
  S->>CR: GPU memory restore
  CR-->>S: restore complete
  Ctrl->>S: wait-cr-done
  R->>R: validate generation/CUDA/cache
  R->>E: re-register KV cache
~~~

## 5. RTP-LLM 当前接入审查

### 5.1 已做对的地方

- 统一 feature gate：RTPLLM_ENABLE_SCR/RTP_LLM_ENABLE_SCR，默认关闭。
- 不在 RTP-LLM 内设置 SCR_PHASE；phase 由 platform/controller 负责。
- lazy import；只导入 scr_template_utils 不初始化 CUDA。
- 区分 wheel-native 与 external-shim，并打印 effective implementation。
- 从主 engine 和可选 draft/MTP/Eagle engine 收集 KV cache Tensor，并按 data pointer 去重。
- EpsilonAdapter 通过 capability/signature 探测 timeout，避免用有副作用的 trial call。
- ScrParticipantManifest 生成连续 participant ID 并验证无重复、无缺号。
- arrival 使用 daemon thread、有限 timeout，不直接阻塞 HTTP/主服务循环。
- RTP-LLM 不直接执行 scr_controller dump/restore，控制面边界正确。

### 5.2 当前风险和建议

#### A. phase 泄漏到数据面

external shim 仍用 SCR_PHASE 判断 active。短期保留兼容，但启动日志必须记录 phase、generation、backend mode，并确保 normal 绝不启动 arrival。长期应让 shim 查询 scheduler activation/generation，SCR_PHASE 降级为兼容字段。

#### B. manifest 不能只靠 rank 猜

建议 controller/launcher 生成只读 participant 清单：

~~~text
scope: serina.wzq.dev.new.manual
generation: <controller generation>
participants:
  backend_rank:0
  backend_rank:1
  backend_manager:0
  backend_vit:0       # 只有 VIT separation 启用时加入
~~~

如果每个容器各自有 scheduler scope，worker_num 是容器 scope 内的 participant 数；如果多个容器共享 ttrpc，则必须使用跨容器全局 manifest。

#### C. after-restore 必须显式补齐

当前 external shim 的 after-restore API 是 no-op。restore 完成后应：

1. controller 更新 generation/release。
2. RTP-LLM 校验 CUDA context 和 model wrapper。
3. 重新发现 KV cache Tensor。
4. 再次 register_kv_caches。
5. health ready 后恢复流量。

不能只看 register_after_restore_func 返回 0。

#### D. fail-open 必须上报 health

普通服务可以 fail-open，但 checkpoint 场景中 arrival timeout 必须成为 controller 的 missing-quorum，而不是继续 dump。建议暴露 registered、arrival_started、arrived、timed_out、restored、generation 等状态。

#### E. generation 必须原子变化

RTP-LLM 已支持 generation 环境变量和 mismatch 日志。controller 每轮 dump/restore 前应原子更新 generation；generation 变化后旧 arrival 结果必须失效。

#### F. timeout 变量需要唯一入口

RTP-LLM 支持多个 timeout alias，而 shim 原生读取 SCR_TIMEOUT。部署只设置 canonical 变量，并打印最终解析值，避免 Python 和 shim timeout 不一致。

## 6. 推荐职责边界

| 层 | 负责 | 不负责 |
|---|---|---|
| controller/supervisor | generation、phase、block/release、调用 dump/restore、quorum | 不写死 Python rank 细节 |
| scheduler/Epsilon/libaion | participant barrier、GPU store、native IPC、CRIU/RunD 协作 | 不理解业务请求 |
| RTP-LLM | cache 注册、CUDA sync、arrival、restore re-register、health | 不调用 controller dump/restore |

正常启动：

~~~text
RTPLLM_ENABLE_SCR unset/false
SCR_ENABLE unset/false
SCR_PHASE normal 或 unset
不启动 Epsilon arrival thread
不注册 KV cache 到 libaion
推理路径与无 SCR 版本一致
~~~

checkpoint：

~~~text
1. controller 生成 generation 和 participant manifest。
2. scheduler health/check 正常。
3. RTP-LLM cache ready 后 register_kv_caches。
4. 每个 participant 一次 arrival。
5. check 确认 quorum。
6. block，停止新请求或切走流量。
7. dump；scheduler 同时处理 GPU memory 与 CPU/process CRIU。
8. wait-cr-done，记录 generation、路径和结果。
~~~

restore：

~~~text
1. controller 选择匹配 generation 的 checkpoint。
2. controller 执行 restore。
3. scheduler/CRIU 恢复 CPU/process/GPU memory。
4. wait-cr-done。
5. controller 更新 release/generation。
6. RTP-LLM 校验并重新注册 KV cache。
7. health ready 后恢复流量。
~~~

## 7. 一键脚本设计要求

脚本放在 sidecar 的独立目录，不改动 scheduler 主进程：

~~~text
/opt/scr-tools/status.sh
/opt/scr-tools/dump.sh
/opt/scr-tools/restore.sh
/opt/scr-tools/roundtrip.sh
~~~

脚本必须：

- 优先解析 /run/scr/scr_controller，允许 SCR_CONTROLLER 覆盖。
- 默认使用 SCR_CR_PATH；路径必须存在且是挂载点。
- 不把 tmpfs 写死；当前真实路径是 ext4。
- 可用 SCR_REQUIRE_TMPFS=1 开启额外 tmpfs 检查。
- 使用 lock 防止并发 dump/restore。
- dump 前解析 check JSON，只有 errno=0 且 checkpoint_ready=true 才执行。
- dump/restore 后统一 wait-cr-done，并支持 SCR_WAIT_TIMEOUT。
- 写入 /run/scr/log/scr-e2e.log。
- 提供 status 和 dry-run，不启动/停止 scheduler，也不修改 SCR_PHASE。
- roundtrip 默认只做 dump 完成后等待，再 restore；真正执行前必须确认会覆盖当前 checkpoint 数据。

当前目录包含 cpu-images、state/blocking、record.json 和约 178GB rootfs.tar；运行 dump 可能覆盖/更新这些数据，不能在生产 checkpoint 目录盲跑。

本次已将脚本部署到 `serina.wzq.dev.new.scr-scheduler` 的 `/opt/scr-tools/`，并完成 `bash -n`、`status`、GPU-only dump dry-run、CPU+CRIU dump dry-run、CPU+CRIU restore dry-run 和 GPU-only roundtrip dry-run。部署时实测 `scr_controller check` 返回 `errno=0` 但 `checkpoint_ready=false`，因此没有执行真实 dump/restore。

## 8. 测试顺序

1. 单卡单 rank：注册、before callback、arrival、GPU-only dump/restore、re-register。
2. 单容器多 rank：验证 manifest ID 和共同 worker_num。
3. prefill/decode 分离：验证是否应分开 scheduler scope。
4. 故意漏掉一个 rank：确认不 dump，并报告 missing quorum。
5. CPU+GPU CRIU：确认 cpu-images/pages/core/pstree 产生，restore 后服务可用。
6. SCR 关闭：确认无 arrival thread、无 libaion 依赖、普通请求延迟不回退。
7. restore 后短请求和长请求各测一次，观察 CUDA context、KV cache 命中和错误。

必须保留的日志字段：

~~~text
scope, generation, phase, backend mode
worker_id, worker_num, role
cache tensor count, total bytes
registration/arrival result
controller check/dump/restore/wait-cr-done result
restore elapsed time
~~~

## 9. 证据和限制

本文中的路径、大小、SHA256、版本、环境变量、挂载类型、历史命令和 controller check 返回值来自目标 sidecar/worker 实测。第三方 libaion.so、scheduler、controller、epsilon daemon 没有源代码；native IPC 内部消息和 GPU store 细节只能依据 CLI、日志、socket、shim 行为描述。

本次调研没有执行真正的 dump 或 restore，避免覆盖现有 checkpoint 数据；只做了 ls、find、stat、mount、env、help、sha256sum、controller check/check-steady-state、历史和日志读取，以及 worker0 代码审查。
