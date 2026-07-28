# Sleep Mode Level 3 Integration

This harness follows the deterministic flow in
`internal_source/freeze_resume_integration`, adapted to the level-3 sleep API:

1. Health and baseline inference.
2. Verify `RUNNING` status and capture GPU/backend PID memory.
3. Call `POST /sleep` with `level=3`.
4. Verify `CHECKPOINTED`, released GPU resources, and zero backend PID GPU memory.
5. Verify inference is rejected and repeated sleep is idempotent.
6. Call `POST /wake_up`.
7. Verify GPU memory restoration, post-wake inference, deterministic output, and final status.

Start a local server in one terminal:

```bash
MODEL_PATH=/path/to/model GPU=3 PORT=39080 \
  ./sleep_mode_integration/start_level3_server.sh
```

Run the deterministic scenario in another terminal:

```bash
GPU=3 PORT=39080 MODEL=deepseek-v4-flash \
  /opt/conda310/bin/python3 sleep_mode_integration/run_level3_scenario.py
```

The launcher uses the TP1 SM100 smoke-test settings from
`internal_source/rtp_llm/test/smoke/BUILD`. It uses the internal integration
flow's `HACK_LAYER_NUM=4` default and limits the E2E KV pool to 64 blocks so the
model and test cache fit on one local L20D. Set
`HACK_LAYER_NUM=0 TEST_BLOCK_NUM=0` to restore full-model production sizing.
`MODEL_PATH` is required; `MODEL_TYPE`, `GPU`, `PORT`, and the remaining settings
can be overridden for another deployment. Level 3 requires the frontend
coordinator and backend ranks to share the host boot and PID namespace only on
the single-node external-controller path. The cross-node path below invokes the
Driver API inside each backend rank and does not inspect remote `/proc`.

For a same-node TP2 run:

```bash
MODEL_PATH=/path/to/model GPU=1,2 TP_SIZE=2 WORLD_SIZE=2 PORT=39080 \
  ./sleep_mode_integration/start_level3_server.sh

GPU=1,2 EXPECTED_RANKS=2 PORT=39080 MODEL=deepseek-v4-flash \
  /opt/conda310/bin/python3 sleep_mode_integration/run_level3_scenario.py
```

## DeepSeek V4 Flash PD CP2 Level 3

The DSV4 prefill implementation represents context parallelism with the TP
process group. Therefore CP2 is encoded as `--tp_size 2` together with
`--cp_rotate_method ALL_GATHER`; it is not a tensor-parallel model-sharding
configuration. The matching decode role is TP1/DP2/EP2 and uses
`--cp_rotate_method PREFILL_CP`.

Run the read-only preflight first. It validates the model, topology text,
ports, GPU availability, RDMA, memlock escalation, CUDA13 Python dependencies,
and the NVLS/multicast keeper without starting or stopping a service:

```bash
MODEL_DIR=/mnt/nas1/hf/DeepSeek-V4-Flash \
DECODE_GPUS=4,5 PREFILL_GPUS=6,7 \
./sleep_mode_integration/preflight_level3_pd_cp2.sh
```

After the selected GPUs are idle, start the two roles in one terminal. Level 3
must be selected explicitly because the shared launcher defaults to level 1:

```bash
SLEEP_MODE_LEVEL=3 CACHE_STORE_RDMA_MODE=1 \
MODEL_DIR=/mnt/nas1/hf/DeepSeek-V4-Flash \
DECODE_GPUS=4,5 PREFILL_GPUS=6,7 \
DECODE_PORT=21000 PREFILL_PORT=22000 \
LOG_DIR=/tmp/diag_pd_level3_cp2 \
./start_rtp_pd_cuda13_sleep.sh
```

Run the three-cycle E2E from another terminal:

```bash
SLEEP_WAKE_CYCLES=3 EXPECTED_RANKS_PER_ROLE=2 \
DECODE_GPUS=4,5 PREFILL_GPUS=6,7 \
DECODE_PORT=21000 PREFILL_PORT=22000 \
DECODE_CONTROL_ADDRESSES=127.0.0.1:21001,127.0.0.1:21010 \
PREFILL_CONTROL_ADDRESSES=127.0.0.1:22001,127.0.0.1:22010 \
SUMMARY_PATH=/tmp/diag_pd_level3_cp2/e2e_summary.json \
/opt/conda310/bin/python3 sleep_mode_integration/run_level3_pd_scenario.py
```

The scenario compares every pre-sleep and post-wake response against both the
first baseline and the CP2 smoke golden, checks all four backend process
identities and epochs, requires zero GPU memory for each backend PID while
checkpointed, and requires each physical GPU to be within 64 MiB of the
configured baseline.

For a continuous physical-memory trace during the scenario:

```bash
while sleep 0.5; do
  date '+%s.%N'
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, '$1 ~ /^[[:space:]]*[4-7]$/ {print}'
done | tee /tmp/diag_pd_level3_cp2/gpu_memory.log
```

## Two-node CP8 Prefill Level 3

Cross-node Level 3 uses rank-local Driver API calls; the frontend never opens a
remote `/proc/<pid>`. The shared distributed TCPStore stores one transaction
manifest containing every rank identity and the distinct node-local multicast
holder identities. Driver operations are sequential in world-rank order:

```text
all ranks SLEEPING
  -> LOCK rank 0..7 -> verify all LOCKED
  -> CHECKPOINT rank 0..7 -> verify all CHECKPOINTED
  -> RESTORE rank 0..7 -> verify all LOCKED
  -> UNLOCK rank 0..7 -> verify all RUNNING
```

Both nodes must run the same build and pass NVIDIA's checkpoint/migration demo
on their installed driver. For the current two-node GB200 pair, node 0 is
`11.139.19.52` and node 1 is `11.139.19.54`. Use the same gang string and base
port on both:

```bash
bazelisk build //rtp_llm:rtp_llm_aarch64 \
  --verbose_failures --config=cuda13_arm --jobs=64

/opt/conda310/bin/pip install --force-reinstall --no-deps \
  bazel-bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-linux_aarch64.whl
```

If the generated filename has a more specific `manylinux_*_aarch64` platform
tag, install that file instead. Verify `uname -m` reports `aarch64`; never copy
or install the `manylinux1_x86_64` wheel on these nodes.

```bash
export MODEL_PATH=/Deepseek-V4-Flash
export GPU=0,1,2,3
export WORLD_SIZE=8
export LOCAL_WORLD_SIZE=4
export TP_SIZE=8
export EP_SIZE=8
export DP_SIZE=1
export ROLE_TYPE=PREFILL
export CP_ROTATE_METHOD=ALL_GATHER
export PORT=22000
export HACK_LAYER_NUM=0
export TEST_BLOCK_NUM=0
export CACHE_STORE_RDMA_MODE=1
export RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER=1
export GANG_CONFIG_STRING='name:dsv4_part0,ip:11.139.19.52,port:22000;name:dsv4_part1,ip:11.139.19.54,port:22000'
```

On node 0:

```bash
export WORLD_RANK=0
./sleep_mode_integration/start_level3_server.sh \
  > /tmp/dsv4-cp8-node0.log 2>&1
```

On node 1:

```bash
export WORLD_RANK=4
./sleep_mode_integration/start_level3_server.sh \
  > /tmp/dsv4-cp8-node1.log 2>&1
```

After both nodes are ready, drive the public endpoint on node 0:

```bash
curl -sS http://11.139.19.52:22000/sleep_status | jq
curl -sS -X POST http://11.139.19.52:22000/sleep \
  -H 'content-type: application/json' \
  -d '{"level":3,"mode":"wait","timeout_ms":3600000}' | jq
curl -sS http://11.139.19.52:22000/sleep_status | jq
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
curl -sS -X POST http://11.139.19.52:22000/wake_up \
  -H 'content-type: application/json' -d '{}' | jq
curl -sS http://11.139.19.52:22000/sleep_status | jq
```

Do not kill or restart either node's holder while the status is
`CHECKPOINTED`. A failed phase is rolled back to all-RUNNING when every rank is
reachable; otherwise the shared manifest is marked `RECOVERY_REQUIRED` and the
instance fails closed.

## Native CUDA Checkpoint Probe

`cuda_checkpoint_native_probe.c` isolates CUDA Driver API checkpoint/restore
from RTP-LLM, Python, PyTorch, ctypes, and the `cuda-checkpoint` CLI. The default
mode uses an external controller process. `--self` follows NVIDIA's R580
migration sample and makes the CUDA target control its own checkpoint.

```bash
gcc -std=c11 -O2 -Wall -Wextra -Werror \
  -I/usr/local/cuda/include \
  sleep_mode_integration/cuda_checkpoint_native_probe.c \
  -L/usr/lib64 -Wl,-rpath,/usr/lib64 -lcuda \
  -o /tmp/cuda_checkpoint_native_probe

CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
  /tmp/cuda_checkpoint_native_probe

CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
  /tmp/cuda_checkpoint_native_probe --self
```

A passing run must complete `Lock`, `Checkpoint`, `Restore`, and `Unlock`, then
verify the deterministic device-memory pattern after restoration. Symbol
presence or a successful `Restore` without a successful `Unlock` is not treated
as checkpoint/restore support.

## Level 3 failure diagnostics

The test environment does not need a working checkpoint Driver API to produce a
useful failure report. Collect the frontend log and every backend-rank log, then
filter the structured Level 3 lines:

```bash
grep -E \
  'level3 |level-3 |sleep lifecycle|multicast keeper|multicast-holder|rtp-mc-shim' \
  /path/to/frontend.log /path/to/backend*.log
```

Interpret the last completed line as follows:

- `checkpoint preflight failed` together with `cuda_result`/`driver_state`
  means the Driver API rejected the operation before GPU resources were
  released. This is the expected fail-safe behavior on an unsupported driver.
- `checkpoint rpc begin` without the matching `checkpoint rpc end` identifies
  a rank blocked inside the Driver API. The line contains its rank, address,
  transaction, epoch, action, and timeout.
- `sleep lifecycle hook end ... success=0` identifies the exact backend release
  or rebuild hook that failed, such as RDMA teardown, KV backing, collective
  rebuild, graph recapture, or health check.
- `distributed manifest` shows the durable transaction phase and every rank's
  last confirmed CUDA state. `RECOVERY_REQUIRED` means do not serve traffic or
  remove the manifest manually; restart or perform explicit recovery.
- `Multicast keeper health probe failed` includes holder identity, topology,
  PID/socket information, and the holder log tail captured before private state
  cleanup.

For a Driver compatibility issue, retain the native probe output together with
`nvidia-smi -q`, `uname -a`, the wheel filename, and the filtered Level 3 logs.
