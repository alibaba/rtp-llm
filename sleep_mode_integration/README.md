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
coordinator and all backend ranks to share the host boot and PID namespace.

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
