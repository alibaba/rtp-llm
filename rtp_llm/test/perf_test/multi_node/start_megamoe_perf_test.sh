#!/bin/bash
# MegaMoE decode perf test (Option B: input_len<=mtpr so prefill fits the 512 cap).
# Modeled on start_mori_perf_test.sh; swaps MoriEP -> MegaMoE and adds the
# FlyDSL/MegaMoE env. All three fused optimizations are ON.
cd /home/admin/qinhanwen/codes/rtp-llm

export PYTHONPATH=/home/admin/qinhanwen/codes/rtp-llm:/home/admin/qinhanwen/codes/FlyDSL:$PYTHONPATH

export ROCM_PATH=/opt/rocm-7.2.0
export HIP_PATH=/opt/rocm-7.2.0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---- backend selection: MegaMoE ----
export USE_MEGAMOE=1
export USE_MORI_EP=0
export USE_DEEPEP_MOE=0
export USE_DEEPEP_LOW_LATENCY=0
export USE_ALL_GATHER=0
export LOAD_PYTHON_MODEL=1

# ---- MegaMoE / FlyDSL knobs (all optimizations on) ----
export MEGAMOE_MAX_TOK=512          # mtpr cap; input_len<=512 keeps prefill within it
export MORI_SHMEM_HEAP_SIZE=8G
export FLYDSL_FUSE_REQUANT=1        # gathered requant (#2)
export FLYDSL_EMIT_DTM=1            # in-kernel dest_tok_map (#1b)
export TEST_BLOCK_NUM=256           # pin KV blocks (397B weights leave less room)
export MAX_CONTEXT_BATCH_SIZE=1     # one-sequence prefill so cur_tok<=mtpr

export WORLD_SIZE=8
export DP_SIZE=1
export TP_SIZE=8
export EP_SIZE=8

model_path=~/.cache/modelscope/hub/models/Qwen/Qwen3.5-397B-A17B/
export TOKENIZER_PATH=$model_path
export CHECKPOINT_PATH=$model_path
export MODEL_TYPE=qwen35_moe
export ACT_TYPE=bf16

export START_PORT=6655
export FT_SERVER_TEST=1
export CONCURRENCY_LIMIT=128

export DEVICE_RESERVE_MEMORY_BYTES=-2048000000
export RESERVER_RUNTIME_MEM_MB=10240

export SEQ_SIZE_PER_BLOCK=1024
export KERNEL_SEQ_SIZE_PER_BLOCK=16

export ROCM_DISABLE_CUSTOM_AG=True
export FT_DISABLE_CUSTOM_AR=True
export ENABLE_CUDA_GRAPH=0
export USE_ASM_PA=0
export WARM_UP=0

export FAKE_BALANCE_EXPERT=0
export GEN_TIMELINE_SYNC=1
export USE_BATCH_DECODE_SCHEDULER=1
export WORKER_INFO_PORT_NUM=10

export IS_DECODE=1
export DECODE_TEST_LENGTH=20
export INPUT_LEN_LIST="[512]"
export BATCH_SIZE_LIST="[1,4,8,16,32,64]"

export DIST_COMM_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN

echo "=== Starting MegaMoE Decode Perf Test (Option B, input_len=512) ==="
echo "MODEL: $MODEL_TYPE"
echo "CHECKPOINT: $CHECKPOINT_PATH"
echo "CONFIG: DP=${DP_SIZE} TP=${TP_SIZE} EP=${EP_SIZE}  MEGAMOE_MAX_TOK=${MEGAMOE_MAX_TOK}"
echo "DECODE: batch_sizes=${BATCH_SIZE_LIST}, input_lens=${INPUT_LEN_LIST}, decode_len=${DECODE_TEST_LENGTH}"
echo "========================================="

exec /opt/conda310/bin/python3.10 rtp_llm/test/perf_test/multi_node/local_server_runner.py 2>&1
