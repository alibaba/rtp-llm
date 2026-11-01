# Cross-host RTP Engram component integration

This harness exercises the real `HostEngramEmbedding`, Triton lookup/hash and
guarded CUDA Graph path. It is not a complete model server or HTTP acceptance
test. Owner and consumers use the same source/dependency versions. Confirm
HOST_NUMA/FABRIC, IMEX authorization and the intended cross-host topology first.

## Owner

Build `native/load_native.py` once in each development environment. Use private
run directories and explicitly reserved GPUs. `owner_fixture.py` creates two
small deterministic layers. `checkpoint_owner.py` writes equivalent standard
safetensors at a fresh local path and exercises the real checkpoint loader.

From the repository root, on the owner host:

```bash
umask 077
engram_owner_run=$(mktemp -d /tmp/engram-owner.XXXXXX)
engram_checkpoint="$engram_owner_run/checkpoint"
export CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_OWNER_UUID
export ENGRAM_NATIVE_BUILD_DIR="$HOME/.cache/rtp_llm/engram_native"
python tools/engram_sidecar/integration/checkpoint_owner.py \
  --directory "$engram_owner_run" --checkpoint "$engram_checkpoint" \
  --device 0 --numa REPLACE_WITH_LOCAL_NUMA_NODE \
  > "$engram_owner_run/owner.jsonl" 2>&1 &
engram_owner_pid=$!
```

Wait for `READY` in the owner's log. The owner holds allocations until an
explicit `STOP` marker. `--timeout-seconds` reports timeout while retaining
published memory; it does not authorize freeing an active allocation. Do not
hard-timeout an owner while consumers may still access its memory.

## Consumer

Use authenticated SSH/SCP between confirmed development-container endpoints.
Create a mode-0700 consumer directory and transfer only `manifest.json` and
`capabilities.json`; enforce 0600 capability-file permissions. Do not transfer
the checkpoint, complete weight table or `STOP`, and do not print capabilities.
Use the same publication for every consumer.

On another host/container, from its repository root:

```bash
export CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_CONSUMER_UUID
export ENGRAM_NATIVE_BUILD_DIR="$HOME/.cache/rtp_llm/engram_native"
engram_consumer_run=/tmp/REPLACE_WITH_PRIVATE_CONSUMER_DIRECTORY
chmod 700 "$engram_consumer_run"
chmod 600 "$engram_consumer_run/capabilities.json"
timeout --kill-after=10s 300s python tools/engram_sidecar/integration/consumer.py \
  --directory "$engram_consumer_run" --device 0 --label cross-host \
  --graph-replays 10000 --exercise-hash-pipeline \
  > "$engram_consumer_run/consumer.jsonl" 2>&1
```

A same-host second-GPU run is useful but cannot replace the cross-host run.
`--exercise-device-guard` is a separate two-visible-GPU test; do not enable it
with the single-GPU command above.

## Acceptance and cleanup

Require successful exit, exact independent expected values, common generation
and allocation IDs, correct pointer base+offset, read-only permissions, and
zero remaining native resource counters after Graph/view/storage cleanup.
A successful import alone is not a successful GPU read. A zero PyTorch table
allocation delta does not imply zero context/page-table/RSS overhead.

Only after **every consumer** has exited successfully with completed cleanup:

```bash
touch "$engram_owner_run/STOP"
wait "$engram_owner_pid"
```

Require owner's `CLOSED` with zero counters. Delete copied private capability
files after cleanup. If a consumer fails or its GPU drain is uncertain, retain
the owner until the process/context state is resolved. `STOP` is a coordinator
authorization, not distributed fencing or automatic discovery.

Full-model checkpoint capacity, full-engine Graph lifetime, request serving,
latency SLOs and hard fault recovery remain separate acceptance gates.
