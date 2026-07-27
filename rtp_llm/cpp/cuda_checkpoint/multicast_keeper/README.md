# CUDA checkpoint multicast keeper

This component keeps NCCL NVLS and PyTorch symmetric-memory multicast enabled
across rank-local CUDA checkpoint/restore. A multicast object's ready fabric
state is held by a separate process while checkpointed ranks destroy and later
rebuild their NCCL and symmetric-memory resources.

The holder is deliberately CUDA-free. Protocol V3 uses explicit `CREATE` and
`FETCH` operations. Every create gets a unique object ID under a random holder
instance ID; fetch requires that exact identity and verifies all object
properties. Size is validation data, never an object key. The transient creator
deposits the ready multicast object's POSIX FD and exits, while the holder keeps
the object alive. The creating rank retains the object identity after
`cuMemRelease`, so a wake rebuild fetches the same keeper object.

Before its first keeper-backed import, each rank retains the primary contexts
for all GPUs visible to that role. NVIDIA's NVLS multi-process import requires
this peer-context participation even though a rank only binds memory on its
local GPU. The retained contexts belong to the rank and are released from the
physical GPUs by its Level3 CUDA checkpoint; the holder remains CUDA-free.

For NCCL's POSIX-handle exchange, the shim exports a sealed 64-byte memfd token.
The versioned token contains the holder instance, object ID, and complete
requested properties. Cross-machine FABRIC exchange uses CUDA's real opaque
64-byte handle. Raw FABRIC imports first pass through unchanged because ordinary
allocations use the same ABI; a later `cuMulticastAddDevice` proves that a
handle is multicast and promotes it to the peer holder. The holder imports it
once, adds its configured local GPU list, and returns node-local POSIX FDs.
Tokens from a restarted holder are rejected instead of accidentally resolving
to a same-size object.

## Safety boundary

The generic CUDA Driver API does not expose a communicator ID to
`cuMulticastCreate`. The shim therefore supports one active locally-created
object for each exact `(size, numDevices, handleTypes, flags)` tuple. A second
simultaneous object with identical properties fails closed; after release, the
single saved identity is treated as the wake rebuild. No call ordinal or timing
heuristic is used. Applications requiring multiple identical-property local
creators need an explicit communicator identity API before enabling this shim.

Single-node POSIX multicast requires `numDevices` to equal the holder's complete
local GPU list. Cross-machine FABRIC additionally requires
`--fabric-team-size N`; every request must use exactly that global `numDevices`.
The generated `keeper.env` exports the same local list and global size, and the
shim rejects a rank unless its visible GPU list exactly matches the holder list.
These checks keep incomplete or mixed NVLink partitions from reaching a CUDA
bind/map that waits for missing team members. `flags` must be zero and
`handleTypes` may only contain POSIX FD and FABRIC.

## Object lifecycle and ownership

Every keeper object has a set of process-incarnation owners. `CREATE` registers
the creator and `IMPORT_ADD` idempotently registers each exact
`(owner_id, owner_generation)` peer. Repeated raw-handle imports during one
process's checkpoint/rebuild cycle do not add references. On normal process exit
the shim releases its peer references; the holder closes the FD only after the
last owner releases it.

`CREATE`/`FETCH`/`RELEASE` carry owner attribution in an 80-byte extended request
(a superset of the 64-byte base request; `PING` and legacy clients keep the base
form, which the holder still accepts). Two owner fields drive lifecycle:

- `owner_id` — a logical, restart-stable owner key. The shim reads
  `RTP_LLM_MC_OWNER_ID`, else derives it from `LOCAL_RANK`/`RANK` (biased by one
  so rank 0 is a real owner), else `0` (anonymous, no generation reclamation).
- `owner_generation` — a per-incarnation nonce, stable across a
  checkpoint/restore cycle but new on every relaunch (`RTP_LLM_MC_OWNER_GENERATION`
  or a random value).

`RELEASE` removes the exact owner's reference. The shim sends it only at process
teardown (a destructor), never during a checkpoint and never on an intermediate
`cuMemRelease`; the peer registration therefore survives repeated rebuilds.
`RELEASE` is authenticated by holder instance, object id, owner id, and owner
generation; a stale, foreign, or unknown release fails closed.

A fresh `CREATE` or `IMPORT_ADD` from a known owner removes that owner's stale
references with a different generation. Other owners keep the shared entry
alive, so backend restart cleanup is independent of which local rank imports
first. Objects of the same owner and same generation are kept.

See [UPSTREAM.md](UPSTREAM.md) for provenance and differences from the nekyia
prototype.

## Build

```bash
bazelisk build \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:keeper_lite_creator \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:keeper_lite_holder \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:mc_shim_unified.so \
  --config=cuda13
```

## Start and launch ranks

Start one holder for the local GPU team before launching any checkpointed rank:

```bash
KEEPER=./bazel-bin/rtp_llm/cpp/cuda_checkpoint/multicast_keeper/multicast_keeper
"${KEEPER}" start --gpus 0,1 --keeper-dir /run/user/${UID}/rtp-llm-mc
source /run/user/${UID}/rtp-llm-mc/keeper.env
exec your-rank-launcher
```

The generated environment:

- sets `RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER=1`, the RTP-LLM opt-in/ready marker;
- sets `NEKYIA_KEEPER_DIR`; the default socket remains
  `$NEKYIA_KEEPER_DIR/mcsk.sock`;
- appends `mc_shim_unified.so` to `LD_PRELOAD` and preserves an existing
  torch_memory_saver preload. The TMS interposer must stay first because it
  resolves the real `cudaMalloc` through `RTLD_NEXT`;
- defaults `NCCL_NVLS_ENABLE=1` and
  `TORCH_SYMM_MEM_DISABLE_MULTICAST=0` only when the caller did not configure
  them. This component never defaults multicast off.

An explicit `--socket` is supported. The launcher then also exports
`RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET` for the shim.

For a cross-machine eight-GPU FABRIC team, start each node's holder with the
same global size and that node's exact local physical GPU list:

```bash
"${KEEPER}" start --gpus 0,1,2,3 --fabric-team-size 8 \
  --keeper-dir /run/user/${UID}/rtp-llm-mc-${JOB_ID}
source /run/user/${UID}/rtp-llm-mc-${JOB_ID}/keeper.env
exec torchrun ...
```

For a foreground command with automatic cleanup:

```bash
"${KEEPER}" run --gpus 0,1 --keeper-dir /tmp/my-keeper -- torchrun ...
```

For a detached service, use `start`, `status`, and `stop`. `SIGTERM` closes all
cached FDs and removes the socket and ready file. Stop the holder only after all
ranks have exited; killing the sole FD holder frees the multicast objects.

## Checkpoint lifecycle

1. Start the holder outside the rank process tree and source `keeper.env`.
2. Launch all ranks with the unified shim preloaded.
3. Before checkpoint, quiesce work and destroy rank-side symmetric-memory
   handles/mappings, NCCL communicators, and other CUDA collectives.
4. Checkpoint only rank PIDs. Never pass the holder PID to `cuda-checkpoint`.
5. Restore and unlock every rank.
6. Rebuild NCCL and symmetric-memory resources. The shim reimports the cached
   multicast object from the surviving holder.

The creator GPU list uses physical CUDA ordinals. The launcher removes
`CUDA_VISIBLE_DEVICES` from the holder process so a rank-specific logical
visibility mapping cannot reinterpret those ordinals.

## Configuration

| Variable | Purpose |
| --- | --- |
| `RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER=1` | Enables keeper interception in the shim |
| `NEKYIA_KEEPER_DIR` | Directory containing `mcsk.sock` |
| `RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET` | Optional exact socket override |
| `RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER_DEBUG=1` | Verbose shim logging |
| `RTP_LLM_MC_KEEPER_GPUS` | Launcher default for `--gpus` |
| `RTP_LLM_MC_LOCAL_GPUS` | Exact holder GPU list exported to ranks for FABRIC validation |
| `RTP_LLM_MC_FABRIC_TEAM_SIZE` | Launcher default for `--fabric-team-size`; exact global FABRIC `numDevices` |
| `RTP_LLM_MC_CREATOR_TIMEOUT_MS` | Creator timeout, default 120 seconds |
| `RTP_LLM_MC_HOLDER_IO_TIMEOUT_MS` | Holder request/reply I/O timeout, default 1 second |
| `RTP_LLM_MC_REQUEST_TIMEOUT_MS` | Shim FETCH/connect deadline, default 5 seconds |
| `RTP_LLM_MC_CREATE_TIMEOUT_MS` | Shim CREATE/connect deadline, default 125 seconds |
| `RTP_LLM_MC_RELEASE_TIMEOUT_MS` | Shim teardown RELEASE/connect deadline, default 1 second |
| `RTP_LLM_MC_OWNER_ID` | Restart-stable owner key; falls back to `LOCAL_RANK`/`RANK` |
| `RTP_LLM_MC_OWNER_GENERATION` | Per-incarnation nonce; defaults to a random value |

Without the opt-in marker, the preloaded shim passes multicast driver calls
through unchanged.

## Tests

CPU tests cover unique same-size objects, exact rebuild reuse, stale holder
tokens, subgroup/property rejection, half/silent clients, creator timeout,
readiness, duplicate-holder protection, CUDA-free holder operation, signal
cleanup, `RELEASE` freeing a slot and re-registration, fail-closed release of
unknown/foreign/stale objects, owner-generation orphan reclamation, and
capacity-exhaustion fail-closed, exact FABRIC team contracts, ordinary FABRIC
import passthrough, AddDevice-based multicast promotion and unknown-size
correlation, idempotent peer references, last-owner release, and backend restart
with different first importers:

```bash
bazelisk test \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:multicast_keeper_test \
  --config=cuda13
```

The manual GPU target runs real NCCL plus PyTorch symmetric memory before and
after gang CUDA checkpoint/restore. It requires at least two NVLink GPUs. An
explicit `CUDA_CHECKPOINT_BIN` or `cuda-checkpoint` from `PATH` takes priority;
when neither exists, the test automatically uses
`rtp_llm.utils.checkpoint_controller.LibCudaCheckpointDriver`:

```bash
GPUS=0,1 \
  bazelisk test \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:multicast_keeper_gpu_test \
  --config=cuda13 --test_output=streamed
```

The test asserts nonzero symmetric-memory multicast pointers before and after
restore, identical collective output, unchanged rank/holder PIDs, and a live
holder throughout the checkpoint window.
