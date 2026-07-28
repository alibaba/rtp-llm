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

The sources are shared, but the three runtime artifacts are native ELF files.
Build them on the matching architecture; do not copy an aarch64 wheel to an
x86_64 node or vice versa.

```bash
# GB200/GB300 aarch64
bazelisk build \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:keeper_lite_creator \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:keeper_lite_holder \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:mc_shim_unified.so \
  --config=cuda13_arm

# x86_64 CUDA 13
bazelisk build \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:keeper_lite_creator \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:keeper_lite_holder \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:mc_shim_unified.so \
  --config=cuda13
```

## Production runtime contract

The complete CUDA `rtp_llm` wheel ships these executable runtime artifacts at
stable package-relative paths:

```text
rtp_llm/cpp/cuda_checkpoint/multicast_keeper/keeper_lite_holder
rtp_llm/cpp/cuda_checkpoint/multicast_keeper/keeper_lite_creator
rtp_llm/cpp/cuda_checkpoint/multicast_keeper/mc_shim_unified.so
```

The frontend-only wheel does not contain them. CPU and ROCm wheels also omit
the CUDA-only payload. Container images need only install the complete CUDA
wheel; they must not copy `bazel-bin` or rebuild the keeper during image
startup.

Production Level3 startup is owned by the RTP-LLM process supervisor. It
resolves the three paths from the installed `rtp_llm` package, starts the holder
before any backend rank, waits for `--ready-file` plus a successful protocol PING,
and passes `--parent-pid` with its own PID. On Linux the holder arms
`PR_SET_PDEATHSIG=SIGTERM` and verifies the parent after arming it, so supervisor
death cannot leave an unowned holder alive. The supervisor injects the keeper
environment and shim into backend rank environments directly; production code
does not source `keeper.env` or invoke the shell launcher.

The holder is outside the checkpoint PID set. A holder exit is fatal for the
current Level3 instance: do not restart it underneath restored ranks, because
the replacement has a different holder identity and no longer owns their
multicast objects.

The independent Python entry point is
`rtp_llm.utils.multicast_keeper.MulticastKeeperRuntime`. It selects
`single_node_posix` when `world_size == local_world_size`, otherwise
`cross_node_fabric`. The latter starts one holder on each node and injects the
global FABRIC team size while retaining only that node's physical GPU list.
Cross-node handle publication and barriers use the existing RTP-LLM lifecycle
TCPStore; holders remain node-local and never form a second control plane.

The architecture-neutral runtime test starts the native holder in both modes,
checks its protocol identity, verifies every ELF matches the current host, and
preloads the real shim into Python. Run it under both native build configs:

```bash
bazelisk test //rtp_llm/utils/test:multicast_keeper_runtime_test \
  --config=cuda13_arm --test_output=errors

bazelisk test //rtp_llm/utils/test:multicast_keeper_runtime_test \
  --config=cuda13 --test_output=errors
```

## Checkpoint lifecycle

1. The node supervisor starts and verifies the holder, then launches ranks with
   an explicit environment containing the keeper socket and shim preload.
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
| `RTP_LLM_MC_KEEPER_GPUS` | Optional physical GPU-list override for the supervisor |
| `RTP_LLM_MC_LOCAL_GPUS` | Exact holder GPU list injected into ranks for FABRIC validation |
| `RTP_LLM_MC_FABRIC_TEAM_SIZE` | Exact global FABRIC `numDevices`, injected for cross-node runs |
| `RTP_LLM_MC_CREATOR_TIMEOUT_MS` | Creator timeout, default 120 seconds |
| `RTP_LLM_MC_HOLDER_IO_TIMEOUT_MS` | Holder request/reply I/O timeout, default 1 second |
| `RTP_LLM_MC_REQUEST_TIMEOUT_MS` | Shim FETCH/connect deadline, default 5 seconds |
| `RTP_LLM_MC_CREATE_TIMEOUT_MS` | Shim CREATE/connect deadline, default 125 seconds |
| `RTP_LLM_MC_RELEASE_TIMEOUT_MS` | Shim teardown RELEASE/connect deadline, default 1 second |
| `RTP_LLM_MC_OWNER_ID` | Restart-stable owner key; falls back to `LOCAL_RANK`/`RANK` |
| `RTP_LLM_MC_OWNER_GENERATION` | Per-incarnation nonce; defaults to a random value |

Without the opt-in marker, the preloaded shim passes multicast driver calls
through unchanged.
