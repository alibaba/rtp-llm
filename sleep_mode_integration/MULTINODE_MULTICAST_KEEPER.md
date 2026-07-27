# Two-node multicast keeper validation

This is a component test for the multicast keeper, CUDA FABRIC handles, NCCL
NVLS, and PyTorch symmetric memory on two GB200/GB300 nodes. It does not enable
RTP-LLM Level 3 multi-node checkpointing. The frontend multi-node guard remains
in place until checkpoint orchestration and durable per-node holder manifests
exist.

The test creates one CUDA-free holder per node and per role/job. Every local
rank uses the same ordered local GPU team, while
`RTP_MC_TEST_GLOBAL_TEAM_SIZE` is the explicit full cross-node team size. The
worker validates the global rank/GPU/holder map, performs the initial symmetric
memory rendezvous, runs multiple destroy/rebuild generations, verifies the NCCL
collective result and a nonzero `multicast_ptr`, and checks that the node-local
holder identity remains stable.

After the successful rounds, one rank stops its node-local holder while the
current NCCL group is still usable. All ranks exchange their local holder
readiness and must reject the next destructive phase. This validates peer
holder failure as a bounded, fail-closed precondition instead of relying on a
rebuild timeout.

## Build

Build the same commit and CUDA configuration on both nodes:

```bash
bazelisk build \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:multinode_multicast_keeper_test \
  --config=cuda13
```

The test intentionally requires the driver, CUDA runtime, PyTorch, and NCCL
versions to match across nodes. It also requires every selected GPU to report
FABRIC `Completed/Success` and, by default, requires the IMEX channel at
`/dev/nvidia-caps-imex-channels/channel0`. Override
`RTP_MC_TEST_IMEX_CHANNEL` if the deployment exposes a different channel.

## Bazel runner

Choose one reachable address on node 0 and a unique job ID. Run preflight on
both nodes before starting the test. The examples use two GPUs per node and one
prefill role.

Node 0:

```bash
export RTP_MC_TEST_JOB_ID=gb300-nvls-001
export RTP_MC_TEST_ROLE=prefill
export RTP_MC_TEST_NODE_RANK=0
export RTP_MC_TEST_NNODES=2
export RTP_MC_TEST_GLOBAL_TEAM_SIZE=4
export RTP_MC_TEST_LOCAL_GPUS=0,1
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=39420

bazelisk run \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:multinode_multicast_keeper_test \
  --config=cuda13 -- preflight
```

Node 1 uses the same values except for its node rank and, if necessary, its
physical GPU list:

```bash
export RTP_MC_TEST_JOB_ID=gb300-nvls-001
export RTP_MC_TEST_ROLE=prefill
export RTP_MC_TEST_NODE_RANK=1
export RTP_MC_TEST_NNODES=2
export RTP_MC_TEST_GLOBAL_TEAM_SIZE=4
export RTP_MC_TEST_LOCAL_GPUS=0,1
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=39420

bazelisk run \
  //rtp_llm/cpp/cuda_checkpoint/multicast_keeper:multinode_multicast_keeper_test \
  --config=cuda13 -- preflight
```

After both preflights pass, replace `preflight` with `run` on both nodes. Start
node 0 first and node 1 immediately afterward. The default run executes three
successful rebuilds and then kills node 1's holder. Each node must print
`MULTINODE_MULTICAST_TEST_PASS`.

Use a different `RTP_MC_TEST_ROLE`, `MASTER_PORT`, and preferably job ID when
testing decode concurrently. This yields distinct paths such as:

```text
/tmp/rtp_mc_multinode/gb300-nvls-001/prefill/node-0/keeper
/tmp/rtp_mc_multinode/gb300-nvls-001/prefill/node-1/keeper
/tmp/rtp_mc_multinode/gb300-nvls-001/decode/node-0/keeper
/tmp/rtp_mc_multinode/gb300-nvls-001/decode/node-1/keeper
```

Artifacts are retained under the job directory. Reusing an existing job path
is rejected so a stale socket or log cannot be mistaken for the current run.

## torchrun mapping

The Bazel runner launches the local ranks directly so it can own and verify the
node-local holder PID. Its rank mapping is the same as a balanced two-node
`torchrun` launch:

```text
WORLD_SIZE       = RTP_MC_TEST_GLOBAL_TEAM_SIZE
LOCAL_WORLD_SIZE = number of RTP_MC_TEST_LOCAL_GPUS
RANK              = NODE_RANK * LOCAL_WORLD_SIZE + LOCAL_RANK
```

For integration with an existing `torchrun` scheduler, start one holder for
each `(job, role, node)` through the production launcher and source its generated
`keeper.env`. The global team contract is mandatory. The following preparation
is performed independently on each node:

```bash
KEEPER=./bazel-bin/rtp_llm/cpp/cuda_checkpoint/multicast_keeper/multicast_keeper
KEEPER_DIR="/tmp/rtp_mc_multinode/${RTP_MC_TEST_JOB_ID}/${RTP_MC_TEST_ROLE}/node-${RTP_MC_TEST_NODE_RANK}/keeper"

"${KEEPER}" start \
  --gpus "${RTP_MC_TEST_LOCAL_GPUS}" \
  --fabric-team-size "${RTP_MC_TEST_GLOBAL_TEAM_SIZE}" \
  --keeper-dir "${KEEPER_DIR}"
source "${KEEPER_DIR}/keeper.env"

export RTP_MC_TEST_HOLDER_PID="$(awk '{print $1}' "${KEEPER_DIR}/holder.pid")"
export RTP_MC_TEST_NNODES=2
export RTP_MC_TEST_SUCCESS_ROUNDS=3
export RTP_MC_TEST_FAIL_HOLDER_NODE=1
export RTP_MC_TEST_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
export RTP_MC_TEST_FABRIC_STATUS=Completed/Success,Completed/Success
export RTP_MC_TEST_LOCAL_GPU_UUIDS="$(
  for gpu in ${RTP_MC_TEST_LOCAL_GPUS//,/ }; do
    nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader
  done | paste -sd,
)"
```

After preparation, the corresponding launch has this shape on each node:

```bash
CUDA_VISIBLE_DEVICES="${RTP_MC_TEST_LOCAL_GPUS}" torchrun \
  --nnodes=2 --nproc-per-node=2 \
  --node-rank="${RTP_MC_TEST_NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  rtp_llm/cpp/cuda_checkpoint/multicast_keeper/tests/multinode_multicast_worker.py
```

Use the Bazel runner for the failure-injection acceptance test because it
passes the exact holder PID and verifies that the expected holder, rather than
an unrelated process, exited. With direct `torchrun`, stop the surviving holder
manually after the test.

## Acceptance signals

The run is successful only when all of the following are present:

- `TOPOLOGY_OK` from global rank 0;
- one `READY` and the configured number of `ROUND_OK` records per rank;
- a nonzero `multicast_ptr` in every generation;
- stable holder identities and distinct holder identities between nodes;
- identical all-reduce results in every generation;
- a FABRIC-capable request (`handles=0x8` or `handles=0x9`) in each holder log;
- NVLS evidence in NCCL logs;
- `FAIL_CLOSED` and `TEST_PASS` from every rank after peer-holder failure.

If the two machines are not members of the same usable FABRIC/IMEX domain, the
preflight or initial rendezvous must fail. Do not disable the checks and treat
an NVLS-off result as a passing GB200/GB300 cross-node keeper validation.
