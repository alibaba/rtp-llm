# DSV4 MTP fake decode host memory leak

## Summary

On 2026-08-13, a DSV4 decode deployment showed linear cgroup memory growth after restart even though it received no inference traffic. The growth came from pinned host buffers retained by the MTP speculative-prefill CUDA Graph model while DP synchronization kept the idle engine executing fake decode streams.

The affected deployment used:

- MTP speculative decoding;
- decode role with `dp_size > 1`;
- `ENABLE_CUDA_GRAPH=1`;
- a separate `sp_prefill_draft_model_` for speculative prefill.

BlockTreeCache was disabled on decode. The leak did not originate from BlockTreeCache, CacheStore, RDMA buffers, or GPU KV cache.

## Production evidence

The following observations were collected directly from the host and container:

- All four rank processes grew at the same rate.
- Over a 20-second sample, each rank added about 17 MiB of anonymous memory and about 4 MiB of `/dev/zero (deleted)` mappings.
- The aggregate container RSS slope was approximately 15--16 GiB per hour.
- GPU memory for all rank processes remained unchanged over the same sample.
- Per-thread minor-fault deltas were attributed to `normal_engine_loop`; CacheStore, BlockTree, ACCL, and monitoring threads showed no corresponding growth.
- Access logs contained health checks but no inference requests.

The `/dev/zero (deleted)` mappings were 2 MiB allocations associated with CUDA/PyTorch pinned host memory. This explains why cgroup memory increased while GPU memory stayed flat.

## Trigger path

For `dp_size > 1` and TP rank 0, `FIFOSchedulerBase` enables fake-stream filling. `FIFOScheduler::schedule()` wakes every 10 ms even when the request queues are empty. `NormalEngine::mayAddFakeStream()` then creates an MTP fake decode stream for the decode role.

The fake stream still executes the MTP target-verify and draft-prefill path. With CUDA Graph enabled, draft prefill runs through:

```cpp
sp_prefill_draft_model_->forward(model_input)
```

`PyWrappedModel::tensorHoldHostAndToCuda()` retains pinned CPU tensors in the model's private `buffer_holder_` until `PyWrappedModel::releaseBuffers()` is called.

Before this fix, MTP phase boundaries and return paths released only `model_` and `draft_model_`. They never released `sp_prefill_draft_model_`. In particular, the fake-stream early return executed roughly every 10 ms and retained another set of pinned host buffers on every iteration.

The fake stream is required by the current DP/EP execution model. Removing it is not part of this fix.

## Commit and branch history

The separate speculative-prefill model was introduced by:

- `99844ec2c354223d4322a1050c280b921755e407`
- Author: `tanboyu.tby`
- Subject: `feat: support sp prefill cuda graph`
- Date: 2026-04-08

That change added `sp_prefill_draft_model_` and routed speculative-prefill forward calls through it, but did not add the model to the buffer-release paths.

A later branch fixed the leak explicitly in:

- `6cd3d59884e6db75ff1295f2ce7df3aa95fcc64c`
- Author: `huzetao.hzt`
- Subject: `fix: fix mtp mem-leak`
- Date: 2026-05-19

The deployed branch contained `99844ec2` but not that fix. The deployment image commit `63bdeae00561233e620448774d45d787d5ffd582` did not introduce the MTP leak; it inherited the missing release from its branch history.

The original DSV4 branch at `6bfd3d2ca37c01e06f2790f6542024f058e5f9ca` uses `releaseAllModelBuffers()` to release:

1. executor-owned staging tensors;
2. target-model buffers;
3. draft-model buffers;
4. the optional speculative-prefill draft-model buffers.

It invokes the helper at MTP phase boundaries and before fake-stream and normal return paths.

## Fix

This change ports the original branch's unified release pattern into the deployed branch:

```cpp
void MtpExecutor::releaseAllModelBuffers() {
    buffer_holder_.release();
    model_->releaseBuffers();
    draft_model_->releaseBuffers();
    if (sp_prefill_draft_model_) {
        sp_prefill_draft_model_->releaseBuffers();
    }
}
```

All existing paired `model_`/`draft_model_` release sites in MTP prefill and decode now use the helper. This covers:

- the phase boundary before model forward;
- warm-up, empty-stream, non-TP-rank-0, and fake-stream early returns;
- successful prefill and decode dispatch returns.

Releasing at the next phase boundary bounds the holder lifetime during ordinary execution. Releasing after `cudaSyncAndCheck()` on the fake path also guarantees that asynchronous H2D copies have finished before their pinned sources are dropped.

## Validation on a DSV4 machine

Use the same model and topology as the affected deployment, including `dp_size=4`, decode role, MTP, and `ENABLE_CUDA_GRAPH=1`.

1. Start the decode service and leave it without inference traffic for at least 20 minutes after CUDA Graph warm-up.
2. Record cgroup `memory.current` and the `anon`, `file`, and `shmem` fields from `memory.stat` every 10 seconds.
3. For every rank process, sample `Rss`, `Anonymous`, and `Private_Dirty` from `/proc/<pid>/smaps_rollup`.
4. Count `/dev/zero (deleted)` mappings in `/proc/<pid>/maps` and verify that their count reaches a plateau after allocator warm-up.
5. Verify with `nvidia-smi` that GPU memory remains stable.
6. Send normal prefill/decode traffic and verify request correctness, MTP acceptance metrics, CUDA Graph execution, and stable host memory after traffic stops.

Expected result: a bounded allocator warm-up is acceptable, but anonymous RSS and `/dev/zero` mapping count must no longer show the previous linear 15--16 GiB/hour slope.

## Local verification status

No build or tests were run in the local checkout because the required DSV4 CUDA 13/Blackwell runtime and model environment are unavailable. The change was checked by source comparison against the original DSV4 branch and the historical MTP memory-leak fixes. Runtime validation must be performed on a compatible deployment machine.
