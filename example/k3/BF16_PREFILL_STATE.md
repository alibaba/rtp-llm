# K3 BF16 Prefill state storage

Implementation baseline: `origin/feat/k3_dev` at
`1527d574f6fd21c4c61e80884cd4b6dc9ac0b2bb`.

Set `SSM_STATE_DTYPE=bf16` on Prefill and `SSM_STATE_DTYPE=fp32` on
Decode. Each process keeps one LINEAR pool. The default remains FP32 in
`start_kimi_k3_pd.sh`.

For the existing two-host smoke driver, use:

```bash
export PREFILL_SSM_STATE_DTYPE=bf16
export DECODE_SSM_STATE_DTYPE=fp32
export LOAD_METHOD=fastsafetensors
# Set the driver's role-local checkpoint, container, repository and endpoint
# options for the selected hosts before launching it.
python3 example/k3/kimi_k3_full_model_two_host_pd_smoke_driver.py --remote-detached
```

The driver defaults both role dtypes to FP32. It forwards each role's setting
as `SSM_STATE_DTYPE` to the service launcher.

## Numerical boundaries

Gather reads either supported pool dtype into contiguous FP32 states. cuLA
continues to use its existing mixed precision operations, FP32 checkpoint
workspace and FP32 current-state registry. Internal chunks of one request
continue from that registry. Persistent prefix hits reload the stored state.
Scatter converts directly through the destination pointer, so BF16 storage
rounds each published checkpoint without allocating a BF16 checkpoint copy.
The pool's existing dtype-derived block stride and conv offsets remain local
to each process.

The terminal state is also rounded in the Prefill pool. Before CacheStore
publication, its SSM segment alone is widened into an owned FP32 tensor. Decode
therefore receives the exact FP32 representation of the **stored BF16** value;
widening does not recover the original FP32 mantissa. Conv segment addresses
and lengths remain unchanged. Decode and MTP keep their existing FP32 state
updates.

## Wire compatibility and ownership

`GenerateRequestPB.prefill_ssm_state_dtype` still describes Prefill storage.
Field 23, `prefill_ssm_transfer_dtype`, describes the transmitted SSM segment.
A oneof supplies presence with the repository's protobuf toolchain. New K3
Prefill advertises FP32 transfer for both supported storage dtypes.

| Prefill storage | Wire declaration | Decode storage | Result |
| --- | --- | --- | --- |
| FP32 | absent (legacy) | FP32 | accepted |
| FP32 | FP32 | FP32 | accepted, original address |
| BF16 | FP32 | FP32 | accepted, widened terminal segment |
| BF16 | absent | FP32 | rejected |
| either | BF16 | FP32 | rejected |
| either | FP32 | BF16 | rejected |

Other storage types are rejected. Existing conv and topology checks still
apply. Upgrade Decode first: an old Decode rejects a new BF16 Prefill because
its storage dtype differs.

Both synchronous and asynchronous writers use `runtimeWriteCacheStore`. The
writer waits for the producer event, selects the source device, performs the
conversion and records a private ready event after it. It does not rerecord
the shared producer event. Each SSM `BlockBuffer` owns its tensor through an
aliasing shared pointer, independently of the store callback. CacheStore's
existing request/remote-read ownership and cleanup govern release.

The existing registration path may copy an unregistered staging allocation
into registered pinned host memory. This adds transfer work and must be
included in handoff measurements. No persistent staging pool is introduced.
The conversion follows the existing terminal-only publication plan; it does
not cast the entire pool or change conv layout to match Decode.

## Validation and rollback

The recurrent-cache test compares BF16 scatter with PyTorch conversion and
FP32 gather with the saved values. It includes sparse physical IDs, padded
blocks, zero prefixes, partial pages and invalid slots. The transfer test
covers synchronous/background publication, unchanged conv segments, source
SSM reuse after conversion, private ready events, callback-independent
ownership and release. The protocol test serializes every supported and
rejected dtype combination, including legacy field absence.

Run GPU and model validation before deployment. Four-layer connectivity is
not evidence of full-model numerical accuracy. Compare the new path with an
FP32 pool reference that explicitly rounds at the same persistent-cache
boundaries; report the separate difference from the unrounded FP32 baseline.
Measure resident pool bytes, FP32 workspaces and transient staging separately.
Only the SSM portion of the Prefill pool uses two bytes per element.

To roll back, set Prefill `SSM_STATE_DTYPE=fp32`, restart the service and rebuild
its cache. Do not change a live pool's dtype or reuse old BF16 cache contents.
