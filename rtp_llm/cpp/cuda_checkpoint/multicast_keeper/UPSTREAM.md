# Upstream provenance

This directory is an engineering port of the CUDA checkpoint multicast keeper
prototype from:

- Repository: `git@code.alibaba-inc.com:foundation_models/nekyia.git`
- Branch: `cuda-ckpt-symmem-keeper`
- Commit: `5e417f2cba5f4ecf73ba7ab5bb3241473cc4bc6d`
- Commit title: `symmem+ckpt: full end-to-end (MegaMoE + dual NVLS multicast + keeper + CRIU-to-disk) O1==O2; lightweight ~0-mem keeper`
- Author: `风夏 <liukan.lk@alibaba-inc.com>`
- Source directory: `examples/megamoe_ckpt_keeper/`

The main source inputs were `keeper_lite_creator.cu`,
`keeper_lite_holder.c`, `keeper_nccl.cu`, `mc_shim_unified.c`,
`run_final.sh`, and `README.md`.

## Port changes

- Replaced the unsafe upstream size-only wire format with protocol V3. `CREATE`
  returns a unique holder-instance/object identity; `FETCH` requires that exact
  identity and validates all requested properties.
- Replaced the single-FD proof-of-concept holder with an object-ID keyed FD
  table. Same-size creates are independent and stale holder tokens fail closed.
- Replaced the upstream `NEKYIAMC` token with a versioned 64-byte token carrying
  holder instance, object ID, and validated properties. POSIX tokens are sealed
  memfds; cross-node FABRIC exchange uses CUDA's opaque 64-byte handle.
- The holder starts a short-lived creator on a cache miss. Creator and holder
  exchange the new FD over a private socketpair; the holder process itself does
  not link or initialize CUDA.
- Generalized the creator from two fixed GPUs to a validated GPU list.
- Added bounded holder/client I/O, bounded creator termination, strict full-team
  property checks, explicit opt-in, readiness, deterministic signal cleanup,
  Bazel targets, CPU lifecycle tests, and a scriptable GPU checkpoint test.
- Kept `NEKYIA_KEEPER_DIR/mcsk.sock` as the default endpoint. Protocol V3 is
  deliberately not wire-compatible with the upstream size-only protocol.

The code here is intentionally isolated. The port does not modify RTP-LLM's
collective or RPC lifecycle implementation.
