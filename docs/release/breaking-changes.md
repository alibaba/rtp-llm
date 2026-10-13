# Breaking changes

本文档记录可能影响现有部署的默认行为或协议变更，便于撰写 release notes 与 PR 说明。英文摘要见各节 **Summary**。

---

## JIT cache unified local root and remote snapshot boundary

**Introduced in:** [PR #1112](https://github.com/alibaba/rtp-llm/pull/1112) (JIT remote cache).

**Summary:** Managed JIT toolchains now write their compilation caches under one
fixed local root, `/tmp/rtp-llm/.jit_cache/v1/<scope_id>/<component>/`, where
`<scope_id>` digests the build environment (GPU architecture, CUDA/ROCm toolkit,
C++ runtime, torch ABI, compile flags, and managed package versions) — for
example `/tmp/rtp-llm/.jit_cache/v1/3d8e437f489bb70e/torch_extensions/`. Setting
`REMOTE_JIT_DIR` additionally restores and publishes
`<REMOTE_JIT_DIR>/v1/<scope_id>/<time_ns>-<host>.jit_snapshot.tar.zst` snapshots;
it governs remote access only, so local redirection stays on when it is empty.

**Impact and migration:** Component cache variables you do not set explicitly are
redirected: `FLASHINFER_WORKSPACE_BASE`, `DG_JIT_CACHE_DIR`,
`TRTLLM_DG_CACHE_DIR`, `TILELANG_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`,
`AITER_JIT_DIR`, `FLYDSL_RUNTIME_CACHE_DIR`, `TVM_FFI_CACHE_DIR`,
`CUTE_DSL_CACHE_DIR`, and `TRITON_CACHE_DIR` — preset one to keep its existing
path. Retain old caches until the new layout is verified, and copy artifacts only
between identical `scope_id`s. `DG_JIT_REMOTE_CACHE_DIR` and `deep_gemm_python/`
are not migrated.

`REMOTE_JIT_DIR` must be a trusted, writable absolute path or FUSE URI and needs
capacity monitoring; the local root is shared under `/tmp` as `01777` plus a
default ACL, so artifacts and snapshots can fill a tmpfs. Production images need
`setfacl`/`getfacl` — without them setup fails open and stays local.
`JIT_CACHE_SETUP_TIMEOUT_S` (default `180`; `-1` waits without limit) bounds the
snapshot restore only: build-scope probing runs before it and can add tens of
seconds on a cold host. `JIT_CACHE_RESTORED`, `JIT_CACHE_FAIL_OPEN`, and a
`JIT shared root ... mode ...` line report status.

The local root is fixed, not configurable: ninja depfiles and compiled artifacts
embed absolute paths, so a snapshot restored under a different root cannot be
reused. Opt out of the feature rather than relocating it.

**Rollback:** Unset `REMOTE_JIT_DIR` to stop remote restore and publication.
`--manage_jit_cache=0` (`MANAGE_JIT_CACHE=0`) disables the feature outright: no
local redirection, no snapshot restore or publication, and neither the component
cache variables nor the local root is touched.

**Security:** Use this only inside a trusted, single-tenant container. Any
participating UID can replace artifacts that a peer process later `dlopen`s, and
remote snapshots carry no provenance check; the path, lock, and tar guards protect
against accidents, not against a malicious co-tenant. Trust a remote writer
exactly as much as you would trust it to execute code inside your container.

---

## `worker_info_port_num` default `8` → `9` (DashSc gRPC / port layout)

**Introduced in:** [PR #813](https://github.com/alibaba/rtp-llm/pull/813) (DashSc gRPC, `develop/dash_sc`).

**Summary:** The CLI/env default for `--worker_info_port_num` / `WORKER_INFO_PORT_NUM` changed from **8** to **9** (`rtp_llm/server/server_args/server_group_args.py`, `rtp_llm/config/py_config_modules.py` `MIN_WORKER_INFO_PORT_NUM`). Worker listen ports are laid out as
`base = start_port + rank_id * worker_info_port_num`, then fixed offsets (RPC, cache store, HTTP, embedding RPC, DashSc gRPC at **base + 8**, etc.).

**Impact:**

- **Single rank (`rank_id = 0`)**: `base` is unchanged (still `start_port`). Offsets within the block still land on the same absolute ports as before **only if** you also did not rely on “next rank” occupying a specific slot; DashSc gRPC now uses **base + 8**, which required a wider stride so ranks do not overlap.
- **Multiple ranks / distributed setups** that relied on the **old stride 8**: bases for `rank_id ≥ 1` **shift** (`start_port + rank * 8` → `start_port + rank * 9`). Service discovery, firewalls, and docs that hard-coded old ports **must be updated**.

**Migration:**

1. Reconfigure discovery and firewall rules to the new layout (default `9`).
2. Do not set `--worker_info_port_num 8` or `WORKER_INFO_PORT_NUM=8` for services with DashSc gRPC enabled; startup validation requires `worker_info_port_num >= 9`.

See also: [DashSc gRPC — listen ports](../backend/DashSc-gRPC.md).
