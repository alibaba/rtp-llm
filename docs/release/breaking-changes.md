# Breaking changes

本文档记录可能影响现有部署的默认行为或协议变更，便于撰写 release notes 与 PR 说明。英文摘要见各节 **Summary**。

---

## `worker_info_port_num` default `8` → `9` (Bailian gRPC / port layout)

**Introduced in:** [PR #813](https://github.com/alibaba/rtp-llm/pull/813) (Bailian gRPC, `develop/bailian`).

**Summary:** The CLI/env default for `--worker_info_port_num` / `WORKER_INFO_PORT_NUM` changed from **8** to **9** (`rtp_llm/server/server_args/server_group_args.py`, `rtp_llm/config/py_config_modules.py` `MIN_WORKER_INFO_PORT_NUM`). Worker listen ports are laid out as
`base = start_port + rank_id * worker_info_port_num`, then fixed offsets (RPC, cache store, HTTP, embedding RPC, Bailian gRPC at **base + 8**, etc.).

**Impact:**

- **Single rank (`rank_id = 0`)**: `base` is unchanged (still `start_port`). Offsets within the block still land on the same absolute ports as before **only if** you also did not rely on “next rank” occupying a specific slot; Bailian gRPC now uses **base + 8**, which required a wider stride so ranks do not overlap.
- **Multiple ranks / distributed setups** that relied on the **old default 8** without explicitly setting `worker_info_port_num`: bases for `rank_id ≥ 1` **shift** (`start_port + rank * 8` → `start_port + rank * 9`). Service discovery, firewalls, and docs that hard-coded old ports **must be updated**, or you must pin the old layout.

**Migration:**

1. Prefer **reconfiguring** discovery and firewall rules to the new layout (default `9`).
2. To **preserve the previous stride** (not recommended unless you understand port overlap with new services): set `--worker_info_port_num 8` or `WORKER_INFO_PORT_NUM=8` explicitly, and verify it does not collide with Bailian gRPC or other offsets in your topology.

See also: [Bailian gRPC — listen ports](../backend/Bailian-gRPC.md).
