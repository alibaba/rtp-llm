# PACE integration smoke

## Run

Use an x86_64 Linux CUDA13 RTP environment and a non-root user with
noninteractive `sudo`. A CUDA device is required even for TCP. The environment
needs Bazel, Python 3, `unshare`, `nsenter`, `setpriv`, `ip`, `mount`, `curl`,
`rpm2cpio`, `cpio`, and compatible PACE/KVCM runtime libraries. If running in
an existing container, it must allow namespaces, `tmpfs` mounts, and veth pairs.

From the RTP checkout:

```bash
scripts=rtp_llm/test/smoke/scripts
bash "$scripts/run_pace_smoke.sh"
```

The default is a TENT TCP SDK round trip with pinned binaries. Local runs and
`remote cache ci` use the same namespace launcher and start no extra containers.
Provider has a separate IP and `/dev/shm`; RTP shares both with Consumer.
`sudo` is used only for namespace setup and TCP fault control. Services and tests
run as the original user.

Logs and counters are under `pace/` in Bazel's undeclared test outputs. Set
`BAZEL_OUTPUT_USER_ROOT` to reuse a Bazel cache. Prebuilt versions and SHA256
checksums are in the [artifact manifest](../../../../3rdparty/pace_test_runtime/repositories.bzl);
the test account needs access to those artifacts.

## Choose a test

Set the environment variables on the launcher. The test reads the mode from the
fixture; do not pass a second mode to Bazel.

| Launcher settings | Checks |
| --- | --- |
| `PACE_SMOKE_TRANSPORT=tcp` (default) | TENT TCP read/write with byte-for-byte comparison. |
| `PACE_SMOKE_FAULTS=1` (TCP) | Slow reads, slow writes, and packet loss cause SDK timeouts; payload I/O recovers after each fault is cleared. No model required. |
| `PACE_SMOKE_FAULTS=provider-offline` (TCP) | Provider loss during model inference. A freeze is not required. |
| `PACE_SMOKE_TRANSPORT=rdma` or `gdr` | TENT payload and transport counters. GDR must be used in both directions. |
| `PACE_SMOKE_TRANSPORT=aft-rdma` | Native AFT with CPU buffers; checks that Provider serves an RDMA read. |
| `PACE_SMOKE_TRANSPORT=aft-gdr` | Native AFT with GPU buffers; reads use GPUDirect RDMA, writes use staging. |
| `PACE_SMOKE_TRANSPORT=aft` | Provider loss with native AFT and GDR reads. Requires a freeze, Meta eviction, and Consumer buffer-safety proof. |

Both Provider-offline cases require a remote cache hit before the fault, matching
output after recompute, and another successful request on the same RTP process
with Provider still down. Any frozen blocks must release within 10 seconds of
Provider pause, with forced release disabled.

- **RDMA:** Set `PACE_SMOKE_RDMA_DEVICE` to an active HCA name or `auto`.
  RDMA needs a shared RDMA namespace and unlimited `memlock`; GDR also needs
  a compatible GPU. CI uses `auto` and skips AFT cases without the required
  hardware. TENT can override `PACE_SMOKE_RDMA_PORT` (default `1`) and
  `PACE_SMOKE_GID_INDEX` (default `0`); AFT uses these defaults.
- **TCP faults:** The kernel needs `prio`, `flower`, and `netem`. The x86 CUDA13
  prebuilt runtime provides `tc` if it is missing from the system.
- **Model cases:** Use a checkout with `internal_source`. The launcher verifies
  `CHECKPOINT_PATH` or downloads a pinned Qwen2.5-0.5B-Instruct checkpoint before
  isolating the network. Set `HF_ENDPOINT` to use a mirror.

## Build from three commits

```bash
PACE_SMOKE_KVCM_INTERNAL_REPO="$KVCM_INTERNAL_REPO" PACE_SMOKE_MEMPOOL_REPO="$MEMPOOL_REPO" \
  bash "$scripts/run_pace_smoke.sh" \
  --source "$KVCM_PUBLIC_SHA" "$KVCM_INTERNAL_SHA" "$MEMPOOL_SHA"
```

Use full 40-character SHAs and repository URLs or absolute paths. Source builds
require access to all three repos, CMake, a C++ toolchain, libibverbs development
libraries, and `rpmbuild`. The Manager and SDK come from the same build. For the
AFT model case, the Consumer and SDK must support late read-safety queries.

`PACE_SMOKE_CACHE_DIR` holds artifacts, model files, and completed source builds.
The launcher reuses source builds with the same three SHAs and CUDA architectures;
use a separate cache when changing toolchains. To use an existing build, set
`PACE_SMOKE_BIN_DIR` to its `pace/bin` directory and pass its `--bazelrc` instead of
`--source`.
