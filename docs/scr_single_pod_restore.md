# Single-Pod SCR restore

This branch adds opt-in stable loopback communication for ranks that share one
Pod network namespace. It builds on `feat/dsv4_on_dev_scr` at
`7eb9d7bc790b561e347e3bf4cc3c168c39787844`.

Set `RTP_LLM_SCR_LOCAL_COMM=1` on both the checkpoint seed and restore workload.
Set `LOCAL_WORLD_SIZE` to `WORLD_SIZE`. RPC fan-out, TCPStore and NCCL rendezvous
then use `127.0.0.1`; the actual worker addresses remain available for discovery.
The mode rejects multiple nodes, incomplete rank sets and mixed member addresses.
After restore, the saved `self.ip` may differ from newly resolved member addresses;
that difference alone must not invalidate the topology.

Local health polling uses numeric loopback to avoid transient resolver netlink
sockets during checkpoint. With `SCR_ENABLE=1` and `SCR_PHASE=checkpoint`, grammar
sandbox workers are created on first validation rather than eagerly at startup.
Do not send grammar validation traffic before checkpoint: it can instantiate the
sandbox pool. Ordinary startup retains eager pool creation.

The internal entrypoint sets `NCCL_SOCKET_IFNAME=lo` in this mode and preloads the
SCR-injected NCCL interposer for the checkpoint phase. The interposer and the
underlying NCCL implementation must exist before startup. The image retains the
GCC 12 library search path needed by runtime JIT compilation.

The fused RoPE call site supports both legacy and current rtp-kernel wrappers:
the current API takes `position_ids` in prefill and separate position IDs plus
sequence lengths in decode. Kernel feature detection is cached, uses the Python
signature, and does not change tensors or native modules at runtime.

## Validation and acceptance boundary

The preceding runtime-overlay experiment completed a two-rank DeepSeek V4 Flash
Prefill checkpoint and restore with a changed Pod IP. Restored startup warmup
passed, followed by the same 31 output token IDs as the normal and seed baselines.
This is evidence for the fixes; the new source-built image still needs acceptance.
It is not a claim about multi-node, Decode/PD, throughput, or arbitrary grammars.

For acceptance, use the final image directly through its packaged entrypoint.
Remove bootstrap source rewriting and the hostPath kernel wheel. Retain the SCR
injected mounts, shared FUSE memory, child-only model mount, and the checkpoint
storage configuration from the successful experiment. The model path must remain
outside dumped writable paths. Provision enough checkpoint storage for both ranks.

Require all of the following for the same attempt:

1. Seed startup warmup and semantic baseline succeed.
2. Checkpoint reaches completion for the exact seed container.
3. Restore uses a distinct container, preferably with a changed Pod IP.
4. All participants restore; startup warmup and readiness pass without restarts.
5. Restored semantic output matches the baseline token IDs.

The environment flag is opt-in because loopback only works for ranks sharing one
network namespace. Leave it unset for multi-node deployments.

## Monitoring connections at the checkpoint boundary

During an SCR checkpoint or restore phase, each participant creates its Python
and native metric registries at the normal initialization points, but defers the
external Kmonitor transport. The Python reporter does not create its Flume client
or reporting thread. The native reporter initializes the factory configuration
in manual mode and accepts metric registration, but does not start its metrics
system or create the configured sink.

After `epsilon.snapstart_checkpoint()` returns, including after a checkpoint
error, both reporters activate their external transport. They reread the Hippo
runtime environment and rebuild the sink and identity tags so a restored process
does not report with the seed Pod's host or container IP. Python also replaces
stale runtime tags when rendering data points that were registered before the
checkpoint. Native reporting applies the refreshed runtime identity at the
publish boundary, so a metric declared before the checkpoint can retain its
existing handle while its emitted records use the restored Pod's IP tags. The
native library must provide the matching lifecycle hooks; mixing new Python
helpers with an older loaded native library rejects checkpoint participation
rather than silently retaining sockets. Ordinary serving without SCR keeps the
original eager reporting behavior.

CPU validation covers deferred Python transport activation, real TCP
closure/reconnection, refreshed runtime identity, and native metric registration
retention across sink replacement. Full acceptance must additionally verify
dump/restore, restored inference, and resumed reporting in a fresh image.
These hooks address the configured built-in sink; custom sinks and independently
retained sink references require their own external-connection lifecycle checks.
