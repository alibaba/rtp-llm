# Single-Pod SCR restore

The RTP-LLM participation switch is `RTPLLM_ENABLE_SCR=1`; the external
controller supplies `SCR_PHASE` and drives checkpoint/restore. RTP-LLM registers
its resources and waits at the template barrier before starting service and the
engine computation loop.

Loopback is selected automatically when SCR is enabled, `SCR_PHASE` is
`checkpoint` or `restore`, and the configured `WORLD_SIZE` equals
`LOCAL_WORLD_SIZE` (both positive). The dump phase uses the existing protocol
value `checkpoint`. No separate local-communication environment switch is used.
For this single-Pod template, TCPStore, NCCL rendezvous, rank registration and
local RPC fan-out use `127.0.0.1`. Normal startup and multi-node services retain
their configured/discovered addresses. A topology identified as local still
requires a complete, consistent rank set before publishing local routes.

External PD peers must receive a routable KV endpoint. For loopback members,
`cache_store_advertise_ip()` supplies the current Pod IP while local RPC addresses
stay on loopback. This address conversion is part of the feature. Explicit
non-loopback endpoint-manifest addresses retain their authority. Local health
polling always targets the local server using numeric loopback.

After a successful barrier, process identity is refreshed before component fixup
and release. See [runtime fixup and audit](scr_runtime_fixup.md) for supported
restore inputs. Grammar workers are first created by validation requests after
release. The restore environment reader does not rewrite NCCL variables or
inject transport libraries.

Endpoint manifests refresh application routing; they do not rebuild existing
TCPStore/NCCL objects. Single-Pod P/D creates CacheStore and external RPC
connections for the first time after release; it does not require an additional
transport-ready manifest. Existing multi-node readiness checks remain separate.
Acceptance must include actual GPU collectives and cross-Pod KV transfer.

## Validation and acceptance boundary

For acceptance, use the final image directly through its packaged entrypoint.
Remove bootstrap source rewriting and the hostPath kernel wheel. Retain the SCR
injected mounts, shared FUSE memory, child-only model mount, and the checkpoint
storage configuration from the successful experiment. The model path must remain
outside dumped writable paths. Provision enough checkpoint storage for both ranks.

Require all of the following for the same attempt:

1. Seed model/executor initialization reaches the pre-service barrier.
2. Checkpoint reaches completion for the exact seed container.
3. Restore uses a distinct container, preferably with a changed Pod IP.
4. All participants restore; startup warmup and readiness pass without restarts.
5. Restored semantic output matches the baseline token IDs.

The automatic topology check limits rank loopback communication to services
whose ranks share one Pod network namespace. Multi-node services retain their
normal transport configuration; they still require separate restore acceptance.

## Monitoring connections at the checkpoint boundary

During an SCR checkpoint or restore phase, each participant creates its Python
and native metric registries at the normal initialization points, but defers the
external Kmonitor transport. The Python reporter does not create its Flume client
or reporting thread. The native reporter initializes the factory configuration
in manual mode and accepts metric registration, but does not start its metrics
system or create the configured sink.

The upstream template lifecycle releases both reporters after the Epsilon
barrier and restore fixup. A failed barrier leaves deferred reporters inactive.
The main parent participates through the same lifecycle wrapper as its children,
so its deferred Python reporter is also released. Both reporters then activate
their external transport. They reread the Hippo
runtime environment and construct the sink with current identity tags so a restored process
does not report with the seed Pod's host or container IP. Python also replaces
stale runtime tags when rendering data points that were registered before the
checkpoint. Native reporting applies the refreshed runtime identity at the
publish boundary, so a metric declared before the checkpoint can retain its
existing handle while its emitted records use the restored Pod's IP tags. The
native library must provide the matching lifecycle hooks; mixing new Python
helpers with an older loaded native library cannot complete identity fixup and
reporter activation. Ordinary serving without SCR keeps the
original eager reporting behavior.

CRIU can preserve `RequestedIP` from the seed. The unified restore fixup reads
fresh identity inputs (or resolves the current Pod hostname) before metrics
start. Failure to obtain a usable Pod IP rejects normal release. Rebuilding the
native configuration also replaces its common tag map. Fresh-image acceptance
checks both `container_ip` and `host` on actual emitted native records.

CPU validation covers deferred Python/native transport activation, current
runtime identity and metric registration retention. Full acceptance must also
verify real dump/restore, restored inference and reporting in a fresh image.
