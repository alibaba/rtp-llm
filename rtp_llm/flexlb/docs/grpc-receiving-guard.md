# Master gRPC receiving budget

The master reserves a receiving slot and an initial byte allowance as soon as HTTP/2 HEADERS arrive. As a partial request body arrives, `ServerStreamTracer.inboundWireSize` charges body wire bytes to its lane. If new HEADERS or a body increment cannot fit, the call is closed RESOURCE_EXHAUSTED. A separate watchdog closes incomplete requests independently of the application executor, including clients with no deadline.

This is a **logical receiving wire-byte budget**, not Netty pinned/reserved memory or a hard total JVM Direct limit. The callback runs after buffers have been retained and flow-control credit has been returned. Netty retained slices can keep larger backing allocations alive than their wire-byte count. Initial allowances and headroom mitigate ordinary read-window/allocator overhead but do not prove a physical-byte bound.

Decoded unary half-close ends receiving and cancels the watchdog before invoking business handling. Accepted scheduler queue time, inference time and response time are outside receiving. Schedule and Cancel have separate receiving counts, byte budgets and executors. Existing application-queue admission also remains: new calls hitting a full lane's application queue return RESOURCE_EXHAUSTED before payload deserialization. Accepted message/terminal callbacks cannot be discarded on executor rejection or shutdown because they own message cleanup.

## Configuration at master startup

Fields are under `loadBalanceConfig.grpcServer`:

| Field | Default | Meaning |
| --- | --- | --- |
| `receiveBudgetBytes` | `0` | Zero derives one quarter of JVM MaxDirectMemory. Explicit nonzero values must be positive and no greater than MaxDirectMemory. |
| `maxReceivingRequests` | `0` | Zero derives the Schedule count cap below. A positive count can reduce it, while retaining the lane's body-byte allowance. |
| `maxReceivingControlRequests` | `0` | Zero derives a separate Cancel count cap. An explicit positive count changes its reservation partition. |
| `receiveTimeoutMs` | `30000` | HEADERS to decoded half-close. A shorter client deadline takes precedence. Does not limit accepted business queue duration. |

Let `B` be the total budget and `A = 8 MiB` the initial per-call allowance for the configured 4 MiB receive window plus frame/allocator headroom.

- `N = min(1024, floor(B / (2*A)))`. At least two slots are required, so `B` must be at least 32 MiB.
- Default Cancel slots: `C = max(1, min(8, floor(N/8)))`.
- Default Schedule slots: `S = N-C`.
- Cancel byte budget: `C * 2*A`; Schedule byte budget: `B - C*2*A`.
- Explicit Schedule/Cancel counts must each be positive after defaults are resolved, and their sum must be no greater than `N`.

Thus HEADERS-only reservations consume at most half of either lane's budget, leaving body-charge headroom even at the count cap. The 1024 total count cap bounds concurrent receiving leases/watchdogs; it is independent of business in-flight requests. Invalid budgets/counts fail startup instead of silently rounding up reservations.

| JVM Direct limit | Derived wire-byte budget | Schedule / Cancel slots | Schedule / Cancel byte budgets |
| --- | --- | --- | --- |
| 64 GiB | 16 GiB | 1016 / 8 | 16256 MiB / 128 MiB |
| 8 GiB | 2 GiB | 120 / 8 | 1920 MiB / 128 MiB |
| 2 GiB | 512 MiB | 28 / 4 | 448 MiB / 64 MiB |

The protocol maximum message size is unchanged. A request can nevertheless exceed the available lane byte budget before reaching that maximum, especially with smaller Direct settings or large Cancel payloads. Such resource admission returns RESOURCE_EXHAUSTED. An explicit receiving budget/count configuration must be selected with actual message sizes and overlap in mind; increasing it does not repair buffer ownership leaks. The startup log records total/lane budgets, initial allowance, count limits and budget/Direct ratio.

The final 64 GiB and 8 GiB policies are regression-tested with 32 simultaneous HEADERS followed by concurrency32 ×512 calls with8KiB bodies. Both accept all calls and return logical charges/timers and actual pooled payload pins to zero while TCP remains open. This demonstrates the chosen small-message regression, not production throughput. New receive metrics allow rollout monitoring of the real receiving stage, previously absent from business QPS/inflight metrics.

## Cleanup ownership

A transport `streamClosed` event alone cannot return an active handler's slot or byte charges: a complete message producer may still be queued in the application's serial executor and hold Direct buffers. Normal cleanup returns both at decoded half-close, onCancel or onComplete. Before a handler starts, no application message has been requested, so transport closure can return that lease directly. A refused body increment keeps the initial allowance and all prior charges until that same reliable cleanup fence; it does not make possibly retained bytes available to another call merely because closure was requested.

This implementation is validated with grpc-java **1.65.1**. Its SerializingExecutor implements Runnable and Executor. Error escaping a callback resets STOPPED but skips its normal pending-queue reschedule. The guard restarts that same per-call serial executor with a no-op without returning its reservation. If startCall throws before a terminal listener exists, after deframer closure the guard places the cleanup marker at the tail of that same serial queue, behind previously queued producers. Unsupported runnable shapes fail closed rather than reusing unproven resources. Dependency upgrades require ownership-ordering and real-network regression tests.

The watchdog uses remove-on-cancel scheduled futures and disables delayed execution after shutdown, preventing a QPS×timeout heap queue of cancelled timers. Server shutdown drains accepted RPCs before stopping the guard and its control executor; interrupted shutdown does not discard accepted callback tasks. Closing the guard closes receiving calls, stops its watchdog and waits for the watchdog thread to terminate.

## Monitoring semantics

FlexMonitor reports these gauges once per second independently of optional MeterRegistry:

- `app.grpc.server.receiving.active`, `app.grpc.server.receiving.control.active`
- `app.grpc.server.receiving.budget.used.bytes`, `app.grpc.server.receiving.control.budget.used.bytes`
- `app.grpc.server.receiving.rejected.total`, `app.grpc.server.receiving.timed.out.total`
- `app.grpc.server.receiving.timers.active`
- `app.grpc.server.executor.queue.size`, `app.grpc.server.executor.active.threads`
- `app.grpc.server.control.queue.size`, `app.grpc.server.control.active.threads`

`budget.used.bytes` is the sum of initial allowances plus accepted wire-body increments for calls still receiving/awaiting safe cleanup. It must not be displayed as Netty pinned or reserved memory. Both `.total` values are process-cumulative gauges: query deltas/rates, account for restarts, and do not sum sampled values. Receiving rejections include HEADERS/count/byte-budget admission failures, separate from application-queue saturation.

Monitor the logical budgets alongside actual pinned/reserved Direct, JVM Direct, errors, body timeouts, both executor queues and service latency. Compression expansion/heap, post-close frames already authorized/read, unsolicited bytes after requested unary messages, socket buffers, backing fragmentation, client/response allocators and service-owned buffers remain outside exact wire accounting. Real TCP multiplexing with a fixed64KiB receive buffer has reproduced256 bytes of slow-stream body wire retaining16MiB pinned backing while this call's logical charge was about8MiB; RST cleaned both on the same TCP connection. That experiment fixed the receive buffer at64KiB; the production default adaptive allocator was not validated by that experiment, and no additional amplification experiment was run after the stop request. No mathematical pinned-memory guarantee is claimed.
