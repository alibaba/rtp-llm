# Master gRPC receiving budget

The master reserves one receiving slot as soon as HTTP/2 HEADERS arrive. It does not reserve hypothetical body bytes. As a request body arrives, `ServerStreamTracer.inboundWireSize` charges the received body wire bytes to its lane. A new call exceeding the count limit is closed with `RESOURCE_EXHAUSTED: receiving request limit reached`; a body increment exceeding the byte budget is closed with `RESOURCE_EXHAUSTED: receiving byte budget full`. A separate watchdog closes incomplete requests independently of the application executor, including clients with no deadline.

This is a **logical receiving body-byte budget**, not Netty pinned/reserved memory or a hard total JVM Direct limit. The callback runs after buffers have been retained and flow-control credit has been returned. Netty retained slices can keep larger backing allocations alive than their wire-byte count. Headers, buffers waiting for demand, and allocator overhead are not charged as body bytes. The independent receiving count and timeout also bound calls whose body has not arrived or whose callbacks are delayed.

Decoded unary half-close ends receiving and cancels the watchdog before invoking business handling. Accepted scheduler queue time, inference time and response time are outside receiving. Schedule and Cancel have separate receiving counts, byte budgets and executors. Existing application-queue admission also remains: new calls hitting a full lane's application queue return RESOURCE_EXHAUSTED before payload deserialization. Accepted message/terminal callbacks cannot be discarded on executor rejection or shutdown because they own message cleanup.

## Configuration at master startup

Fields are under `loadBalanceConfig.grpcServer`:

| Field | Default | Meaning |
| --- | --- | --- |
| `receiveBudgetBytes` | `0` | Zero derives one quarter of JVM MaxDirectMemory. Explicit nonzero values must be positive and no greater than MaxDirectMemory. |
| `maxReceivingRequests` | `1024` | Concurrent Schedule calls from HEADERS to decoded half-close or terminal cleanup. Independent of message size, byte budget, executor size, and accepted business requests. Explicit positive values are used directly; zero remains an alias for 1024. |
| `maxReceivingControlRequests` | `128` | Independent concurrent Cancel receiving limit. Explicit positive values are used directly; zero remains an alias for 128. |
| `receiveTimeoutMs` | `30000` | HEADERS to decoded half-close. A shorter client deadline takes precedence. Does not limit accepted business queue duration. |

The count limits are configuration values, with no conversion from bytes to request slots. For example, `maxReceivingRequests=4096` permits up to 4096 receiving calls, subject to actual body bytes and the executor's separate overload policy. It does not reserve 4096 times a maximum message size.

Let `B` be the total body budget. Cancel gets `min(128 MiB, B / 8)` and Schedule gets the remainder. This keeps Cancel independent of Schedule pressure. Changing either count limit does not change this byte partition. Each lane needs a positive byte budget, so `B` must be at least 8 bytes. Production budgets should accommodate the expected body sizes.

| JVM Direct limit | Derived wire-byte budget | Schedule / Cancel slots | Schedule / Cancel byte budgets |
| --- | --- | --- | --- |
| 64 GiB | 16 GiB | 1024 / 128 | 16256 MiB / 128 MiB |
| 16 GiB | 4 GiB | 1024 / 128 | 3968 MiB / 128 MiB |
| 8 GiB | 2 GiB | 1024 / 128 | 1920 MiB / 128 MiB |
| 2 GiB | 512 MiB | 1024 / 128 | 448 MiB / 64 MiB |

The protocol maximum message size is unchanged. A request can nevertheless exceed the available lane byte budget before reaching that maximum, especially with smaller Direct settings or large Cancel payloads. Such resource admission returns RESOURCE_EXHAUSTED. The startup log records count limits, timeout and total/lane body budgets. These settings take effect at server startup.

Regression coverage uses 32 simultaneous HEADERS followed by 512 calls at concurrency 32 with 8 KiB bodies, including the 2 GiB Direct configuration that previously allowed only 28 receiving calls. HEADERS consume zero body budget. Tests also verify that excessive body bytes are rejected and charged bytes remain held until queued producers are cleaned up.

## Cleanup ownership

A transport `streamClosed` event alone cannot return an active handler's slot or byte charges: a complete message producer may still be queued in the application's serial executor and hold Direct buffers. Normal cleanup returns both at decoded half-close, onCancel or onComplete. Before a handler starts, no application message has been requested, so transport closure can return that lease directly. A refused body increment keeps all prior charges until that same reliable cleanup fence; it does not make possibly retained bytes available to another call merely because closure was requested.

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

`budget.used.bytes` is the sum of accepted wire-body increments for calls still receiving/awaiting safe cleanup. It no longer includes a fixed initial allowance for every call. It must not be displayed as Netty pinned or reserved memory. Both `.total` values are process-cumulative gauges: query deltas/rates, account for restarts, and do not sum sampled values. Receiving rejections include count and body-budget failures, separate from application-queue saturation.

Monitor the logical budgets alongside actual pinned/reserved Direct, JVM Direct, errors, body timeouts, both executor queues and service latency. Compression expansion/heap, post-close frames already authorized/read, unsolicited bytes after requested unary messages, socket buffers, backing fragmentation, client/response allocators and service-owned buffers remain outside exact wire accounting. A real TCP experiment with a fixed 64 KiB receive buffer reproduced 256 body wire bytes retaining 16 MiB of pinned backing. RST cleaned both on the same TCP connection. Body-byte accounting therefore does not guarantee a physical memory bound.
