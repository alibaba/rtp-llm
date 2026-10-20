# Routing hotspot comparison

Build the baseline and current FlexLB reactors, including test classes. Use JDK 21:

```sh
JAVA_HOME=/path/to/jdk21 python3 tools/routing-hotspots/run.py /path/to/baseline/rtp_llm/flexlb /tmp/routing-hotspots
```

Both revisions run the same source and formula. The benchmark uses real Java objects,
without Mockito instrumentation. Scheduler threads and RPCs are disabled; queue
mutation goes through `PrefillState` under its actual ownership lock. Reflection is
used only to access that lock during fixture setup and controlled mutations.

Cases cover 10/100 endpoints, 32/1024 active requests, 1/64 planner threads, and
maxRequests=1024 with a 700 ms prediction budget. Each round changes scheduling
inputs or removes/reinserts one member per endpoint, then runs concurrent scans.
Each scan includes `captureRouteProjectionInputs`; projection cases also run the
actual formula and route timeline. Capture-only scans repeat eight times per
invalidation; projection scans repeat twice. Membership never changes midway
through a scan, so checksums are deterministic. Concurrent mutation correctness
is covered separately by the scheduler and endpoint tests.

Three separate JVMs per revision alternate execution order. Each case has two
warm-up rounds and three measured rounds of at least 200 ms. Index mutation cases
use 100,000 remove/add pairs per round. Reports use the median of nine samples.
Wall ns/op for concurrent planners describes aggregate throughput, **not individual
request latency**. CPU ns/op sums measured planner threads and the coordinator;
GC/JIT CPU is excluded. Bytes/op includes both planners and coordinator. Pool
submission and coordination overhead is included in wall time. This is a local
algorithm benchmark, not an online CPU-utilization or TTFT forecast.

`ProjectionDifferential` independently compares complete selected memberships,
raw predictions, readiness plans, and candidate records in 2,000 seeded cases,
including priority/FIFO, collection expiry, overflow-scale tokens, probe positions,
committed work, and both delivery policies. The runner verifies identical hashes.
