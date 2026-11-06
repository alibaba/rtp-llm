# Planning-local prediction cache comparison

Build both revisions with JDK 21, including `IncrementalPredictionTest` and the
current `ProjectionPredictionCacheTest`. Then run from the current FlexLB root:

```sh
JAVA_HOME=/path/to/jdk21 python3 tools/prediction-cache/run.py /path/to/baseline/rtp_llm/flexlb /tmp/planning-comparison
```

Both revisions execute the same Java source and DeepSeek v4 formula. Baseline
selection is not replaced with a slower algorithm. Each operation projects one
request across five endpoint snapshots. Cases include Batch and Route delivery,
128/512 queued requests, and diverse shapes that defeat cross-request memoization.
The batch request cap is 64. Standard cases enable a 700 ms prediction budget;
`route512_wide` uses 1,000,000 ms to exercise longer growing prefixes, while
`route512_no_budget` disables prefix prediction as a control.
The snapshots are immutable; RPC and scheduler threads are excluded.

Three JVM forks per revision alternate execution order. Each case has five warmup
rounds and three measured rounds of at least 300 ms. The report uses the median
of nine measured samples. `ns`, `cpu`, and `bytes` are per five-endpoint selection;
CPU/allocation cover the calling thread, excluding GC and compiler threads.
This benchmark measures local projection cost, not production JVM CPU or TTFT.
It does not benchmark a complete global queue or failed admission retries.

The runner also checks the exact checksum of 2,000 seeded selections, 2,000 plans,
and 4,000 candidate projections, including FIFO/priority and expiry behavior.
`summary.json`, per-fork logs, and differential signatures are saved in the output
directory. Batch is a control: its append-only formula prediction already exists
in the baseline. The change reuses Route planning offsets during service projection
and eliminates rescanning old members as a prefix grows.

For endpoint-prefix-cache comparisons, use a baseline containing the preceding
Route prefix optimization. Snapshot instances are endpoint-local and reused
across consecutive probes. `batch128_invalidated` constructs a fresh snapshot on
every endpoint probe to measure cache-build overhead without any warm hits.
The snapshot memo only retains ready groups entirely before the insertion point,
including their one-member boundary lookahead. Expiration, model/delivery changes,
or queue/work/scheduling-input version changes prevent reuse. Warm-vs-uncached
differential and concurrent-publication checks live in `QueuePrefixCacheTest`.

## Contended planner benchmark

```sh
JAVA_HOME=/path/to/jdk21 python3 tools/prediction-cache/concurrent.py /path/to/baseline/rtp_llm/flexlb /tmp/concurrent-planning
```

This compares the pre-endpoint-cache revision with the current revision using
identical source, formula, probes, and 128-member queues. Cases use 1/8/64 calling
threads, a shared endpoint, ten shared endpoints, and shared snapshots replaced on
every operation. Replacement preserves request data to keep the correctness oracle
constant while exercising version invalidation and concurrent cache publication.
Every projection is checked against an expected complete candidate record.

Three JVM forks per revision alternate order. Each case runs a 700 ms warmup and
two 700 ms measurement rounds. Metrics are the median of six rounds: wall-clock
throughput, calling-thread CPU/allocation per operation, histogram P99 (1% buckets),
and summed calling-thread Java monitor blocked time divided by operation count.
An operation visits one or ten endpoints as named. Monitor time covers all Java
monitors on the worker threads; use a separate JFR to attribute contention.
The shared atomic operation sequence and correctness checks add harness overhead
equally to both versions. This is a closed-loop local projection stress test,
without RPC, actual queue mutation, admission commits/retries, or full scheduler
latency. Thread counts above available cores also exercise oversubscription.

For a focused 64-thread shared-endpoint JFR, pass any second argument to
`ConcurrentBench`; profile timings must not be mixed with benchmark timings.
