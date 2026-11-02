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
