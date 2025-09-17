# Performance Profiling

This document introduces how to do performance profiling for the framework.

## Model Timeline

The most straightforward way of analyzing model performance is to print the execution timeline of the model. RTP-LLM provides functionality to directly dump model execution timeline into perfetto trace json.

### Per Request Timeline

Add option `"gen_timeline": True,` in `generate_config` of request could generate timeline for current request. The complete request looks like
``` json
{
    "prompt": "Hello, ",
    "generate_config": {
        "gen_timeline": True,
    }
}
```

The generated timeline json file is located at the workdir where you started rtp-llm.

### Service-Wide Timeline

You may use start arg `--gen_timeline_sync` or specify env `GEN_TIMELINE_SYNC=1` to enable service level timeline profiling. When this option is used, every model request would generate a profiling timeline.

### How to visualize timeline

Open [perfetto ui](https://ui.perfetto.dev/) to load the timeline json and visualize. You may also use perfetto sql to perform quantized analyzation.

### Query SQLs Example

Here are some query sqls that are useful for performance analyze.

1. stat are kernel latencies

``` sql
SELECT
    ROUND(AVG(dur) / 1000, 2) AS avg_us,
    MAX(dur) / 1000 AS max_us,
    MIN(dur) / 1000 AS min_us,
    ROUND(SUM(dur) / 1000.0 / 1000, 2) AS sum_ms,
    ROUND(SUM(dur) * 100.0 / (SELECT SUM(dur) FROM slice WHERE (category = 'kernel' OR category = 'gpu_memcpy')), 1) AS percent,
    COUNT(name) AS count,
    name
FROM
    (SELECT * FROM slice)
WHERE
    (category = 'kernel' OR category = 'gpu_memcpy')
GROUP BY
    name
ORDER BY
    SUM(dur) DESC;
```


2. analyze empty time slots

``` sql
WITH cte AS (
    SELECT name, ts, dur,
           ROW_NUMBER() OVER (ORDER BY ts) AS seqnum
    FROM slice
    WHERE (category = 'kernel' OR category = 'gpu_memcpy')
)
SELECT slice.ts, slice.name, tprev.name,
       slice.ts - COALESCE(tprev.ts, 0) - tprev.dur AS diff
FROM cte slice
LEFT OUTER JOIN cte tprev
ON slice.seqnum = tprev.seqnum + 1
ORDER BY diff DESC
```

3. sum up all empty slots on timeline
``` sql
SELECT SUM(diff)
FROM (
    WITH cte AS (
        SELECT
            name,
            ts,
            dur,
            ROW_NUMBER() OVER (ORDER BY ts) AS seqnum
        FROM slice
        WHERE (category = 'kernel' OR category = 'gpu_memcpy')
    )
    SELECT
        slice.name,
        slice.ts - COALESCE(tprev.ts, 0) - tprev.dur AS diff
    FROM cte slice
    LEFT OUTER JOIN cte tprev
    ON slice.seqnum = tprev.seqnum + 1
)
```

## use nsight

You may also use NVIDIA Nsight or other hardware-manufacturer provided profiling tools.

To enable nsight profiling with rtp-llm, you could add nsys binary and options before the start command you actually execut:

``` bash
    /opt/nvidia/nsight-systems/2025.1.1/bin/nsys profile \
    -c cudaProfilerApi \
    -b none \
    --wait=primary \
    --cpuctxsw=none \
    --sample=none \
    --trace='cuda,nvtx' \
    --trace-fork-before-exec=true
    /opt/conda310/bin/python -m rtp_llm.start_server
```

or use bazel `--run_under` option to run a bazel target with nsight profiling

``` bash
bazelisk test //rtp_llm/cpp/normal_engine/test:engine_test  --config=cuda12_6  \
  --run_under="/usr/local/cuda/bin/nsys profile \
    --sampling-period 125000 \
    --trace='cuda,nvtx,osrt,cublas,cudnn' \
    --trace-fork-before-exec=true \
    -o /tmp/report.rep"
```
