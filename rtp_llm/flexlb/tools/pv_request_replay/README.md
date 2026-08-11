# FlexLB PV request replay

This tool turns a FlexLB PV log time window into two shareable artifacts:

- `analysis.xlsx`: request-level route, cache-hit, WorkerStatus, and decision-snapshot analysis. It can be opened directly with Apple Numbers.
- `replay.html`: a self-contained request replay page. It has no server or CDN dependency and can be shared as one file.

The automation contract is raw `pv.log` data rather than a `.numbers` document:

```text
FlexLB pv.log -> analysis.xlsx -> replay.html
```

## One-command usage

The current test workspace and deployment are the defaults, so a normal run only needs a local-log time window:

```bash
cd rtp_llm/flexlb/tools/pv_request_replay

python3 generate_replay.py all \
  --start '2026-08-11 01:55:00' \
  --end   '2026-08-11 02:35:00'
```

Override the target when needed:

```bash
python3 generate_replay.py all \
  --workspace ai-lab-test \
  --deployment flexlb-hongyi-test-v1-flexlb \
  --start '2026-08-11 01:55:00+08:00' \
  --end   '2026-08-11 02:35:00+08:00' \
  --output-dir /path/to/output
```

Naive timestamps are interpreted as `Asia/Shanghai`. The collector is read-only: it resolves running instances with `dashctl`, reads current and rotated `pv.log` files with paginated `tail`, and writes local snapshots.

By default it includes available log records from five minutes before the requested window through ten minutes after it. The command does not sleep waiting for a future tail boundary. The workbook still includes only routes whose `requestTimeMs` is in `[start, end)`; the extra records are used to join delayed cache and WorkerStatus feedback.

## Two-stage usage

Collect once, then rebuild the workbook/page without reading the instance again:

```bash
python3 generate_replay.py collect \
  --start '2026-08-11 01:55:00' \
  --end   '2026-08-11 02:35:00' \
  --output-dir /path/to/output

python3 generate_replay.py build \
  --input /path/to/output \
  --start '2026-08-11 01:55:00' \
  --end   '2026-08-11 02:35:00'
```

To rebuild only the HTML from an existing workbook:

```bash
python3 generate_replay.py html \
  --input-xlsx /path/to/analysis.xlsx \
  --output-html /path/to/replay.html
```

## Output layout

```text
output/
  collect_manifest.json
  manifest.json
  raw/<flexlb-instance>/pv.log.snapshot
  analysis.xlsx
  replay.html
```

`collect_manifest.json` records the requested window, collection grace period, resolved instances, files and line counts read, actual parsed log coverage, and warnings. `manifest.json` adds join/output summaries.

The request window is applied to the routing record's `requestTimeMs`. Collection continues beyond the requested end by a configurable completion grace so that delayed `cache_hit_comparison` and `prefill_worker_status` records can still join to requests inside the window.

Every command returns a non-zero exit status when collection or request joins are partial, even if non-strict mode produced inspectable artifacts. Use `--strict` to stop before HTML generation when log coverage is incomplete or any routed request lacks cache/WorkerStatus first-token telemetry.

## Semantics and limitations

- The page ends a request at observed first token. PV does not contain Chat/Decode completion, so the page does not claim full request completion.
- Decision Top5 rows are the facts recorded at route time. Host lifecycle buckets are reconstructed from request timestamps.
- If the source only has terminal WorkerStatus, RUNNING water level and step progress are interpolated between RUNNING and first-token boundaries; they are not a historical sequence of per-step snapshots.
- Multiple FlexLB instances are kept separate while joining records. This avoids accidentally joining identical request IDs across instances.
- Automatic deployment resolution sees the instances that are RUNNING at collection time. For a historical window that crossed a rollout or scale event, pass each still-accessible historical instance explicitly with repeated `--instance`, or build from an exported PV log bundle. The manifest cannot claim coverage for an instance that no longer exists.
- A `.numbers` file is intentionally not generated. Numbers can open `analysis.xlsx`, while Python-native Numbers generation is not a stable automation interface.

## Dependencies and tests

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
```
