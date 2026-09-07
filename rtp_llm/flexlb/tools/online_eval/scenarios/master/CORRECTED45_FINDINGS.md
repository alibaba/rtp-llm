# Corrected Master runtime findings: fixed45

Runtime source: `45e20061610c1c3950710d7d931eec885e849726`.
Run: `yaml_agent1_master_corrected45_20260908`.
Evidence root: `/tmp/agent1-yaml-evidence/master-corrected45`.
This note covers only the completed batch-window and single-nonbatch samples
inspected by agent4. It is not complete ten-instance acceptance or a replacement
for the original5aff results. The rest of the run remains independently tracked.

## Wraparound: the intended ledger check now runs and times out

| Profile | Stage | Duration | Samples | Scheduler min/max |
|---|---|---:|---:|---:|
| batch-window | clean_a / master_inflight_clean | 10001ms | 20 | 8 / 12 |
| single-nonbatch | clean_a / master_inflight_clean | 10001ms | 20 | 6 / 11 |

Neither observed a zero scheduler count, so neither could satisfy all-owner
clean. In batch-window, the first sample had Prefill batch counts4/4 and
Decode loads2/3/2/2; the final sample had2/3 and2/3/2/2. These are actual
separate owner counters, not synonyms for engine slots or KV references.

The old failback_wraparound performs the same A-target clean check for10s
with .5s polling, after killB and the10s switch window, while its150s traffic
runner is still active. The corrected stage preserves that condition. This
failure is distinct from5aff's60s overly strict master_ready gate. Active
traffic and nonzero observations do not establish a resource leak; there is
no justification here to extend the timeout, stop traffic early, or remove
an owner from the predicate.

## Freeze: retries exist but their send timestamps are outside the predicate

| Profile | Continuity | All events | Actual failover events | Judged events | Judged failovers |
|---|---|---:|---:|---:|---:|
| batch-window | all3 checks PASS; scheduler12 to12 | 2105 | 8 | 500 | 0 |
| single-nonbatch | all3 checks PASS; after scheduler8 | 2105 | 8 | 358 | 0 |

Both fail at retry_seen. The eight batch-window retries were issued at
long-freeze-start+.022 through+.322s; their completion wall times were
+16.263 through+16.508s and all were successful. The eight single-nonbatch
retries were issued at+.031 through+.331s. Thus zero judged failovers does
not mean no retry occurred.

Both old master_freeze and the new YAML select the judgment window from
freeze-start+29s toCONT, using send_start_epoch_ms preferentially through
rows_between / the corresponding explicit window action. All observed
failover requests in these samples were sent before that window. This note
records an unsatisfied existing predicate, not permission to change its
window or evidence type. It does not assert why retry completion occurred
at the observed time or extrapolate to uninspected profiles.

## Additional run boundaries

Batch-window dual-kill is reported PASS on the corrected configuration;
quota reached a new ready-stage missing-endpoint-ledger ERROR after passing
the formerly broken blocked stage. Quota's distinct Prefill-only liveness
repair is5c30d4fc2b4ec9134f1f3783571ace021d2aa721 and requires integration
with the previous blocked-sampling fix plus a new run. Current45e results
must not be rewritten as that repair's outcome. All inspected result cleanup
callbacks passed; independent process/port/lock/run-release acceptance is
owned by the execution and review tasks.

## Raw file hashes

Paths below are relative to the evidence root above. Hashes identify the
actual local raw files inspected, not a claim that every remote artifact or
all remaining groups has already been independently accepted.

| File | SHA256 |
|---|---|
| `batch-window/lane0/part0-yaml/instances/instance-2a4bd9e058561f63527a4c7de9f11bd362176f9d292a9171ed2ca124e943df77/result.json` | `2023ea9fe9f159820f39bb52835ee0346ae9decd43f2f4326ca5652060433e3d` |
| `batch-window/lane0/part0-yaml/instances/instance-2a4bd9e058561f63527a4c7de9f11bd362176f9d292a9171ed2ca124e943df77/master-inflight-12.json` | `8721db9b52c84aa2f47a926b2a1ab0efc93b2095f9f3ca429cf19d6bc771300b` |
| `batch-window/lane0/part0-yaml/instances/instance-bc329be9cb4ace28b0a25a7a6d12b86332d1d64572a508911439f43d286dfcf7/result.json` | `4bd067b43205414503fba26b7d92e053b77c91d8b05600048021d782cdff3d3b` |
| `batch-window/lane0/part0-yaml/instances/instance-bc329be9cb4ace28b0a25a7a6d12b86332d1d64572a508911439f43d286dfcf7/ha-client-1/traffic_out/client_events.jsonl` | `885429df714e5dbc2de8bc6f931845d32486e6e4a03a6258585634a3e125bda3` |
| `single-nonbatch/lane0/part0-yaml/instances/instance-7cea58e65937e27659b2ce4d9db263ce5d06e809fc0b2c35d0ff886214092adb/result.json` | `785ab15f540cdd4b622ba6b5f50ed6fd48153695202c9b69064ceaa39c408fff` |
| `single-nonbatch/lane0/part0-yaml/instances/instance-7cea58e65937e27659b2ce4d9db263ce5d06e809fc0b2c35d0ff886214092adb/ha-client-1/traffic_out/client_events.jsonl` | `dbb58d1ed29f051054420138b9918a1cb13f0453feda99e42240127b077f87a5` |
| `single-nonbatch/lane0/part0-yaml/instances/instance-31737f4752e9c0352c6d51f7845ed2844eebdfae721d8709099e97de52656a16/result.json` | `da2f78fb7f75a93371a1bdfed9e54e6ecf3a8b499890d8249b8d3525ec8148be` |
| `single-nonbatch/lane0/part0-yaml/instances/instance-31737f4752e9c0352c6d51f7845ed2844eebdfae721d8709099e97de52656a16/master-inflight-12.json` | `09ec94aea9017656e31b9d2970017e19d7b360432ceffd5a9c8542cdbf6b7e9c` |


## Final ten-instance run outcomes

All ten corrected45 instances are now complete. Direct inspection of their
result.json files agrees with run-summary.json:2 PASS,3 FAIL,4 TIMEOUT,1 ERROR;
63 actual checks were executed and48 cleanup callbacks all passed. This is
executed coverage, not all-green acceptance. The execution owner reports full
resource release; independent raw-run acceptance remains a separate review.

| Exact instance | Status | Failed boundary |
|---|---|---|
| `client_fallback_failback::wraparound::batch-window` | TIMEOUT | clean_a |
| `client_fallback_failback::wraparound::single-batch` | TIMEOUT | clean_a |
| `client_fallback_failback::wraparound::single-nonbatch` | TIMEOUT | clean_a |
| `client_fallback_failback::wraparound::window-nonbatch` | TIMEOUT | clean_a |
| `master_dispatch_quota::single_prefill_ttl::batch-window` | ERROR | ready |
| `master_lifecycle::freeze_short_long::batch-window` | FAIL | retry_seen |
| `master_lifecycle::freeze_short_long::single-batch` | PASS | none |
| `master_lifecycle::freeze_short_long::single-nonbatch` | FAIL | retry_seen |
| `master_lifecycle::freeze_short_long::window-nonbatch` | FAIL | retry_seen |
| `master_lifecycle::kill_dual_b_to_a::batch-window` | PASS | none |

Single-batch freeze satisfies the existing predicate:2105 total events,
8 actual failovers,120 judged events and8 judged failovers. Their send times
are freeze-start+29.763 through+30.007s, inside the old judgment window.
Continuity also passes, with the same process and scheduler22 to14.
Window-nonbatch has2105 total events and8 failovers, but358 judged events
contain zero failovers; continuity passes with scheduler10 to5. Its wrap
samples have scheduler8..12 throughout the20-sample10s check. These profile
observations prevent extrapolating the first two freeze failures to all four.

Quota's next run uses independent fixed source
5344b02991b3f68d15f1624d40682c446f97ca7d, whose sole commit over45e is the
reviewed5c30 Prefill liveness correction. Both quota fixes are present. The
47 combined Master tests and exact quota parent dry-run passed. No result
from that later runtime is claimed in this note.

Additional evidence hashes, relative to the same corrected45 root:

| File | SHA256 |
|---|---|
| `run-summary.json` | `5147d0176d1835213eee2db6d4a5012b176940b3237d834bf888a7906b8677c6` |
| `single-batch/lane0/part0-yaml/instances/instance-8af5e93c4b04d24d406edd66f1588d4fa783d5b8bed2f5612f65ded145605596/result.json` | `2558b004388acd602581fd553f70f1d21cbfef0cf83da48a5d8ce5c4ffc2aab6` |
| `single-batch/lane0/part0-yaml/instances/instance-6963fa230a065f51134739effabf0e1f7b5cdecd4f063d1b27d76243fdac9157/result.json` | `f9e1449b81c0f9fa90fce5ec14da9b4a9f56a9ffd39508d302dec53252fda16a` |
| `window-nonbatch/lane0/part0-yaml/instances/instance-691d571c59ccaa2228f9e5929c7ce79eb850e3e1de4cc8707589dde2b8d50e36/result.json` | `d341a0395cb8f873ece2b8c58f83c10f1d11499413359a06200ce04b3ecaba90` |
| `window-nonbatch/lane0/part0-yaml/instances/instance-5976904fb61806850038312c6233919c35295ff313e4f848382f845ee8d6494d/result.json` | `b07b134113b5737f9513cca5708dabf430b4ac183881a8d36d242b712f4210c6` |
| `single-batch/lane0/part0-yaml/instances/instance-8af5e93c4b04d24d406edd66f1588d4fa783d5b8bed2f5612f65ded145605596/ha-client-1/traffic_out/client_events.jsonl` | `8c59de3585887e24bbb857d405d119515f923e4b790e06d6ec685003df8bdd1b` |
| `window-nonbatch/lane0/part0-yaml/instances/instance-691d571c59ccaa2228f9e5929c7ce79eb850e3e1de4cc8707589dde2b8d50e36/ha-client-1/traffic_out/client_events.jsonl` | `eb0d0ded56f20c57df8c42d8c082c2962c1ba407440197d344f5aac664637580` |
