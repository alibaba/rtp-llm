# iGraph bucket input for the CSR constraint-tree Master

Status (2026-09-08, numeric-bucket update): Local acceptance passed: 66 tests,
zero failures/errors/skips, with numeric 4000-bucket E2E and a 2.5-million-item
synthetic merge. See `build_logs/igraph_numeric_acceptance_20260908.log`.
The earlier acceptance passed 60 tests,
including three native C++ Worker HTTP E2E cases and synthetic 2-million-item
input. The local Master Java package and packaged class-loading/CSR smoke check
also passed. See `build_logs/igraph_local_acceptance_20260908.md` for evidence.
The native tokenizer fix has additionally passed real-model startup and full
C-coded SID publication with fixed 1024/1500-beam replay; see
`build_logs/csr_saro_release_validation_20260908.md`. No live iGraph request or
current-production-workload acceptance has been performed. Do not treat the
local results as production acceptance. A subsequent packaged Java SDK single-key
probe failed at service discovery (`no host to srv`) in the local container,
whose jmenv reports `daily`; no table response or live completeness was obtained.

## Scope and data contract

SARO maintains eligible items in a dedicated KKV table. Master reads a known
finite set of pkeys through bounded asynchronous point queries and merges the
results itself. This is not iGraph scan, server-side aggregation or a snapshot
API. No Table Service, DFS or new multi-Master coordination is introduced.
The existing SID mapping -> CSR -> HTTP Worker publication path is reused.

Writer/reader agreement for this implementation:

- `BUCKET_ALGORITHM=CRC32` (legacy default):
  `bucket = unsigned_crc32(UTF8(item_id)) % bucket_count`.
- `BUCKET_ALGORITHM=ITEM_ID_MOD`: unsigned decimal numeric `item_id % bucket_count`.
  Arithmetic is exact even beyond 64 bits; signs, whitespace, non-ASCII digits
  and floating-point notation are rejected. Leading zeros do not change the
  numeric bucket, but the original item ID remains the skey.
- `pkey = key_prefix + decimal(bucket)`; no zero padding. Defaults: 4096 buckets,
  e.g. prefix `gul_item_bucket_` gives keys `gul_item_bucket_0` ... `_4095`.
- Explicit empty `KEY_PREFIX` supports plain numeric pkeys. Omitting the variable
  is an error. `ITEM_ID_MOD`, count `4000`, empty prefix yields `"0"` ... `"3999"`.
- `skey = item_id`; value field `sid` contains the C-coded SID, not token IDs.
- Hash the exact canonical item ID string; do not use Java/Python's built-in
  string hash, signed CRC conversion, whitespace or inconsistent leading zeros.
  Test vector: CRC32(`123456789`) = `0xcbf43926`, bucket 2342 for 4096 buckets.
- Bucket algorithm/count/prefix remain fixed while incremental writes are active. Changing
  either requires a freshly initialized table/keyspace and a coordinated switch.
- The table contains **only eligible items**. Downstream-ineligible items must
  be deleted by `(pkey, item_id)`, or expire through a verified source TTL policy.
  This adapter does not interpret arbitrary business status/expiry fields.
- SID changes overwrite the same item record. Master deduplicates SID after
  reading items; deleting one of several items sharing a SID does not remove
  that SID as long as another eligible item still has it.
- Reader accepts variable C-symbol syntax, but conversion follows the existing
  SARO API: `ConstraintTreeSidMapping.convert` currently requires exactly two
  C-symbols. This change does not expand that API. An unsupported SID causes
  build failure, never a partial tree.

4096 buckets give an average of about 488 rows for 2 million items; this is
not a per-bucket bound. The default accepted maximum is 2000 rows, with a query
for 2001 rows to detect overflow. Tune using actual distribution and measured RT.
Set `SOURCE_ROW_LIMIT` to the known server/index cap (e.g. `2000`): reaching
that number, including exact equality, rejects the round. Default `0` disables
this extra guard for legacy sources. Fewer rows do NOT prove completeness after
historical index truncation and deletes; reconcile against an authoritative source.

## Completeness and freshness limits (deployment prerequisites)

1. Verify initialization is complete **and visible to query replicas** before
   setting `SOURCE_READY=true`. This operator acknowledgement is not an automatic
   SARO watermark protocol or an authoritative count check.
2. Confirm with iGraph that table construction does not truncate retained skeys
   and query/seek/response limits allow the configured maximum **plus one**.
   The sentinel detects our requested limit, NOT hidden lower server-side caps.
3. All configured buckets must succeed. One error, timeout, malformed row,
   wrong configured bucket, duplicate item, reported hot-key/degradation, or overflow
   aborts the round. No partial input is submitted. Empty individual buckets
   are legitimate; an all-empty round retains the old tree and reports failure.
   A valid empty pool cannot currently be published as a deny-all tree.
4. Reads happen at different times, so this is eventually consistent input,
   not a fixed-version source snapshot. Deletes/updates during a round can be
   reflected only in the next round. Explicitly acknowledge this with
   `ALLOW_NON_ATOMIC_READ=true`; retain BE's live eligibility filtering.
   This flag does not create a snapshot or prove the business accepts stale items.
5. Normal freshness includes source propagation + up to one polling interval +
   read/build/publication time; failures may extend staleness further. Monitor
   refresh failures and the actual active tree age. Do not claim a strict
   10-minute bound or strict current-pool enforcement from this input path.

## Master configuration

The feature is disabled unless `CONSTRAINT_TREE_IGRAPH_ENABLED=true`. Environment
variables map to Spring properties `constraint.tree.igraph.*`.

```text
CONSTRAINT_TREE_IGRAPH_ENABLED=true
CONSTRAINT_TREE_IGRAPH_MODEL=gul_item
CONSTRAINT_TREE_IGRAPH_TABLE=<eligible-item-table>
CONSTRAINT_TREE_IGRAPH_SEARCH_DOMAIN=<query VIP domain>
CONSTRAINT_TREE_IGRAPH_UPDATE_DOMAIN=<update VIP domain required by SDK initialization>
CONSTRAINT_TREE_IGRAPH_CLUSTER=DEFAULT
CONSTRAINT_TREE_IGRAPH_SRC=whale_constraint_tree
CONSTRAINT_TREE_IGRAPH_PKEY_FIELD=pkey
CONSTRAINT_TREE_IGRAPH_ITEM_FIELD=item_id
CONSTRAINT_TREE_IGRAPH_SID_FIELD=sid
CONSTRAINT_TREE_IGRAPH_KEY_PREFIX=gul_item_bucket_
CONSTRAINT_TREE_IGRAPH_BUCKET_ALGORITHM=CRC32
CONSTRAINT_TREE_IGRAPH_BUCKET_COUNT=4096
CONSTRAINT_TREE_IGRAPH_CONCURRENCY=16
CONSTRAINT_TREE_IGRAPH_MAX_ROWS_PER_BUCKET=2000
CONSTRAINT_TREE_IGRAPH_SOURCE_ROW_LIMIT=2000
CONSTRAINT_TREE_IGRAPH_QUERY_TIMEOUT_MS=5000
CONSTRAINT_TREE_IGRAPH_ROUND_TIMEOUT_SECONDS=300
CONSTRAINT_TREE_IGRAPH_RETRIES=1
CONSTRAINT_TREE_IGRAPH_INTERVAL_SECONDS=600
CONSTRAINT_TREE_IGRAPH_SOURCE_READY=false
CONSTRAINT_TREE_IGRAPH_ALLOW_NON_ATOMIC_READ=false
CONSTRAINT_TREE_IGRAPH_DRY_RUN=true
```

Replace table/domain placeholders and field names. Enable the final two flags
only after the prerequisites above are met. The update domain is an SDK builder
requirement; our adapter never calls a write API. Do not embed personal credentials.

Start with `DRY_RUN=true`: the active Master traverses and validates all buckets,
reports item/SID counts, maximum bucket occupancy and read duration, but NEVER
submits a CSR build. The two readiness acknowledgements are not required in this
mode. Successful source status is `VALIDATED_NO_PUBLISH`, not proof of source
completeness. Failed reads/cap checks report `FAILED` without partial publication.
After source reconciliation and accepting non-atomic reads, configure
`DRY_RUN=false`, `SOURCE_READY=true`, `ALLOW_NON_ATOMIC_READ=true` and restart
through the normal deployment process. These are startup properties, not HTTP
switches. Dry run does not disable manual builds or existing tree reconciliation.

For a numeric-source deployment override:

```text
CONSTRAINT_TREE_IGRAPH_BUCKET_ALGORITHM=ITEM_ID_MOD
CONSTRAINT_TREE_IGRAPH_BUCKET_COUNT=4000
CONSTRAINT_TREE_IGRAPH_KEY_PREFIX=
CONSTRAINT_TREE_IGRAPH_PKEY_FIELD=item_bucket_id
CONSTRAINT_TREE_IGRAPH_CONCURRENCY=4
CONSTRAINT_TREE_IGRAPH_RETRIES=0
```

In zone JSON, preserve empty prefix as `["CONSTRAINT_TREE_IGRAPH_KEY_PREFIX", ""]`.
MODEL must be the actual Worker discovery model key, not an inferred project name.

Only the active Master polls. An additional round is skipped while a read is
running, or while a build/publication runs in publishing mode. New artifact versions use at least wall-clock
milliseconds and exceed the local latest accepted version; this is an artifact
version, not an iGraph data version. Keep a single active Master and avoid
concurrent manual full-input submissions from a different versioning scheme.

```bash
# Read status of source fetching (SUBMITTED does NOT mean Workers activated).
curl http://MASTER/rtp_llm/constraint_tree/source/status
# Manually enqueue a refresh; 202 means queued, 409 means busy/not ready.
curl -X POST http://MASTER/rtp_llm/constraint_tree/source/refresh
# Existing build/publication status remains authoritative for Worker activation.
curl http://MASTER/rtp_llm/constraint_tree/status
```

The first automatic read starts after one configured interval. Manual refresh
can start it earlier. Protect these management routes with existing deployment
network/access controls; no new authentication mechanism is added here.

To test one numeric bucket without starting Master, use the packaged classpath
and the same `CONSTRAINT_TREE_IGRAPH_*` endpoint/table/field environment:

```bash
timeout 30s "$JAVA_HOME/bin/java" --class-path 'rtp_llm/flexlb/flexlb-api/target/FlexLB/BOOT-INF/classes:rtp_llm/flexlb/flexlb-api/target/FlexLB/BOOT-INF/lib/*' rtp_llm/flexlb/scripts/IgraphReadProbe.java 0
```

The probe only queries one key and prints counts, never SIDs or tree publication.
Successful zero rows show an empty response, not a populated/correct table. Keep
an outer timeout because SDK discovery initialization can precede query timeouts.

## Packaging

The open-source common module contains only `SidBucketClient`. The SDK adapter
lives in `internal_source/java/igraph` and is included by FlexLB's internal
profile. The existing CI already installs `flexlb-common`, builds internal Java
modules, and packages FlexLB. No CI YAML or Dockerfile changes were made.

Pinned SDK: `com.alibaba.etao.igraph:igraph-client:2.1.7`; its required provided
dependencies are declared in the internal module. Dependency resolution, tests,
local Master packaging, packaged class loading and a small CSR encode/decode
check have passed. The adapter excludes EagleEye's extra SLF4J binding so the
package retains a single Logback implementation. This is not a full service
startup against real SDK discovery endpoints. The SDK bean is not created with
the feature disabled.

## Local verification

Use the repository `test-execution` skill; execute inside the existing development
container as the non-root owner. No Bazel/source/CI changes are needed for the
Java reader itself. For a missing/stale native Worker binary, run the skill's
precheck and wrapper before rebuilding its existing target.

Inside the container, set `JAVA_HOME`, `MAVEN_CMD` and `MAVEN_SETTINGS` to the
working local Java 21/Maven/internal-mirror configuration, then:

```bash
export CONSTRAINT_TREE_CPP_WORKER_BINARY=/absolute/repo/bazel-bin/rtp_llm/cpp/api_server/test/constraint_tree_test_server
bash rtp_llm/flexlb/scripts/test_igraph_constraint_tree.sh
```

The script refuses to run outside a container, as root, or without the native
test Worker, to avoid silently skipping its HTTP E2E. It performs:

- Internal adapter unit tests using real SDK types and mocked SDK responses.
- Reader tests for bounded async requests, SID dedup, timeout, retry, cancellation,
  malformed/overflow input, and same-SID item deletion.
- Poller tests for readiness/leadership gates, non-overlap, source-failure
  isolation, scheduler recovery, environment configuration, monotonically
  increasing local versions and HTTP control routes.
- Synthetic 2-million-item / 2-million-SID / 4096-bucket reader test, followed by
  fingerprinting, SID mapping, CSR construction and codec checks. Separate
  timings are recorded for local merge, mapping and construction; none is real
  iGraph RT or GPU latency. The existing million-variable-length-token-path
  construction regression also runs.
- Numeric 4000-bucket bounded-concurrency tests, exact large-ID modulo, server
  cap equality rejection, dry-run/no-publication and 2.5-million-item merge test.
- Fake bucket source -> real Java CSR builder/publisher -> native C++ HTTP Worker,
  failure retaining old tree, subsequent update and current/backup checks;
  existing mapping/restart/publication regressions are included.

The numeric update run had 7 internal SDK tests and 59 API tests (66 total).
After testing, build the local Java package inside the same container:

```bash
# From rtp_llm/flexlb; internal modules were installed by the test script.
"$MAVEN_CMD" -s "$MAVEN_SETTINGS" -B -pl flexlb-api -am -Pinternal -DskipTests clean package
# From the repository root, check only the packaged classes/dependencies.
"$JAVA_HOME/bin/java" --class-path 'rtp_llm/flexlb/flexlb-api/target/FlexLB/BOOT-INF/classes:rtp_llm/flexlb/flexlb-api/target/FlexLB/BOOT-INF/lib/*' rtp_llm/flexlb/scripts/IgraphPackageSmoke.java
```

Artifact: `rtp_llm/flexlb/flexlb-api/target/ai-whale.tgz`. This is the local
Master Java package, not a Whale inference image or an uploaded release.

Missing real-service acceptance inputs: test iGraph table, query/update domains,
read permission, completed initial data, and confirmed truncation/retention limits.
Use real data to measure bucket occupancy, per-query P50/P99, whole-round time,
maximum concurrent requests, Master heap/GC, and comparison with an authoritative
source dump before enabling publication. No live iGraph write/load test is
authorized or performed by the local test script.

The earlier C-tokenizer wrapper source defect was fixed and verified against
the real checkpoint vocabulary and actual inference Worker startup. Full C-coded
SID publication and fixed 1024/1500-beam historical-prompt replay passed on the
local BF16 checkpoint/H20, not the current online checkpoint/L20. See
`build_logs/csr_saro_release_validation_20260908.md` for exact evidence and limits.
The local Master Java package does not include the native Worker fix: deployments
must already run the mapping-aware CSR Worker. The numeric-bucket update changes
only Master; compatible Workers from the previous mapped-CSR release need not
be rebuilt or replaced.

## First test deployment

1. Upgrade all inference Workers to the mapping-aware build, with
   `CONSTRAINT_TREE_REQUIRED=true`, `WARM_UP=0`, `ACT_TYPE=bf16`, and no legacy
   `TREE_DECODE_CONFIG`. Keep business requests away until their trees are ready.
2. Deploy the matching Master image, initially with iGraph disabled. Verify
   Worker discovery and `/constraint_tree_mapping_status` on every Worker.
   Mixed fingerprints block publication until the model rollout is consistent.
3. Supply the real iGraph configuration above and verify initialization, delete
   semantics, bucket agreement and server limits. Only then enable source reads
   and acknowledge the two readiness/consistency flags.
4. Trigger `/rtp_llm/constraint_tree/source/refresh`. Check source item counts,
   then Master `READY` and every Worker's active version/digests. The default
   60-second reconciliation can lag behind actual Worker activation.
5. Send isolated test requests using fixed beams. The root must contain at least
   that many distinct first-token candidates. Existing requests with non-empty
   `variable_num_beams` are rejected; coordinate any TPP config change explicitly.

Master and Worker are separate release artifacts. Do not deploy only the Java
`ai-whale.tgz` and assume it updates the native inference implementation.
