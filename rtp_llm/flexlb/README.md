# FlexLB - Intelligent Load Balancer for AI Model Inference

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java](https://img.shields.io/badge/Java-21-red.svg)](https://www.oracle.com/java/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.1-brightgreen.svg)](https://spring.io/projects/spring-boot)

FlexLB is a high-performance, intelligent load balancer specifically designed for AI model inference workloads. It provides compute-token routing, caching mechanisms, and automatic failover to optimize the performance and reliability of AI service deployments.

## Features

- **Smart Load Balancing**: Multiple strategies including cost-based routing, shortest TTFT, and cache affinity
- **Advanced Caching**: KV cache management for improved performance
- **Health Monitoring**: Real-time worker health checking and automatic failover
- **Reactive Architecture**: Built on Spring WebFlux for high concurrency
- **gRPC Support**: Native gRPC client implementation for backend services
- **Metrics & Monitoring**: Prometheus metrics integration
- **Master Election**: ZooKeeper-based master election for high availability

## Architecture

FlexLB has seven independent business modules:

- **flexlb-api**: Application entry, service endpoints and wiring
- **flexlb-config**: Configuration loading and validation
- **flexlb-balance**: Request scheduling, queues and route selection
- **flexlb-worker**: Worker state, endpoint resources, synchronization and Engine communication
- **flexlb-cache**: Cache matching, indexes and KVCM integration
- **flexlb-master**: Leader election and leader change notifications
- **flexlb-util**: Shared components, connections, discovery and common definitions

The reactor also verifies the Mock Engine test tool. See [architecture](docs/architecture/00-overview.md)
for package ownership and dependency direction.

## Quick Start

### Prerequisites

- Java 21
- Maven 3.6+ (optional, project includes Maven Wrapper)
- ZooKeeper (optional, for master election)

### Build

This project includes Maven Wrapper, so you don't need to install Maven separately.

#### Using Maven Wrapper (Recommended)

**Unix/Linux/macOS:**

```bash
./mvnw clean package -DskipTests
```

**Windows:**

```bash
mvnw.cmd clean package -DskipTests
```

#### Using System Maven

```bash
mvn clean package -DskipTests
```

#### Maven Wrapper Benefits

- **Environment Consistency**: Ensures all developers use the same Maven version
- **Simplified CI/CD**: No need to pre-install Maven in build environments
- **Version Lock**: Project specifies the exact Maven version, avoiding compatibility issues

#### Maven Wrapper Files

The following Maven Wrapper files are included in the project (do not delete):

```
├── mvnw              # Unix/Linux/macOS script
├── mvnw.cmd          # Windows script
└── .mvn/
    └── wrapper/
        ├── maven-wrapper.jar        # Core Maven Wrapper JAR
        └── maven-wrapper.properties # Configuration file
```

### Code Formatting

FlexLB development follows the repository-scoped
[development skill](skills/flexlb-development/SKILL.md) and the
[member organization rules](AGENTS.md#java-成员组织顺序).
The skill is discoverable from the repository root and its subdirectories through
the root `.agents/skills/flexlb-development` link.

From `rtp_llm/flexlb`, check or apply member ordering across **all FlexLB Java
modules, tests, and tools**, excluding generated `target` directories:

```bash
python3 -m pip install -r tools/java-member-order-requirements.txt
python3 tools/order_java_members.py --check
python3 tools/order_java_members.py --apply
./mvnw spotless:apply -Pspotless-check
./mvnw spotless:check -Pspotless-check
```

Pass specific files or directories to narrow the scope. `--diff` previews changes;
`--patch` emits a patch without writing files. The tool orders declarations by
kind, visibility, and source call relationships, preserving method contents and
initializer order. Class workflow comments require human review: the tool does
not infer business importance or interpret lifecycle prose.

Run the ordering tool's tests with
`python3 -m unittest discover -s tools -p 'test_order_java_members.py'`.

FlexLB uses the repository-root [pre-commit](https://pre-commit.com/) configuration.
Java is formatted with Spotless and google-java-format 1.28.0 (AOSP four-space style,
preserving Javadoc layout); Python uses the repository's Black and isort versions.
C/C++/CUDA files use the existing clang-format configuration; Protocol Buffers use
clang-format's Google style. Shell scripts use shfmt. JavaScript, HTML, JSON, YAML,
XML and Markdown use Prettier (with its official XML plugin).

Enable the repository's Git hooks once per checkout (run from the repository root):

```bash
python3 -m pip install --user pre-commit  # or install in your Python environment
git config --local core.hooksPath .githooks
python3 -m pre_commit run --hook-stage pre-commit
```

Each `git commit` checks the staged changes. FlexLB Java changes first check member
ordering and the presence and multiline format of purpose/flow Javadocs on changed
named classes, including nested classes. Unchanged historical class documentation is
not backfilled by this gate. Simple records with no custom behavior, field-only DTOs
and simple enums do not require redundant class comments. Units, special values and
other implicit contracts still require review and appropriate documentation.
The Python hook installs its pinned parser dependencies
in an isolated environment. Java then runs read-only `spotless:check` across the full
FlexLB reactor. Reviewers still assess the accuracy and brevity of the descriptions.
Do not bypass failed checks with `--no-verify`, `SKIP`, or disabled hooks.

The other formatters process staged files. If a formatter changes a file,
review and stage the result, then commit again. Unstaged changes are preserved by pre-commit.
Java formatting requires JDK 21; Maven is supplied by `mvnw`. The first run downloads
the formatter dependencies. Compilation and tests remain separate build commands.

To format all FlexLB code, run from the repository root:

```bash
git ls-files -z rtp_llm/flexlb | xargs -0 python3 -m pre_commit run --files
```

To apply or check Java formatting across all Maven modules:

```bash
cd rtp_llm/flexlb
./mvnw spotless:apply -Pspotless-check
./mvnw spotless:check -Pspotless-check
```

### Configuration

`FLEXLB_CONFIG` is a required JSON environment variable. It configures scheduling,
delivery, routing, worker synchronization, and observability; file paths are not accepted.
The online loader accepts schema 3 only. Duplicate keys, unknown or inactive fields,
`null`, scalar coercion, numeric enums, and trailing JSON fail startup.

| Setting                                             | Schema 3 behavior                                                                                                                                                                         |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prefill ownership                                   | Work remains tracked until authoritative completion, safe rollback or retirement. PDFUSION has no distinct Prefill-completion signal, so its ownership lasts through request termination. |
| `requestLifecycle.request.timeoutMs`                | Required positive integer; maximum request inactivity in milliseconds, renewed by matching Engine request status. No default.                                                             |
| `workerRegistry.health.cleanupIntervalMs`           | Positive integer, default 3000 ms; scan interval for retiring workers whose status exceeds `statusStaleAfterMs`.                                                                          |
| `scheduler.queueTimeoutMs`                          | QUEUE only, positive, default 3600000 ms; actual queue TTL. DIRECT has no queue timer and rejects this field.                                                                             |
| Removed routing controls                            | Prefill candidate randomization/LRU and outlier filters, legacy Decode decay controls/outlier filters, and output-estimate truncation are rejected.                                       |
| `router.roles.decode.costEstimator.expression`      | Default `kvcache_used_ratio`; smallest cost wins within the same availability tier. Using `max_running_size` requires a positive `availability.maxEngineRequests`.                        |
| `scheduler.ordering.preemption.timeoutMs`           | Default 1000 ms, positive; Engine-owned Decode preemption only. Starts after the Cancel ACK phase and bounds the wait for the Engine terminal. ACK timeout stays internal, 50 ms.         |
| `scheduler.ordering.preemption.allowedVictimStages` | PRIORITY supports `DECODE_ENGINE_OWNED`, including when `preemption` is omitted. An explicit non-empty list replaces the defaults.                                                        |

The request timeout below is an explicit workload value:

```bash
export FLEXLB_CONFIG='{
  "schemaVersion": 4,
  "scheduler": {
    "type": "QUEUE",
    "queueTimeoutMs": 3600000,
    "ordering": {"type": "FIFO"}
  },
  "requestLifecycle": {
    "request": {"timeoutMs": 60000}
  },
  "router": {
    "roles": {
      "prefill": {"maxComputeTokens": 1048576},
      "decode": {
        "costEstimator": {"expression": "kvcache_used_ratio"},
        "availability": {"maxKvUsagePercent": 90, "maxEngineRequests": 128}
      }
    }
  }
}'
```

Omit `router.roles.decode.costEstimator` to balance by KV usage ratio. Set its
`expression` to `running_size` to balance by request load, or use a formula to combine metrics:

```json
{
  "schemaVersion": 4,
  "requestLifecycle": { "request": { "timeoutMs": 60000 } },
  "router": {
    "roles": {
      "decode": {
        "costEstimator": {
          "expression": "0.3 * running_size / max_running_size + 0.7 * kvcache_used_ratio"
        },
        "availability": { "maxEngineRequests": 128 }
      }
    }
  }
}
```

`workerRegistry`, `observability.cacheHit`, and `router.groupSelector` remain available.
Only schema 3 is accepted. Write the configuration directly with explicit values for
`requestLifecycle.request.timeoutMs`, which has no default. A matching Engine request
status renews this deadline. Once it expires, FlexLB releases the exact local
request and its reservations even if delivery acknowledgement or Engine request status is missing.
Only priority preemption may send an Engine Cancel RPC. Delivery uncertainty,
request inactivity expiry, and client-cancellation bookkeeping remain local to the Master.
`requestLifecycle.decision.lifetime` defaults to 2.0. There is no legacy-schema conversion layer.

`MODEL_SERVICE_CONFIG` still describes service discovery and endpoint topology; it is
not a second FlexLB behavior configuration:

```bash
export MODEL_SERVICE_CONFIG='{
    "service_id": "aigc.text-generation.generation.engine_service",
    "load_balance": true,
    "role_endpoints": [
        {
            "group": "blue-group",
            "prefill_endpoint": {
                "path": "/",
                "protocol": "http",
                "address": "com.blue.prefill"
            },
            "decode_endpoint": {
                "path": "/",
                "protocol": "http",
                "address": "com.blue.decode"
            }
        },
        {
            "group": "green-group",
            "prefill_endpoint": {
                "path": "/",
                "protocol": "http",
                "address": "com.green.prefill"
            },
            "decode_endpoint": {
                "path": "/",
                "protocol": "http",
                "address": "com.green.decode"
            }
        }
    ]
}'
```

Static IP lists use `MODEL_SERVICE_CONFIG.hosts`:

```json
"hosts": {
  "com.blue.prefill": ["10.0.0.1:8000", "10.0.0.2:8000"],
  "com.blue.decode": ["10.0.0.3:9000"]
}
```

The key matches the endpoint's `address`; ports follow its `protocol`.
Use `hosts` or `discovery_file`, not both.

Local file discovery uses `MODEL_SERVICE_CONFIG.discovery_file`, pointing to a
JSON mapping of service domains to HTTP host:port lists. Production discovery
providers continue to resolve their service domains.

Master configuration uses `FLEXLB_CONFIG` and `MODEL_SERVICE_CONFIG`.
Leader election and logging are configured through `FLEXLB_CONFIG.consistency`
and `FLEXLB_CONFIG.observability.logging`; the separate `FLEXLB_SYNC_CONSISTENCY_CONFIG`
and `LOG_LEVEL` environment variables are no longer consumed. Spring binds environment
variables to infrastructure properties, including `SERVER_PORT` and
`FLEXLB_MONITOR_PROVIDER`. Ports, RPC transport and logging also accept standard
command-line properties, such as `--server.port` and
`--flexlb.engine-grpc.enqueue-timeout-ms`.
HA currently retains `HIPPO_ROLE` as its existing election group identifier.

The gRPC server executor and shutdown quiet period are configured in `FLEXLB_CONFIG`:

```json
"grpcServer": {
  "executorCoreSize": 1000,
  "executorMaxSize": 1000,
  "executorQueueSize": 1000,
  "shutdownQuietPeriodMs": 5000
}
```

All three executor settings default to 1000. Core size may be 0; maximum size and queue size
must be positive, and maximum size must be at least core size.
`shutdownQuietPeriodMs` (静默时间) defaults to 5000 ms and must be positive.
The quiet period starts when shutting down; each Schedule arrival restarts it.
After it elapses, gRPC waits for accepted RPCs to finish before serving resources
are destroyed. Normal idle serving does not start an exit timer.

### Single-request scheduling

QUEUE considers one request at a time, using FIFO or PRIORITY ordering. It selects workers,
registers capacity and returns the complete route immediately. When no worker has enough
capacity, the request waits in the global queue for a capacity notification.
DIRECT performs the same selection immediately and reports capacity failures to its caller.

Prefill work is the input tokens that require computation after cache reuse.
Each worker is limited by router.roles.prefill.maxComputeTokens (default 1048576).
With cacheAffinity configured, prefer cache hits within cacheAffinity.maxComputeTokens;
when omitted, that cap inherits the worker limit. Full candidates are skipped.
No collection window, future batch simulation, learned timing or batch RPC is involved.

Configuration example: [QUEUE + PRIORITY](docs/config-examples/flexlb-queue-priority.json).

Decode freezes each route attempt's request identity, priority, complete prompt-plus-output
demand, capacity limits and admission mode before selecting workers. DIRECT requires
immediately available dispatch capacity. QUEUE without reclamation can retain a physically
feasible route while Decode is busy and wait for a permit at delivery. With Decode reclamation
enabled, placement also counts queued reservations and can reclaim allowed lower-priority owners.
Selection prefers workers with current dispatch capacity over feasible workers that must wait.
Within that availability tier, Decode minimizes `router.roles.decode.costEstimator.expression`,
which defaults to `kvcache_used_ratio`. Formulas support `+`, `-`, `*`, `/`, `^`,
parentheses and `sqrt`, `log`, `exp`, `abs`, `max`, `min`, `pow`, and `param(name, initialValue)` functions.
Decode formulas do not use `sum`. Available variables are:

| Variable             | Meaning                                                                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `running_size`       | Engine-owned requests, including accepted and running requests, plus local request reservations                                           |
| `max_running_size`   | Fixed `router.roles.decode.availability.maxEngineRequests`; must be configured as a positive value when the expression uses this variable |
| `kvcache_used`       | Used KV tokens plus local predicted KV reservations                                                                                       |
| `kvcache_capacity`   | Total KV token capacity                                                                                                                   |
| `kvcache_used_ratio` | `kvcache_used / kvcache_capacity`; zero for unknown capacity of zero, without capping ratios above one                                    |

Request and KV measurements include queued reservations and ownership retained during preemption.
Equal costs rotate. Workers with a non-finite formula result are excluded from the preferred
availability tier; if that tier has no finite result, routing fails with a formula error.
Reservation and delivery always recheck current
inventory under the selected generation's lock. `maxKvUsagePercent` defaults to 90 and must
be in `[1, 100]`; zero and negative values are rejected during configuration validation.
`maxEngineRequests`, when configured, covers the ownership scope of the current admission stage,
including dispatch shadows and permits; it is not the Engine's physical RUNNING concurrency.

See [FlexLB scheduling and configuration](docs/priority-scheduler-delivery-modes.md)
for lifecycle behavior, capacity accounting, configuration defaults, and the mode matrix.

### Run

```bash
java -jar flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar \
--server.port=7002 \
--management.server.port=8804 \
--spring.profiles.active=test
```

The service will start on port 7002 with management endpoints on port 8804.

## API Documentation

### Health Check

```
GET /actuator/health
```

### Load Balance Status Sync

```
POST /load-balance/prefill/consistency/syncStatus
Content-Type: application/json
Authorization: Bearer <token>

{
    "roleId": "model_service_id"
}
```

### Master Notification

```
POST /load-balance/prefill/consistency/notifyMaster
Content-Type: application/json
Authorization: Bearer <token>

{
    "reqIp": "client.ip.address",
    "roleId": "model_service_id"
}
```

### Follower recovery after one forward

Schedule forwards at most once. If the receiver is also a follower, it returns
`NOT_MASTER` (8517) before admission; the sender may run the existing local route.
No-master and connection-establishment failures also allow local routing.
This includes `UnknownHostException` and `UnresolvedAddressException` causes
on an `UNAVAILABLE` result; error message text alone does not permit replay.

Other `UNAVAILABLE` or `DEADLINE_EXCEEDED` failures can happen after admission.
These failures are returned directly without cancellation or local replay.
Business errors and cancelled or expired callers also do not trigger local
execution. Unresolved forwards that may already be owned by the Master retain
`SCHEDULING_DEADLINE_EXCEEDED` (8511) as a terminal response to prevent frontend replay;
this code does not prove that the failure was a timeout. Local scheduling
exceptions use 8511 for typed timeouts and `DISPATCH_FAILED` (8510) otherwise;
caller cancellation uses `REQUEST_CANCELLED` (8504).

Local recovery keeps the request id and caller Context. State queries and cancel
requests first check local ownership, so a recovering follower can manage its
own request. This does not provide cross-node deduplication for frontend retries.

## Configuration reference

- **FlexLB behavior**: one strict JSON document in `FLEXLB_CONFIG`.
- **Prefill work**: input tokens minus reusable cache tokens; no execution-time model.
- **Prefill capacity**: `router.roles.prefill.maxComputeTokens`.
- **Routing parameters**: Prefill execution-time estimation/cache affinity and Decode
  cost formulas/admission thresholds under `router.roles`.
- **Traffic group selection**: `router.groupSelector` inside the same document.
- **Backend topology**: `MODEL_SERVICE_CONFIG`.
- **ZooKeeper consistency**: `FLEXLB_SYNC_CONSISTENCY_CONFIG`.

## Monitoring

FlexLB provides comprehensive monitoring through:

- Prometheus metrics endpoint: `/actuator/prometheus`
- Health checks: `/actuator/health`
- Application info: `/actuator/info`

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Checked routing performance

The normal Maven `test` phase runs the 750P/750D Schedule RPC performance matrix
in `flexlb-api`, with separate 8 GiB JVMs for DIRECT and QUEUE. After channel readiness and 64 low-rate priming requests, independent cases run
3,000 and 10,000 QPS, with 10 seconds of warmup and 10 seconds of measurement per
rate. The fixture uses 1,024 input tokens, eight configured output tokens,
real loopback gRPC and production routing, with simulated worker completion.

Run this check on B300 inside `luoli_gpu` with JDK 21. RSS sampling requires Linux
`/proc/self/status`. CPU 100% represents one logical processor. Process CPU and RSS
include the Master, colocated load client, and simulated completion worker.

`flexlb-api/src/test/resources/route-performance-limits.json` defines the required
QPS ratio, client RPC p50/average/p99 latency, average/peak process CPU, and peak
RSS. Missing resource samples, RPC errors, incomplete server completions, or failure
to select all 750 workers of either role fail the test. JSON measurements are saved
under `flexlb-api/target/route-performance/`, with Surefire reports for all twelve capacity/mode/rate cases. A failed case does not prevent the remaining cases from running.

`tools/run_queue_performance.sh` runs this same matrix without the ordinary UT suite.

### Capacity pressure matrix

The same automatic test phase covers PREFILL_LIMITED and DECODE_LIMITED for both
DIRECT and QUEUE at 3,000 and 10,000 offered QPS. All scenarios retain 750 workers
per role. Prefill pressure limits each worker to 1,024 compute tokens for 1,024-token
requests. Decode pressure limits each worker to one admitted request. Simulated
completion is delayed by 1,000 ms, producing a nominal bottleneck of 750 requests/s.
The other role retains its ordinary capacity. QUEUE has a 250 ms scheduling timeout.

Pressure limits live under `pressure` in the limits JSON. DIRECT retains the normal
RT limits. QUEUE checks p50/average RT at 275 ms and p99 at 500 ms. Both require at
least 90% of offered QPS to receive a response, at least 50% expected capacity
rejections/timeouts, and 450–950 successful requests/s. CPU average limits are 600%
at 3k and 1,500% at 10k, with peak limits twice those values and the same 7 GiB RSS
limit. Unexpected protocol/transport errors fail the test.

The cases also check actual worker registration bounds, occupancy of the finite
role, empty P/D resource ledgers after completion, and admission of a fresh low-rate
request batch after the overload. Expected rejections and successful throughput are
reported separately, so rejecting all requests cannot satisfy the pressure check.
