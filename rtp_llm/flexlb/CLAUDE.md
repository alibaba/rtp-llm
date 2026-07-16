# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlexLB is a high-performance, intelligent load balancer designed for AI model inference workloads. It provides advanced load balancing strategies, request batching, caching mechanisms, and automatic failover to optimize AI service deployments.

Built on Spring Boot 2.7.18 with reactive architecture (WebFlux), targeting Java 21.

## Module Architecture

FlexLB is a multi-module Maven project with the following structure:

### flexlb-api
Web layer providing HTTP endpoints and reactive web services. Runs the main Spring Boot application with actuator endpoints for health monitoring and Prometheus metrics.

Key classes:
- `HttpLoadBalanceServer`: Main HTTP service for load balancing requests
- `HealthCheckServer`: Health check endpoints
- Configuration: `application.yml` (reactive mode, ports 7001/8803)

New HTTP endpoints in `HttpLoadBalanceServer`:
- `POST /rtp_llm/schedule`: Main load balance endpoint
- `POST /rtp_llm/batch_schedule`: Resolve N worker targets in one shot (`{"batch_count": N}` → `{"server_status": [...]}`); single-role deployments only, count capped by `batchScheduleMaxCount`
- `POST /rtp_llm/master/info`: Get master info
- `POST /rtp_llm/schedule_snapshot`: Dump LB status
- `POST /rtp_llm/notify_master`: Notify participant of master change
- `POST /rtp_llm/update_log_level`: Debug log level control
- `GET /rtp_llm/queue_snapshot`: Get queue snapshot

### flexlb-common
Shared utilities, data models, exception handling, and common configurations used across all modules.

Key classes:
- `ServerStatus`: Worker node status representation
- `Request`/`Response`: API request/response models
- `RoleType`: Enum defining worker roles (PREFILL, DECODE, PDFUSION, VIT) with resourceMeasureIndicator field
- `LoadBalanceStrategyEnum`: Available load balancing strategies
- `ConfigService`: Configuration interface for environment variables

RoleType enhancements:
- `resourceMeasureIndicator`: Field for resource availability tracking (WAIT_TIME, REMAINING_KV_CACHE)
- `getStrategy()`: Per-role strategy selection
- `getErrorType()`: Role-specific error mapping
- `ResourceMeasureIndicatorEnum`: WAIT_TIME, REMAINING_KV_CACHE

### flexlb-grpc
gRPC client implementation for model service communication. Contains protocol buffer definitions and generated stubs for communicating with backend AI worker nodes.

### flexlb-sync
Core load balancing logic, scheduling strategies, and worker status synchronization. This is the heart of the load balancing system.

Key concepts:
- **Router pattern**: `Router` interface + `DefaultRouter` implementation for multi-role request routing
- **LoadBalancer pattern**: Strategy interface for worker selection (Random, WeightedCache, ShortestTTFT)
- **Queue-based scheduling**: `QueueManager` + `RequestScheduler` for async request processing
- **Dynamic resource management**: `DynamicWorkerManager` for adaptive capacity control
- **Worker synchronization**: Periodic gRPC-based status sync (`GrpcWorkerStatusRunner`)
- **Master election**: ZooKeeper-based leader election (`ZookeeperMasterElectService`)
- **Graceful lifecycle**: Hook-based online/shutdown management

Queue scheduling components:
- `QueueManager`: Manages request queue with configurable capacity, timeout handling, and request cancellation
- `RequestScheduler`: Worker thread pool that consumes queue and routes requests (configurable pool size)
- `RouteService`: High-level routing service supporting queue/direct routing modes

Resource management components:
- `DynamicWorkerManager`: Adjusts worker capacity based on resource water levels
- `ResourceMeasure`: Interface for resource availability abstraction (PrefillResourceMeasure, DecodeResourceMeasure)
- `ResourceMeasureFactory`: Factory for creating resource measures
- `ReducibleSemaphore`: Semaphore that supports reducing permits

Lifecycle hook interfaces:
- `AppOnlineHooker`: Online service hooks (replaces OnlineListener)
- `AppShutDownHooker`: Shutdown service hooks (replaces ShutdownListener)

Hook implementations:
- `ActiveRequestShutdownHooker`: Waits for active requests to complete
- `HealthCheckHooker`: Manages health check state during lifecycle
- `LbConsistencyHooker`: Manages ZooKeeper consistency during lifecycle
- `QueryWarmerHooker`: Warms up routing cache on startup

See flexlb-sync/CLAUDE.md for detailed module-specific guidance.

### flexlb-cache
KV cache management for improving inference performance by tracking and matching cached computation blocks across workers.

Key classes:
- `KvCacheManager`: High-level cache management API
- `GlobalCacheIndex`: Global hash table for cache block tracking
- `EngineLocalView`: Per-worker cache state tracking

## Development Commands

### Build
```bash
# Using Maven Wrapper (recommended)
./mvnw clean package -DskipTests

# Build without tests
./mvnw clean package -DskipTests

# Full build with tests
./mvnw clean install
```

### Run Application
```bash
# Run the main application
java -jar flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar \
  --server.port=7002 \
  --management.server.port=8804 \
  --spring.profiles.active=test

# Required environment variables must be set:
# - FLEXLB_CONFIG: Load balance strategy, timeouts, batch settings
# - MODEL_SERVICE_CONFIG: Backend worker endpoints
# - FLEXLB_SYNC_CONSISTENCY_CONFIG: ZooKeeper configuration (optional)
```

### Testing
```bash
# Run all tests
./mvnw test

# Run tests for specific module
./mvnw test -pl flexlb-sync

# Run specific test class
./mvnw test -Dtest=DefaultRouterTest

# Run specific test method
./mvnw test -Dtest=DefaultRouterTest#testRouteSuccess
```

### Code Formatting
```bash
# Check code formatting
./mvnw spotless:check -Pspotless-check

# Auto-format code
./mvnw spotless:apply -Pspotless-check
```

### Build Specific Module
```bash
# Build only flexlb-sync
./mvnw clean package -pl flexlb-sync -DskipTests

# Build module with dependencies
./mvnw clean package -pl flexlb-api -am -DskipTests
```

## Key Architecture Concepts

### Role-Based Multi-Stage Routing

FlexLB routes inference requests through multiple worker stages based on model requirements:

1. **PREFILL**: Initial token processing and KV cache generation
2. **DECODE**: Autoregressive token generation
3. **PDFUSION**: Prefill-Decode fusion workers (combined processing)
4. **VIT**: Vision-language model processing

The `DefaultRouter` orchestrates routing across these stages. If a later stage fails (e.g., DECODE unavailable), the system rolls back local state changes for earlier stages (flexlb-sync/src/main/java/org/flexlb/balance/scheduler/DefaultRouter.java:93).

### Load Balancing Strategies

Four strategies are available (registered with `LoadBalanceStrategyFactory`):

- **RANDOM**: Random worker selection
- **SHORTEST_TTFT**: Select worker with shortest Time-To-First-Token (default for PDFUSION/PREFILL)
- **WEIGHTED_CACHE**: Cache-aware selection prioritizing workers with matching KV cache blocks (default for DECODE)
- **ROUND_ROBIN**: Cursor-based round-robin. No load awareness, much cheaper than SHORTEST_TTFT — no resource scan or scoring; selection is a cursor bump plus an O(alive) liveness filter (`RoundRobinLoadBalancer`). Supports both `select` and batch-aware `selectBatch`; cursors are keyed per `(role, group)`. Use when worker fleets are typically uniform; trades off the ability to avoid hot workers under load skew.

Each `RoleType` can use a different strategy. See `LoadBalanceStrategyEnum` in flexlb-common.

### Queue-Based Request Scheduling

FlexLB supports two routing modes controlled by `FLEXLB_CONFIG.enableQueueing`:

**Direct Mode** (queue disabled): Requests route directly to workers, returning immediate success/failure.

**Queue Mode** (queue enabled): Requests enter a blocking queue and are processed asynchronously by worker threads:

- `QueueManager`:
  - Manages `BlockingDeque<BalanceContext>` with max capacity `FLEXLB_CONFIG.maxQueueSize`
  - `tryRouteAsync()`: Non-blocking attempt to enqueue with timeout
  - `offerToHead()`: Priority insertion for retries (e.g., DECODE retry after PREFILL success)
  - `takeRequest()`: Worker thread consumption
  - `snapshotQueue()`: Debugging snapshot of queue state
  - Handles request cancellation and timeout

- `RequestScheduler`:
  - Fixed worker thread pool (size: `FLEXLB_CONFIG.scheduleWorkerSize`)
  - Polls queue and calls `RouteService.routeRequest()`
  - Retry mechanism for resource-unavailable errors (NO_X_WORKER)
  - Graceful shutdown with 10-second timeout

- `RouteService`:
  - `routeRequest()`: Main routing entry point
  - Supports queue mode (async) and direct mode (sync)
  - `cancelRequest()`: Request cancellation via sequence ID

**Request Lifecycle in Queue Mode**:
1. Client submits request → `QueueManager.tryRouteAsync()`
2. Request enqueued with `enqueueTime` and `sequenceId`
3. Worker thread dequeues → `RequestScheduler` processes
4. Routes through `DefaultRouter`
5. If resource unavailable → retry via `offerToHead()`
6. Response completes the `CompletableFuture<BalanceContext>`

### Dynamic Resource Management

FlexLB dynamically adjusts worker capacity based on resource availability:

- `DynamicWorkerManager`:
  - Periodically recalculates capacity (interval: `FLEXLB_CONFIG.resourceCheckIntervalMs`)
  - Uses `ReducibleSemaphore` for dynamic permit management
  - Gradual adjustment (step size = 1) to avoid oscillation
  - Water level calculation determines when to increase/decrease capacity

- `ResourceMeasure` interface:
  - `PrefillResourceMeasure`: Uses `WAIT_TIME` indicator for resource calculation
  - `DecodeResourceMeasure`: Uses `REMAINING_KV_CACHE` indicator
  - `getWaterLevel()`: Returns 0-100% based on worker resource metrics

- `ReducibleSemaphore`:
  - Extends standard semaphore with permit reduction capability
  - Used by `DynamicWorkerManager` to adjust capacity atomically

- `ResourceMeasureFactory`:
  - Creates appropriate `ResourceMeasure` based on `RoleType.resourceMeasureIndicator`

**Capacity Adjustment Logic**:
1. Calculate water level across all workers of a role
2. If water level < threshold → increase capacity
3. If water level > threshold → decrease capacity
4. Apply changes via `ReducibleSemaphore.reducePermits()` / `release()`

### Worker Status Synchronization

Worker health and capacity information is synchronized asynchronously:

- `GrpcWorkerStatusRunner`: Periodically fetches worker status via gRPC
- `EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP`: Shared concurrent map of worker states
- `GrpcCacheStatusCheckRunner`: Syncs KV cache information with `KvCacheManager`

Routing reads from these shared data structures which are concurrently updated by background threads.

### KV Cache Management

The flexlb-cache module maintains a two-level hash table:

1. **Global index**: Maps cache block hashes to workers containing those blocks
2. **Local view**: Tracks per-worker cache state

During routing, the system queries matching cache blocks to prefer workers with relevant cached data, reducing computation overhead.

### Graceful Lifecycle Hooks

FlexLB provides a hook-based system for managing application lifecycle events gracefully:

- **Lifecycle interfaces**:
  - `AppOnlineHooker`: Hooks executed during online phase
  - `AppShutDownHooker`: Hooks executed during shutdown phase

- **Lifecycle services**:
  - `GracefulLifecycleReporter`: Reports lifecycle events to metrics
  - `GracefulOnlineService`: Manages online phase with priority-ordered hook listeners
  - `GracefulShutdownService`: Manages shutdown phase with hook listeners

- **Hook implementations** (executed in priority order):
  - `ActiveRequestShutdownHooker`: Waits for active requests to complete before shutdown
  - `HealthCheckHooker`: Manages health check state during lifecycle transitions
  - `LbConsistencyHooker`: Manages ZooKeeper consistency during lifecycle
  - `QueryWarmerHooker`: Warms up routing cache on startup

**Lifecycle Flow**:
1. **Online phase**: `GracefulOnlineService` executes `AppOnlineHooker` implementations
2. **Shutdown phase**: `GracefulShutdownService` executes `AppShutDownHooker` implementations
3. Each hook reports status via `GracefulLifecycleReporter`
4. Hooks execute in priority order; a failed hook may prevent subsequent hooks

### Master Election and Consistency

For high availability, FlexLB uses ZooKeeper-based master election:

- `ZookeeperMasterElectService`: Handles leader election
- `LBStatusConsistencyService`: Manages master-slave state consistency
- Only the elected master performs routing decisions

## Configuration

FlexLB reads configuration from environment variables:

### FLEXLB_CONFIG (required)
```json
{
  "deploy": "DISAGGREGATED",
  "loadBalanceStrategy": "SHORTEST_TTFT",
  "prefillBatchWaitTimeMs": 100,
  "kvCache": "LOCAL_STATIC",
  "staticCacheBlockSize": 500,
  "batchSize": 1,
  "prefillLbTimeoutMs": 300,
  "prefillGenerateTimeoutMs": 5000,
  "enableGrpcPrefillMaster": false,
  "enableQueueing": true,
  "maxQueueSize": 1000,
  "scheduleWorkerSize": 10,
  "resourceCheckIntervalMs": 5000
}
```

New configuration fields:
- `enableQueueing`: Enable/disable queue-based routing (default: true)
- `maxQueueSize`: Maximum queue capacity for `QueueManager`
- `scheduleWorkerSize`: Worker thread pool size for `RequestScheduler`
- `resourceCheckIntervalMs`: Resource check interval for `DynamicWorkerManager`
- `batchLoadBalanceStrategy`: Strategy for `/batch_schedule`, decoupled from `/schedule`'s (default: `ROUND_ROBIN`, the only batch-capable strategy today)
- `batchScheduleMaxCount`: Upper bound accepted for `batch_count` on `/batch_schedule` (default: 1000)
- `engineType`: `LLM` (default) or `EMBEDDING`. EMBEDDING flips three behaviors at once: liveness trusts the service-discovery host list instead of gRPC probing (embedding engines expose no `GetWorkerStatus`), `/batch_schedule` targets carry `arpc_port` instead of `grpc_port`, and the single-call `/schedule` path is refused (batch-only). EMBEDDING requires load-unaware strategies (`ROUND_ROBIN`/`RANDOM`) for all deployed roles — boot fails otherwise (`validateEngineTypeConfig`)
- `discoveryFailureGraceMs`: How long a service-discovery outage may keep already-known workers routable before they age out (default: 300000). An **empty** result from a lookup that otherwise "succeeded" counts as an outage, not as an empty fleet: a discovery client that swallows a failed lookup reports it exactly like a fleet that scaled to zero, so an empty list never overwrites non-empty known state — it rides this same grace window, past which the workers age out normally. A fleet that genuinely scaled to zero therefore drains after the grace window rather than instantly. Partial shrinkage (a non-empty list that dropped some hosts) is unambiguous and still takes effect immediately.

Env override semantics (`EnvConfigOverrides`, applies to every field above via `FIELD_NAME_UPPER_SNAKE` env vars — e.g. `BATCH_SCHEDULE_MAX_COUNT`): **behavior changed vs. older builds** — an invalid *enum* value (e.g. a mistyped `LOAD_BALANCE_STRATEGY` or `ENGINE_TYPE`) now fails startup instead of being silently ignored, and enum values are now case-insensitive (a lowercase value that older builds ignored now takes effect). Audit lingering env vars before upgrading. Numeric typos still log-and-keep-default; booleans follow `Boolean.parseBoolean` (anything but "true" reads as false).

### MODEL_SERVICE_CONFIG (required)
```json
{
  "prefill_endpoint": {
    "path": "/",
    "protocol": "http",
    "type": "SpecifiedIpPortList",
    "address": "[\"localhost:8080\"]"
  },
  "service_id": "model.service"
}
```

### FLEXLB_SYNC_CONSISTENCY_CONFIG (optional, for master election)
ZooKeeper connection configuration for distributed coordination.

### DISPATCH_CONFIG (optional, opt-in)

A non-blank `fePoolServiceId` is the enable signal (there is no separate `enabled` flag): every dispatcher bean is gated on `dispatch.fe-pool-service-id`. Either style sets it — the `DISPATCH_FE_POOL_SERVICE_ID` env (Spring relaxed binding) or a `fePoolServiceId` inside the `DISPATCH_CONFIG` JSON (expanded into the property at startup by `DispatchConfigEnvironmentPostProcessor`), so configuring everything through `DISPATCH_CONFIG` alone enables the dispatcher too. When enabled, FlexLB serves `/dispatcher/<original_fe_path>` on its 7001 listener. Requests are matched against the hard-coded **batch endpoint registry** (`BatchEndpointSpec.SPECS`); registered paths whose array field is present (as a JSON array) are split across the FE pool and merged; any other JSON-object body on a registered path — and every unregistered path — is passthrough-forwarded verbatim to one FE. For `/v1/embeddings` the array only counts as a batch when every element is a string (`splitRequiresStringItems`); a single multimodal input expressed as `List[ContentPart]`/`List[ChatMessage]` is passthrough-forwarded whole, not split per element. For the `prompt_batch` endpoints (`/`, `/batch_infer`) a body that carries a companion field FE positionally aligns to the prompt count but the dispatcher does not slice — top-level `images`/`urls` (each `list[list]`) or a list-form `generate_config.adapter_name` — is likewise passthrough-forwarded whole (`requiresWholeBody`), since a split chunk would carry the full-length companion against a shorter prompt slice and FE's `request_extractor` would reject every chunk. Only non-JSON-object bodies (empty, malformed, top-level array) on registered paths are rejected with 400.

```json
{
  "fePoolServiceId": "rtp_llm.frontend.service",
  "subBatch": "count:5",
  "batchTimeoutMs": 30000,
  "probePath": "/frontend_health",
  "preAssignBe": true
}
```

- `subBatch`: chunk-splitting DSL — `count:N` (exactly N chunks, default), `size:N` (≤N items per chunk), bare integer = `size:N`.
- `batchTimeoutMs`: per sub-call wait for FE response headers (default 30000). For non-streaming generation endpoints FE sends headers only after the whole chunk finishes generating, so this must cover one chunk's full generation time; tune down for embedding-only deployments. The body read is separately capped at `batchTimeoutMs + bodyReadMarginMs`, and the passthrough path bounds its headers wait with the same knob (its body stream has its own 10-min inactivity cap). The headers window starts at subscribe, so it also includes connection acquisition and inbound request-body upload time — relevant on the passthrough path, whose catch-all traffic can carry large bodies. A non-streaming passthrough request whose TTFB exceeds `batchTimeoutMs` (e.g. a single long generation) gets a 502: deployments with such traffic must raise `batchTimeoutMs` or have those clients hit FE directly. For the same reason, before tuning down for an embedding-only deployment, verify the passthrough traffic's TTFB also fits the lower value.
- `bodyReadMarginMs`: extra budget past `batchTimeoutMs` for reading the FE response body (default 30000). `batchTimeoutMs` only bounds time-to-headers; this caps the whole call so an FE that sends headers and then stalls mid-body cannot pin the request and its pooled connection forever. Raise it for FE fleets on slow links.
- `probePath`: FE liveness probe path (`/frontend_health` for rtp_llm, `/health` for vLLM).
- `preAssignBe`: BE pre-assignment toggle (see Known limits below). Note `/dispatcher/_dryrun` ignores this default and never pre-assigns unless called with `?pre_assign=true`: resolving BE targets advances master's round-robin cursor, and a diagnostic must not perturb live distribution.

**FE pool and empty discovery:** an empty discovery snapshot never displaces a non-empty FE pool — it is indistinguishable from a swallowed lookup failure, and accepting it would drop every FE at once and fail 100% of batch traffic. Liveness is not discovery's job here: `FeHealthChecker` probes the known FEs directly, so a fleet that genuinely went away is taken out of rotation by the probe. A cold pool (nothing known yet) still accepts an empty snapshot, and any non-empty answer replaces the retained one.

Loading order: defaults → `DISPATCH_CONFIG` JSON → per-field `DISPATCH_*` env overrides (e.g. `DISPATCH_BATCH_TIMEOUT_MS`, `DISPATCH_PROBE_PATH`), matching the `FLEXLB_CONFIG` contract. Connection-pool/timeout knobs that are never operator-tuned (connect timeout, max connections, pending acquire, stream duration cap, max response bytes) are constants in `DispatcherConfiguration` / `FeClient` / `PassthroughClient`.

**Batch endpoint registry (built-in):**

| Path under `/dispatcher/` | Request array field | Response array field | Failure shape | Cross-chunk aggregation |
|---|---|---|---|---|
| `/` | `prompt_batch` | `response_batch` | `null` | — |
| `/batch_infer` | `prompt_batch` | `response_batch` | `null` | — |
| `/v1/batch/chat/completions` | `requests` | `responses` | `{index, error: {code, message}}` | — |
| `/v1/embeddings` | `input` (when list) | `data` | `{index, embedding: null, error: <reason>}` | `data[i].index` renumbered to absolute offset; `usage.{prompt_tokens, total_tokens}` summed across successful sub-bodies |

**Header & query relay:** both paths relay the caller's end-to-end headers (hop-by-hop filtered per RFC 7230 §6.1, shared via `DispatcherHeaders`) and the original query string, so a request does not lose its `Authorization`/tenant/tracing context merely because it was batch-shaped and took the split path. The fanout path additionally drops `accept-encoding` (it parses each FE body, so a gzipped response would break the parse) and `content-type` (each chunk body is re-serialized as JSON).

**Client-facing error text:** failure reasons in the response body are a bounded set — `fe_client_error`, `fe_server_error`, `fe_unavailable`, `malformed_sub_batch` — never the raw exception text, which embeds the FE address. Full detail goes to the rate-limited WARN and `pv.log`.

**Partial-failure contract:**
- HTTP 200 on full success or any partial success.
- HTTP 500 only when **every** sub-batch failed — except when every FE-reachable sub-batch failed with the *same* FE 4xx (a client error), in which case that shared 4xx is returned instead of masking it as 500. Transport/pick failures carry no HTTP status and don't vote, so they can't hide a 4xx the reachable chunks agreed on.
- On partial success, the response body contains an extra top-level object: `_partial_failure: { failed_count: N, total_count: M, failed_indices: [...] }`.
- Failed positions in the response array are filled in-place by the per-endpoint failure factory (see table). **Indices are preserved** so callers can correlate failures back to input positions.

**Migration:**
- **Service-discovery empty-vs-failure contract (behavior change):** `ServiceDiscovery.getHosts` now means "empty list = empty fleet, failed lookup = throw". `NoOpServiceDiscovery` accordingly **throws** on a malformed `DOMAIN_ADDRESS:<addr>` value (e.g. `ip` with no port) instead of the older return-empty; the internal `VipServerDiscovery` likewise propagates a failed VipServer lookup instead of swallowing it into an empty list. Audit `DOMAIN_ADDRESS` values before upgrading — a value that used to be silently ignored now fails the lookup (which the engine rides out via `discoveryFailureGraceMs`, then ages the workers out). A genuinely empty fleet is unaffected.
- Pre-dispatcher clients calling `<fe>/batch_infer` keep working — they hit FE directly. To opt in, change the URL to `<master>:7001/dispatcher/batch_infer` (everything else stays the same; the registered field names match FE's existing wire format).
- Streaming endpoints (e.g. `/v1/chat/completions` with `stream=true`) work through the passthrough as long as `PassthroughClient.STREAM_TIMEOUT_MS` (10 min) exceeds the longest expected response time.
- Direct-to-FE remains the bypass for any client that can't change URLs.

**Known limits (deferred):**
- Bare `POST /dispatcher` (no trailing slash) does not match the root batch route; it is passthrough-forwarded with the path normalized to `/`, i.e. it reaches one FE unsplit.
- Bare `POST /` aliases `/batch_infer`: it batches only when the body carries `prompt_batch` (rtp_llm FE historically exposes batch generation on the root path with the same wire shape). The `prompt: [...]` variant is NOT batched (known FE-side footgun) — such requests fall through to passthrough-forward to a single FE.
- `request_id` set by `frontend_server.py` overwrites any upstream id — dispatcher to FE trace linkage is broken. Tracked in `project_frontend_request_id_overwrite.md`.
- BE pre-assignment is enabled by default (`DISPATCH_PRE_ASSIGN_BE=true`) and applies **only to the `prompt_batch` endpoints** (`/`, `/batch_infer`): FE's pydantic models for `/v1/batch/chat/completions` and `/v1/embeddings` ignore unknown top-level fields, so a stamped `generate_config` would be silently dropped — the dispatcher skips the `/batch_schedule` round-trip for those endpoints entirely. For the prompt_batch endpoints, the dispatcher resolves N BE targets via master `/rtp_llm/batch_schedule` and stamps each chunk's `generate_config.role_addrs` with `{role, ip, http_port, grpc_port}` so FE skips its own master round-trip (existing FE path: `backend_rpc_server_visitor.route_ips` honors non-empty `role_addrs`). **FE version precondition**: the FE build must include `RoleAddr.validate_role` (`@field_validator("role", mode="before")` in `rtp_llm/config/generate_config.py`, on main since `53dc319bd`); older FE builds leave `role_addrs` as `list[dict]` and 500 every stamped request at `model_rpc_client`'s `addr.role` — verified in production 2026-05-28. Against an FE fleet of unknown vintage, start with `DISPATCH_PRE_ASSIGN_BE=false`. `/batch_schedule`'s strategy is decoupled from `/schedule`'s via `FlexlbConfig.batchLoadBalanceStrategy` (default `ROUND_ROBIN`). Failed pre-assigned BE targets still do not auto-failover at the dispatcher; FE handles dead-target fallback by re-invoking `/schedule`.
- Embedding variants (`/v1/embeddings/{dense,sparse,colbert,similarity}`, `/v1/reranker`, `/v1/classifier`) — not in the registry yet; add one row each after verifying wire shape.

## Important Implementation Details

### LoadBalancer Registration
All `LoadBalancer` implementations must register with `LoadBalanceStrategyFactory` during Spring initialization. Use `@DependsOn` annotation to ensure proper initialization order (see `DefaultRouter`).

### Rollback Mechanism
When multi-stage routing partially fails, the system must rollback local state updates. See `DefaultRouter.roolBackRoutingFailure()` which calls `LoadBalancer.rollBack()` for each successfully routed stage.

### Concurrent Data Access
`EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP` is shared between routing threads (reading) and sync threads (writing). Updates are performed atomically using proper synchronization.

### Queue Concurrency
The request queue is a `BlockingDeque<BalanceContext>` accessed by both HTTP request threads (for enqueueing) and worker scheduler threads (for dequeueing). Use non-blocking operations (`offer()`, `poll()`) for thread-safe access.

### BalanceContext Extensions
`BalanceContext` (request state) includes queue-related fields:
- `future`: `CompletableFuture<BalanceContext>` for async response
- `cancelled`: AtomicBoolean for request cancellation
- `retryCount`: Number of retry attempts
- `enqueueTime`: Timestamp when request entered queue
- `dequeueTime`: Timestamp when request left queue
- `sequenceId`: Unique request identifier for cancellation

Methods:
- `cancel()`: Mark request as cancelled
- `isCancelled()`: Check if request is cancelled
- `incrementRetryCount()`: Increment retry counter

### Reactive Programming
The flexlb-api module uses Spring WebFlux for non-blocking reactive request handling. All HTTP endpoints return `Mono` or `Flux` types.

## Testing Strategy

- Unit tests use JUnit 5 and Mockito 5.20.0
- Mock external dependencies (gRPC clients, cache managers, config service)
- Test classes mirror source structure
- Focus on routing logic, strategy selection, error handling, and rollback behavior

## Monitoring and Observability

FlexLB provides comprehensive monitoring through Spring Boot Actuator:

- `/actuator/health`: Health check endpoint
- `/actuator/prometheus`: Prometheus metrics
- `/actuator/info`: Application information

OpenTelemetry integration for distributed tracing (configured via `OTEL_EXPORTER_OTLP_ENDPOINT`).

Monitoring enhancements:
- `RoutingQueueReporter`: Reports queue size, wait time, execution time metrics
- `ResourceMonitorReporter`: Reports resource utilization metrics
- `ActiveRequestCounter`: Tracks concurrent active requests

## Error Types

### Queue Errors
- `QUEUE_FULL`: Request rejected because queue is at capacity (maxQueueSize)
- `QUEUE_TIMEOUT`: Request waited in queue longer than configured timeout
- `REQUEST_CANCELLED`: Request cancelled by client or system during queue wait

### Worker Errors
- `NO_PREFILL_WORKER`: No available Prefill workers
- `NO_DECODE_WORKER`: No available Decode workers
- `NO_PDFUSION_WORKER`: No available Pdfusion workers
- `NO_VIT_WORKER`: No available Vit workers

Worker errors can trigger retry logic in the queue scheduler when resource-unavailable conditions occur.

## Commit Message Format

Follow Conventional Commits specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

Examples:
- `feat(router): add cache-aware routing strategy`
- `fix(grpc): handle connection timeout gracefully`
- `refactor(LoadBalancer): rename method getLoadBalanceStrategy to getLoadBalancer`

## Java Version and Dependencies

- **Java**: 21 (required)
- **Spring Boot**: 2.7.18
- **Project Reactor**: 2024.0.10
- **gRPC**: 1.65.0
- **Apache Curator**: 5.4.0 (ZooKeeper client)
- **Mockito**: 5.20.0 (testing)
- **Netty**: 4.1.127.Final

JVM args required for Java 21 module system (see pom.xml spring-boot-maven-plugin configuration).

## Internal vs Open Source Profiles

The project supports two Maven profiles:

- **opensource** (default): No internal dependencies
- **internal**: Auto-activated when `../../../internal_source` exists, enables KMonitor and VipServer integrations

Most development uses the opensource profile.

## Important Reminders
1. Do what is asked; no more, no less.
2. Don't keep reading the file back and forth. If you need to make changes, do it quickly.
3. Always prefer editing existing files over creating new ones.
3. Do not proactively create documentation files (*.md) or README files unless explicitly requested.
4. When fixing issues in code, such as using solution A to fix problem X, don't write comments that explain why solution A was used to fix problem X. Make the code appear as if problem X never existed in the first place. For example, avoid comments like:
// Request queue (using configured capacity parameter to control queue size, avoiding race conditions)
private final BlockingDeque<BalanceContext> queue;

The parenthetical content in such comments is unnecessary because it makes readers wonder about a problem X they weren't aware of. The code should look naturally correct from the beginning.
5. To run Maven commands, use the Maven wrapper from rtp_llm/flexlb directory: `./mvnw`
6. **IMPORTANT**: Do not repeatedly read the same file multiple times. Once you have sufficient context from a file read, proceed to edit directly. Avoid excessive redundant Read operations on the same file or code snippets.
