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
- **LoadBalanceStrategy pattern**: Strategy interface for worker selection (Random, WeightedCache, ShortestTTFT)
- **Queue-based scheduling**: `PriorityScheduler` + per-prefill `WorkerBatcher` queues
- **Resource measurement**: Endpoint resource views used by routing strategies
- **Worker synchronization**: Periodic gRPC-based status sync (`GrpcWorkerStatusRunner`)
- **Master election**: ZooKeeper-based leader election (`ZookeeperMasterElectService`)
- **Graceful lifecycle**: Hook-based online/shutdown management

Queue scheduling components:
- `PriorityScheduler`: Canonical lifecycle owner for every QUEUE ordering and dispatcher combination
- `WorkerBatcher`: Per-prefill queue and dispatch loop for BATCH and NON_BATCH delivery
- `RouteService`: High-level service that delegates DIRECT requests to the router and QUEUE requests to `PriorityScheduler`

Resource management components:
- `ResourceMeasure`: Interface for resource availability abstraction (PrefillResourceMeasure, DecodeResourceMeasure)
- `ResourceMeasureFactory`: Factory for creating resource measures

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
# Run all functional tests (performance regressions are excluded)
./mvnw test

# Run FlexLB Sync performance regressions in a dedicated Maven invocation
./mvnw -Psync-performance-regression -pl flexlb-sync -am test

# Run the FlexLB API end-to-end performance regression separately
./mvnw -Papi-performance-regression -pl flexlb-api -am test

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

Four baseline strategies are available (registered with `LoadBalanceStrategyFactory`):

- **RANDOM**: Random worker selection
- **COST_BASED_PREFILL**: Select worker with lowest cost for prefill requests
- **COST_BASED_DECODE**: Select worker with lowest cost for decode requests
- **SHORTEST_TTFT**: Select worker with lowest predicted TTFT (prefill time + queue time) using candidate pool mechanism (RATIO/FIXED modes) with CAS fairness

Each `RoleType` can use a different compatible strategy. The public choices are tagged
selectors under `FLEXLB_CONFIG.router.roles`: PREFILL/PDFUSION use `RANDOM` or
`ESTIMATED_TTFT`, DECODE uses `RANDOM` or `KV_USAGE_WEIGHTED_RANDOM`, and VIT uses
`RANDOM`. Under `ESTIMATED_TTFT`, `LEAST_RECENTLY_USED_IN_POOL` maps to the shortest-TTFT
candidate-pool path; the other candidate choices map to cost-based prefill selection.
Cache affinity is enabled only by including `router.roles.prefill.cacheAffinity`, with
`maxExtraTtftMs` and `minPrefixHitPercent`; omit the object to disable it.

### Queue-Based Request Scheduling

Scheduling and dispatch are independent tagged choices in `FLEXLB_CONFIG`:

- `scheduler.type=DIRECT`: Routes immediately through `DefaultRouter`.
- `scheduler.type=QUEUE`: Uses `PriorityScheduler` for lifecycle, capacity, cancellation, and timeout ownership. Queue ordering is `FIFO` or `PRIORITY`.
- `dispatcher.type=NON_BATCH`: A `WorkerBatcher` dispatches one request immediately.
- `dispatcher.type=BATCH`: A `WorkerBatcher` collects and dispatches a bounded batch.

Every QUEUE combination follows the same lifecycle: `RouteService` submits to
`PriorityScheduler`, the scheduler selects a prefill endpoint, and that endpoint's
`WorkerBatcher` performs the configured delivery. There is no secondary routing queue or
resource-unavailable retry loop.

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

### FLEXLB_CONFIG (single public behavior document)
```json
{
  "schemaVersion": 1,
  "scheduler": {
    "type": "QUEUE",
    "ordering": {"type": "FIFO"}
  },
  "dispatcher": {"type": "NON_BATCH"}
}
```

The parser is strict: fields belonging to an inactive scheduler, ordering, or dispatcher
variant are rejected instead of being silently ignored.

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

## Important Implementation Details

### LoadBalanceStrategy Registration
All `LoadBalanceStrategy` implementations must register with `LoadBalanceStrategyFactory` during Spring initialization. Use `@DependsOn` annotation to ensure proper initialization order (see `DefaultRouter`).

### Rollback Mechanism
When multi-stage routing partially fails, the system must rollback local state updates. See `DefaultRouter.roolBackRoutingFailure()` which calls `LoadBalanceStrategy.rollBack()` for each successfully routed stage.

### Concurrent Data Access
`EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP` is shared between routing threads (reading) and sync threads (writing). Updates are performed atomically using proper synchronization.

### Queue Concurrency
`PriorityScheduler` owns request lifecycle and global capacity. Each prefill endpoint owns a
bounded `WorkerBatcher` queue. Reservation and release paths must remain idempotent across
completion, timeout, and cancellation races.

### BalanceContext Extensions
`BalanceContext` (request state) includes queue-related fields:
- `future`: `CompletableFuture<Response>` for async response
- `enqueueTime`: Timestamp when request entered queue
- `schedulingMetadata`: Immutable request id, priority, and absolute expiration metadata

Methods:
- Cancellation and lifecycle state are owned by `PriorityScheduler`, keyed by request id.

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
- `BatchSchedulerReporter`: Reports canonical worker-queue size and wait-time metrics
- `PrioritySchedulerReporter`: Reports priority admission and lifecycle metrics
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
- `refactor(LoadBalanceStrategy): rename method getLoadBalanceStrategy to getLoadBalancer`

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
