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
# - WHALE_SYNC_LB_CONSISTENCY_CONFIG: ZooKeeper configuration (optional)
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

Three strategies are available (registered with `LoadBalanceStrategyFactory`):

- **RANDOM**: Random worker selection
- **SHORTEST_TTFT**: Select worker with shortest Time-To-First-Token
- **WEIGHTED_CACHE**: Cache-aware selection prioritizing workers with matching KV cache blocks

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
  "loadBalanceStrategy": "ROUND_ROBIN_LOWEST_CONCURRENCY",
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

### WHALE_SYNC_LB_CONSISTENCY_CONFIG (optional, for master election)
ZooKeeper connection configuration for distributed coordination.

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
