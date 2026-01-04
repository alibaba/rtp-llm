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
- `HttpMasterLoadBalanceServer`: Main HTTP service for load balancing requests
- `HealthCheckServer`: Health check endpoints
- Configuration: `application.yml` (reactive mode, ports 7001/8803)

### flexlb-common
Shared utilities, data models, exception handling, and common configurations used across all modules.

Key classes:
- `ServerStatus`: Worker node status representation
- `MasterRequest`/`MasterResponse`: API request/response models
- `RoleType`: Enum defining worker roles (PREFILL, DECODE, PDFUSION, VIT)
- `LoadBalanceStrategyEnum`: Available load balancing strategies
- `ConfigService`: Configuration interface for environment variables

### flexlb-grpc
gRPC client implementation for model service communication. Contains protocol buffer definitions and generated stubs for communicating with backend AI worker nodes.

### flexlb-sync
Core load balancing logic, scheduling strategies, and worker status synchronization. This is the heart of the load balancing system.

Key concepts:
- **Router pattern**: `Router` interface + `DefaultRouter` implementation for multi-role request routing
- **LoadBalancer pattern**: Strategy interface for worker selection (Random, WeightedCache, ShortestTTFT)
- **Worker synchronization**: Periodic gRPC-based status sync (`GrpcWorkerStatusRunner`)
- **Master election**: ZooKeeper-based leader election (`ZookeeperMasterElectService`)

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
# - WHALE_MASTER_CONFIG: Load balance strategy, timeouts, batch settings
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

### Master Election and Consistency

For high availability, FlexLB uses ZooKeeper-based master election:

- `ZookeeperMasterElectService`: Handles leader election
- `LBStatusConsistencyService`: Manages master-slave state consistency
- Only the elected master performs routing decisions

## Configuration

FlexLB reads configuration from environment variables:

### WHALE_MASTER_CONFIG (required)
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
  "enableGrpcPrefillMaster": false
}
```

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
Do what is asked; no more, no less.
Always prefer editing existing files over creating new ones.
Do not proactively create documentation files (*.md) or README files unless explicitly requested.
