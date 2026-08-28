# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

flexlb-sync is the core load balancing module of FlexLB. It handles:
- Load balancing strategy execution (`RandomStrategy`, cost-based Prefill/Decode, and `ShortestTTFTStrategy`)
- Worker node status synchronization via gRPC
- Master election using ZooKeeper
- Request routing across different role types (PREFILL, DECODE, PDFUSION, VIT)
- Integration with flexlb-cache for KV cache management

## Key Architecture Concepts

### Router and LoadBalanceStrategy Pattern
- `Router` interface defines routing contract for incoming requests
- `DefaultRouter` orchestrates routing across multiple role types (e.g., PREFILL → DECODE)
- `LoadBalanceStrategy` interface handles worker selection for a single role type
- Each role type can use different load balancing strategies

### Role-Based Routing
The system routes requests through multiple worker types based on model requirements:
- **PREFILL**: Initial token processing
- **DECODE**: Token generation
- **PDFUSION**: Prefill-Decode fusion workers
- **VIT**: Vision-language model processing

Key classes:
- `RoleType` (in flexlb-common): Enum defining worker roles
- `DefaultRouter`: Implements multi-role routing with rollback support
- `BalanceContext`: Carries request context through routing pipeline

### Worker Status Synchronization
- `GrpcWorkerStatusRunner`: Periodically fetches worker status via gRPC
- `EngineWorkerStatus`: Maintains global worker status mapping
- `ModelWorkerStatus`: Per-model worker information
- `GrpcCacheStatusCheckRunner`: Syncs KV cache status with flexlb-cache module

### Master Election
- `ZookeeperMasterElectService`: ZooKeeper-based leader election
- `LBStatusConsistencyService`: Handles master-slave consistency

## Development Commands

### Build this module only
```bash
# From flexlb-sync directory
mvn clean package -DskipTests

# From parent directory
mvn clean package -pl flexlb-sync -DskipTests
```

### Run tests
```bash
# Run all functional tests in this module
mvn test

# Run this module's performance regressions in a dedicated invocation
mvn test -Psync-performance-regression

# Run a specific test class
mvn test -Dtest=DefaultRouterTest

# Run a specific test method
mvn test -Dtest=DefaultRouterTest#testRouteSuccess
```

### Code formatting check
```bash
# From parent directory
mvn spotless:check -Pspotless-check

# Auto-format code
mvn spotless:apply -Pspotless-check
```

## Project Structure

```
flexlb-sync/
├── balance/
│   ├── scheduler/
│   │   ├── Router.java              # Core routing interface
│   │   └── DefaultRouter.java       # Multi-role router implementation
│   └── strategy/
│       ├── LoadBalanceStrategy.java        # Load balancing interface
│       ├── RandomStrategy.java      # Random selection strategy
│       ├── ShortestTTFTStrategy.java   # TTFT-based strategy
│       ├── CostBasedPrefillStrategy.java   # Predicted Prefill cost strategy
│       └── CostBasedDecodeStrategy.java    # KV-weighted Decode strategy
├── consistency/
│   ├── MasterElectService.java      # Master election interface
│   └── ZookeeperMasterElectService.java  # ZK implementation
├── sync/
│   ├── runner/
│   │   ├── GrpcWorkerStatusRunner.java    # Worker status sync
│   │   └── GrpcCacheStatusCheckRunner.java # Cache status sync
│   └── status/
│       ├── EngineWorkerStatus.java     # Global worker status
│       └── ModelWorkerStatus.java      # Per-model status
└── service/
    ├── RouteService.java            # High-level routing service
    └── grpc/
        └── EngineGrpcService.java   # gRPC client for workers
```

## Key Dependencies

This module depends on:
- **flexlb-common**: Shared data models (`BalanceContext`, `ServerStatus`, `RoleType`, `WorkerStatus`)
- **flexlb-cache**: KV cache management (`CacheAwareService`, `KvCacheManager`)
- **flexlb-grpc**: gRPC protocol definitions and clients

External dependencies:
- Spring Boot 2.7.18 (WebFlux for reactive programming)
- Apache Curator 5.4.0 (ZooKeeper client)
- gRPC 1.65.0
- Caffeine (local caching)
- OpenTelemetry (distributed tracing)

## Important Implementation Notes

### Rollback Mechanism
When routing fails for a later role type (e.g., PREFILL succeeds but DECODE fails),
`DefaultRouter` closes the exact `SelectedRole` capabilities already selected and
rolls back any direct-placement endpoint reservations.

### Load Balancer Selection
Load-balance strategies are ordinary Spring leaf beans. `ConfiguredLoadBalanceSelector`
matches the request's sealed role selector configuration and requires exactly one supporting
strategy. Do not introduce static registration, bean-name lookup, fallbacks, or `@DependsOn`
ordering to wire strategies.

### Worker Status Updates
Worker status is updated asynchronously by scheduled runners. `EndpointRegistry`
owns generation-fenced endpoint publication, while `EngineWorkerStatus` exposes
immutable routing snapshots and exact captures to strategies.

### Cache Integration
The module calls `CacheAwareService` from flexlb-cache to update cache information
received from workers. See `GrpcCacheStatusCheckRunner`.

## Configuration

This module reads configuration from:
- `FLEXLB_CONFIG`: Load balance strategy, timeouts, batch settings
- `FLEXLB_SYNC_CONSISTENCY_CONFIG`: ZooKeeper connection, master election
- `MODEL_SERVICE_CONFIG`: Backend worker endpoints

Configuration is injected via `ConfigService` interface (implementation in flexlb-common).

## Testing Strategy

- Unit tests use Mockito 5.20.0 (no PowerMock needed with Java 21)
- Test classes mirror source structure (e.g., `DefaultRouterTest` for `DefaultRouter`)
- Mock external dependencies (gRPC clients, cache managers, config service)
- Focus on routing logic, strategy selection, and error handling

## Important Reminders
Do what is asked; no more, no less.
Always prefer editing existing files over creating new ones.
Do not proactively create documentation files (*.md) or README files unless explicitly requested.
