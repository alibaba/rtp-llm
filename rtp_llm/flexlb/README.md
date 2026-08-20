# FlexLB - Intelligent Load Balancer for AI Model Inference

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java](https://img.shields.io/badge/Java-8+-red.svg)](https://www.oracle.com/java/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.1-brightgreen.svg)](https://spring.io/projects/spring-boot)

FlexLB is a high-performance, intelligent load balancer specifically designed for AI model inference workloads. It provides advanced load balancing strategies, request batching, caching mechanisms, and automatic failover to optimize the performance and reliability of AI service deployments.

## Features

- **Smart Load Balancing**: Multiple strategies including cost-based routing, shortest TTFT, and cache affinity
- **Request Batching**: Intelligent batching of inference requests to improve throughput
- **Advanced Caching**: KV cache management for improved performance
- **Health Monitoring**: Real-time worker health checking and automatic failover
- **Reactive Architecture**: Built on Spring WebFlux for high concurrency
- **gRPC Support**: Native gRPC client implementation for backend services
- **Metrics & Monitoring**: Prometheus metrics integration
- **Master Election**: ZooKeeper-based master election for high availability

## Architecture

FlexLB consists of four main modules:

- **flexlb-api**: Web layer providing HTTP endpoints and reactive web services
- **flexlb-common**: Shared utilities, data models, exception handling, and common configurations
- **flexlb-grpc**: gRPC client implementation for model service communication
- **flexlb-sync**: Core load balancing logic, scheduling strategies, and worker status synchronization

## Quick Start

### Prerequisites

- Java 8 or higher
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

### Configuration

`FLEXLB_CONFIG` is the single public configuration document for FlexLB scheduling,
dispatch, routing, worker-state synchronization, and observability. It is JSON carried
directly in the environment variable; a file-path form is not supported.

The parser is strict: duplicate keys, unknown fields, fields from inactive tagged
variants, `null`, scalar coercion, numeric enum values, and trailing JSON are rejected at
startup. Optional fields must be omitted rather than set to `null`. If the environment
variable is absent, the code defaults to `QUEUE + FIFO + BATCH` and the remaining model
defaults.

The following example activates every major configuration section:

```bash
export FLEXLB_CONFIG='{
  "schemaVersion": 1,
  "scheduler": {
    "type": "QUEUE",
    "queueTimeoutMs": 3600000,
    "ordering": {
      "type": "PRIORITY",
      "defaultPriority": 50,
      "preemption": {
        "allowedVictimStages": [
          "PREFILL_QUEUED",
          "DECODE_RESERVED",
          "DECODE_ENGINE_OWNED"
        ],
        "engineCancellation": {
          "ackTimeoutMs": 50,
          "completionTimeoutMs": 1000
        }
      }
    },
    "capacity": {
      "maxOutstandingRequestsGlobal": 100000
    },
    "lifecycle": {
      "staleInflightTimeoutMs": 300000,
      "deliveredNotAcceptedTimeoutMs": 30000,
      "maxDeliveredNotAcceptedRequestsGlobal": 200
    }
  },
  "dispatcher": {
    "type": "BATCH",
    "maxRequests": 8,
    "maxCollectionWaitMs": 300,
    "maxWaitingRequestsPerPrefillWorker": 1024,
    "earlyDispatchPredictedExecutionMs": 100,
    "maxInflightBatchesPerPrefillWorker": 2,
    "enqueueRpcTimeoutMs": 5000
  },
  "router": {
    "availabilityHysteresisPercent": 15,
    "groupSelector": {
      "defaultTargets": [
        {"group": "default-group", "weight": 1}
      ],
      "rules": [
        {
          "name": "long-context",
          "match": {"inputTokens": {"min": 8192}},
          "targets": [
            {"group": "long-context-group", "weight": 1}
          ]
        }
      ]
    },
    "roles": {
      "prefill": {
        "availability": {
          "maxPendingRequests": 64
        },
        "executionTimeEstimator": {
          "type": "FORMULA",
          "expression": "sum(computeTokens) + 0.3*sum(hitCacheTokens)"
        },
        "selector": {
          "type": "ESTIMATED_TTFT",
          "candidateChoice": {
            "type": "RANDOM_WITHIN_TOLERANCE",
            "relativeTolerance": 0.1,
            "minimumToleranceMs": 20,
            "outlierRejection": {
              "maxPendingVsAverageMultiplier": 3.0,
              "maxWaitVsAverageMultiplier": 3.0
            }
          }
        },
        "cacheAffinity": {
          "maxExtraTtftMs": 100,
          "minPrefixHitPercent": 5
        }
      },
      "decode": {
        "availability": {
          "maxKvUsagePercent": 90,
          "maxEngineRequests": 128
        },
        "kvReservation": {
          "maxOutputTokensForEstimate": 1000
        },
        "selector": {
          "type": "KV_USAGE_WEIGHTED_RANDOM",
          "decayPerToken": 0.001,
          "outlierRejection": {
            "maxEngineLoadVsAverageMultiplier": 3.0,
            "maxKvUsedVsAverageMultiplier": 3.0
          }
        }
      },
      "vit": {
        "selector": {"type": "RANDOM"}
      }
    }
  },
  "workerRegistry": {
    "health": {
      "statusPollIntervalMs": 20,
      "statusRpcTimeoutMs": 5000,
      "statusStaleAfterMs": 10000
    },
    "cacheStatus": {
      "targetDiffSize": 30,
      "minRefreshIntervalMs": 50,
      "maxRefreshIntervalMs": 3000,
      "fullSnapshotDebugMode": false
    }
  },
  "observability": {
    "cacheHit": {
      "recentKeyWindow": {
        "writeEnabled": true,
        "durationMs": 1800000,
        "maxKeyOccurrences": 10000000
      },
      "metricsEnabled": true,
      "requestTraceLogEnabled": false,
      "theoryLog": {
        "path": "/home/admin/ai-whale/logs/master_theory_hit.log"
      }
    }
  }
}'
```

`MODEL_SERVICE_CONFIG` still describes service discovery and endpoint topology; it is
not a second FlexLB behavior configuration:

```bash
export MODEL_SERVICE_CONFIG='{
    "service_id": "model.service",
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

### Scheduler, ordering, and dispatcher

These are separate concepts:

| Scheduler | Queue ordering | Dispatcher | Behavior |
| --- | --- | --- | --- |
| `DIRECT` | not applicable | `NON_BATCH` | Route immediately; the frontend sends the request |
| `QUEUE` | `FIFO` | `NON_BATCH` | Queue lifecycle with arrival-order, one immediate route decision per request |
| `QUEUE` | `PRIORITY` | `NON_BATCH` | Queue lifecycle with priority-order, one immediate route decision per request |
| `QUEUE` | `FIFO` | `BATCH` | Arrival-order collection followed by Master `EnqueueBatch` |
| `QUEUE` | `PRIORITY` | `BATCH` | Priority-order collection followed by Master `EnqueueBatch` |

`DIRECT + BATCH` is invalid. `FIFO` and `PRIORITY` only describe the order of
requests owned by `QUEUE`; neither is a synonym for direct routing or batching.
`PRIORITY` adds `defaultPriority` and optional preemption. It does not add an SLO
budget, length buckets, or priority-dependent TTL multipliers. `QUEUE` uses one
absolute scheduling expiration derived from FlexLB admission time plus
`scheduler.queueTimeoutMs`; retries and preemption do not reset it. `DIRECT` has
no scheduling timeout.

`BATCH` dispatches on batch size, maximum collection wait, or the optional predicted
execution threshold. `NON_BATCH` has no collection window or target batch size.

Production-style examples migrated from the former field-level environment variables:

- [QUEUE + PRIORITY + NON_BATCH](docs/config-examples/flexlb-queue-priority-non-batch.json)
- [QUEUE + PRIORITY + BATCH](docs/config-examples/flexlb-queue-priority-batch.json)

DIRECT uses the same role routing configuration as QUEUE. For example, a compact
DIRECT configuration with explicit random prefill/decode selection is:

```bash
export FLEXLB_CONFIG='{
  "schemaVersion": 1,
  "scheduler": {"type": "DIRECT"},
  "dispatcher": {"type": "NON_BATCH"},
  "router": {
    "roles": {
      "prefill": {"selector": {"type": "RANDOM"}},
      "decode": {"selector": {"type": "RANDOM"}},
      "vit": {"selector": {"type": "RANDOM"}}
    }
  }
}'
```

PREFILL and PDFUSION share the prefill selector. Prefill selector types are
`RANDOM` and `ESTIMATED_TTFT`; the latter supports `BEST_ONLY`,
`RANDOM_WITHIN_TOLERANCE`, or `LEAST_RECENTLY_USED_IN_POOL` candidate choice.
The candidate pool for `LEAST_RECENTLY_USED_IN_POOL` is tagged as either
`{"type":"RATIO","ratio":0.3,"minimumWorkers":1}` or
`{"type":"FIXED","workers":2}`. Decode selector types are `RANDOM` and
`KV_USAGE_WEIGHTED_RANDOM`; VIT currently supports `RANDOM`.

Cache affinity is enabled by including `router.roles.prefill.cacheAffinity` and is
valid only with `ESTIMATED_TTFT`. Omit the object to disable it. Decode admission is
controlled by the optional positive
`router.roles.decode.availability.maxEngineRequests`; omit it for no FlexLB-side
request-count cap.

See [QUEUE ordering and dispatcher modes](docs/priority-scheduler-delivery-modes.md)
for the QUEUE lifecycle, accounting invariants, complete scheduler/dispatcher
parameter reference, and mode matrix.

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

## Configuration reference

- **FlexLB behavior**: one strict JSON document in `FLEXLB_CONFIG`.
- **Prefill execution formula**:
  `router.roles.prefill.executionTimeEstimator.expression` when estimator type is
  `FORMULA`.
- **Routing strategy parameters**: the tagged selector objects under
  `router.roles.prefill`, `router.roles.decode`, and `router.roles.vit`.
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
