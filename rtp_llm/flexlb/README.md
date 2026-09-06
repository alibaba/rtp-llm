# FlexLB - Intelligent Load Balancer for AI Model Inference

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java](https://img.shields.io/badge/Java-21-red.svg)](https://openjdk.org/projects/jdk/21/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.18-brightgreen.svg)](https://spring.io/projects/spring-boot)

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

- Java 21 (matches `java.version` in `pom.xml`; see `.sdkmanrc` for the recommended distribution)
- ZooKeeper (optional, for master election)

The project includes the Maven Wrapper (`./mvnw` on Unix/Linux/macOS, `mvnw.cmd` on Windows), so no separate Maven installation is required.

### Build

See [AGENTS.md](AGENTS.md) for the full, authoritative list of build and test commands. The most common one:

```bash
./mvnw clean package -DskipTests
```

### Configuration

`FLEXLB_CONFIG` is the single public configuration document for FlexLB scheduling,
dispatch, routing, worker-state synchronization, and observability. It is JSON carried
directly in the environment variable; a file-path form is not supported.

`schemaVersion` is numeric: `0` identifies the historical combined document and `1`
identifies the current standard document. A document without `schemaVersion` defaults
to `0`. `FLEXLB_CONFIG_SCHEMA_VERSION` supplies the fallback version when the document
does not declare one; an explicit version in Nacos or another source takes precedence,
so a standard document with `"schemaVersion": 1` can replace an environment-selected
compatibility configuration without changing the environment variable first.

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
    "logging": {
      "level": "info",
      "stdoutEnabled": false
    },
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
  },
  "serviceDiscovery": {
    "connectTimeoutMs": 500,
    "readTimeoutMs": 500,
    "pollIntervalMs": 1000,
    "maxIdleConnections": 5,
    "keepAliveDurationMs": 300000
  },
  "cacheMatching": {"type": "LOCAL_SYNC"},
  "optimizer": {
    "enabled": false,
    "discoveryPollIntervalMs": 1000
  },
  "consistency": {"type": "NONE"},
  "blockHashStrategy": "VLLM",
  "enableFallback": false
}'

# Optional: prometheus, kmonitor, or noop (the default).
# Missing or unsupported values fall back to the NoOp monitor.
export FLEXLB_MONITOR_PROVIDER=prometheus
```

The top-level `enableFallback` switch defaults to `false`. When enabled, the gRPC
schedule endpoint returns `success=false`, code `8600`, and error message `FALLBACK`
before forwarding or routing so the caller can use domain routing. FlexLB behavior is
read only from `FLEXLB_CONFIG`; legacy field-level variables such as `ENABLE_FALLBACK`,
`BLOCK_HASH_STRATEGY`, `FLEXLB_LOG_LEVEL`, and `ENABLE_STDOUT_LOG` are ignored.

`MODEL_SERVICE_CONFIG` still describes service discovery and endpoint topology; it is
not a second FlexLB behavior configuration:

```bash
export MODEL_SERVICE_CONFIG='{
    "service_id": "model.service",
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

Each endpoint must contain exactly one `discovery` object. Supported types are:

- `static-env`: Reads `hosts` directly from the endpoint configuration.
- `vipserver`: Uses `address` as the VipServer service name (internal builds).
- `dashscope`: Uses `address` as the virtual service ID (internal builds). `base_url` defaults to
  `http://127.0.0.1:8880` when omitted.

`worker_status_port` is optional and controls the gRPC port used only for `GetWorkerStatus`.
When omitted, FlexLB uses the endpoint gRPC port (`http` discovery port + 1, or the discovered
port itself when `protocol` is `grpc`).

Discovery runtime policy belongs to `FLEXLB_CONFIG`, not to each topology endpoint:

```json
{
  "serviceDiscovery": {
    "connectTimeoutMs": 500,
    "readTimeoutMs": 500,
    "pollIntervalMs": 1000,
    "maxIdleConnections": 5,
    "keepAliveDurationMs": 300000
  }
}
```

The values shown above are the code defaults and are exposed to discovery providers from the
current FlexLB snapshot. Endpoint `discovery` objects contain only locator data: `type`, optional
`base_url`, and `hosts` for `static-env`. There is no global discovery strategy or fallback.

To query cache matches from KVCM instead of the local cache index, configure KVCM topology in
`MODEL_SERVICE_CONFIG`:

```json
{
  "service_id": "aigc.text-generation.generation.engine_service",
  "role_endpoints": [{
    "group": "default",
    "pd_fusion_endpoint": {
      "address": "v-workers",
      "protocol": "grpc",
      "discovery": {
        "type": "dashscope"
      }
    }
  }],
  "kvcm": {
    "address": "v-kvcm",
    "port": 6381,
    "discovery": {
      "type": "dashscope"
    }
  }
}
```

Select KVCM and configure its runtime behavior in `FLEXLB_CONFIG`:

```json
{
  "cacheMatching": {
    "type": "KVCM",
    "requestTimeoutMs": 500,
    "leaderRefreshIntervalMs": 10000,
    "heartbeatFailureThreshold": 3,
    "queryFailureThreshold": 10,
    "maxQueryRetryCount": 1,
    "recoverySuccessThreshold": 3,
    "p2pHostCount": 0,
    "localStandby": {
      "autoSwitch": true,
      "blockSize": 0,
      "ttlMs": 300000,
      "minimumTtlMs": 100000,
      "ttlReductionStartRatio": 0.8,
      "maximumEntries": 2000000,
      "capacityMultiplier": 10,
      "asyncQueueCapacity": 100000,
      "hashThreadCount": 4,
      "hashQueueCapacity": 100000
    }
  }
}
```

The worker deployment name returned by DashScope discovery is used as the KVCM cache namespace.
The namespace is sent through the KVCM protocol's `instance_id` field.
KVCM communication always uses gRPC and does not require a protocol setting.
The optional KVCM `port` defaults to `6381` and is used with discovered seed IPs only for
`GetClusterInfo`. Subsequent RPCs use the leader host and `meta_rpc_port` returned in
`leader_endpoint`.

Each cache query is retried once by default before that request falls back to Local Standby.
`cacheMatching.maxQueryRetryCount` controls the maximum retry count and does not include the
initial attempt. KVCM is
marked unhealthy after three consecutive `GetClusterInfo` failures or ten logical cache-query
failures after retries are exhausted. It recovers only after three consecutive successful
background `GetClusterInfo` probes. The optional `heartbeatFailureThreshold`,
`queryFailureThreshold`, and `recoverySuccessThreshold` fields override those defaults.
`localStandby.autoSwitch` controls whether an unhealthy KVCM changes subsequent requests to
Local Standby automatically; the current request still falls back after its KVCM retries fail.
Local Standby multiplies each worker's HBM block capacity reported by `GetWorkerStatus` by
`capacityMultiplier`, sums the results, and caps the global metadata budget at
`maximumEntries`. The global TTL starts decreasing linearly at
`ttlReductionStartRatio` utilization, from `ttlMs` to `minimumTtlMs` at full
utilization. Below 80% utilization, cleanup runs every 30 seconds and scans roughly 10% of block
hashes. Between 80% and 90%, it runs every 20 seconds and scans roughly 20%. At or above 90%, it
runs every 10 seconds and scans the full index. The request that first raises utilization to 90%
immediately submits the same cleanup task; a single trigger flag prevents concurrent requests from
submitting duplicates. At the capacity limit, existing mappings remain refreshable but new
mappings are paused until cleanup reduces usage below 100%. Because this is an approximate
metadata budget, concurrent additions may exceed the limit slightly.

`kvcm.namespace` can explicitly override the namespace for every role and group:

```json
{
  "kvcm": {
    "address": "v-kvcm",
    "namespace": "vllm-test-0",
    "discovery": {
      "type": "dashscope"
    }
  }
}
```

When `namespace` is non-blank, it takes priority over deployment names discovered from
worker endpoints. When omitted, FlexLB keeps resolving namespaces by role and group from
worker discovery metadata.

When KVCM is enabled, FlexLB stops polling `GetCacheStatus`. Engines must return
`available_kv_cache`, `total_kv_cache`, and `block_size` from `GetWorkerStatus`.

Optimizer follows the same split: `MODEL_SERVICE_CONFIG.optimizer` contains only
`address`, `port`, `path`, and `discovery`, while `FLEXLB_CONFIG.optimizer.enabled` and
`discoveryPollIntervalMs` control behavior. ZooKeeper consistency is selected by
`FLEXLB_CONFIG.consistency={"type":"ZOOKEEPER",...}` with `connectString`,
`sessionTimeoutMs`, `connectionTimeoutMs`, and `masterRefreshIntervalMs`; `{"type":"NONE"}`
disables it.

### Run

```bash
java -jar flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar \
--spring.profiles.active=test
```

By default, the service starts on port 7001 with management endpoints on port 7002.

## API Documentation

### Health Check
```
GET http://localhost:7002/health
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
  `FORMULA`. Omitting the estimator applies the code default: the production
  DSv4 prefill fit (`RoutingConfig.FormulaEstimatorConfig.DEFAULT_EXPRESSION`).
- **Routing strategy parameters**: the tagged selector objects under
  `router.roles.prefill`, `router.roles.decode`, and `router.roles.vit`.
- **Traffic group selection**: `router.groupSelector` inside the same document.
- **Backend topology**: `MODEL_SERVICE_CONFIG`.
- **Cache matching, Optimizer, discovery runtime policy, and ZooKeeper consistency**:
  their behavior lives in `FLEXLB_CONFIG`; only service locators live in
  `MODEL_SERVICE_CONFIG`.

## Monitoring

FlexLB provides comprehensive monitoring through:

- Prometheus metrics endpoint: `/actuator/prometheus`
- Health checks: `/actuator/health`
- Application info: `/actuator/info`

At Spring startup, the environment document establishes the strict baseline. If
`FLEXLB_NACOS_SERVER_ADDR` is configured, Nacos supplies higher-priority recursive partial
overrides and subsequent valid updates replace the in-memory `FlexlbConfig` snapshot. A missing
field—including a field deleted from Nacos—is a no-op and retains its current in-memory value;
objects merge recursively, while arrays and scalars replace as a whole. Changing a tagged-union
`type` replaces that complete branch. Blank content and `{}` are also no-ops. Unknown fields,
scalar coercion, `null`, and invalid cross-field combinations are rejected; a rejected runtime
update leaves the last-known-good snapshot active. `MODEL_SERVICE_CONFIG` remains a startup-only
topology document and is never overridden by Nacos. Whether a valid FlexLB update takes effect
immediately or after restart is determined by the consuming component, not by the Nacos layer.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
