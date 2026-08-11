# FlexLB - Intelligent Load Balancer for AI Model Inference

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java](https://img.shields.io/badge/Java-21-red.svg)](https://openjdk.org/projects/jdk/21/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.18-brightgreen.svg)](https://spring.io/projects/spring-boot)

FlexLB is a high-performance, intelligent load balancer specifically designed for AI model inference workloads. It provides advanced load balancing strategies, request batching, caching mechanisms, and automatic failover to optimize the performance and reliability of AI service deployments.

## Features

- **Smart Load Balancing**: Multiple strategies including round-robin, lowest concurrency, and shortest TTFT (Time to First Token)
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

Configure the following environment variables:

```bash
export FLEXLB_CONFIG='{
    "deploy":"DISAGGREGATED",
    "loadBalanceStrategy":"ROUND_ROBIN_LOWEST_CONCURRENCY",
    "prefillBatchWaitTimeMs":100,
    "kvCache":"LOCAL_STATIC",
    "staticCacheBlockSize":500,
    "batchSize":1,
    "prefillLbTimeoutMs":300,
    "prefillGenerateTimeoutMs": 5000,
    "enableGrpcPrefillMaster": false
}'

export MODEL_SERVICE_CONFIG='{
    "service_id": "aigc.text-generation.generation.engine_service",
    "role_endpoints": [{
        "group": "default",
        "pd_fusion_endpoint": {
            "address": "local-engine",
            "protocol": "http",
            "path": "/",
            "worker_status_port": 18002,
            "discovery": {
                "type": "static-env",
                "hosts": ["127.0.0.1:8080"]
            }
        }
    }]
}'

# Optional: prometheus, kmonitor, or noop (the default).
# Missing or unsupported values fall back to the NoOp monitor.
export FLEXLB_MONITOR_PROVIDER=prometheus
```

Each endpoint must contain exactly one `discovery` object. Supported types are:

- `static-env`: Reads `hosts` directly from the endpoint configuration.
- `vipserver`: Uses `address` as the VipServer service name (internal builds).
- `dashscope`: Uses `address` as the virtual service ID (internal builds). `base_url` defaults to
  `http://127.0.0.1:8880` when omitted.

`worker_status_port` is optional and controls the gRPC port used only for `GetWorkerStatus`.
When omitted, FlexLB uses the endpoint gRPC port (`http` discovery port + 1, or the discovered
port itself when `protocol` is `grpc`).

DashScope tuning fields are optional and belong to the same `discovery` object:

```json
{
  "type": "dashscope",
  "base_url": "http://127.0.0.1:8880",
  "connect_timeout_ms": 500,
  "read_timeout_ms": 500,
  "poll_interval_ms": 1000,
  "max_idle_connections": 5,
  "keep_alive_duration_ms": 300000
}
```

The values shown above are the code defaults. There is no global discovery strategy or fallback.

To query cache matches from KVCM instead of the local cache index, add a `kvcm`
object at the same level as `role_endpoints`:

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
    "enabled": true,
    "address": "v-kvcm",
    "port": 6381,
    "discovery": {
      "type": "dashscope"
    },
    "request_timeout_ms": 500,
    "max_query_retry_count": 1,
    "leader_refresh_interval_ms": 10000,
    "local_standby": {
      "auto_switch": true,
      "ttl_ms": 300000,
      "minimum_ttl_ms": 100000,
      "ttl_reduction_start_ratio": 0.8,
      "maximum_entries": 2000000,
      "capacity_multiplier": 10
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
`max_query_retry_count` controls the maximum retry count and does not include the initial attempt. KVCM is
marked unhealthy after three consecutive `GetClusterInfo` failures or ten logical cache-query
failures after retries are exhausted. It recovers only after three consecutive successful
background `GetClusterInfo` probes. The optional `heartbeat_failure_threshold`,
`query_failure_threshold`, and `recovery_success_threshold` fields override those defaults.
`local_standby.auto_switch` controls whether an unhealthy KVCM changes subsequent requests to
Local Standby automatically; the current request still falls back after its KVCM retries fail.
Local Standby multiplies each worker's HBM block capacity reported by `GetWorkerStatus` by
`capacity_multiplier`, sums the results, and caps the global metadata budget at
`maximum_entries`. The global TTL starts decreasing linearly at
`ttl_reduction_start_ratio` utilization, from `ttl_ms` to `minimum_ttl_ms` at full
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
    "enabled": true,
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

## Configuration

FlexLB supports various configuration options through environment variables and Spring Boot properties:

- **Load Balancing Strategy**: Configure through `FLEXLB_CONFIG`
- **Backend Services**: Configure through `MODEL_SERVICE_CONFIG`
- **ZooKeeper Settings**: Configure through `FLEXLB_SYNC_CONSISTENCY_CONFIG`
## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
