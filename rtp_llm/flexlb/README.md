# FlexLB - Intelligent Load Balancer for AI Model Inference

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java](https://img.shields.io/badge/Java-8+-red.svg)](https://www.oracle.com/java/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.1-brightgreen.svg)](https://spring.io/projects/spring-boot)

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
    "request_timeout_ms": 50,
    "leader_refresh_interval_ms": 5000
  }
}
```

The worker deployment name returned by DashScope discovery is used as the KVCM cache namespace.
The namespace is sent through the KVCM protocol's `instance_id` field.
KVCM communication always uses gRPC and does not require a protocol setting.
The optional KVCM `port` defaults to `6381` and is used with discovered seed IPs only for
`GetClusterInfo`. Subsequent RPCs use the leader host and `meta_rpc_port` returned in
`leader_endpoint`.

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

## Configuration

FlexLB supports various configuration options through environment variables and Spring Boot properties:

- **Load Balancing Strategy**: Configure through `FLEXLB_CONFIG`
- **Backend Services**: Configure through `MODEL_SERVICE_CONFIG`
- **ZooKeeper Settings**: Configure through `FLEXLB_SYNC_CONSISTENCY_CONFIG`

## Monitoring

FlexLB provides comprehensive monitoring through:

- Prometheus metrics endpoint: `/actuator/prometheus`
- Health checks: `/actuator/health`
- Application info: `/actuator/info`

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
