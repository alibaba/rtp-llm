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
    "enableGrpcPrefillMaster": false,
    "decodeConcurrencyLimit": 32
}'

export TRAFFIC_POLICY_CONFIG='{
    "rules": [
        {
            "name": "vip-api-key",
            "api_keys": ["key-a", "key-b"],
            "target_group": "vip-group"
        },
        {
            "name": "long-context",
            "min_seq_len": 8192,
            "target_group": "long-context-group"
        },
        {
            "name": "weighted-split",
            "min_seq_len": 1,
            "target_groups": [
                {"group": "blue-group", "weight": 80},
                {"group": "green-group", "weight": 20}
            ]
        }
    ]
}'

export STRATEGY_CONFIGS='{
    "shortestTtft": {
        "candidatePool": {
            "mode": "FIXED",
            "size": 1
        }
    }
}'

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

Traffic routing is two-layered: `TRAFFIC_POLICY_CONFIG` selects the target `group`, then each role's load balancing strategy selects the final prefill/decode host inside that group. You can also set `TRAFFIC_POLICY_CONFIG_FILE` to a JSON file path. Standalone traffic policy config takes priority over `trafficPolicy` embedded in `FLEXLB_CONFIG`, and you can replace the active policy at runtime with `POST /rtp_llm/update_traffic_policy`.

Set `decodeConcurrencyLimit` to a positive number to cap each decode worker's in-flight requests. FlexLB counts reported waiting/running tasks plus local in-transit selections, deduplicated by request id. When a decode worker reaches the limit, it is not considered serviceable; values <= 0 disable this FlexLB-side limit.

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
- **Strategy Parameters**: Configure strategy internals through `STRATEGY_CONFIGS`; for example `shortestTtft.candidatePool` controls the ShortestTTFT candidate pool. `mode=RATIO` uses `max(minSize, floor(workerCount * ratio))`, while `mode=FIXED` uses `size`.
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
