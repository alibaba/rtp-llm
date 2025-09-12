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
- Maven 3.6+
- ZooKeeper (optional, for master election)

### Build

```bash
mvn clean package -DskipTests
```

### Configuration

Configure the following environment variables:

```bash
export WHALE_MASTER_CONFIG='{
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
    "prefill_endpoint": {
        "path": "/",
        "protocol": "http",
        "type": "SpecifiedIpPortList",
        "address": "[\"localhost:8080\"]"
    },
    "service_id": "model.service"
}'
```

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

- **Load Balancing Strategy**: Configure through `WHALE_MASTER_CONFIG`
- **Backend Services**: Configure through `MODEL_SERVICE_CONFIG`
- **ZooKeeper Settings**: Configure through `WHALE_SYNC_LB_CONSISTENCY_CONFIG`

See [Configuration Guide](docs/configuration.md) for detailed configuration options.

## Monitoring

FlexLB provides comprehensive monitoring through:

- Prometheus metrics endpoint: `/actuator/prometheus`
- Health checks: `/actuator/health`
- Application info: `/actuator/info`

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

