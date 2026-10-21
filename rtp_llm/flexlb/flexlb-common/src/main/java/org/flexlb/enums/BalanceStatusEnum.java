package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum BalanceStatusEnum {
    SUCCESS(0, "Success"),
    UNKNOWN_ERROR(1000, "Unknown error!"),
    SERVICE_DISCOVERY_ERROR(1001, "Vip server error!"),
    SERVICE_DISCOVERY_TIMEOUT(1002, "Vip server timeout!"),
    RESPONSE_NULL(1004, "Response null!"),

    // Cache related errors (3000-3099)
    CACHE_STATUS_INVALID_VERSION(3000, "Cache status invalid version!"),
    CACHE_UPDATE_FAILED(3001, "Cache update failed!"),
    CACHE_SERVICE_UNAVAILABLE(3002, "Cache service unavailable!"),
    CACHE_GRPC_CONNECTION_FAILED(3003, "Cache gRPC connection failed!"),
    CACHE_GRPC_TIMEOUT(3004, "Cache gRPC timeout!"),

    // Worker Status related errors (3100-3199)
    WORKER_STATUS_GRPC_TIMEOUT(3104, "Worker gRPC timeout!"),
    MASTER_STATUS_MEMORY_EXHAUSTED(3105, "Master status polling memory exhausted!"),
    MASTER_STATUS_EXECUTOR_REJECTED(3106, "Master status executor rejected work!"),
    MASTER_STATUS_PROCESSING_FAILED(3107, "Master status processing failed!"),
    WORKER_STATUS_GRPC_FAILED(3108, "Worker status RPC failed; fault origin unconfirmed!"),
    WORKER_STATUS_CONNECTION_FAILED(3109, "Worker status connection failed!"),
    WORKER_STATUS_SERVER_FAILED(3110, "Worker status reported alive=false!"),
    MASTER_STATUS_BIND_FAILED(3111, "Master socket binding failed!");

    private final int code;

    private final String message;

    BalanceStatusEnum(int code, String message) {
        this.code = code;
        this.message = message;
    }
}
