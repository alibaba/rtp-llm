package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum BalanceStatusEnum {
    SUCCESS(0, "Success"),
    UNKNOWN_ERROR(1000, "Unknown error!"),
    VIPSERVER_ERROR(1001, "Vip server error!"),
    VIPSERVER_TIMEOUT(1002, "Vip server timeout!"),
    RESPONSE_NULL(1004, "Response null!"),

    // Cache related errors (3000-3099)
    CACHE_STATUS_INVALID_VERSION(3000, "Cache status invalid version!"),
    CACHE_UPDATE_FAILED(3001, "Cache update failed!"),
    CACHE_SERVICE_UNAVAILABLE(3002, "Cache service unavailable!"),
    CACHE_GRPC_CONNECTION_FAILED(3003, "Cache gRPC connection failed!"),
    CACHE_GRPC_TIMEOUT(3004, "Cache gRPC timeout!"),

    // Worker Status related errors (3100-3199)
    WORKER_SERVICE_UNAVAILABLE(3102, "Worker service unavailable!"),
    WORKER_STATUS_GRPC_TIMEOUT(3104, "Worker gRPC timeout!")

    ;
    private final int code;

    private final String message;

    BalanceStatusEnum(int code, String message) {
        this.code = code;
        this.message = message;
    }
}
