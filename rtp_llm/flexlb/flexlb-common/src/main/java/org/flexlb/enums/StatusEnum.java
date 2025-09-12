package org.flexlb.enums;

import lombok.Getter;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.exception.AccessDeniedWhaleException;
import org.flexlb.exception.BadRequestWhaleException;
import org.flexlb.exception.BalancingServiceTimeoutException;
import org.flexlb.exception.BalancingWorkerException;
import org.flexlb.exception.BreakerTriggeredWhaleException;
import org.flexlb.exception.ClientDisconnectWhaleException;
import org.flexlb.exception.ConfigException;
import org.flexlb.exception.ConnectionResetException;
import org.flexlb.exception.EngineAbnormalDisconnectException;
import org.flexlb.exception.EngineConcurrencyConflictException;
import org.flexlb.exception.EngineConnectException;
import org.flexlb.exception.EngineInternalException;
import org.flexlb.exception.EngineMallocException;
import org.flexlb.exception.EngineReadTimeoutException;
import org.flexlb.exception.EngineResponseTooLargeException;
import org.flexlb.exception.EngineTimeoutException;
import org.flexlb.exception.EngineUnexpectedException;
import org.flexlb.exception.ExternalServiceException;
import org.flexlb.exception.ImageSpaceTransitionException;
import org.flexlb.exception.JsonMapperException;
import org.flexlb.exception.LoadBalanceException;
import org.flexlb.exception.MallocBlockException;
import org.flexlb.exception.ModelRoutConfigException;
import org.flexlb.exception.NettyCatchWhaleException;
import org.flexlb.exception.PromptEmptyException;
import org.flexlb.exception.PromptExceedException;
import org.flexlb.exception.PromptException;
import org.flexlb.exception.RedisWhaleException;
import org.flexlb.exception.RetryExhaustedException;
import org.flexlb.exception.ServiceDiscoveryException;
import org.flexlb.exception.ThrottlingWhaleException;
import org.flexlb.exception.WhaleException;
import org.flexlb.exception.WorkerInfoServiceException;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

@Getter
public enum StatusEnum {
    /*--------------------------------------------------- 成功 ------------------------------------------------------*/

    SUCCESS(200, "Success", "Success.", WhaleException.class),

    /*------------------------------------------------- 客户端错误 ---------------------------------------------------*/

    BAD_REQUEST(400, "BadRequest", "Bad request!", BadRequestWhaleException.class),
    ACCESS_DENIED(403, "AccessDenied", "Access denied!", AccessDeniedWhaleException.class),
    /**
     * 直接访问worker才会出现
     */
    ENGINE_CONCURRENCY_CONFLICT(409, "EngineConcurrencyConflict", "Engine concurrency conflict!",
                                EngineConcurrencyConflictException.class),
    STATUS_RUNTIME_EXCEPTION_CANCEL(418, "StatusRuntimeExceptionCancel", "Status runtime exception cancel!",
                                    WhaleException.class),
    THROTTLING(429, "Throttling", "Too many requests for inference service!", ThrottlingWhaleException.class),
    BREAKER_TRIGGERED(410, "BreakerTriggered", "Breaker triggered!", BreakerTriggeredWhaleException.class),
    CLIENT_DISCONNECT(499, "ClientDisconnect", "Client disconnected!", ClientDisconnectWhaleException.class),

    /*-------------------------------------------------- 服务端错误 --------------------------------------------------*/

    INTERNAL_ERROR(500, "InternalError", "Internal server error!", WhaleException.class),
    PROMPT_ERROR(507, "PromptError", "Prompt error!", PromptException.class),
    PROMPT_EMPTY_ERROR(510, "PromptEmptyError", "Prompt empty error!", PromptEmptyException.class),
    PROMPT_EXCEED_MAX_TOKENS_ERROR(511, "PromptExceedMaxTokens", "Prompt exceed max tokens error!",
                                   PromptExceedException.class),
    SERVICE_DISCOVERY_ERROR(512, "ServiceDiscoveryError", "Service discovery error!", ServiceDiscoveryException.class),
    MALLOC_BLOCK_ERROR(513, "MallocBlockError", "Malloc block error!", MallocBlockException.class),
    ENGINE_INTERNAL_ERROR(514, "EngineInternalError", "Engine internal server error!", EngineInternalException.class),
    ENGINE_UNEXPECTED_ERROR(516, "EngineUnexpectedError", "Engine unexpected error!", EngineUnexpectedException.class),
    LOAD_BALANCE_ERROR(517, "LoadBalanceError", "Load balance error!", LoadBalanceException.class),
    ROUTE_CONFIG_ERROR(518, "ModelRoutConfigError", "The model is not ready, missing model route config", ModelRoutConfigException.class),
    NETTY_CATCH_ERROR(519, "NettyCatchError", "Netty catch error!", NettyCatchWhaleException.class),
    ENGINE_CONNECT_ERROR(520, "EngineConnectError", "Engine connect error!", EngineConnectException.class),
    REDIS_ERROR(521, "RedisError", "Redis error!", RedisWhaleException.class),
    JSON_MAPPER_ERROR(522, "JsonMapperError", "Json mapper error!", JsonMapperException.class),
    READ_TIME_OUT(523, "ReadEngineTimeout", "Read Engine time out!", EngineReadTimeoutException.class),
    RETRY_EXHAUSTED_EXCEPTION(524, "RetryExhaustedException", "Retry exceed times!", RetryExhaustedException.class),
    EXTERNAL_SERVICE_ERROR(525, "ExternalServiceError", "External service error!", ExternalServiceException.class),
    IMAGE_SPACE_TRANSITION_ERROR(526, "ImageSpaceTransitionError", "Image space transition error", ImageSpaceTransitionException.class),
    CONNECTION_RESET_EXCEPTION(527, "ConnectionResetException", "Connection Reset Exception!", ConnectionResetException.class),
    ENGINE_RESPONSE_TOO_LARGE(528, "EngineResponseTooLarge", "Engine response too large", EngineResponseTooLargeException.class),

    BALANCING_WORKER_EXCEPTION(600, "BalancingWorkerException", "Balancing worker exception!",
                               BalancingWorkerException.class),
    BALANCING_SERVICE_TIMEOUT(601, "BalancingServiceTimeoutException", "Balancing service timeout exception!",
            BalancingServiceTimeoutException.class),
    MALLOC_EXCEPTION(602, "EngineMallocException", "Malloc exception", EngineMallocException.class),
    ENGINE_TIMEOUT_EXCEPTION(603, "EngineTimeoutException", "Engine timeout control: timout error!", EngineTimeoutException.class),
    WORKER_INFO_SERVICE_TIMEOUT(604, "WorkerInfoServiceTimeoutException", "Worker info service timeout exception!", WorkerInfoServiceException.class),
    DIAMOND_CONFIG_EXCEPTION(701, "ConfigException", "config exception!", ConfigException.class),

    ALL(-9, "ALL", "All.", WhaleException.class),

    /*-------------------------------------------------- 可重试错误 80xx ----------------------------------------------*/

    ENGINE_ABNORMAL_DISCONNECT_EXCEPTION(8000, "EngineAbnormalDisconnectException", "Engine abnormal disconnect!", EngineAbnormalDisconnectException.class),

    ;

    private final int code;

    private final String name;

    private final String message;

    private final Class<? extends WhaleException> exceptionClz;

    public static final Map<Integer, Function<String, WhaleException>> statusExceptionMapper = new HashMap<>();

    static {
        for (StatusEnum statusEnum : StatusEnum.values()) {
            statusExceptionMapper.put(statusEnum.code, statusEnum::toException);
        }
    }

    StatusEnum(int code, String name, String message, Class<? extends WhaleException> exceptionClz) {
        this.code = code;
        this.name = name;
        this.message = message;
        this.exceptionClz = exceptionClz;
    }

    public WhaleException toException() {
        return toException("");
    }

    public WhaleException toException(String exceptionMsg) {
        String msg = this.message;
        if (StringUtils.isNotBlank(exceptionMsg)) {
            msg = this.message + ": " + exceptionMsg;
        }
        try {
            return getExceptionClz().getDeclaredConstructor(int.class, String.class, String.class)
                    .newInstance(this.code, this.name, msg);
        } catch (Throwable e) {
            return new WhaleException(this.code, this.name, msg);
        }
    }

    public WhaleException toException(Throwable cause) {
        return toException(null, cause);
    }

    public WhaleException toException(String exceptionMsg, Throwable cause) {
        String msg = this.message;
        if (StringUtils.isNotBlank(exceptionMsg)) {
            msg = this.message + ": " + exceptionMsg;
        }
        try {
            return getExceptionClz().getDeclaredConstructor(int.class, String.class, String.class, Throwable.class)
                    .newInstance(this.code, this.name, msg, cause);
        } catch (Throwable e) {
            return new WhaleException(this.code, this.name, msg, cause);
        }
    }
}
