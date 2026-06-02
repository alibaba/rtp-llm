package org.flexlb.enums;

import lombok.Getter;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.exception.EngineAbnormalDisconnectException;
import org.flexlb.exception.EngineReadTimeoutException;
import org.flexlb.exception.FlexLBException;
import org.flexlb.exception.JsonMapperException;
import org.flexlb.exception.NettyCatchException;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

@Getter
public enum StatusEnum {
    /*---------------------------------------------- Success ------------------------------------------------------*/

    SUCCESS(200, "Success", "Success.", FlexLBException.class),

    /*-------------------------------------------- Server Errors --------------------------------------------------*/

    INTERNAL_ERROR(500, "InternalError", "Internal server error!", FlexLBException.class),
    NETTY_CATCH_ERROR(519, "NettyCatchError", "Netty catch error!", NettyCatchException.class),
    JSON_MAPPER_ERROR(522, "JsonMapperError", "Json mapper error!", JsonMapperException.class),
    READ_TIME_OUT(523, "ReadEngineTimeout", "Read Engine time out!", EngineReadTimeoutException.class),

    ALL(-9, "ALL", "All.", FlexLBException.class),

    /*------------------------------------------- Retryable Errors 80xx -------------------------------------------*/

    ENGINE_ABNORMAL_DISCONNECT_EXCEPTION(8000, "EngineAbnormalDisconnectException", "Engine abnormal disconnect!", EngineAbnormalDisconnectException.class),

    ;

    private final int code;

    private final String name;

    private final String message;

    private final Class<? extends FlexLBException> exceptionClz;

    public static final Map<Integer, Function<String, FlexLBException>> statusExceptionMapper = new HashMap<>();

    static {
        for (StatusEnum statusEnum : StatusEnum.values()) {
            statusExceptionMapper.put(statusEnum.code, statusEnum::toException);
        }
    }

    StatusEnum(int code, String name, String message, Class<? extends FlexLBException> exceptionClz) {
        this.code = code;
        this.name = name;
        this.message = message;
        this.exceptionClz = exceptionClz;
    }

    public FlexLBException toException() {
        return toException("");
    }

    public FlexLBException toException(String exceptionMsg) {
        String msg = this.message;
        if (StringUtils.isNotBlank(exceptionMsg)) {
            msg = this.message + ": " + exceptionMsg;
        }
        try {
            return getExceptionClz().getDeclaredConstructor(int.class, String.class, String.class)
                    .newInstance(this.code, this.name, msg);
        } catch (Throwable e) {
            return new FlexLBException(this.code, this.name, msg);
        }
    }

    public FlexLBException toException(Throwable cause) {
        return toException(null, cause);
    }

    public FlexLBException toException(String exceptionMsg, Throwable cause) {
        String msg = this.message;
        if (StringUtils.isNotBlank(exceptionMsg)) {
            msg = this.message + ": " + exceptionMsg;
        }
        try {
            return getExceptionClz().getDeclaredConstructor(int.class, String.class, String.class, Throwable.class)
                    .newInstance(this.code, this.name, msg, cause);
        } catch (Throwable e) {
            return new FlexLBException(this.code, this.name, msg, cause);
        }
    }
}
