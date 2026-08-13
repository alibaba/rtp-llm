package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonCreator;

/** Error codes defined by the online optimizer protocol. */
public enum OptimizerErrorCode {
    UNSPECIFIED,
    OK,
    UNSUPPORTED,
    INTERNAL_ERROR,
    SERVICE_NOT_READY,
    INVALID_ARGUMENT,
    DUPLICATE_ENTITY,
    REACH_MAX_ENTITY_CAPACITY,
    INSTANCE_NOT_EXIST,
    SERVER_NOT_LEADER,
    IO_ERROR,
    UNKNOWN_ERROR,
    ERROR_MAX;

    @JsonCreator
    public static OptimizerErrorCode fromValue(String value) {
        if (value == null) {
            return UNKNOWN_ERROR;
        }
        for (OptimizerErrorCode errorCode : values()) {
            if (errorCode.name().equalsIgnoreCase(value)) {
                return errorCode;
            }
        }
        return UNKNOWN_ERROR;
    }
}
