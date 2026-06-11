package org.flexlb.dao.optimizer;

/** Error codes defined by the latest online optimizer protobuf contract. */
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
    ERROR_MAX
}
