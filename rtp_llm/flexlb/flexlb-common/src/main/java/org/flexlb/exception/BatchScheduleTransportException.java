package org.flexlb.exception;

import lombok.Getter;

/**
 * Signals that the master node could not be reached when resolving a
 * {@code /batch_schedule} request. Carries the same monitoring code tags
 * ("MASTER_NULL", "TIMEOUT", "CONNECT_FAILED") used by
 * {@code EngineHealthReporter.reportForwardToMasterResult}.
 */
public class BatchScheduleTransportException extends RuntimeException {

    @Getter
    private final String errorCode;

    public BatchScheduleTransportException(String message, String errorCode) {
        super(message);
        this.errorCode = errorCode;
    }
}
