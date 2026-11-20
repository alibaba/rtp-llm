package org.flexlb.cache.kvcm;

/**
 * Reports a KVCM cache lookup failure to its source orchestrator.
 */
public class KvcmQueryException extends RuntimeException {

    public KvcmQueryException(String message) {
        super(message);
    }

    public KvcmQueryException(String message, Throwable cause) {
        super(message, cause);
    }
}
