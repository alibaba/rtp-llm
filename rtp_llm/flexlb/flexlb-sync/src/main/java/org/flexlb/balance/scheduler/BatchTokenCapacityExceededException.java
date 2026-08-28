package org.flexlb.balance.scheduler;

/**
 * The head request's padded compute shape can never fit one batch's hard
 * token capacity, so it must fail explicitly instead of being silently
 * dropped (design doc 8.3).
 *
 * <p>Extends {@link IllegalArgumentException} because an impossible padded
 * request shape is invalid scheduler input; the Auto-TPM path maps it to
 * {@code StrategyErrorType.BATCH_TOKEN_CAPACITY_EXCEEDED}.
 */
public class BatchTokenCapacityExceededException extends IllegalArgumentException {

    public BatchTokenCapacityExceededException(String message) {
        super(message);
    }
}
