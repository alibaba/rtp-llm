package org.flexlb.balance.delivery;

/** A definite negative acknowledgement for one exact EnqueueBatch member. */
public final class DeliveryRejection extends RuntimeException {

    private static final long RETRYABLE_INTERNAL_ERROR_CODE = 13L;

    private final String requestId;
    private final long errorCode;
    private final boolean retryDeadlineExceeded;

    public DeliveryRejection(
            String requestId,
            long errorCode,
            String errorMessage) {
        this(requestId, errorCode, errorMessage, false, null);
    }

    private DeliveryRejection(
            String requestId,
            long errorCode,
            String errorMessage,
            boolean retryDeadlineExceeded,
            DeliveryRejection prior) {
        super(message(
                requestId,
                errorCode,
                errorMessage,
                retryDeadlineExceeded), prior);
        if (requestId == null || requestId.isBlank()) {
            throw new IllegalArgumentException("requestId is required");
        }
        this.requestId = requestId;
        this.errorCode = errorCode;
        this.retryDeadlineExceeded = retryDeadlineExceeded;
    }

    public String requestId() {
        return requestId;
    }

    public long errorCode() {
        return errorCode;
    }

    /** Only an explicit member ACK with code 13 is safe to resubmit. */
    public boolean retryable() {
        return errorCode == RETRYABLE_INTERNAL_ERROR_CODE;
    }

    public boolean retryDeadlineExceeded() {
        return retryDeadlineExceeded;
    }

    public DeliveryRejection atRetryDeadline() {
        if (!retryable()) {
            throw new IllegalStateException(
                    "only a retryable acknowledgement can reach the retry deadline");
        }
        return retryDeadlineExceeded
                ? this
                : new DeliveryRejection(
                        requestId, errorCode, null, true, this);
    }

    private static String message(
            String requestId,
            long errorCode,
            String errorMessage,
            boolean retryDeadlineExceeded) {
        if (retryDeadlineExceeded) {
            return "EnqueueBatch retry deadline exceeded for request "
                    + requestId;
        }
        return "EnqueueBatch rejected request " + requestId
                + " error_code=" + errorCode + ": "
                + (errorMessage == null ? "missing error_info" : errorMessage);
    }
}
