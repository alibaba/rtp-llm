package org.flexlb.balance.delivery;

/** Transport result for one exact delivery claim. */
public record DeliveryResult(Status status, Throwable cause) {

    public DeliveryResult {
        if (status == null) {
            throw new IllegalArgumentException("delivery status is required");
        }
        if ((status == Status.DELIVERED) == (cause != null)) {
            throw new IllegalArgumentException(
                    "only unsuccessful delivery requires a cause");
        }
    }

    public static DeliveryResult delivered() {
        return new DeliveryResult(Status.DELIVERED, null);
    }

    public static DeliveryResult notSent(Throwable cause) {
        return new DeliveryResult(Status.NOT_SENT, cause);
    }

    public static DeliveryResult prefillRejected(Throwable cause) {
        return new DeliveryResult(Status.PREFILL_REJECTED, cause);
    }

    public static DeliveryResult timedOut(Throwable cause) {
        return new DeliveryResult(Status.TIMED_OUT, cause);
    }

    public static DeliveryResult uncertain(Throwable cause) {
        return new DeliveryResult(Status.UNCERTAIN, cause);
    }

    public boolean failed() {
        return status() == Status.PREFILL_REJECTED || status() == Status.NOT_SENT;
    }

    public enum Status {
        DELIVERED,
        /** Local failure before RPC invocation; no remote work was started by this claim. */
        NOT_SENT,
        /** Final per-member EnqueueBatch error; Decode may already own resources. */
        PREFILL_REJECTED,
        TIMED_OUT,
        UNCERTAIN
    }
}
