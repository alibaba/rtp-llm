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

    public static DeliveryResult failed(Throwable cause) {
        return new DeliveryResult(Status.FAILED, cause);
    }

    public static DeliveryResult timedOut(Throwable cause) {
        return new DeliveryResult(Status.TIMED_OUT, cause);
    }

    public static DeliveryResult uncertain(Throwable cause) {
        return new DeliveryResult(Status.UNCERTAIN, cause);
    }

    public enum Status {
        DELIVERED,
        FAILED,
        TIMED_OUT,
        UNCERTAIN
    }
}
