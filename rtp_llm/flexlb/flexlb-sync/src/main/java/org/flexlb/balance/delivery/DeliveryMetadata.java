package org.flexlb.balance.delivery;

/** Immutable metadata captured after one exact delivery selection commits. */
public record DeliveryMetadata(
        String decisionReason,
        int remainingQueueDepth) {

    public DeliveryMetadata {
        if (remainingQueueDepth < 0) {
            throw new IllegalArgumentException(
                    "remainingQueueDepth must be non-negative");
        }
    }
}
