package org.flexlb.balance.delivery;

/** Immutable metadata for one exact delivery decision group. */
public record DeliveryMetadata(String reason, int queueDepth) {

    public DeliveryMetadata {
        if (queueDepth < 0) {
            throw new IllegalArgumentException("queueDepth must be non-negative");
        }
    }
}
