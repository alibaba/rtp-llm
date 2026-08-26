package org.flexlb.balance.delivery;

import java.util.Objects;

/** Immutable metadata for one exact delivery decision group. */
public record DeliveryMetadata(String reason, int queueDepth) {

    public DeliveryMetadata {
        reason = Objects.requireNonNull(reason, "reason");
        if (queueDepth < 0) {
            throw new IllegalArgumentException("queueDepth must be non-negative");
        }
    }
}
