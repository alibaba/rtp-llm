package org.flexlb.balance.scheduler;

import java.util.Objects;

/** Exact identity only. Transport and release facts are immutable Context snapshots. */
public record DeliveryClaim(RequestContext owner, long sequence, RequestRoute item,
                            DeliveryClaimKind kind, long batchId) implements ContextResult.GrantValue {
    public DeliveryClaim {
        Objects.requireNonNull(owner); Objects.requireNonNull(item); Objects.requireNonNull(kind);
        if (kind == DeliveryClaimKind.NONE || kind == DeliveryClaimKind.BATCH_ENQUEUE && batchId <= 0L) {
            throw new IllegalArgumentException("invalid delivery identity");
        }
    }
}
