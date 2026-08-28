package org.flexlb.balance.delivery;

import java.util.List;

/**
 * Mode-neutral owner produced by one canonical queue commit.
 *
 * <p>A delivery strategy supplies the implementation. The scheduler callback
 * only advances it through {@link #deliver(DeliveryMetadata)} or aborts an
 * unconsumed callback through {@link #failBeforeDelivery(Throwable)}; it never
 * inspects the strategy's endpoint, capacity, or transport resources.</p>
 */
public interface CommittedDelivery {

    /** Exact canonical request identities owned by this delivery. */
    List<DeliveryItem> items();

    /** Cross the strategy-specific transport boundary exactly once. */
    void deliver(DeliveryMetadata metadata);

    /** Settle a callback that failed before {@link #deliver} took ownership. */
    void failBeforeDelivery(Throwable cause);
}
