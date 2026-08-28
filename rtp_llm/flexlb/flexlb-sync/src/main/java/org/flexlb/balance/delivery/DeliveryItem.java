package org.flexlb.balance.delivery;

/**
 * Canonical request identity as observed by delivery.
 *
 * <p>The scheduler-owned request object implements this interface directly.
 * Implementations must not create a second request owner or copy lifecycle
 * state into a delivery DTO.
 */
public interface DeliveryItem {

    long requestId();

    int priority();

    long enqueuedAtMs();

    long seqLen();

    long hitCache();
}
