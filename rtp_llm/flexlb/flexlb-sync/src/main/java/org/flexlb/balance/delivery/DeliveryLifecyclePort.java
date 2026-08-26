package org.flexlb.balance.delivery;

/**
 * Scheduler-owned lifecycle boundary consumed by one delivery queue.
 *
 * <p>Every item is the exact canonical identity supplied to delivery. The
 * implementation may validate its concrete scheduler type at this boundary;
 * endpoint and delivery packages never depend on that concrete owner.
 */
public interface DeliveryLifecyclePort {

    /** Reduce an exact queued item whose absolute deadline has elapsed. */
    void onExpired(DeliveryItem exactItem);

    /** Resolve one canonical committed delivery through its sole capability. */
    void onDeliveryCommitted(
            CommittedDelivery delivery,
            DeliveryMetadata metadata);

    /** Reduce an exact item which could not enter the scheduling queue. */
    void onOfferFailure(DeliveryItem exactItem, Throwable cause);

    /** Terminally reduce an admitted item whose delivery could not complete. */
    void onDeliveryFailure(DeliveryItem exactItem, Throwable cause);
}
