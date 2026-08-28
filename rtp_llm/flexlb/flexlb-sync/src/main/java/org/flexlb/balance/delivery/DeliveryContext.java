package org.flexlb.balance.delivery;

import java.util.List;

/**
 * Narrow worker capability used by a delivery strategy. It deliberately does
 * not expose BatcherContext, its queue, lock, Registry, or callback handler.
 */
public interface DeliveryContext<R> {

    /** Canonical scheduler-loop result for an unchanged selection. */
    R noAction();

    /**
     * Advisory fast rejection; {@link #commitPreparedSelection} performs the
     * authoritative identity and expiry check under the queue lock.
     */
    boolean selectionStillOwned(List<DeliveryItem> candidates);

    /**
     * Sole ACTIVE-prefix transaction. It validates exact identity and expiry,
     * invokes commit while holding the queue lock, consumes one terminal
     * boundary, updates the queue generation, and emits terminal failure once.
     */
    CommitResult<R> commitPreparedSelection(
            PreparedSelection selection,
            String decisionReason);

    /** Resolve a typed head boundary when no request acquired capacity. */
    R commitBoundary(SelectionBoundary boundary);

    void handoffCommittedDelivery(
            CommittedDelivery delivery,
            DeliveryMetadata metadata);

    /**
     * Invocation-local strategy capability that mutates ACTIVE ownership.
     * Implementations privately capture their exact Registry commit port before
     * queue mutation; adding another delivery mode does not change this API.
     */
    interface PreparedSelection {

        List<DeliveryItem> items();

        /** First candidate not covered by this prepared prefix, if any. */
        SelectionBoundary boundary();

        /** Move the prepared resources to one callback-owned capability. */
        CommittedDelivery commitOwnershipUnderLock();
    }

    /** The first candidate not covered by the admitted prefix, if any. */
    record SelectionBoundary(
            DeliveryItem item,
            CapacityBoundary result) {
    }

    /**
     * Invocation-local control result for one queue ownership transaction.
     * The loop result is constructed by the scheduler context and is never
     * translated into a second delivery-owned request state.
     */
    sealed interface CommitResult<R>
            permits CommitResult.Committed, CommitResult.NotCommitted {

        R loopResult();

        record Committed<R>(
                CommittedDelivery owner,
                R loopResult)
                implements CommitResult<R> {
        }

        record NotCommitted<R>(R loopResult)
                implements CommitResult<R> {
        }
    }
}
