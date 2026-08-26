package org.flexlb.balance.delivery;

import java.util.List;
import java.util.Objects;

/**
 * Narrow worker capability used by a delivery strategy. It deliberately does
 * not expose BatcherContext, its queue, lock, Registry, or callback handler.
 */
public interface DeliveryContext<R> {

    /** Canonical scheduler-loop result for an unchanged selection. */
    R noAction();

    /** Advisory fast rejection; commitSelection performs the authoritative check. */
    boolean selectionStillOwned(List<DeliveryItem> candidates);

    /**
     * Sole ACTIVE-prefix transaction. It validates exact identity and expiry,
     * invokes commit while holding the queue lock, consumes one terminal
     * boundary, updates the queue generation, and emits terminal failure once.
     */
    SelectionCommit<R> commitSelection(
            CanonicalCommit capability,
            SelectionBoundary boundary,
            String reason);

    /** Resolve a typed head boundary when no request acquired capacity. */
    R commitBoundary(SelectionBoundary boundary);

    void publishCommittedDelivery(
            CommittedDelivery delivery,
            DeliveryMetadata metadata);

    /**
     * Invocation-local strategy capability that mutates ACTIVE ownership.
     * Implementations privately capture their exact Registry commit port before
     * queue mutation; adding another delivery mode does not change this API.
     */
    interface CanonicalCommit {

        List<DeliveryItem> items();

        CommittedDelivery commitUnderLock();
    }

    /** The first candidate not covered by the admitted prefix, if any. */
    record SelectionBoundary(
            DeliveryItem item,
            CapacityBoundary result) {

        public SelectionBoundary {
            Objects.requireNonNull(item, "item");
            Objects.requireNonNull(result, "result");
        }
    }

    /**
     * Invocation-local control result for one queue ownership transaction.
     * The loop result is constructed by the scheduler context and is never
     * translated into a second delivery-owned request state.
     */
    sealed interface SelectionCommit<R>
            permits SelectionCommit.Committed, SelectionCommit.NotCommitted {

        R loopResult();

        record Committed<R>(
                CommittedDelivery owner,
                R loopResult)
                implements SelectionCommit<R> {

            public Committed {
                Objects.requireNonNull(owner, "owner");
                Objects.requireNonNull(loopResult, "loopResult");
            }
        }

        record NotCommitted<R>(R loopResult)
                implements SelectionCommit<R> {

            public NotCommitted {
                Objects.requireNonNull(loopResult, "loopResult");
            }
        }
    }
}
