package org.flexlb.balance.delivery;

import org.flexlb.balance.projection.RouteProjection;


/**
 * Materialized result of crossing one delivery-admission boundary.
 *
 * <p>Concrete capacity kinds never cross an execution port. An unavailable
 * result already contains both its exact wake source and its immutable
 * projection meaning.
 */
public sealed interface CapacityBoundary permits CapacityBoundary.Unavailable,
        CapacityBoundary.OwnershipLost, CapacityBoundary.Failed {

    /** The request remains active and waits for this exact availability source. */
    record Unavailable(
            Availability availability,
            RouteProjection.AdmissionBlockSemantics projectionSemantics)
            implements CapacityBoundary {
    }

    /** Another exact lifecycle reducer already owns the request generation. */
    enum OwnershipLost implements CapacityBoundary {
        INSTANCE
    }

    /** Preparation failed and the exact request must be terminally reduced. */
    record Failed(Throwable cause) implements CapacityBoundary {
    }

    /**
     * Non-blocking wake capability for the exact unavailable admission.
     * Implementations must not acquire endpoint mutation locks while queried.
     */
    interface Availability {

        boolean isAvailable();

        void addListener(Runnable listener);

        void removeListener(Runnable listener);
    }

    /** Either an accepted capability/value or one fully materialized boundary. */
    sealed interface Attempt<T>
            permits Attempt.Accepted, Attempt.Rejected {

        record Accepted<T>(T value) implements Attempt<T> {
        }

        record Rejected<T>(CapacityBoundary boundary) implements Attempt<T> {
        }
    }
}
