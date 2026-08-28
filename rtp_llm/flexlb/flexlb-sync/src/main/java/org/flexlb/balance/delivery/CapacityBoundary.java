package org.flexlb.balance.delivery;

import org.flexlb.balance.projection.RouteProjection;

import java.util.Objects;

/** Materialized result of crossing one delivery-admission boundary. */
public record CapacityBoundary(
        Status status,
        Availability availability,
        RouteProjection.AdmissionBlockSemantics projectionSemantics,
        Throwable cause) {

    public static final CapacityBoundary OWNERSHIP_LOST =
            new CapacityBoundary(Status.OWNERSHIP_LOST, null, null, null);

    public CapacityBoundary {
        Objects.requireNonNull(status, "status");
        boolean unavailable = status == Status.UNAVAILABLE;
        boolean failed = status == Status.FAILED;
        if (unavailable != (availability != null && projectionSemantics != null)
                || failed != (cause != null)
                || (!unavailable
                        && (availability != null || projectionSemantics != null))) {
            throw new IllegalArgumentException(
                    "capacity boundary status requires its exact payload");
        }
    }

    public static CapacityBoundary unavailable(
            Availability availability,
            RouteProjection.AdmissionBlockSemantics projectionSemantics) {
        return new CapacityBoundary(Status.UNAVAILABLE,
                Objects.requireNonNull(availability, "availability"),
                Objects.requireNonNull(projectionSemantics,
                        "projectionSemantics"),
                null);
    }

    public static CapacityBoundary failed(Throwable cause) {
        return new CapacityBoundary(Status.FAILED, null, null,
                Objects.requireNonNull(cause, "cause"));
    }

    public boolean unavailable() {
        return status == Status.UNAVAILABLE;
    }

    public enum Status {
        UNAVAILABLE,
        OWNERSHIP_LOST,
        FAILED
    }

    /**
     * Non-blocking wake capability for the exact unavailable admission.
     * Implementations must not acquire endpoint mutation locks while queried.
     */
    public interface Availability {

        boolean isAvailable();

        void addListener(Runnable listener);

        void removeListener(Runnable listener);
    }

    /** Either an accepted capability/value or one fully materialized boundary. */
    public record Attempt<T>(T value, CapacityBoundary boundary) {

        public Attempt {
            if ((value == null) == (boundary == null)) {
                throw new IllegalArgumentException(
                        "capacity attempt requires exactly one value or boundary");
            }
        }

        public static <T> Attempt<T> accepted(T value) {
            return new Attempt<>(Objects.requireNonNull(value, "value"), null);
        }

        public static <T> Attempt<T> rejected(CapacityBoundary boundary) {
            return new Attempt<>(null,
                    Objects.requireNonNull(boundary, "boundary"));
        }

        public boolean accepted() {
            return value != null;
        }
    }
}
