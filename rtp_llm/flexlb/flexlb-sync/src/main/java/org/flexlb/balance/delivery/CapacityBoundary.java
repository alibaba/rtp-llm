package org.flexlb.balance.delivery;

import org.flexlb.balance.projection.RouteProjection;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Materialized result of crossing one delivery-admission boundary.
 * A delivery wait may carry additional route-publication restrictions;
 * absent projection semantics leave routing forecasts unchanged.
 */
public record CapacityBoundary(
        Status status,
        Resource resource,
        RouteProjection.AdmissionBlockSemantics projectionSemantics,
        Throwable cause) {

    public static final CapacityBoundary OWNERSHIP_LOST =
            new CapacityBoundary(Status.OWNERSHIP_LOST, null, null, null);

    public CapacityBoundary {
        Objects.requireNonNull(status, "status");
        boolean unavailable = status == Status.UNAVAILABLE;
        boolean failed = status == Status.FAILED;
        checkArgument(unavailable == (resource != null)
                && failed == (cause != null)
                && (unavailable
                || projectionSemantics == null),
                "capacity boundary status requires its exact payload");
    }

    public static CapacityBoundary unavailable(
            Resource resource,
            RouteProjection.AdmissionBlockSemantics projectionSemantics) {
        return new CapacityBoundary(Status.UNAVAILABLE,
                Objects.requireNonNull(resource, "resource"),
                Objects.requireNonNull(projectionSemantics,
                        "projectionSemantics"),
                null);
    }

    /** Park delivery without restricting publication into the worker queue. */
    public static CapacityBoundary deliveryUnavailable(Resource resource) {
        return new CapacityBoundary(Status.UNAVAILABLE,
                Objects.requireNonNull(resource, "resource"),
                null, null);
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

    /** Identifies the exhausted resource; waiting belongs to WorkerBatcher and notification to the capacity owner. */
    public enum Resource {
        DISPATCH_EXECUTOR,
        PREFILL_BATCH,
        DECODE_ENGINE
    }

    /** Either an accepted capability/value or one fully materialized boundary. */
    public record Attempt<T>(T value, CapacityBoundary boundary) {

        public Attempt {
            checkArgument((value == null) != (boundary == null),
                    "capacity attempt requires exactly one value or boundary");
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
