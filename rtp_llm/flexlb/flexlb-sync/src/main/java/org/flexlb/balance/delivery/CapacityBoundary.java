package org.flexlb.balance.delivery;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Materialized result of crossing one delivery-admission boundary.
 * The boundary carries only the resource fact; queue projection interprets it.
 */
public record CapacityBoundary(Status status, Resource resource, Throwable cause) {

    public static final CapacityBoundary OWNERSHIP_LOST =
            new CapacityBoundary(Status.OWNERSHIP_LOST, null, null);

    public CapacityBoundary {
        Objects.requireNonNull(status, "status");
        boolean unavailable = status == Status.UNAVAILABLE;
        boolean failed = status == Status.FAILED;
        checkArgument(
                unavailable == (resource != null)
                        && failed == (cause != null)
                        && (unavailable || resource == null),
                "capacity boundary status requires its exact payload");
    }

    public static CapacityBoundary unavailable(Resource resource) {
        return new CapacityBoundary(Status.UNAVAILABLE, Objects.requireNonNull(resource), null);
    }

    public static CapacityBoundary failed(Throwable cause) {
        return new CapacityBoundary(Status.FAILED, null, Objects.requireNonNull(cause, "cause"));
    }

    public boolean unavailable() {
        return status == Status.UNAVAILABLE;
    }

    public enum Status {
        UNAVAILABLE,
        OWNERSHIP_LOST,
        FAILED
    }

    /** Shared capacity failure descriptions for direct admission and queued delivery. */
    @lombok.Getter
    @lombok.RequiredArgsConstructor
    public enum Resource {
        DISPATCH_EXECUTOR("DELIVERY_CAPACITY_BATCH_ADMISSION"),
        PREFILL_BATCH("Prefill batch capacity exhausted"),
        PREFILL_REQUEST("Prefill request capacity exhausted"),
        DECODE_ENGINE("DELIVERY_CAPACITY_DECODE_ENGINE");

        private final String description;
    }

    /** Either an accepted capability/value or one fully materialized boundary. */
    public record Attempt<T>(T value, CapacityBoundary boundary) {

        public Attempt {
            checkArgument(
                    (value == null) != (boundary == null),
                    "capacity attempt requires exactly one value or boundary");
        }

        public static <T> Attempt<T> accepted(T value) {
            return new Attempt<>(Objects.requireNonNull(value, "value"), null);
        }

        public static <T> Attempt<T> rejected(CapacityBoundary boundary) {
            return new Attempt<>(null, Objects.requireNonNull(boundary, "boundary"));
        }

        public boolean accepted() {
            return value != null;
        }
    }
}
