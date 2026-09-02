package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.Optional;
import java.util.OptionalLong;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/** Exact request-generation ownership required by delivery strategies. */
public interface SlotDeliveryPort {

    /** Execute preparation only while the exact item is still queue-owned. */
    <T> Optional<T> prepareIfOwned(
            ScheduledRequest exactItem,
            Supplier<T> preparation);

    /**
     * Atomically transfer endpoint admission and claim the slot's delivery
     * point-of-no-return.
     *
     * <p>The endpoint handoff is a local, synchronous leaf operation executed
     * while the exact request slot is locked. It may acquire endpoint-local
     * locks and publish capacity signals in their documented order, but must
     * not perform I/O, await external completion, or call back into a request
     * slot, delivery lifecycle, or user code.
     *
     * @return the exact claim, or {@code null} when another reducer already
     *         owns this request generation
     */
    Claim tryClaimForDelivery(
            ScheduledRequest exactItem,
            Identity identity,
            BooleanSupplier endpointHandoff);

    /**
     * Apply one terminal completion to the exact claim. An unknown, stale, or
     * already-completed claim is an illegal lifecycle transition and must
     * throw.
     */
    void complete(Claim exactClaim, Completion completion);

    /** Terminally reduce an exact prepared item which acquired no claim. */
    void failPrepared(ScheduledRequest exactItem, Throwable cause);

    /** Opaque proof of the exact slot point-of-no-return. */
    interface Claim {

        ScheduledRequest item();
    }

    /** Canonical delivery identity installed at the point-of-no-return. */
    record Identity(
            ConfirmationBoundary boundary,
            OptionalLong correlationId) {

        public Identity {
            if (correlationId.isPresent()
                    && correlationId.getAsLong() <= 0L) {
                throw new IllegalArgumentException(
                        "delivery correlation id must be positive");
            }
            if ((boundary == ConfirmationBoundary.EXTERNAL_ACK)
                    != correlationId.isPresent()) {
                throw new IllegalArgumentException(
                        "external acknowledgement requires one correlation id");
            }
        }

        public static Identity externalAcknowledgement(long correlationId) {
            return new Identity(
                    ConfirmationBoundary.EXTERNAL_ACK,
                    OptionalLong.of(correlationId));
        }

        public static Identity commitConfirmation() {
            return new Identity(
                    ConfirmationBoundary.COMMIT_CONFIRMED,
                    OptionalLong.empty());
        }

        public long requiredCorrelationId() {
            return correlationId.orElseThrow(() ->
                    new IllegalStateException(
                            "delivery identity has no correlation id"));
        }

        public enum ConfirmationBoundary {
            EXTERNAL_ACK,
            COMMIT_CONFIRMED
        }
    }

    /** Terminal transport result for one exact claim. */
    record Completion(Status status, Throwable cause) {

        public Completion {
            if (status == null) {
                throw new IllegalArgumentException("delivery status is required");
            }
            if ((status == Status.DELIVERED) == (cause != null)) {
                throw new IllegalArgumentException(
                        "only unsuccessful delivery requires a cause");
            }
        }

        public static Completion delivered() {
            return new Completion(Status.DELIVERED, null);
        }

        public static Completion failed(Throwable cause) {
            return new Completion(Status.FAILED, cause);
        }

        public static Completion timedOut(Throwable cause) {
            return new Completion(Status.TIMED_OUT, cause);
        }

        public static Completion uncertain(Throwable cause) {
            return new Completion(Status.UNCERTAIN, cause);
        }

        public enum Status {
            DELIVERED,
            FAILED,
            TIMED_OUT,
            UNCERTAIN
        }
    }

}
