package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.VictimResolution;

import java.util.Objects;

/** Exact victim and attempt identity, including its read-only completion. */
public final class PreemptionRegistration {
    @lombok.Getter
    @lombok.experimental.Accessors(fluent = true)
    private final RequestContext owner;

    @lombok.Getter
    @lombok.experimental.Accessors(fluent = true)
    private final long attemptToken;

    @lombok.Getter
    @lombok.experimental.Accessors(fluent = true)
    private final DecodeResources.ReservationHandle reservation;

    @lombok.Getter
    @lombok.experimental.Accessors(fluent = true)
    private final CancelTarget cancelTarget;

    /**
     * Intermediate Scheduler progress after one cancel reply was interpreted.
     *
     * <p>None of these values means that the request is closed or that the Decode ledger was
     * released. Those decisions require the matching Context and endpoint evidence.
     */
    public enum Reply {
        /** The reply was accepted as progress, but request and ledger settlement are pending. */
        WAITING,
        /** The remote worker did not own this work; this is not completion proof. */
        NOT_FOUND,
        /** The reply could not advance the matching attempt; this is not completion proof. */
        UNKNOWN
    }

    public PreemptionRegistration(
            RequestContext owner,
            long attemptToken,
            DecodeResources.ReservationHandle reservation,
            CancelTarget cancelTarget) {
        this.owner = owner;
        this.attemptToken = attemptToken;
        this.reservation = reservation;
        this.cancelTarget = cancelTarget;

        Objects.requireNonNull(owner);
        Objects.requireNonNull(reservation);
        Objects.requireNonNull(cancelTarget);
        if (!cancelTarget.isRoutable()) {
            throw new IllegalArgumentException("victim has no routable Cancel target");
        }
    }

    private volatile java.util.concurrent.CompletableFuture<VictimResolution> completion;

    @SuppressWarnings("rawtypes")
    private static final java.util.concurrent.atomic.AtomicReferenceFieldUpdater<
                    PreemptionRegistration, java.util.concurrent.CompletableFuture>
            COMPLETION =
                    java.util.concurrent.atomic.AtomicReferenceFieldUpdater.newUpdater(
                            PreemptionRegistration.class,
                            java.util.concurrent.CompletableFuture.class,
                            "completion");

    private java.util.concurrent.CompletableFuture<VictimResolution> completionFuture() {
        var current = completion;
        if (current != null) return current;
        var created = new java.util.concurrent.CompletableFuture<VictimResolution>();
        return COMPLETION.compareAndSet(this, null, created) ? created : completion;
    }

    public java.util.concurrent.CompletionStage<VictimResolution> completion() {
        return completionFuture().minimalCompletionStage();
    }

    public VictimResolution result() {
        var current = completion;
        return current == null ? null : current.getNow(null);
    }

    public void complete(VictimResolution result) {
        completionFuture().complete(Objects.requireNonNull(result));
    }
}
