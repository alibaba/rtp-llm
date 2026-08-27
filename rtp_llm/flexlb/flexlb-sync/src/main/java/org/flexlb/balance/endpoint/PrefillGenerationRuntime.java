package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.DeliveryLifecyclePort;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FlexlbConfig;

import java.util.List;
import java.util.Map;

/**
 * Endpoint-owned runtime for one Prefill worker generation.
 *
 * <p>The endpoint owns this interface and its {@link PrefillWorkLedger}; the
 * scheduling implementation remains private to composition. Queue mutations
 * accept only canonical {@link DeliveryItem} identities, so neither scheduler
 * request state nor queue implementation types cross this boundary.</p>
 */
public interface PrefillGenerationRuntime {

    /**
     * Exact, one-shot hold on this generation's queue and pending seats.
     * Closing an uncommitted hold releases only that hold.
     */
    interface PreparedOffer extends AutoCloseable {

        /**
         * Make this hold non-revocable before an irreversible cross-role step.
         * Repeated calls return {@code true} while the live hold remains sealed;
         * {@code false} means it was revoked or has already been consumed.
         */
        boolean seal();

        /** Convert this hold into the exact canonical ACTIVE item. */
        void commit(DeliveryItem exactItem);

        @Override
        void close();
    }

    /** Sole construction boundary for a generation runtime. */
    @FunctionalInterface
    interface Factory {
        /**
         * Construct but do not start or call back through {@code endpoint}.
         * The endpoint is still inside its constructor and publishes the
         * returned pair only after both final ownership fields are assigned.
         */
        Generation create(
                String endpointId,
                PrefillEndpoint endpoint,
                FlexlbConfig config,
                DeliveryStrategy deliveryStrategy,
                DeliveryLifecyclePort deliveryLifecycle);
    }

    /** Exact construction transfer: runtime and its canonical ledger. */
    record Generation(
            PrefillGenerationRuntime runtime,
            PrefillWorkLedger ledger) {
    }

    /**
     * Immutable queue state materialized under the runtime's queue lock.
     * {@code waitingCount} includes ACTIVE items and live prepared holds;
     * {@code items} contains only canonical ACTIVE identities.
     */
    record QueueSnapshot(
            String endpointId,
            long queueVersion,
            int queueCapacity,
            long waitingCount,
            long pendingCount,
            long maxPendingRequests,
            List<DeliveryItem> items) {
        public QueueSnapshot {
            if (queueVersion < 0L) {
                throw new IllegalArgumentException(
                        "queueVersion must be non-negative");
            }
            if (queueCapacity < 0) {
                throw new IllegalArgumentException(
                        "queueCapacity must be non-negative");
            }
            if (waitingCount < 0L
                    || pendingCount < 0L
                    || maxPendingRequests <= 0L) {
                throw new IllegalArgumentException(
                        "waiting/pending counts must be non-negative with a positive maximum");
            }
            items = List.copyOf(items);
            if (waitingCount < items.size()
                    || pendingCount < waitingCount) {
                throw new IllegalArgumentException(
                        "waiting must cover ACTIVE items and pending must cover waiting");
            }
        }

    }

    enum QueueReplacementStatus {
        SUCCESS,
        CONFLICT,
        DECLINED
    }

    /** Result of one atomic exact-victim replacement. */
    record QueueReplacement(QueueReplacementStatus status) {
    }

    /** Start this exact generation once, after endpoint construction. */
    void start();

    /**
     * Stop new queue work, drain all ACTIVE items, and return only after the
     * worker thread has exited. Repeated calls observe the same completed
     * generation cleanup.
     */
    Throwable stopAndAwait();

    /** Publish one exact item after the endpoint facade validates its pin. */
    boolean offer(DeliveryItem exactItem);

    /**
     * Hold one queue/pending seat before cross-role admission. A {@code null}
     * result means capacity is temporarily full and no lower-priority OPEN
     * hold is eligible for replacement.
     */
    PreparedOffer prepareOffer(long requestId, int priority);

    /** Stable generation-local wake source for queue/pending offer capacity. */
    CapacityBoundary.Availability offerAvailability();

    /** Monotonic generation-local epoch advanced whenever an offer seat frees. */
    long offerCapacityEpoch();

    /** Remove only the supplied canonical ACTIVE identity. */
    boolean removeQueued(DeliveryItem exactItem, String reason);

    /** Replace exact queued victims with one incoming canonical identity. */
    QueueReplacement replaceQueued(
            List<DeliveryItem> exactVictims,
            DeliveryItem incoming);

    QueueSnapshot captureQueueSnapshot();

    int queueSize();

    /** Canonical pending count, including uncommitted prepared offers. */
    long pendingRequestCount();

    Map<Integer, Integer> queueSizeByPriority();

    RouteProjection.Inputs captureRouteProjectionInputs();

    RouteProjection.DeliveryProjection deliveryProjection();

    void signalSchedulingInputsChanged();
}
