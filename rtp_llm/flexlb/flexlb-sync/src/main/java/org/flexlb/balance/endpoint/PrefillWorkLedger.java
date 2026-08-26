package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.dao.master.WorkerStatus;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.Function;
import java.util.function.LongPredicate;

/**
 * Canonical Prefill ownership ledger for one endpoint generation.
 *
 * <p>The endpoint keeps this port private. Admission code receives only the
 * exact opaque capabilities returned by endpoint methods; it cannot inspect
 * registry state, lease state, queue storage, or cleanup resources.
 */
public interface PrefillWorkLedger {

    /** Exact endpoint-generation handoff moved into committed ownership. */
    @FunctionalInterface
    interface GenerationHandoff {
        void close();
    }

    /** Provisional capacity ownership; closing rolls back only while open. */
    interface Reservation extends AutoCloseable {
        @Override
        void close();
    }

    /** One exact route reservation within a prepared group. */
    interface RouteReservation extends Reservation {
        List<CommittedHandoff> commitGroupUnderLock(
                List<DeliveryItem> exactItems,
                List<RouteReservation> exactReservations);
    }

    /** One exact batch reservation anchored by its prepared head. */
    interface BatchReservation extends Reservation {
        long batchId();

        CommittedHandoff commitUnderLock(
                List<DeliveryItem> exactItems,
                long predictedMs);
    }

    /** Idempotent post-commit endpoint-generation handoff. */
    interface CommittedHandoff extends AutoCloseable {
        @Override
        void close();
    }

    /** Thread-confined rollback capability for one provisional DIRECT owner. */
    interface DirectRegistration extends AutoCloseable {
        void commit();

        @Override
        void close();
    }

    /** Opaque exact guard for one committed request generation. */
    interface Protection {
    }

    enum CapacityStatus {
        ACQUIRED,
        CAPACITY_FULL,
        REQUEST_NOT_ACTIVE,
        REQUEST_ALREADY_RESERVED,
        BATCH_ID_ALREADY_RESERVED,
        ENDPOINT_RETIRED
    }

    record RouteReservationResult(
            CapacityStatus status,
            RouteReservation reservation) {
        public RouteReservationResult {
            Objects.requireNonNull(status, "status");
            if ((status == CapacityStatus.ACQUIRED)
                    != (reservation != null)) {
                throw new IllegalArgumentException(
                        "only ACQUIRED may carry a route reservation");
            }
        }
    }

    record BatchReservationResult(
            CapacityStatus status,
            BatchReservation reservation) {
        public BatchReservationResult {
            Objects.requireNonNull(status, "status");
            if ((status == CapacityStatus.ACQUIRED)
                    != (reservation != null)) {
                throw new IllegalArgumentException(
                        "only ACQUIRED may carry a batch reservation");
            }
        }
    }

    sealed interface WorkerStatusFact
            permits ActiveWorkerStatusFact, TerminalWorkerStatusFact {
        DeliveryItem item();
    }

    record ActiveWorkerStatusFact(DeliveryItem item)
            implements WorkerStatusFact {
        public ActiveWorkerStatusFact {
            Objects.requireNonNull(item, "item");
        }
    }

    enum TerminalFactKind {
        COMPLETED,
        FAILED,
        PRIORITY_CANCELED
    }

    record TerminalWorkerStatusFact(
            DeliveryItem item,
            TerminalFactKind kind,
            long errorCode) implements WorkerStatusFact {
        public TerminalWorkerStatusFact {
            Objects.requireNonNull(item, "item");
            Objects.requireNonNull(kind, "kind");
        }
    }

    /** Immutable materialized outcome of one canonical status reduction. */
    interface StatusReconciliation {
        List<WorkerStatusFact> schedulerFacts();

        List<BatchCompletion> batchCompletions();

        /** Callback failure captured after the canonical facts were reduced. */
        Throwable publicationFailure();
    }

    record BatchCompletion(
            long batchId,
            PrefillBatchFeatures originalFeatures,
            long predictedWorkMs,
            long actualWorkMs,
            boolean successfulCompletion,
            boolean learningEligible) {
        public BatchCompletion {
            Objects.requireNonNull(originalFeatures, "originalFeatures");
        }
    }

    record Retirement(
            List<DeliveryItem> ownedItems,
            List<BatchCompletion> batchCompletions,
            Throwable invariantFailure) {
        public Retirement {
            ownedItems = List.copyOf(ownedItems);
            batchCompletions = List.copyOf(batchCompletions);
        }
    }

    record Stats(
            int locallyOwnedRequests,
            int individuallyOwnedRequests,
            int batchCount,
            long maxObservedAgeMs) {
        public Stats {
            if (locallyOwnedRequests < 0 || individuallyOwnedRequests < 0
                    || batchCount < 0 || maxObservedAgeMs < 0L) {
                throw new IllegalArgumentException(
                        "Prefill ledger stats must be non-negative");
            }
        }
    }

    RouteReservationResult reserveRoute(
            DeliveryItem exactItem,
            long predictedMs,
            int maximum,
            GenerationHandoff generationHandoff);

    BatchReservationResult reserveBatch(
            DeliveryItem exactHead,
            long batchId,
            int maximum,
            GenerationHandoff generationHandoff);

    CapacityBoundary.Availability routeAvailability(int maximum);

    CapacityBoundary.Availability batchAvailability(int maximum);

    /** Returns {@code null} when the request id already has a live owner. */
    DirectRegistration tryRegisterDirect(long requestId, long predictedMs);

    boolean terminalizeCommittedItem(DeliveryItem exactItem);

    /** Returns {@code null} when no exact individual owner can be protected. */
    Protection tryAcquireProtection(DeliveryItem exactItem);

    /** Returns {@code null} when the exact batch member cannot be protected. */
    Protection tryAcquireBatchProtection(
            long batchId,
            DeliveryItem exactItem);

    List<BatchCompletion> releaseProtection(
            Protection exactProtection,
            Function<List<DeliveryItem>, OptionalLong> repredictor);

    StatusReconciliation reconcileWorkerStatus(
            WorkerStatus.StatusObservation observation,
            Function<List<DeliveryItem>, OptionalLong> repredictor,
            Runnable committedPublication,
            Runnable failedReduction);

    Retirement retireGenerationOwnership();

    int evictExpiredIndividuals(
            long ttlMs,
            LongPredicate schedulerOwnsRequest);

    int evictExpiredBatches(
            long ttlMs,
            LongPredicate schedulerOwnsRequest);

    Stats stats();

    long pendingRequestCount();

    WorkSnapshot committedSnapshot();
}
