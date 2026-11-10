package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.util.Failures;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Holds one selected batch and its temporary member permits until the dispatch handoff.
 * Endpoint commits queue membership; Scheduler records request delivery; Dispatcher sends it.
 * Closing this batch returns only its temporary permits, not request resource registrations.
 */
@lombok.Getter
@lombok.experimental.Accessors(fluent = true)
public final class Delivery implements AutoCloseable {
    long batchId;

    List<Member> members;
    WorkSnapshot precedingWork;

    long predictedMs;

    PrefillTimePredictor.PrefillBatchFeatures features;
    PrefillTimePredictor.Evaluator evaluator;
    String reason;

    int remainingQueueDepth;
    private boolean committed;
    private boolean removedFailedMember;
    long selectionTimeMs;

    RouteResult bypassedHead;
    DeliveryExecution.DeliveryPermit executionPermit;

    final PrefillEndpoint endpoint;

    final boolean batch;

    final long[] predictions;

    int maxInflightBatches;

    CapacityBoundary boundary;
    RouteResult blocked;

    public Delivery(
            PrefillEndpoint endpoint,
            boolean batch,
            int size,
            PrefillTimePredictor.Evaluator evaluator,
            String reason) {
        this.endpoint = endpoint;
        this.batch = batch;
        this.evaluator = evaluator;
        this.reason = reason;
        members = new ArrayList<>(size);
        predictions = batch ? null : new long[size];
    }

    /**
     * Endpoint calls this only after validating every member in its state lock.
     */
    public void committed(
            WorkSnapshot precedingWork, int remainingQueueDepth, boolean removedFailedMember) {
        checkState(!committed, "delivery already committed");
        this.precedingWork = precedingWork;
        this.remainingQueueDepth = remainingQueueDepth;
        this.removedFailedMember = removedFailedMember;
        committed = true;
    }

    public List<RouteResult> items() {
        return members.stream().map(Member::route).toList();
    }

    public RouteResult failedMember() {
        return boundary != null && boundary.status() == CapacityBoundary.Status.FAILED
                ? blocked
                : null;
    }

    @Override
    public void close() {
        Throwable failure = executionPermit == null ? null : Failures.close(executionPermit);
        failure = Failures.append(failure, releaseMembers());
        Failures.rethrow(failure, "temporary delivery resource cleanup failed");
    }

    Throwable releaseMembers() {
        Throwable failure = null;
        for (Member member : members) failure = Failures.run(failure, member::close);
        return failure;
    }

    /**
     * Identity of one request handoff, shared by Context and transport callbacks until cleanup.
     * The Decode permit is temporary; the completion signal lasts until this request is released.
     */
    public static final class Member implements AutoCloseable {
        @SuppressWarnings("rawtypes")
        private static final java.util.concurrent.atomic.AtomicReferenceFieldUpdater<
                        Delivery.Member, java.util.concurrent.CompletableFuture>
                COMPLETION =
                        java.util.concurrent.atomic.AtomicReferenceFieldUpdater.newUpdater(
                                Delivery.Member.class,
                                java.util.concurrent.CompletableFuture.class,
                                "completion");

        @lombok.Getter
        @lombok.experimental.Accessors(fluent = true)
        private final RouteResult route;

        @lombok.Getter
        @lombok.experimental.Accessors(fluent = true)
        private final DecodeEndpoint.EngineDispatchPermit decode;

        @lombok.Getter
        @lombok.experimental.Accessors(fluent = true)
        private Delivery.SendMode kind = Delivery.SendMode.NONE;

        @lombok.Getter
        @lombok.experimental.Accessors(fluent = true)
        private long batchId;

        private volatile java.util.concurrent.CompletableFuture<Void> completion;

        public Member(RouteResult route, DecodeEndpoint.EngineDispatchPermit decode) {
            this.route = Objects.requireNonNull(route);
            this.decode = decode;
        }

        public RequestContext owner() {
            return route.ctx();
        }

        @Override
        public void close() {
            if (decode != null) {
                decode.release();
            }
        }

        public java.util.concurrent.CompletionStage<Void> completion() {
            return completionFuture().minimalCompletionStage();
        }

        void initialize(Delivery.SendMode kind, long batchId) {
            checkState(this.kind == Delivery.SendMode.NONE, "delivery member already claimed");
            checkArgument(
                    kind != Delivery.SendMode.NONE
                            && (kind != Delivery.SendMode.BATCH_ENQUEUE || batchId > 0L),
                    "invalid delivery identity");
            this.kind = Objects.requireNonNull(kind);
            this.batchId = batchId;
        }

        void recordFailure(Throwable failure) {
            route.ctx().data().getRequestScheduler().taskPool.recordFailure(failure);
        }

        static CapacityBoundary.Attempt<Member> prepare(
                RouteResult route, AdmissionCapacity capacity) {
            var endpoint = route.decodeEndpoint();
            var reservation = route.decodeReservation();
            if (endpoint == null && reservation == null) {
                return CapacityBoundary.Attempt.accepted(new Member(route, null));
            }
            if (endpoint == null || reservation == null) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.failed(
                                new IllegalStateException(
                                        "Decode reservation is unavailable: request_id="
                                                + route.requestId())));
            }
            DecodeEndpoint.EngineDispatchPermit permit = null;
            try {
                var acquisition = endpoint.acquireDispatchPermit(reservation, capacity);
                switch (acquisition.status()) {
                    case ACQUIRED, ALREADY_ACCEPTED -> permit = acquisition.value();
                    case CAPACITY_FULL -> {
                        return CapacityBoundary.Attempt.rejected(
                                CapacityBoundary.unavailable(
                                        CapacityBoundary.Resource.DECODE_ENGINE));
                    }
                    case NOT_OWNED, NOT_QUEUED, ENDPOINT_RETIRED -> {
                        return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
                    }
                    case ALREADY_ACQUIRED -> {
                        return CapacityBoundary.Attempt.rejected(
                                CapacityBoundary.failed(
                                        new IllegalStateException(
                                                "Decode dispatch permit already acquired: request_id="
                                                        + route.requestId())));
                    }
                }
                return CapacityBoundary.Attempt.accepted(new Member(route, permit));
            } catch (Throwable failure) {
                if (permit != null) {
                    failure = Failures.run(failure, permit::release);
                }
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(failure));
            }
        }

        void complete() {
            completionFuture().complete(null);
        }

        private java.util.concurrent.CompletableFuture<Void> completionFuture() {
            var current = completion;
            if (current != null) return current;
            var created = new java.util.concurrent.CompletableFuture<Void>();
            return COMPLETION.compareAndSet(this, null, created) ? created : completion;
        }
    }

    /**
     * Transport result for one exact delivery claim.
     */
    public record Result(Status status, Throwable cause) {

        public Result {
            checkArgument(status != null, "delivery status is required");
            checkArgument(
                    (status == Status.DELIVERED) != (cause != null),
                    "only unsuccessful delivery requires a cause");
        }

        public static Result delivered() {
            return new Result(Status.DELIVERED, null);
        }

        public static Result notSent(Throwable cause) {
            return new Result(Status.NOT_SENT, cause);
        }

        public static Result prefillRejected(Throwable cause) {
            return new Result(Status.PREFILL_REJECTED, cause);
        }

        public static Result uncertain(Throwable cause) {
            return new Result(Status.UNCERTAIN, cause);
        }

        public boolean failed() {
            return status() == Status.PREFILL_REJECTED || status() == Status.NOT_SENT;
        }

        public enum Status {
            DELIVERED,
            /**
             * Local failure before RPC invocation; no remote work was started by this claim.
             */
            NOT_SENT,
            /**
             * Final per-member EnqueueBatch error; Decode may already own resources.
             */
            PREFILL_REJECTED,
            UNCERTAIN
        }
    }

    /**
     * Whether unavailable capacity may shorten an already ordered selection.
     */
    public enum PrefixPolicy {
        ALL_SELECTED,
        ORDERED_PREFIX
    }

    /**
     * Delivery mode, also used by request diagnostics before a delivery is claimed.
     */
    public enum SendMode {
        /**
         * The request is still wholly owned by the scheduler.
         */
        NONE,

        /**
         * An EnqueueBatch operation may have made the request visible to the engine.
         */
        BATCH_ENQUEUE,

        /**
         * The route decision may have been returned to the caller.
         */
        ROUTE_DECISION
    }
}
