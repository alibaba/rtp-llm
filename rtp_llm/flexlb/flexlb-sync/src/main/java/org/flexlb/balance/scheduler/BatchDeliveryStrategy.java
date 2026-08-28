package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.BiConsumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import static org.flexlb.balance.scheduler.PrefillAdmissionResources.createCommittedOwner;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.missingEndpoint;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.prepareMember;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.preserveRejectedCause;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rejectedPrefill;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rollbackMember;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rollbackReservation;

/** EnqueueBatch admission, ownership, transport, telemetry, and projection. */
public final class BatchDeliveryStrategy implements DeliveryStrategy {

    private static final RouteProjection.DeliveryProjection PROJECTION =
            new BatchProjection();
    private static final RouteProjection.AdmissionBlockSemantics
            CAPACITY_BLOCK = new RouteProjection.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RouteProjection.AfterProbeAdmission.BLOCKED,
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RoleType.PREFILL);

    private final Supplier<CapacityBoundary.Attempt<PreparedSubmission>>
            prepareSubmission;
    private final LongSupplier batchIds;
    private final RequestRegistry requests;
    private final DeliveryMetrics telemetry;

    public BatchDeliveryStrategy(
            Supplier<CapacityBoundary.Attempt<PreparedSubmission>>
                    prepareSubmission,
            LongSupplier batchIds,
            RequestRegistry requests,
            DeliveryMetrics telemetry) {
        this.prepareSubmission = Objects.requireNonNull(
                prepareSubmission, "prepareSubmission");
        this.batchIds = Objects.requireNonNull(batchIds, "batchIds");
        this.requests = Objects.requireNonNull(requests, "requests");
        this.telemetry = Objects.requireNonNull(telemetry, "telemetry");
    }

    @Override
    public Transaction prepare(
            List<ScheduledRequest> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        if (candidates.isEmpty()) {
            throw new IllegalArgumentException(
                    "batch delivery requires at least one candidate");
        }
        return admit(candidates, evaluator, plannedPredictionMs);
    }

    private BatchTransaction admit(
            List<ScheduledRequest> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        ScheduledRequest head = candidates.get(0);

        CapacityBoundary.Attempt<BatchTransaction> groupAttempt =
                requests.prepareIfOwned(head, () -> prepareAdmission(head))
                        .orElseGet(() -> BatchDeliveryStrategy
                                .<BatchTransaction>ownershipLost());
        if (!groupAttempt.accepted()) {
            return BatchTransaction.blocked(
                    this,
                    head,
                    groupAttempt.boundary());
        }

        BatchTransaction transaction = groupAttempt.value();
        List<ScheduledRequest> admitted = new ArrayList<>(candidates.size());
        ScheduledRequest blockedItem = null;
        CapacityBoundary blockedResult = null;
        try {
            for (int index = 0; index < candidates.size(); index++) {
                ScheduledRequest item = candidates.get(index);
                CapacityBoundary.Attempt<ScheduledRequest> attempt =
                        requests.prepareIfOwned(
                                item, () -> transaction.append(item))
                                .orElseGet(() -> BatchDeliveryStrategy
                                        .<ScheduledRequest>ownershipLost());
                if (attempt.accepted()) {
                    admitted.add(item);
                } else {
                    blockedItem = item;
                    blockedResult = attempt.boundary();
                    break;
                }
            }
            if (admitted.isEmpty()) {
                transaction.select(
                        List.of(), 0L, null, blockedItem, blockedResult);
                return transaction;
            }
            long predictedMs = plannedPredictionMs.isPresent()
                    && sameIdentitySequence(admitted, candidates)
                    ? plannedPredictionMs.getAsLong()
                    : PrefillPredictionBoundary.predictCommittedBatchMs(
                            evaluator,
                            PrefillBatchFeatures.from(
                                    admitted,
                                    ScheduledRequest::seqLen,
                                    ScheduledRequest::hitCache));
            transaction.select(
                    admitted,
                    predictedMs,
                    evaluator,
                    blockedItem,
                    blockedResult);
            return transaction;
        } catch (Throwable failure) {
            Throwable cleanup = close(transaction);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
    }

    private CapacityBoundary.Attempt<BatchTransaction> prepareAdmission(
            ScheduledRequest head) {
        try {
            CapacityBoundary.Attempt<
                    PreparedSubmission> submissionAttempt =
                    Objects.requireNonNull(
                            prepareSubmission.get(), "submission attempt");
            if (!submissionAttempt.accepted()) {
                return rejected(submissionAttempt.boundary());
            }
            PreparedSubmission submission =
                    submissionAttempt.value();
            final long batchId;
            final PrefillEndpoint prefill;
            try {
                batchId = batchIds.getAsLong();
                if (batchId <= 0L) {
                    throw new IllegalStateException(
                            "batch id supplier returned a non-positive id");
                }
                prefill = head.prefillEp();
                if (prefill == null) {
                    throw missingEndpoint("Prefill", head);
                }
            } catch (Throwable failure) {
                Throwable cleanup = close(submission);
                if (cleanup != null) {
                    if (cleanup != failure) {
                        cleanup.addSuppressed(failure);
                    }
                    return failed(cleanup);
                }
                return failed(failure);
            }
            try {
                return accepted(new BatchTransaction(
                        this,
                        batchId,
                        submission,
                        prefill));
            } catch (Throwable failure) {
                Throwable cleanup = close(submission);
                if (cleanup != null && cleanup != failure) {
                    failure.addSuppressed(cleanup);
                }
                return failed(failure);
            }
        } catch (Throwable failure) {
            return failed(failure);
        }
    }

    @Override
    public double projectGroupDurationMs(
            List<ScheduledRequest> items,
            PrefillTimePredictor.Evaluator evaluator) {
        return PrefillPredictionBoundary.predictDecisionGroupMs(
                evaluator,
                PrefillBatchFeatures.from(
                        items,
                        ScheduledRequest::seqLen,
                        ScheduledRequest::hitCache));
    }

    @Override
    public RouteProjection.DeliveryProjection projectionPolicy() {
        return PROJECTION;
    }

    private void deliverCommitted(
            BatchTransaction batch,
            String decisionReason,
            int remainingQueueDepth) {
        List<ScheduledRequest> original = batch.items();
        List<ClaimedMember> claimed = new ArrayList<>(original.size());
        DispatchGate gate = null;
        Throwable handoffFailure = null;
        long deliveredPredictionMs = batch.predictedMs();
        try {
            for (int index = 0; index < original.size(); index++) {
                ScheduledRequest item = original.get(index);
                try {
                    RequestRegistry.DeliveryClaim claim =
                            requests.tryClaimBatchDelivery(
                                    item,
                                    batch.batchId(),
                                    () -> batch.transferToEndpoint(item));
                    if (claim == null) {
                        continue;
                    }
                    claimed.add(new ClaimedMember(item, claim));
                } catch (Throwable claimFailure) {
                    requests.failPrepared(item, claimFailure);
                }
            }

            if (!claimed.isEmpty()) {
                List<ScheduledRequest> submitted = claimed.stream()
                        .map(ClaimedMember::item)
                        .toList();
                if (submitted.size() != original.size()) {
                    deliveredPredictionMs =
                            PrefillPredictionBoundary.predictCommittedBatchMs(
                                    batch.evaluator(),
                                    PrefillBatchFeatures.from(
                                            submitted,
                                            ScheduledRequest::seqLen,
                                            ScheduledRequest::hitCache));
                }
                gate = new DispatchGate(
                        claimed, requests);
                batch.submit(
                        submitted,
                        deliveredPredictionMs,
                        decisionReason,
                        remainingQueueDepth,
                        gate);
                batch.transportAccepted();
            }
        } catch (Throwable failure) {
            handoffFailure = failure;
        } finally {
            try {
                batch.closeCapabilities();
            } catch (Throwable releaseFailure) {
                handoffFailure = append(handoffFailure, releaseFailure);
            }
            if (gate != null) {
                try {
                    gate.open();
                } catch (Throwable callbackFailure) {
                    handoffFailure = append(
                            handoffFailure, callbackFailure);
                }
            }
        }

        if (handoffFailure != null) {
            if (batch.transportOwned()) {
                throw propagate(handoffFailure);
            } else {
                Throwable completionFailure = null;
                for (ClaimedMember member : claimed) {
                    try {
                        requests.complete(
                                member.claim(),
                                DeliveryResult.failed(handoffFailure));
                    } catch (Throwable memberFailure) {
                        completionFailure = append(
                                completionFailure, memberFailure);
                    }
                }
                batch.transportFailed();
                if (completionFailure != null) {
                    throw propagate(append(
                            handoffFailure, completionFailure));
                }
            }
        }

        if (batch.transportOwned()) {
            telemetry.batchDispatched(
                    batch.batchId(),
                    decisionReason,
                    remainingQueueDepth,
                    claimed.stream().map(ClaimedMember::item).toList(),
                    deliveredPredictionMs);
        }
    }

    private Throwable failCommitted(
            BatchTransaction batch,
            Throwable cause) {
        Throwable cleanup = null;
        try {
            batch.closeSubmission();
        } catch (Throwable failure) {
            cleanup = append(cleanup, failure);
        }
        for (ScheduledRequest item : batch.items()) {
            try {
                requests.failPrepared(item, cause);
            } catch (Throwable failure) {
                cleanup = append(cleanup, failure);
            }
        }
        try {
            batch.closeAdmission();
        } catch (Throwable failure) {
            cleanup = append(cleanup, failure);
        }
        return cleanup;
    }

    private static boolean sameIdentitySequence(
            List<ScheduledRequest> left,
            List<ScheduledRequest> right) {
        if (left.size() != right.size()) {
            return false;
        }
        for (int index = 0; index < left.size(); index++) {
            if (left.get(index) != right.get(index)) {
                return false;
            }
        }
        return true;
    }

    private static <T> CapacityBoundary.Attempt<T> accepted(T value) {
        return CapacityBoundary.Attempt.accepted(value);
    }

    private static <T> CapacityBoundary.Attempt<T> rejected(
            CapacityBoundary boundary) {
        return CapacityBoundary.Attempt.rejected(boundary);
    }

    private static <T> CapacityBoundary.Attempt<T> ownershipLost() {
        return rejected(CapacityBoundary.OWNERSHIP_LOST);
    }

    private static <T> CapacityBoundary.Attempt<T> failed(Throwable cause) {
        return rejected(CapacityBoundary.failed(cause));
    }

    private static Throwable close(AutoCloseable capability) {
        try {
            capability.close();
            return null;
        } catch (Throwable failure) {
            return failure;
        }
    }

    private static Throwable append(Throwable first, Throwable next) {
        if (next == null) {
            return first;
        }
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException("batch delivery failed", failure);
    }

    /** One owner and one explicit state machine for the complete batch flow. */
    private static final class BatchTransaction implements Transaction {
        private enum Phase {
            PREPARED,
            COMMITTED,
            INFLIGHT,
            TERMINAL
        }

        private final BatchDeliveryStrategy owner;
        private final long batchId;
        private final PrefillEndpoint prefill;
        private PreparedSubmission submission;
        private PrefillState.BatchReservation reservation;
        private ArrayList<PrefillAdmissionResources.Member> members;
        private PrefillAdmissionResources.CommittedAdmissionOwner
                committedAdmission;
        private List<ScheduledRequest> items = List.of();
        private long predictedMs;
        private PrefillTimePredictor.Evaluator evaluator;
        private ScheduledRequest blockedItem;
        private CapacityBoundary blockedResult;
        private Phase phase;

        private BatchTransaction(
                BatchDeliveryStrategy owner,
                long batchId,
                PreparedSubmission submission,
                PrefillEndpoint prefill) {
            this.owner = owner;
            this.batchId = batchId;
            this.submission = submission;
            this.prefill = prefill;
            this.members = prefill == null ? null : new ArrayList<>(1);
            this.phase = Phase.PREPARED;
        }

        private static BatchTransaction blocked(
                BatchDeliveryStrategy owner,
                ScheduledRequest blockedItem,
                CapacityBoundary blockedResult) {
            BatchTransaction transaction = new BatchTransaction(
                    owner, 0L, null, null);
            transaction.blockedItem = blockedItem;
            transaction.blockedResult = blockedResult;
            transaction.phase = Phase.TERMINAL;
            return transaction;
        }

        private synchronized CapacityBoundary.Attempt<ScheduledRequest> append(
                ScheduledRequest exact) {
            requirePrepared("append");
            requireAdmissionPrepared("append");
            try {
                CapacityBoundary.Attempt<ScheduledRequest> acceptedItem =
                        accepted(exact);
                members.ensureCapacity(Math.addExact(members.size(), 1));
                boolean first = members.isEmpty();
                if (first) {
                    PrefillState.ReservationResult<
                            PrefillState.BatchReservation> result =
                            prefill.reserveBatch(
                                    exact,
                                    batchId,
                                    exact.maxInflightDeliveriesPerPrefillWorker());
                    if (result.status()
                            != PrefillState.CapacityStatus.ACQUIRED) {
                        return rejectedPrefill(
                                exact,
                                result.status(),
                                CapacityBoundary.unavailable(
                                        prefill.batchAdmissionAvailability(
                                                exact.maxInflightDeliveriesPerPrefillWorker()),
                                        CAPACITY_BLOCK));
                    }
                    reservation = result.reservation();
                }
                CapacityBoundary.Attempt<PrefillAdmissionResources.Member>
                        memberAttempt = prepareMember(exact);
                CapacityBoundary.Attempt<ScheduledRequest> result;
                if (memberAttempt.accepted()) {
                    members.add(memberAttempt.value());
                    result = acceptedItem;
                } else {
                    result = rejected(memberAttempt.boundary());
                }
                if (first && !result.accepted()) {
                    return abortAdmission(result.boundary());
                }
                return result;
            } catch (Throwable failure) {
                if (members.isEmpty()) {
                    return abortAdmission(failure);
                }
                return failed(failure);
            }
        }

        private <T> CapacityBoundary.Attempt<T> abortAdmission(
                CapacityBoundary boundary) {
            Throwable rollbackFailure = rollbackAdmission(null);
            if (rollbackFailure == null) {
                return rejected(boundary);
            }
            preserveRejectedCause(rollbackFailure, boundary);
            return failed(rollbackFailure);
        }

        private <T> CapacityBoundary.Attempt<T> abortAdmission(
                Throwable failure) {
            return failed(rollbackAdmission(failure));
        }

        private synchronized void select(
                List<ScheduledRequest> exactItems,
                long exactPredictionMs,
                PrefillTimePredictor.Evaluator exactEvaluator,
                ScheduledRequest exactBlockedItem,
                CapacityBoundary exactBlockedResult) {
            requirePrepared("select");
            items = List.copyOf(exactItems);
            predictedMs = exactPredictionMs;
            evaluator = exactEvaluator;
            blockedItem = exactBlockedItem;
            blockedResult = exactBlockedResult;
        }

        @Override
        public List<ScheduledRequest> items() {
            return items;
        }

        @Override
        public ScheduledRequest blockedItem() {
            return blockedItem;
        }

        @Override
        public CapacityBoundary blockedResult() {
            return blockedResult;
        }

        @Override
        public synchronized void commitUnderLock() {
            requirePrepared("commit");
            requireAdmissionPrepared("commit");
            if (items.isEmpty()) {
                throw new IllegalStateException(
                        "empty batch transaction cannot commit");
            }
            if (!PrefillAdmissionResources.sameIdentitySequence(
                    members, items)) {
                throw new IllegalArgumentException(
                        "batch commit does not match prepared identities");
            }
            if (reservation == null || members.isEmpty()) {
                throw new IllegalStateException(
                        "batch commit requires at least one prepared member");
            }
            PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted =
                    createCommittedOwner(members, 1);
            PrefillState.CommittedHandoff handoff =
                    reservation.commit(items, predictedMs);
            exactCommitted.bindBatchHandoff(handoff);
            committedAdmission = exactCommitted;
            reservation = null;
            members = null;
            phase = Phase.COMMITTED;
        }

        @Override
        public synchronized void handoff(
                String decisionReason, int remainingQueueDepth) {
            requirePhase(Phase.COMMITTED, "deliver");
            try {
                owner.deliverCommitted(
                        this, decisionReason, remainingQueueDepth);
            } catch (Throwable failure) {
                if (phase != Phase.COMMITTED) {
                    throw propagate(failure);
                }
                Throwable cleanup = owner.failCommitted(this, failure);
                if (cleanup != null && cleanup != failure) {
                    failure.addSuppressed(cleanup);
                }
                phase = Phase.TERMINAL;
                throw propagate(failure);
            }
            if (phase == Phase.COMMITTED) {
                phase = Phase.TERMINAL;
            }
        }

        @Override
        public synchronized void abort(Throwable cause) {
            if (phase != Phase.COMMITTED) {
                return;
            }
            Throwable cleanup = owner.failCommitted(this, cause);
            phase = Phase.TERMINAL;
            if (cleanup != null) {
                throw propagate(cleanup);
            }
        }

        @Override
        public synchronized void close() {
            if (phase != Phase.PREPARED) {
                return;
            }
            phase = Phase.TERMINAL;
            Throwable failure = rollbackAdmission(null);
            failure = BatchDeliveryStrategy.append(
                    failure, BatchDeliveryStrategy.close(submission));
            submission = null;
            if (failure != null) {
                throw propagate(failure);
            }
        }

        private long batchId() {
            return batchId;
        }

        private long predictedMs() {
            return predictedMs;
        }

        private PrefillTimePredictor.Evaluator evaluator() {
            return evaluator;
        }

        private boolean transferToEndpoint(ScheduledRequest exactItem) {
            requirePhase(Phase.COMMITTED, "transfer admission");
            return committedAdmission.transferToEndpoint(exactItem);
        }

        private void submit(
                List<ScheduledRequest> exactItems,
                long exactPredictionMs,
                String decisionReason,
                int remainingQueueDepth,
                BiConsumer<ScheduledRequest, DeliveryResult> observer) {
            requirePhase(Phase.COMMITTED, "submit");
            if (remainingQueueDepth < 0) {
                throw new IllegalArgumentException(
                        "remainingQueueDepth must be non-negative");
            }
            submission.submitBatch(
                    exactItems,
                    batchId,
                    exactPredictionMs,
                    decisionReason,
                    observer);
        }

        private void transportAccepted() {
            requirePhase(Phase.COMMITTED, "accept transport");
            phase = Phase.INFLIGHT;
        }

        private void transportFailed() {
            requirePhase(Phase.COMMITTED, "fail transport");
            phase = Phase.TERMINAL;
        }

        private boolean transportOwned() {
            return phase == Phase.INFLIGHT;
        }

        private void closeCapabilities() {
            Throwable failure = null;
            try {
                closeAdmission();
            } catch (Throwable admissionFailure) {
                failure = admissionFailure;
            }
            try {
                closeSubmission();
            } catch (Throwable submissionFailure) {
                failure = BatchDeliveryStrategy.append(
                        failure, submissionFailure);
            }
            if (failure != null) {
                throw propagate(failure);
            }
        }

        private void closeAdmission() {
            PrefillAdmissionResources.CommittedAdmissionOwner exactAdmission =
                    committedAdmission;
            if (exactAdmission != null) {
                committedAdmission = null;
                exactAdmission.close();
            }
        }

        private void closeSubmission() {
            PreparedSubmission exactSubmission = submission;
            if (exactSubmission != null) {
                submission = null;
                exactSubmission.close();
            }
        }

        private void requirePrepared(String operation) {
            if (phase != Phase.PREPARED
                    || submission == null) {
                throw new IllegalStateException(
                        "cannot " + operation + " batch transaction in " + phase);
            }
        }

        /** Detach once, then attempt every admission rollback leaf. */
        private Throwable rollbackAdmission(Throwable priorFailure) {
            PrefillState.Reservation exactReservation;
            ArrayList<PrefillAdmissionResources.Member> exactMembers;
            synchronized (this) {
                if (members == null) {
                    return priorFailure;
                }
                exactReservation = reservation;
                reservation = null;
                exactMembers = members;
                members = null;
            }
            Throwable failure = priorFailure;
            for (PrefillAdmissionResources.Member member : exactMembers) {
                failure = rollbackMember(member, failure);
            }
            return rollbackReservation(exactReservation, failure);
        }

        private void requireAdmissionPrepared(String operation) {
            if (members == null) {
                throw new IllegalStateException(
                        operation + " requires PREPARED batch admission");
            }
        }

        private void requirePhase(Phase expected, String operation) {
            if (phase != expected) {
                throw new IllegalStateException(
                        "cannot " + operation + " batch transaction in " + phase);
            }
        }
    }

    /**
     * One dispatch-capacity permit prepared before the canonical commit.
     * A successful submit transfers ownership to the dispatcher, making the
     * following close a no-op. Closing an unused permit releases its capacity;
     * repeated submission is an invariant violation.
     */
    public interface PreparedSubmission extends AutoCloseable {
        void submitBatch(
                List<ScheduledRequest> exactItems,
                long batchId,
                long predictedMs,
                String decisionReason,
                BiConsumer<ScheduledRequest, DeliveryResult> observer);

        @Override
        void close();
    }

    private record ClaimedMember(
            ScheduledRequest item,
            RequestRegistry.DeliveryClaim claim) {
    }

    private static final class DispatchGate
            implements BiConsumer<ScheduledRequest, DeliveryResult> {
        private final Map<ScheduledRequest, RequestRegistry.DeliveryClaim>
                claimsByItem;
        private final RequestRegistry requests;
        private boolean deferred = true;
        private List<Event> events;

        private DispatchGate(
                List<ClaimedMember> members,
                RequestRegistry requests) {
            this.requests = requests;
            this.claimsByItem = new IdentityHashMap<>(members.size());
            for (ClaimedMember member : members) {
                RequestRegistry.DeliveryClaim previous = claimsByItem.put(
                        member.item(), member.claim());
                if (previous != null) {
                    throw new IllegalArgumentException(
                            "duplicate batch delivery identity");
                }
            }
        }

        private void open() {
            List<Event> pending;
            synchronized (this) {
                deferred = false;
                pending = events;
                events = null;
            }
            if (pending != null) {
                Throwable failure = null;
                for (Event event : pending) {
                    try {
                        invoke(event);
                    } catch (Throwable callbackFailure) {
                        failure = append(failure, callbackFailure);
                    }
                }
                if (failure != null) {
                    throw propagate(failure);
                }
            }
        }

        @Override
        public void accept(
                ScheduledRequest item,
                DeliveryResult completion) {
            publish(new Event(item, completion));
        }

        private void publish(Event event) {
            synchronized (this) {
                if (deferred) {
                    if (events == null) {
                        events = new ArrayList<>();
                    }
                    events.add(event);
                    return;
                }
            }
            invoke(event);
        }

        private void invoke(Event event) {
            RequestRegistry.DeliveryClaim claim = claimFor(event.item());
            if (claim == null) {
                throw new IllegalStateException(
                        "batch completion referenced an unsubmitted identity");
            }
            requests.complete(claim, event.completion());
        }

        private RequestRegistry.DeliveryClaim claimFor(ScheduledRequest item) {
            return claimsByItem.get(item);
        }
    }

    private record Event(
            ScheduledRequest item,
            DeliveryResult completion) {
    }

    private static final class BatchProjection
            implements RouteProjection.DeliveryProjection {

        @Override
        public RouteProjection.GroupPlanning planning(
                RouteProjection.Predictions predictions) {
            return new RouteProjection.GroupPlanning() {
                private List<GroupPlanner.Item> previous = List.of();
                private double[] durations = new double[0];
                private boolean[] computed = new boolean[0];

                @Override
                public double durationMs(
                        List<GroupPlanner.Item> candidatePrefix,
                        int requiredThroughIndex) {
                    if (requiredThroughIndex < 0
                            || requiredThroughIndex >= candidatePrefix.size()) {
                        throw new IndexOutOfBoundsException(
                                requiredThroughIndex);
                    }
                    if (!isIdentityPrefix(previous, candidatePrefix)) {
                        previous = List.copyOf(candidatePrefix);
                        durations = new double[candidatePrefix.size()];
                        computed = new boolean[candidatePrefix.size()];
                    } else if (durations.length < candidatePrefix.size()) {
                        durations = java.util.Arrays.copyOf(
                                durations, candidatePrefix.size());
                        computed = java.util.Arrays.copyOf(
                                computed, candidatePrefix.size());
                        previous = List.copyOf(candidatePrefix);
                    }
                    if (!computed[requiredThroughIndex]) {
                        durations[requiredThroughIndex] =
                                predictions.batchPlanningDurationMs(
                                        candidatePrefix.subList(
                                                0,
                                                requiredThroughIndex + 1));
                        computed[requiredThroughIndex] = true;
                    }
                    return durations[requiredThroughIndex];
                }
            };
        }

        @Override
        public RouteProjection.GroupService service(
                GroupPlanner.Plan<GroupPlanner.Item> plan,
                RouteProjection.Predictions predictions) {
            return new RouteProjection.GroupService() {
                private final long[] completionOffsets =
                        new long[plan.items().size()];
                private final boolean[] computed =
                        new boolean[plan.items().size()];

                @Override
                public long completionOffsetMs(int memberIndex) {
                    if (memberIndex < 0 || memberIndex >= plan.items().size()) {
                        throw new IndexOutOfBoundsException(memberIndex);
                    }
                    if (!computed[memberIndex]) {
                        completionOffsets[memberIndex] =
                                predictions.batchDurationMs(
                                        plan.items().subList(
                                                0, memberIndex + 1));
                        computed[memberIndex] = true;
                    }
                    return completionOffsets[memberIndex];
                }

                @Override
                public long totalDurationMs() {
                    return completionOffsetMs(plan.items().size() - 1);
                }
            };
        }

        private static boolean isIdentityPrefix(
                List<GroupPlanner.Item> previous,
                List<GroupPlanner.Item> next) {
            if (previous.size() > next.size()) {
                return false;
            }
            for (int index = 0; index < previous.size(); index++) {
                if (previous.get(index) != next.get(index)) {
                    return false;
                }
            }
            return true;
        }
    }
}
