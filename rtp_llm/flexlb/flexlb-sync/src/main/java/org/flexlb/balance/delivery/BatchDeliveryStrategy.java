package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.BiConsumer;

/** EnqueueBatch admission, ownership, transport, telemetry, and projection. */
public final class BatchDeliveryStrategy implements DeliveryStrategy {

    private static final RouteProjection.DeliveryProjection PROJECTION =
            new BatchProjection();

    private final BatchSubmissionPort submissionPort;
    private final PrefillAdmissionPort admissionPort;
    private final SlotDeliveryPort slotPort;
    private final DeliveryMetrics telemetry;

    public BatchDeliveryStrategy(
            BatchSubmissionPort submissionPort,
            PrefillAdmissionPort admissionPort,
            SlotDeliveryPort slotPort,
            DeliveryMetrics telemetry) {
        this.submissionPort = Objects.requireNonNull(
                submissionPort, "submissionPort");
        this.admissionPort = Objects.requireNonNull(
                admissionPort, "admissionPort");
        this.slotPort = Objects.requireNonNull(slotPort, "slotPort");
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
                slotPort.prepareIfOwned(head, () -> prepareAdmission(head))
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
                        slotPort.prepareIfOwned(
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
                    BatchSubmissionPort.PreparedSubmission> submissionAttempt =
                    submissionPort.tryPrepareSubmission();
            if (!submissionAttempt.accepted()) {
                return rejected(submissionAttempt.boundary());
            }
            BatchSubmissionPort.PreparedSubmission submission =
                    submissionAttempt.value();
            PrefillAdmissionPort.PreparedAdmission admission = null;
            try {
                CapacityBoundary.Attempt<
                        PrefillAdmissionPort.PreparedAdmission>
                        admissionAttempt =
                        admissionPort.tryBegin(head);
                if (!admissionAttempt.accepted()) {
                    Throwable cleanup = close(submission);
                    if (cleanup != null) {
                        CapacityBoundary boundary = admissionAttempt.boundary();
                        if (boundary.status() == CapacityBoundary.Status.FAILED
                                && boundary.cause() != cleanup) {
                            cleanup.addSuppressed(boundary.cause());
                        }
                        return failed(cleanup);
                    }
                    return rejected(admissionAttempt.boundary());
                }
                admission = admissionAttempt.value();
                long batchId = admission.correlationId()
                        .orElseThrow(() -> new IllegalStateException(
                                "batch admission returned no correlation id"));
                if (batchId <= 0L) {
                    throw new IllegalStateException(
                            "batch admission returned a non-positive "
                                    + "correlation id");
                }
                return accepted(new BatchTransaction(
                        this,
                        batchId,
                        submission,
                        admission));
            } catch (Throwable failure) {
                Throwable cleanup = admission == null
                        ? null : close(admission);
                cleanup = append(cleanup, close(submission));
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
            DeliveryMetadata metadata) {
        List<ScheduledRequest> original = batch.items();
        List<ClaimedMember> claimed = new ArrayList<>(original.size());
        DispatchGate gate = null;
        Throwable handoffFailure = null;
        long deliveredPredictionMs = batch.predictedMs();
        try {
            for (int index = 0; index < original.size(); index++) {
                ScheduledRequest item = original.get(index);
                try {
                    SlotDeliveryPort.Claim claim =
                            slotPort.tryClaimForDelivery(
                                    item,
                                    SlotDeliveryPort.Identity
                                            .externalAcknowledgement(
                                            batch.batchId()),
                                    () -> batch.transferToEndpoint(item));
                    if (claim == null) {
                        continue;
                    }
                    claimed.add(new ClaimedMember(item, claim));
                } catch (Throwable claimFailure) {
                    slotPort.failPrepared(item, claimFailure);
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
                        claimed, slotPort);
                batch.submit(
                        new BatchSubmissionPort.Command(
                                submitted,
                                batch.batchId(),
                                deliveredPredictionMs,
                                metadata),
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
                        slotPort.complete(
                                member.claim(),
                                SlotDeliveryPort.Completion.failed(
                                        handoffFailure));
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
                    metadata,
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
                slotPort.failPrepared(item, cause);
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
        private BatchSubmissionPort.PreparedSubmission submission;
        private PrefillAdmissionPort.PreparedAdmission preparedAdmission;
        private PrefillAdmissionPort.CommittedAdmission committedAdmission;
        private List<ScheduledRequest> items = List.of();
        private long predictedMs;
        private PrefillTimePredictor.Evaluator evaluator;
        private ScheduledRequest blockedItem;
        private CapacityBoundary blockedResult;
        private Phase phase;

        private BatchTransaction(
                BatchDeliveryStrategy owner,
                long batchId,
                BatchSubmissionPort.PreparedSubmission submission,
                PrefillAdmissionPort.PreparedAdmission admission) {
            this.owner = owner;
            this.batchId = batchId;
            this.submission = submission;
            this.preparedAdmission = admission;
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
                ScheduledRequest exactItem) {
            requirePrepared("append");
            return preparedAdmission.tryAppend(exactItem, 0L);
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
            if (items.isEmpty()) {
                throw new IllegalStateException(
                        "empty batch transaction cannot commit");
            }
            committedAdmission = preparedAdmission.commitPreparedUnderLock(
                    items, predictedMs);
            preparedAdmission = null;
            phase = Phase.COMMITTED;
        }

        @Override
        public synchronized void handoff(DeliveryMetadata metadata) {
            requirePhase(Phase.COMMITTED, "deliver");
            try {
                owner.deliverCommitted(this, metadata);
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
            Throwable failure = BatchDeliveryStrategy.close(preparedAdmission);
            preparedAdmission = null;
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
                BatchSubmissionPort.Command command,
                BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> observer) {
            requirePhase(Phase.COMMITTED, "submit");
            submission.submitBatch(command, observer);
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
            PrefillAdmissionPort.CommittedAdmission exactAdmission =
                    committedAdmission;
            if (exactAdmission != null) {
                committedAdmission = null;
                exactAdmission.close();
            }
        }

        private void closeSubmission() {
            BatchSubmissionPort.PreparedSubmission exactSubmission = submission;
            if (exactSubmission != null) {
                submission = null;
                exactSubmission.close();
            }
        }

        private void requirePrepared(String operation) {
            if (phase != Phase.PREPARED
                    || preparedAdmission == null
                    || submission == null) {
                throw new IllegalStateException(
                        "cannot " + operation + " batch transaction in " + phase);
            }
        }

        private void requirePhase(Phase expected, String operation) {
            if (phase != expected) {
                throw new IllegalStateException(
                        "cannot " + operation + " batch transaction in " + phase);
            }
        }
    }

    private record ClaimedMember(
            ScheduledRequest item,
            SlotDeliveryPort.Claim claim) {
    }

    private static final class DispatchGate
            implements BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> {
        private final Map<ScheduledRequest, SlotDeliveryPort.Claim> claimsByItem;
        private final SlotDeliveryPort slotPort;
        private boolean deferred = true;
        private List<Event> events;

        private DispatchGate(
                List<ClaimedMember> members,
                SlotDeliveryPort slotPort) {
            this.slotPort = slotPort;
            this.claimsByItem = new IdentityHashMap<>(members.size());
            for (ClaimedMember member : members) {
                SlotDeliveryPort.Claim previous = claimsByItem.put(
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
                SlotDeliveryPort.Completion completion) {
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
            SlotDeliveryPort.Claim claim = claimFor(event.item());
            if (claim == null) {
                throw new IllegalStateException(
                        "batch completion referenced an unsubmitted identity");
            }
            slotPort.complete(claim, event.completion());
        }

        private SlotDeliveryPort.Claim claimFor(ScheduledRequest item) {
            return claimsByItem.get(item);
        }
    }

    private record Event(
            ScheduledRequest item,
            SlotDeliveryPort.Completion completion) {
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
