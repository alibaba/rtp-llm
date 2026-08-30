package org.flexlb.balance.delivery;

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
    private final DeliveryTelemetry telemetry;

    public BatchDeliveryStrategy(
            BatchSubmissionPort submissionPort,
            PrefillAdmissionPort admissionPort,
            SlotDeliveryPort slotPort,
            DeliveryTelemetry telemetry) {
        this.submissionPort = Objects.requireNonNull(
                submissionPort, "submissionPort");
        this.admissionPort = Objects.requireNonNull(
                admissionPort, "admissionPort");
        this.slotPort = Objects.requireNonNull(slotPort, "slotPort");
        this.telemetry = Objects.requireNonNull(telemetry, "telemetry");
    }

    @Override
    public PreparedDelivery prepare(
            List<DeliveryItem> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        if (candidates.isEmpty()) {
            throw new IllegalArgumentException(
                    "batch delivery requires at least one candidate");
        }
        return admit(candidates, evaluator, plannedPredictionMs);
    }

    private Admission admit(
            List<DeliveryItem> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        DeliveryItem head = candidates.get(0);

        CapacityBoundary.Attempt<Prepared> groupAttempt =
                slotPort.prepareIfOwned(head, () -> prepareAdmission(head))
                        .orElseGet(() -> BatchDeliveryStrategy
                                .<Prepared>ownershipLost());
        if (groupAttempt
                instanceof CapacityBoundary.Attempt.Rejected<Prepared> blocked) {
            return Admission.empty(
                    this,
                    new SelectionBoundary(
                            head, blocked.boundary()));
        }

        Prepared prepared = ((CapacityBoundary.Attempt.Accepted<Prepared>)
                groupAttempt).value();
        List<DeliveryItem> admitted = new ArrayList<>(candidates.size());
        SelectionBoundary boundary = null;
        try {
            for (int index = 0; index < candidates.size(); index++) {
                DeliveryItem item = candidates.get(index);
                CapacityBoundary.Attempt<DeliveryItem> attempt =
                        slotPort.prepareIfOwned(
                                item, () -> prepared.append(item))
                                .orElseGet(() -> BatchDeliveryStrategy
                                        .<DeliveryItem>ownershipLost());
                if (attempt
                        instanceof CapacityBoundary.Attempt.Accepted<
                                DeliveryItem>) {
                    admitted.add(item);
                } else {
                    boundary = new SelectionBoundary(
                            item,
                            ((CapacityBoundary.Attempt.Rejected<DeliveryItem>)
                                    attempt).boundary());
                    break;
                }
            }
            if (admitted.isEmpty()) {
                return new Admission(
                        this,
                        List.of(),
                        prepared,
                        0L,
                        null,
                        boundary);
            }
            long predictedMs = plannedPredictionMs.isPresent()
                    && sameIdentitySequence(admitted, candidates)
                    ? plannedPredictionMs.getAsLong()
                    : PrefillPredictionBoundary.predictCommittedBatchMs(
                            evaluator,
                            PrefillBatchFeatures.from(
                                    admitted,
                                    DeliveryItem::seqLen,
                                    DeliveryItem::hitCache));
            return new Admission(
                    this,
                    admitted,
                    prepared,
                    predictedMs,
                    evaluator,
                    boundary);
        } catch (Throwable failure) {
            Throwable cleanup = close(prepared);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
    }

    private CapacityBoundary.Attempt<Prepared> prepareAdmission(
            DeliveryItem head) {
        try {
            CapacityBoundary.Attempt<
                    BatchSubmissionPort.PreparedSubmission> submissionAttempt =
                    submissionPort.tryPrepareSubmission();
            if (submissionAttempt
                    instanceof CapacityBoundary.Attempt.Rejected<
                            BatchSubmissionPort.PreparedSubmission> rejected) {
                return rejected(rejected.boundary());
            }
            BatchSubmissionPort.PreparedSubmission submission =
                    ((CapacityBoundary.Attempt.Accepted<
                            BatchSubmissionPort.PreparedSubmission>)
                            submissionAttempt).value();
            PrefillAdmissionPort.PreparedAdmission admission = null;
            try {
                CapacityBoundary.Attempt<
                        PrefillAdmissionPort.PreparedAdmission>
                        admissionAttempt =
                        admissionPort.tryBegin(head);
                if (admissionAttempt
                        instanceof CapacityBoundary.Attempt.Rejected<
                                PrefillAdmissionPort.PreparedAdmission>
                                rejected) {
                    Throwable cleanup = close(submission);
                    if (cleanup != null) {
                        if (rejected.boundary()
                                instanceof CapacityBoundary.Failed failed
                                && failed.cause() != cleanup) {
                            cleanup.addSuppressed(failed.cause());
                        }
                        return failed(cleanup);
                    }
                    return rejected(rejected.boundary());
                }
                admission =
                        ((CapacityBoundary.Attempt.Accepted<
                                PrefillAdmissionPort.PreparedAdmission>)
                                admissionAttempt).value();
                long batchId = admission.correlationId()
                        .orElseThrow(() -> new IllegalStateException(
                                "batch admission returned no correlation id"));
                if (batchId <= 0L) {
                    throw new IllegalStateException(
                            "batch admission returned a non-positive "
                                    + "correlation id");
                }
                return accepted(new Prepared(
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
            List<DeliveryItem> items,
            PrefillTimePredictor.Evaluator evaluator) {
        return PrefillPredictionBoundary.predictDecisionGroupMs(
                evaluator,
                PrefillBatchFeatures.from(
                        items,
                        DeliveryItem::seqLen,
                        DeliveryItem::hitCache));
    }

    @Override
    public RouteProjection.DeliveryProjection projectionPolicy() {
        return PROJECTION;
    }

    private void deliverCommitted(
            Committed batch,
            DeliveryMetadata metadata) {
        List<DeliveryItem> original = batch.items();
        List<ClaimedMember> claimed = new ArrayList<>(original.size());
        DispatchGate gate = null;
        Throwable handoffFailure = null;
        long deliveredPredictionMs = batch.predictedMs();
        try {
            for (int index = 0; index < original.size(); index++) {
                DeliveryItem item = original.get(index);
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
                List<DeliveryItem> submitted = claimed.stream()
                        .map(ClaimedMember::item)
                        .toList();
                if (submitted.size() != original.size()) {
                    deliveredPredictionMs =
                            PrefillPredictionBoundary.predictCommittedBatchMs(
                                    batch.evaluator(),
                                    PrefillBatchFeatures.from(
                                            submitted,
                                            DeliveryItem::seqLen,
                                            DeliveryItem::hitCache));
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
                                new SlotDeliveryPort.Completion.Failed(
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

    private Throwable failCommitted(Committed batch, Throwable cause) {
        Throwable cleanup = null;
        try {
            batch.closeSubmission();
        } catch (Throwable failure) {
            cleanup = append(cleanup, failure);
        }
        for (DeliveryItem item : batch.items()) {
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
            List<DeliveryItem> left,
            List<DeliveryItem> right) {
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
        return new CapacityBoundary.Attempt.Accepted<>(value);
    }

    private static <T> CapacityBoundary.Attempt<T> rejected(
            CapacityBoundary boundary) {
        return new CapacityBoundary.Attempt.Rejected<>(boundary);
    }

    private static <T> CapacityBoundary.Attempt<T> ownershipLost() {
        return rejected(CapacityBoundary.OwnershipLost.INSTANCE);
    }

    private static <T> CapacityBoundary.Attempt<T> failed(Throwable cause) {
        return rejected(new CapacityBoundary.Failed(cause));
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

    /** Temporary owner of every resource for one admitted batch prefix. */
    static final class Admission implements PreparedDelivery {
        private final BatchDeliveryStrategy owner;
        private final List<DeliveryItem> items;
        private final long predictedMs;
        private final PrefillTimePredictor.Evaluator evaluator;
        private final SelectionBoundary boundary;
        private Prepared prepared;

        private Admission(
                BatchDeliveryStrategy owner,
                List<DeliveryItem> items,
                Prepared prepared,
                long predictedMs,
                PrefillTimePredictor.Evaluator evaluator,
                SelectionBoundary boundary) {
            this.owner = owner;
            this.items = items;
            this.prepared = prepared;
            this.predictedMs = predictedMs;
            this.evaluator = evaluator;
            this.boundary = boundary;
        }

        private static Admission empty(
                BatchDeliveryStrategy owner,
                SelectionBoundary boundary) {
            return new Admission(
                    owner, List.of(), null,
                    0L, null, boundary);
        }

        @Override
        public List<DeliveryItem> items() {
            return items;
        }

        @Override
        public SelectionBoundary boundary() {
            return boundary;
        }

        @Override
        public synchronized Committed commitOwnershipUnderLock() {
            Prepared exactPrepared = prepared;
            assert !items.isEmpty() && exactPrepared != null
                    : "empty or resolved batch admission cannot commit";
            Committed committed = exactPrepared.commitPreparedUnderLock(
                    owner,
                    items,
                    predictedMs,
                    evaluator);
            prepared = null;
            return committed;
        }

        @Override
        public synchronized void close() {
            Prepared exactPrepared = prepared;
            if (exactPrepared == null) {
                return;
            }
            prepared = null;
            exactPrepared.close();
        }
    }

    /** Private admission transaction whose non-null capabilities are its state. */
    private static final class Prepared implements AutoCloseable {
        private final long batchId;
        private BatchSubmissionPort.PreparedSubmission submission;
        private PrefillAdmissionPort.PreparedAdmission admission;

        private Prepared(
                long batchId,
                BatchSubmissionPort.PreparedSubmission submission,
                PrefillAdmissionPort.PreparedAdmission admission) {
            this.batchId = batchId;
            this.submission = submission;
            this.admission = admission;
        }

        private synchronized CapacityBoundary.Attempt<DeliveryItem> append(
                DeliveryItem exactItem) {
            requirePrepared("append");
            return admission.tryAppend(exactItem, 0L);
        }

        private synchronized Committed commitPreparedUnderLock(
                BatchDeliveryStrategy owner,
                List<DeliveryItem> exactItems,
                long predictedMs,
                PrefillTimePredictor.Evaluator evaluator) {
            requirePrepared("commit");
            Committed committed = new Committed(
                    owner,
                    exactItems,
                    submission,
                    batchId,
                    predictedMs,
                    evaluator);
            PrefillAdmissionPort.CommittedAdmission committedAdmission =
                    admission.commitPreparedUnderLock(exactItems, predictedMs);
            committed.bind(committedAdmission);
            admission = null;
            submission = null;
            return committed;
        }

        @Override
        public synchronized void close() {
            PrefillAdmissionPort.PreparedAdmission exactAdmission = admission;
            BatchSubmissionPort.PreparedSubmission exactSubmission = submission;
            if (exactAdmission == null && exactSubmission == null) {
                return;
            }
            admission = null;
            submission = null;
            Throwable failure = BatchDeliveryStrategy.close(exactAdmission);
            failure = BatchDeliveryStrategy.append(
                    failure, BatchDeliveryStrategy.close(exactSubmission));
            if (failure != null) {
                throw propagate(failure);
            }
        }

        private void requirePrepared(String operation) {
            if (admission == null || submission == null) {
                throw new IllegalStateException(
                        "cannot " + operation
                                + " after batch admission ownership was resolved");
            }
        }
    }

    /** Sole post-Registry batch owner. */
    private static final class Committed implements Handoff {
        private enum DeliveryPhase {
            COMMITTED,
            INFLIGHT,
            TERMINAL
        }

        private final BatchDeliveryStrategy owner;
        private final List<DeliveryItem> items;
        private final long batchId;
        private final long predictedMs;
        private final PrefillTimePredictor.Evaluator evaluator;
        private PrefillAdmissionPort.CommittedAdmission admission;
        private BatchSubmissionPort.PreparedSubmission submission;
        private DeliveryPhase phase = DeliveryPhase.COMMITTED;

        private Committed(
                BatchDeliveryStrategy owner,
                List<DeliveryItem> items,
                BatchSubmissionPort.PreparedSubmission submission,
                long batchId,
                long predictedMs,
                PrefillTimePredictor.Evaluator evaluator) {
            this.owner = owner;
            this.items = List.copyOf(items);
            this.submission = submission;
            this.batchId = batchId;
            this.predictedMs = predictedMs;
            this.evaluator = evaluator;
        }

        /** Bind the capability returned by the canonical commit without work. */
        private void bind(
                PrefillAdmissionPort.CommittedAdmission exactAdmission) {
            admission = exactAdmission;
        }

        @Override
        public List<DeliveryItem> items() {
            return items;
        }

        public long batchId() {
            return batchId;
        }

        public long predictedMs() {
            return predictedMs;
        }

        private PrefillTimePredictor.Evaluator evaluator() {
            return evaluator;
        }

        @Override
        public synchronized void deliver(
                DeliveryMetadata metadata) {
            if (phase != DeliveryPhase.COMMITTED) {
                throw new IllegalStateException(
                        "batch cannot deliver in " + phase);
            }
            try {
                owner.deliverCommitted(this, metadata);
            } catch (Throwable failure) {
                if (phase != DeliveryPhase.COMMITTED) {
                    throw propagate(failure);
                }
                Throwable cleanup = owner.failCommitted(this, failure);
                if (cleanup != null && cleanup != failure) {
                    failure.addSuppressed(cleanup);
                }
                phase = DeliveryPhase.TERMINAL;
                throw propagate(failure);
            }
            if (phase == DeliveryPhase.COMMITTED) {
                phase = DeliveryPhase.TERMINAL;
            }
        }

        @Override
        public synchronized void failBeforeDelivery(Throwable cause) {
            if (phase != DeliveryPhase.COMMITTED) {
                return;
            }
            Throwable cleanup = owner.failCommitted(this, cause);
            phase = DeliveryPhase.TERMINAL;
            if (cleanup != null) {
                throw propagate(cleanup);
            }
        }

        private boolean transferToEndpoint(DeliveryItem exactItem) {
            if (phase != DeliveryPhase.COMMITTED) {
                throw new IllegalStateException(
                        "batch admission cannot transfer in " + phase);
            }
            return admission.transferToEndpoint(exactItem);
        }

        private void submit(
                BatchSubmissionPort.Command command,
                BiConsumer<DeliveryItem, SlotDeliveryPort.Completion> observer) {
            if (phase != DeliveryPhase.COMMITTED) {
                throw new IllegalStateException(
                        "batch cannot submit in " + phase);
            }
            submission.submitBatch(command, observer);
        }

        private void transportAccepted() {
            if (phase != DeliveryPhase.COMMITTED) {
                throw new IllegalStateException(
                        "batch transport cannot accept in " + phase);
            }
            phase = DeliveryPhase.INFLIGHT;
        }

        private void transportFailed() {
            if (phase != DeliveryPhase.COMMITTED) {
                throw new IllegalStateException(
                        "batch transport cannot fail in " + phase);
            }
            phase = DeliveryPhase.TERMINAL;
        }

        private boolean transportOwned() {
            return phase == DeliveryPhase.INFLIGHT;
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
                failure = append(failure, submissionFailure);
            }
            if (failure != null) {
                throw propagate(failure);
            }
        }

        private void closeAdmission() {
            PrefillAdmissionPort.CommittedAdmission exactAdmission = admission;
            if (exactAdmission != null) {
                admission = null;
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
    }

    private record ClaimedMember(
            DeliveryItem item,
            SlotDeliveryPort.Claim claim) {
    }

    private static final class DispatchGate
            implements BiConsumer<DeliveryItem, SlotDeliveryPort.Completion> {
        private final Map<DeliveryItem, SlotDeliveryPort.Claim> claimsByItem;
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
                DeliveryItem item,
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

        private SlotDeliveryPort.Claim claimFor(DeliveryItem item) {
            return claimsByItem.get(item);
        }
    }

    private record Event(
            DeliveryItem item,
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
