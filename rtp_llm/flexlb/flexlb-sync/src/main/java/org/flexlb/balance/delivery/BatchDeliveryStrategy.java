package org.flexlb.balance.delivery;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.prefill.BatchReservation;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.Reservation;
import org.flexlb.balance.endpoint.prefill.ReservationResult;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.PrefillAdmissionResources;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
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

/** EnqueueBatch admission, ownership, transport, telemetry, and estimate. */
public final class BatchDeliveryStrategy implements DeliveryStrategy {

    private final Supplier<DeliveryBlockReason.Attempt<PreparedSubmission>>
            prepareSubmission;
    private final LongSupplier batchIds;
    private final RequestRegistry requests;
    private final DeliveryMetrics telemetry;

    public BatchDeliveryStrategy(Supplier<DeliveryBlockReason.Attempt<PreparedSubmission>> prepareSubmission,
                                 LongSupplier batchIds,
                                 RequestRegistry requests,
                                 DeliveryMetrics telemetry) {
        this.prepareSubmission = prepareSubmission;
        this.batchIds = batchIds;
        this.requests = requests;
        this.telemetry = telemetry;
    }

    @Override
    public DeliveryTransaction prepare(DeliveryGroup group) {
        List<ScheduledRequest> candidates = group.requests();
        PrefillTimePredictor.Evaluator evaluator = group.evaluator();
        OptionalLong plannedPredictionMs = group.plannedPredictionMs();
        ScheduledRequest head = candidates.get(0);

        DeliveryBlockReason.Attempt<BatchTransaction> groupAttempt =
                requests.prepareBatchDelivery(head, this);
        if (!groupAttempt.accepted()) {
            return BatchTransaction.blocked(
                    this,
                    new BlockedRequest(head, groupAttempt.blockReason()));
        }

        BatchTransaction transaction = groupAttempt.value();
        List<ScheduledRequest> admitted = new ArrayList<>(candidates.size());
        BlockedRequest blocked = null;
        try {
            for (int index = 0; index < candidates.size(); index++) {
                ScheduledRequest item = candidates.get(index);
                DeliveryBlockReason.Attempt<ScheduledRequest> attempt =
                        requests.prepareBatchMember(item, transaction);
                if (attempt.accepted()) {
                    admitted.add(item);
                } else {
                    blocked = new BlockedRequest(item, attempt.blockReason());
                    break;
                }
            }
            if (admitted.isEmpty()) {
                transaction.select(List.of(), 0L, null, blocked);
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
                    blocked);
            return transaction;
        } catch (Throwable failure) {
            Throwable cleanup = close(transaction);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
    }

    public DeliveryBlockReason.Attempt<BatchTransaction> prepareAdmission(ScheduledRequest head) {
        try {
            DeliveryBlockReason.Attempt<PreparedSubmission> submissionAttempt = prepareSubmission.get();
            if (!submissionAttempt.accepted()) {
                return rejected(submissionAttempt.blockReason());
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

    private void deliverCommitted(
            BatchTransaction batch,
            String decisionReason,
            int remainingQueueDepth,
            WorkView precedingWork,
            BatchSender sender) {
        List<ScheduledRequest> original = batch.requests();
        List<ClaimedMember> claimed = new ArrayList<>(original.size());
        DispatchGate gate = null;
        Throwable handoffFailure = null;
        long deliveredPredictionMs = batch.predictedMs();
        try {
            for (int index = 0; index < original.size(); index++) {
                ScheduledRequest item = original.get(index);
                try {
                    DeliveryClaim claim =
                            requests.claimBatchDelivery(
                                    item, batch);
                    if (claim == null) {
                        continue;
                    }
                    claimed.add(new ClaimedMember(item, claim));
                } catch (Throwable claimFailure) {
                    requests.failDeliveryPreparation(item, claimFailure);
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
                for (ClaimedMember member : claimed) {
                    requests.setDeliveryPrediction(member.claim(), precedingWork, deliveredPredictionMs);
                }
                gate = new DispatchGate(
                        claimed, requests);
                batch.send(
                        sender, submitted,
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
                        member.claim().complete(DeliveryResult.notSent(handoffFailure));
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
        for (ScheduledRequest item : batch.requests()) {
            try {
                requests.failDeliveryPreparation(item, cause);
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

    private static <T> DeliveryBlockReason.Attempt<T> accepted(T value) {
        return DeliveryBlockReason.Attempt.accepted(value);
    }

    private static <T> DeliveryBlockReason.Attempt<T> rejected(
            DeliveryBlockReason boundary) {
        return DeliveryBlockReason.Attempt.rejected(boundary);
    }

    private static <T> DeliveryBlockReason.Attempt<T> failed(Throwable cause) {
        return rejected(DeliveryBlockReason.failed(cause));
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
    public static final class BatchTransaction implements DeliveryTransaction {
        private enum Phase {
            PREPARED,
            COMMITTED,
            SUBMITTED,
            INFLIGHT,
            TERMINAL
        }

        private final BatchDeliveryStrategy owner;
        private final long batchId;
        private final PrefillEndpoint prefill;
        private PreparedSubmission submission;
        private BatchReservation reservation;
        private ArrayList<PrefillAdmissionResources.Member> members;
        private PrefillAdmissionResources.CommittedAdmissionOwner
                committedAdmission;
        private List<ScheduledRequest> items = List.of();
        private long predictedMs;
        private PrefillTimePredictor.Evaluator evaluator;
        private BlockedRequest blocked;
        // 进入 SUBMITTED 后，只有已被执行器接收的任务才能推进本事务；
        // 调度器的 rollbackCommitted 和 close 只能观察它的状态，不能回收其中的归属。
        private volatile Phase phase;

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
                BlockedRequest blocked) {
            BatchTransaction transaction = new BatchTransaction(
                    owner, 0L, null, null);
            transaction.blocked = blocked;
            transaction.phase = Phase.TERMINAL;
            return transaction;
        }

        public synchronized DeliveryBlockReason.Attempt<ScheduledRequest> append(ScheduledRequest exact) {
            requirePrepared("append");
            requireAdmissionPrepared("append");
            try {
                DeliveryBlockReason.Attempt<ScheduledRequest> acceptedItem =
                        accepted(exact);
                members.ensureCapacity(Math.addExact(members.size(), 1));
                boolean first = members.isEmpty();
                if (first) {
                    ReservationResult<BatchReservation> result =
                            prefill.reserveBatch(exact, batchId, exact.maxInflightBatchesPerPrefillWorker());
                    if (!result.isSuccess()) {
                        return rejectedPrefill(
                                exact,
                                result.status(),
                                DeliveryBlockReason.deliveryUnavailable(
                                        prefill.batchAdmissionAvailability(
                                                exact.maxInflightBatchesPerPrefillWorker())));
                    }
                    reservation = result.reservation();
                }
                DeliveryBlockReason.Attempt<PrefillAdmissionResources.Member>
                        memberAttempt = prepareMember(exact);
                DeliveryBlockReason.Attempt<ScheduledRequest> result;
                if (memberAttempt.accepted()) {
                    members.add(memberAttempt.value());
                    result = acceptedItem;
                } else {
                    result = rejected(memberAttempt.blockReason());
                }
                if (first && !result.accepted()) {
                    return abortAdmission(result.blockReason());
                }
                return result;
            } catch (Throwable failure) {
                if (members.isEmpty()) {
                    return abortAdmission(failure);
                }
                return failed(failure);
            }
        }

        private <T> DeliveryBlockReason.Attempt<T> abortAdmission(
                DeliveryBlockReason boundary) {
            Throwable rollbackFailure = rollbackAdmission(null);
            if (rollbackFailure == null) {
                return rejected(boundary);
            }
            preserveRejectedCause(rollbackFailure, boundary);
            return failed(rollbackFailure);
        }

        private <T> DeliveryBlockReason.Attempt<T> abortAdmission(
                Throwable failure) {
            return failed(rollbackAdmission(failure));
        }

        private synchronized void select(
                List<ScheduledRequest> exactItems,
                long exactPredictionMs,
                PrefillTimePredictor.Evaluator exactEvaluator,
                BlockedRequest exactBlocked) {
            requirePrepared("select");
            items = List.copyOf(exactItems);
            predictedMs = exactPredictionMs;
            evaluator = exactEvaluator;
            blocked = exactBlocked;
        }

        @Override
        public List<ScheduledRequest> requests() {
            return items;
        }

        @Override
        public BlockedRequest blocked() {
            return blocked;
        }

        @Override
        public synchronized WorkView commit() {
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
                    createCommittedOwner(members);
            PrefillState.CommittedHandoff handoff =
                    reservation.commit(items, predictedMs);
            exactCommitted.bindPrefillHandoff(handoff);
            committedAdmission = exactCommitted;
            reservation = null;
            members = null;
            phase = Phase.COMMITTED;
            return handoff.precedingWork();
        }

        @Override
        public synchronized void deliver(String decisionReason, int remainingQueueDepth, WorkView precedingWork) {
            requirePhase(Phase.COMMITTED, "submit delivery");
            phase = Phase.SUBMITTED;
            try {
                // The strategy retains admission ownership while waiting for
                // an executor. Request claims are acquired only when it runs.
                submission.submit(sender -> sendInExecutor(decisionReason, remainingQueueDepth, precedingWork, sender));
            } catch (Throwable failure) {
                failUnsentDelivery(failure);
                throw propagate(failure);
            }
        }

        private void sendInExecutor(String decisionReason,
                                    int remainingQueueDepth,
                                    WorkView precedingWork,
                                    BatchSender sender) {
            requirePhase(Phase.SUBMITTED, "sendInExecutor");
            try {
                owner.deliverCommitted(this, decisionReason, remainingQueueDepth,
                        precedingWork, sender);
            } catch (Throwable failure) {
                failUnsentDelivery(failure);
                throw propagate(failure);
            }
            if (phase == Phase.SUBMITTED) {
                phase = Phase.TERMINAL;
            }
        }

        private void failUnsentDelivery(Throwable failure) {
            if (phase != Phase.SUBMITTED) {
                return;
            }
            Throwable cleanup = owner.failCommitted(this, failure);
            phase = Phase.TERMINAL;
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
        }

        @Override
        public synchronized void rollbackCommitted(Throwable cause) {
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

        public long batchId() {
            return batchId;
        }

        private long predictedMs() {
            return predictedMs;
        }

        private PrefillTimePredictor.Evaluator evaluator() {
            return evaluator;
        }

        public boolean transferToEndpoint(ScheduledRequest exactItem) {
            requirePhase(Phase.SUBMITTED, "transfer admission");
            return committedAdmission.transferToEndpoint(exactItem);
        }

        private void send(
                BatchSender sender,
                List<ScheduledRequest> exactItems,
                long exactPredictionMs,
                String decisionReason,
                int remainingQueueDepth,
                BiConsumer<ScheduledRequest, DeliveryResult> observer) {
            requirePhase(Phase.SUBMITTED, "send");
            if (remainingQueueDepth < 0) {
                throw new IllegalArgumentException(
                        "remainingQueueDepth must be non-negative");
            }
            sender.sendBatch(
                    exactItems,
                    batchId,
                    exactPredictionMs,
                    decisionReason,
                    observer);
        }

        private void transportAccepted() {
            requirePhase(Phase.SUBMITTED, "accept transport");
            phase = Phase.INFLIGHT;
        }

        private void transportFailed() {
            requirePhase(Phase.SUBMITTED, "fail transport");
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
            Reservation exactReservation;
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
     * One executor-capacity permit prepared before commit. Successful submission
     * transfers the permit to the executor; close then becomes a no-op. The
     * strategy still owns the batch admission until Delivery runs and settles it.
     */
    public interface PreparedSubmission extends AutoCloseable {
        void submit(Delivery delivery);

        @Override
        void close();
    }

    /** Runs once on the dispatch executor, with no further queue before send. */
    @FunctionalInterface
    public interface Delivery {
        void run(BatchSender sender);
    }

    /** Transport consumes the strategy's final batch; it never selects members. */
    @FunctionalInterface
    public interface BatchSender {
        void sendBatch(
                List<ScheduledRequest> exactItems,
                long batchId,
                long predictedMs,
                String decisionReason,
                BiConsumer<ScheduledRequest, DeliveryResult> observer);
    }

    private record ClaimedMember(
            ScheduledRequest item,
            DeliveryClaim claim) {
    }

    private static final class DispatchGate
            implements BiConsumer<ScheduledRequest, DeliveryResult> {
        private final Map<ScheduledRequest, DeliveryClaim>
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
                DeliveryClaim previous = claimsByItem.put(
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
            DeliveryClaim claim = claimFor(event.item());
            if (claim == null) {
                throw new IllegalStateException(
                        "batch completion referenced an unsubmitted identity");
            }
            claim.complete(event.completion());
        }

        private DeliveryClaim claimFor(ScheduledRequest item) {
            return claimsByItem.get(item);
        }
    }

    private record Event(
            ScheduledRequest item,
            DeliveryResult completion) {
    }
}
