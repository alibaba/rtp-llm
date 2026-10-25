package org.flexlb.balance.endpoint.prefill.batch;

import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.endpoint.EndpointGenerationLifecycle;
import org.flexlb.balance.endpoint.prefill.LeaseState;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.RequestEntry;
import org.flexlb.balance.endpoint.prefill.Reservation;
import org.flexlb.balance.endpoint.prefill.ReservationResult;
import org.flexlb.balance.endpoint.prefill.TerminalObservation;
import org.flexlb.balance.estimate.WorkView.Phase;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.function.ToLongFunction;

/**
 * Prefill 上只给 BATCH 用的部分：batch 预占和槽位计数、整批提交、按引擎状态重新预测、清理过期 batch。
 * 请求表和锁仍归 PrefillState，这里的方法和 PrefillState 共用同一把锁。
 */
public final class BatchPrefill {
    private final PrefillState prefillState;
    private int batchLeasesInUse;

    public BatchPrefill(PrefillState prefillState) {
        this.prefillState = prefillState;
    }

    public ReservationResult<BatchReservation> reserveBatch(
            ScheduledRequest exactHead,
            long batchId,
            int maximum,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        requirePositiveBatchLimit(maximum);
        ScheduledRequest head = exactHead;
        prefillState.lock.lock();
        try {
            RequestEntry entry = prefillState.requests.get(head.requestId());
            if (entry == null || !entry.activeIdentity(head)) {
                return ReservationResult.fail(ReservationResult.Status.REQUEST_LEFT_QUEUE);
            }
            if (entry.reservation != null) {
                return ReservationResult.fail(ReservationResult.Status.REQUEST_ALREADY_RESERVED);
            }
            if (findBatchReservationUnderLock(batchId) != null) {
                return ReservationResult.fail(ReservationResult.Status.BATCH_ID_ALREADY_RESERVED);
            }
            if (batchLeasesInUse >= maximum) {
                return ReservationResult.fail(ReservationResult.Status.CAPACITY_FULL);
            }
            BatchReservation lease =
                    new BatchReservation(prefillState, entry, head.requestId(), batchId, generationHandoff);
            entry.reservation = lease;
            batchLeasesInUse++;
            prefillState.recordMutationUnderLock();
            return ReservationResult.success(lease);
        } finally {
            prefillState.lock.unlock();
        }
    }

    public int batchLeasesInUseUnderLock() {
        prefillState.requireLock();
        return batchLeasesInUse;
    }

    private boolean batchCapacityAvailable(int maximum) {
        prefillState.lock.lock();
        try {
            return batchLeasesInUse < maximum;
        } finally {
            prefillState.lock.unlock();
        }
    }

    public DeliveryBlockReason.Availability batchAvailability(int maximum) {
        requirePositiveBatchLimit(maximum);
        return new CapacityAvailability(maximum);
    }

    private static void requirePositiveBatchLimit(int maximum) {
        if (maximum <= 0) {
            throw new IllegalArgumentException("maximumInflightBatches must be positive");
        }
    }

    /**
     * Exact wake capability permanently paired with this worker runtime.
     */
    private final class CapacityAvailability implements DeliveryBlockReason.Availability {
        private final int maximum;
        private Runnable subscribed;

        private CapacityAvailability(int maximum) {
            this.maximum = maximum;
        }

        @Override
        public boolean isAvailable() {
            return batchCapacityAvailable(maximum);
        }

        @Override
        public synchronized void addListener(Runnable listener) {
            if (listener != prefillState.capacityAvailable) {
                throw new IllegalArgumentException("Prefill availability requires its exact worker wake callback");
            }
            if (subscribed != null && subscribed != listener) {
                throw new IllegalStateException("Prefill availability already has a listener");
            }
            subscribed = listener;
        }

        @Override
        public synchronized void removeListener(Runnable listener) {
            if (subscribed == listener) {
                subscribed = null;
            }
        }
    }

    public PrefillState.CommittedHandoff commitBatch(
            BatchReservation lease,
            List<ScheduledRequest> items,
            long predictedMs) {
        prefillState.lock.lock();
        try {
            return commitBatchUnderLock(lease, items, predictedMs);
        } finally {
            prefillState.lock.unlock();
        }
    }

    private PrefillState.CommittedHandoff commitBatchUnderLock(BatchReservation lease,
                                                               List<ScheduledRequest> items,
                                                               long predictedMs) {
        prefillState.requireLock();
        List<ScheduledRequest> members = prefillState.validateActiveGroup(items);
        RequestEntry head = prefillState.requests.get(lease.headRequestId);
        ScheduledRequest headItem = head == null ? null : head.activeItem;
        if (lease.state != LeaseState.OPEN || head == null || head.reservation != lease || !members.contains(headItem)
                || openGenerationHandoff(lease) == null) {
            throw new IllegalStateException("batch commit does not own exact OPEN lease batch_id=" + lease.batchId);
        }
        if (findBatchProgressUnderLock(lease.batchId) != null) {
            throw new IllegalStateException("batch id already committed batch_id=" + lease.batchId);
        }
        for (ScheduledRequest item : members) {
            RequestEntry member = prefillState.requests.get(item.requestId());
            Reservation expected = item == headItem ? lease : null;
            if (member.reservation != expected) {
                throw new IllegalStateException(
                        "batch member owns another exact reservation request_id=" + item.requestId());
            }
        }
        long nowMs = prefillState.clock.getAsLong();
        BatchProgress work = new BatchProgress(lease, predictedMs,
                PrefillBatchFeatures.from(members, ScheduledRequest::seqLen, ScheduledRequest::hitCache), nowMs);
        PrefillState.CommittedHandoff committedHandoff = new PrefillState.CommittedHandoff(openGenerationHandoff(lease),
                prefillState.capturePrecedingWorkUnderLock(members, nowMs));
        for (ScheduledRequest item : members) {
            prefillState.removeValidatedActiveIndex(item);
        }
        moveGenerationHandoffToOwnedUnderLock(lease, committedHandoff);
        for (ScheduledRequest item : members) {
            prefillState.requests.get(item.requestId()).commitBatch(work);
        }
        prefillState.committedWorkCapture = null;
        prefillState.recordMutationUnderLock();
        return committedHandoff;
    }

    public void prepareBatchPredictionsUnderLock(Set<LiveBatchMembers> changedBatches,
                                          Map<String, TerminalObservation> terminals,
                                          IdentityHashMap<BatchProgress, Phase> batchPhases,
                                          ToLongFunction<List<ScheduledRequest>> repredictor,
                                          IdentityHashMap<BatchProgress, Long> predictions) {
        prefillState.requireLock();
        for (LiveBatchMembers liveMembers : changedBatches) {
            BatchProgress batch = liveMembers.batch();
            if (batch.engineStarted() || batchPhases.get(batch) == Phase.ENGINE_RUNNING) {
                continue;
            }
            List<ScheduledRequest> survivors = new ArrayList<>(liveMembers.members().size());
            boolean executionObserved = false;
            for (RequestEntry member : liveMembers.members()) {
                TerminalObservation terminal = terminals.get(member.requestId);
                if (terminal == null) {
                    survivors.add(member.committedItem);
                } else {
                    // Completion can arrive before the first RUNNING observation.
                    executionObserved |= terminal.errorCode() == 0L || terminal.executionTimeMs() > 0L;
                }
            }
            if (executionObserved || survivors.isEmpty()) {
                continue;
            }
            survivors.sort(
                    Comparator.comparingLong(ScheduledRequest::enqueueSeq).thenComparing(ScheduledRequest::requestId));
            long prediction = repredictor.applyAsLong(survivors);
            predictions.put(batch, prediction);
        }
    }

    public int evictExpiredBatches(long ttlMs, Predicate<String> retainForSchedulerCleanup) {
        int evicted = 0;
        boolean capacityReleased = false;
        prefillState.lock.lock();
        try {
            long nowMs = prefillState.clock.getAsLong();
            for (LiveBatchMembers liveMembers : liveBatchMembersUnderLock().values()) {
                boolean retained = nowMs - liveMembers.batch().lastObservedAtMs() < Math.max(0L, ttlMs);
                for (RequestEntry entry : liveMembers.members()) {
                    retained |= retainForSchedulerCleanup.test(entry.requestId);
                }
                if (retained) {
                    continue;
                }
                for (RequestEntry entry : List.copyOf(liveMembers.members())) {
                    capacityReleased |= prefillState.settleUnderLock(entry,
                            TerminalObservation.external(entry.requestId), liveMembers, new ArrayList<>(0));
                }
                evicted++;
            }
        } finally {
            prefillState.lock.unlock();
            prefillState.notifyCapacityAvailable(capacityReleased);
        }
        return evicted;
    }

    public static boolean matchesObservedBatch(RequestEntry entry, long observedBatchId) {
        if (entry.batchProgress == null) {
            return true;
        }
        return observedBatchId > 0L && entry.batchProgress.batchId() == observedBatchId;
    }

    public static Phase strongerEnginePhase(Phase left, Phase right) {
        return left == Phase.ENGINE_RUNNING || right == Phase.ENGINE_RUNNING
                ? Phase.ENGINE_RUNNING
                : Phase.ENGINE_QUEUED;
    }

    public IdentityHashMap<BatchProgress, LiveBatchMembers> liveBatchMembersUnderLock() {
        prefillState.requireLock();
        IdentityHashMap<BatchProgress, LiveBatchMembers> membersByBatch = new IdentityHashMap<>();
        for (RequestEntry entry : prefillState.requests.values()) {
            if (entry.batchProgress != null) {
                membersByBatch.computeIfAbsent(entry.batchProgress, LiveBatchMembers::new).add(entry);
            }
        }
        return membersByBatch;
    }

    /**
     * Build only the exact batch needed by a single-item terminal path.
     */
    public LiveBatchMembers liveBatchMembersUnderLock(BatchProgress exactBatch) {
        prefillState.requireLock();
        LiveBatchMembers liveMembers = new LiveBatchMembers(exactBatch);
        for (RequestEntry entry : prefillState.requests.values()) {
            if (entry.batchProgress == exactBatch) {
                liveMembers.add(entry);
            }
        }
        return liveMembers;
    }

    private BatchReservation findBatchReservationUnderLock(long batchId) {
        for (RequestEntry entry : prefillState.requests.values()) {
            if (entry.reservation instanceof BatchReservation reservation && reservation.batchId == batchId) {
                return reservation;
            }
            if (entry.batchProgress != null && entry.batchProgress.batchId() == batchId) {
                return entry.batchProgress.lease();
            }
        }
        return null;
    }

    private EndpointGenerationLifecycle.HandoffPermit openGenerationHandoff(BatchReservation lease) {
        prefillState.requireLock();
        return lease.generationHandoff;
    }

    /**
     * Move the exact handoff out of the OPEN admission before commit publishes.
     */
    private void moveGenerationHandoffToOwnedUnderLock(BatchReservation lease,
                                                       PrefillState.CommittedHandoff committedHandoff) {
        prefillState.requireLock();
        if (lease.state != LeaseState.OPEN || lease.generationHandoff == null
                || committedHandoff.generationHandoff != lease.generationHandoff) {
            throw new IllegalStateException("Prefill commit requires an exact OPEN generation handoff");
        }
        lease.generationHandoff = null;
        lease.state = LeaseState.OWNED;
    }

    private BatchProgress findBatchProgressUnderLock(long batchId) {
        for (RequestEntry entry : prefillState.requests.values()) {
            if (entry.batchProgress != null && entry.batchProgress.batchId() == batchId) {
                return entry.batchProgress;
            }
        }
        return null;
    }

    public void releaseBatchSlotUnderLock() {
        prefillState.requireLock();
        if (batchLeasesInUse <= 0) {
            throw new IllegalStateException("Prefill batch capacity accounting underflow");
        }
        batchLeasesInUse--;
    }

    public void resetBatchLeasesInUseUnderLock() {
        prefillState.requireLock();
        batchLeasesInUse = 0;
    }
}
