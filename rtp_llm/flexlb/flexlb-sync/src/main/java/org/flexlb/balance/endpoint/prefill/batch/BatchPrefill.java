package org.flexlb.balance.endpoint.prefill.batch;

import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.endpoint.EndpointGenerationLifecycle;
import org.flexlb.balance.endpoint.prefill.CommittedHandoff;
import org.flexlb.balance.endpoint.prefill.FinishReport;
import org.flexlb.balance.endpoint.prefill.LeaseState;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.RequestEntry;
import org.flexlb.balance.endpoint.prefill.Reservation;
import org.flexlb.balance.endpoint.prefill.ReservationResult;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.estimate.WorkView.Phase;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
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
            if (findBatchReservation(batchId) != null) {
                return ReservationResult.fail(ReservationResult.Status.BATCH_ID_ALREADY_RESERVED);
            }
            if (batchLeasesInUse >= maximum) {
                return ReservationResult.fail(ReservationResult.Status.CAPACITY_FULL);
            }
            BatchReservation lease =
                    new BatchReservation(prefillState, entry, head.requestId(), batchId, generationHandoff);
            entry.reservation = lease;
            batchLeasesInUse++;
            prefillState.recordMutation();
            return ReservationResult.success(lease);
        } finally {
            prefillState.lock.unlock();
        }
    }

    public int batchLeasesInUse() {
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

    public CommittedHandoff commitBatch(BatchReservation lease, List<ScheduledRequest> items, long predictedMs) {
        prefillState.lock.lock();
        try {
            List<ScheduledRequest> members = prefillState.validateActiveGroup(items);
            RequestEntry head = prefillState.requests.get(lease.headRequestId);
            ScheduledRequest headItem = head == null ? null : head.activeItem;
            if (lease.state != LeaseState.OPEN || head == null || head.reservation != lease
                    || !members.contains(headItem) || openGenerationHandoff(lease) == null) {
                throw new IllegalStateException("batch commit does not own exact OPEN lease batch_id=" + lease.batchId);
            }
            if (findBatchProgress(lease.batchId) != null) {
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
            CommittedHandoff committedHandoff = new CommittedHandoff(openGenerationHandoff(lease),
                    prefillState.capturePrecedingWork(members, nowMs));
            for (ScheduledRequest item : members) {
                prefillState.removeValidatedActiveIndex(item);
            }
            moveGenerationHandoffToOwned(lease, committedHandoff);
            for (ScheduledRequest item : members) {
                prefillState.requests.get(item.requestId()).commitBatch(work);
            }
            prefillState.committedWorkCapture = null;
            prefillState.recordMutation();
            return committedHandoff;
        } finally {
            prefillState.lock.unlock();
        }
    }

    public void prepareBatchPredictions(Set<LiveBatchMembers> changedBatches,
                                        Map<String, FinishReport> finishReports,
                                        IdentityHashMap<BatchProgress, Phase> batchPhases,
                                        ToLongFunction<List<ScheduledRequest>> predictBatchTime,
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
                FinishReport finishReport = finishReports.get(member.requestId);
                if (finishReport == null) {
                    survivors.add(member.committedItem);
                } else {
                    // Completion can arrive before the first RUNNING observation.
                    executionObserved |= finishReport.errorCode() == 0L || finishReport.executionTimeMs() > 0L;
                }
            }
            if (executionObserved || survivors.isEmpty()) {
                continue;
            }
            survivors.sort(
                    Comparator.comparingLong(ScheduledRequest::enqueueSeq).thenComparing(ScheduledRequest::requestId));
            long prediction = predictBatchTime.applyAsLong(survivors);
            predictions.put(batch, prediction);
        }
    }

    public int evictExpiredBatches(long ttlMs, Predicate<String> retainForSchedulerCleanup) {
        int evicted = 0;
        boolean capacityReleased = false;
        prefillState.lock.lock();
        try {
            long nowMs = prefillState.clock.getAsLong();
            for (LiveBatchMembers liveMembers : liveBatchMembers().values()) {
                boolean retained = nowMs - liveMembers.batch().lastObservedAtMs() < Math.max(0L, ttlMs);
                for (RequestEntry entry : liveMembers.members()) {
                    retained |= retainForSchedulerCleanup.test(entry.requestId);
                }
                if (retained) {
                    continue;
                }
                for (RequestEntry entry : List.copyOf(liveMembers.members())) {
                    capacityReleased |= prefillState.removeCommittedEntry(entry, FinishReport.local(entry.requestId),
                            liveMembers, new ArrayList<>(0));
                }
                evicted++;
            }
        } finally {
            prefillState.lock.unlock();
            prefillState.notifyCapacityAvailable(capacityReleased);
        }
        return evicted;
    }

    /**
     * 找出这次上报里有成员完成的批次，后面要重新预测它们剩余的耗时。
     */
    public static Set<LiveBatchMembers> collectChangedBatches(
            Collection<RequestEntry> finishedEntries,
            IdentityHashMap<BatchProgress, LiveBatchMembers> membersByBatch) {
        Set<LiveBatchMembers> changedBatches = Collections.newSetFromMap(new IdentityHashMap<>());
        for (RequestEntry entry : finishedEntries) {
            LiveBatchMembers liveMembers = entry.batchProgress == null ? null : membersByBatch.get(entry.batchProgress);
            if (liveMembers != null) {
                changedBatches.add(liveMembers);
            }
        }
        return changedBatches;
    }

    /**
     * 找出这次上报后所有成员都已完成的批次。
     */
    public static List<FinishedBatch> collectFinishedBatches(
            IdentityHashMap<BatchProgress, LiveBatchMembers> membersByBatch,
            Map<String, FinishReport> finishReports) {
        List<FinishedBatch> finishedBatches = new ArrayList<>(Math.min(finishReports.size(), membersByBatch.size()));
        for (LiveBatchMembers liveMembers : membersByBatch.values()) {
            FinishedBatch finishedBatch = liveMembers.finishedBatchIfAllFinish(finishReports);
            if (finishedBatch != null) {
                finishedBatches.add(finishedBatch);
            }
        }
        return finishedBatches;
    }

    /**
     * 判断这个请求是不是批次里最后一个还没结束的成员；最后一个成员结束时要关闭批次预占。
     */
    public static boolean isLastLiveMember(RequestEntry entry, LiveBatchMembers liveMembers) {
        if (liveMembers == null || liveMembers.batch() != entry.batchProgress
                || !liveMembers.members().contains(entry)) {
            throw new IllegalStateException("missing live batch members for request_id=" + entry.requestId);
        }
        return liveMembers.members().size() == 1;
    }

    /**
     * 把结束的成员从批次里拿掉，并按它的结束结果更新批次进度。
     */
    public static void removeFinishedMember(RequestEntry entry,
                                            LiveBatchMembers liveMembers,
                                            FinishReport finishReport,
                                            long nowMs) {
        entry.batchProgress.updateProgressFromFinishedTask(finishReport.reportedByWorker(), finishReport.errorCode(),
                finishReport.executionTimeMs(), nowMs);
        liveMembers.remove(entry);
    }

    /**
     * 检查 Prefill 下线时还没关闭的 batch 预占。正常情况下这时只剩已提交的预占；
     * 还没提交的预占说明有人漏了回滚，把它的 generation handoff 交给调用方关闭，并记一条错误。
     */
    public static Throwable checkRetiringBatchLease(BatchReservation batch,
                                                    List<EndpointGenerationLifecycle.HandoffPermit> orphanedHandoffs,
                                                    Throwable invariantFailure) {
        if (batch.state == LeaseState.OPEN) {
            if (batch.generationHandoff == null) {
                return PrefillState.appendRetirementInvariant(invariantFailure,
                        "OPEN Prefill batch lease lost its generation handoff");
            }
            orphanedHandoffs.add(batch.generationHandoff);
            return PrefillState.appendRetirementInvariant(invariantFailure,
                    "retirement reached an OPEN Prefill batch lease");
        }
        if (batch.state != LeaseState.OWNED || batch.generationHandoff != null) {
            return PrefillState.appendRetirementInvariant(invariantFailure,
                    "canonical retirement owner has an invalid batch lease state");
        }
        return invariantFailure;
    }

    /**
     * 这些批次里最久没有收到引擎状态的那个，距今多少毫秒。
     */
    public static long maxBatchAgeMs(Set<BatchProgress> batches, long nowMs) {
        long maxAgeMs = 0L;
        for (BatchProgress batch : batches) {
            maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - batch.lastObservedAtMs()));
        }
        return maxAgeMs;
    }

    /**
     * 为每个批次生成负载视图里的一条记录：成员、阶段和预计剩余耗时。
     */
    public static List<WorkView.BatchWork> buildBatchWork(IdentityHashMap<BatchProgress, List<String>> batchMembers,
                                                          long nowMs) {
        List<WorkView.BatchWork> batches = new ArrayList<>(batchMembers.size());
        for (Map.Entry<BatchProgress, List<String>> members : batchMembers.entrySet()) {
            BatchProgress batch = members.getKey();
            batches.add(new WorkView.BatchWork(batch.batchId(), members.getValue(), batch.phase(),
                    OptionalLong.of(batch.remainingWorkMsAt(nowMs))));
        }
        return batches;
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

    public IdentityHashMap<BatchProgress, LiveBatchMembers> liveBatchMembers() {
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
    public LiveBatchMembers liveBatchMembers(BatchProgress exactBatch) {
        prefillState.requireLock();
        LiveBatchMembers liveMembers = new LiveBatchMembers(exactBatch);
        for (RequestEntry entry : prefillState.requests.values()) {
            if (entry.batchProgress == exactBatch) {
                liveMembers.add(entry);
            }
        }
        return liveMembers;
    }

    private BatchReservation findBatchReservation(long batchId) {
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
    private void moveGenerationHandoffToOwned(BatchReservation lease, CommittedHandoff committedHandoff) {
        prefillState.requireLock();
        if (lease.state != LeaseState.OPEN || lease.generationHandoff == null
                || committedHandoff.generationHandoff != lease.generationHandoff) {
            throw new IllegalStateException("Prefill commit requires an exact OPEN generation handoff");
        }
        lease.generationHandoff = null;
        lease.state = LeaseState.OWNED;
    }

    private BatchProgress findBatchProgress(long batchId) {
        for (RequestEntry entry : prefillState.requests.values()) {
            if (entry.batchProgress != null && entry.batchProgress.batchId() == batchId) {
                return entry.batchProgress;
            }
        }
        return null;
    }

    public void releaseBatchSlot() {
        prefillState.requireLock();
        if (batchLeasesInUse <= 0) {
            throw new IllegalStateException("Prefill batch capacity accounting underflow");
        }
        batchLeasesInUse--;
    }

    public void resetBatchLeasesInUse() {
        prefillState.requireLock();
        batchLeasesInUse = 0;
    }
}
