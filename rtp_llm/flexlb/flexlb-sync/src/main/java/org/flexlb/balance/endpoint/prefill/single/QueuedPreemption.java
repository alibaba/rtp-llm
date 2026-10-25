package org.flexlb.balance.endpoint.prefill.single;

import org.flexlb.balance.endpoint.prefill.LeaseState;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.RequestEntry;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 按优先级抢占 Prefill 上还在排队的 NON_BATCH 请求：挑出优先级更低、预占还没提交的排队请求，
 * 用新来的高优先级请求替换它们。请求表和锁仍归 PrefillState，这里的方法和 PrefillState 共用同一把锁。
 */
public final class QueuedPreemption {
    private final PrefillState prefillState;

    public QueuedPreemption(PrefillState prefillState) {
        this.prefillState = prefillState;
    }

    public boolean canPreemptQueuedRequest(int priority, long requestLimit) {
        prefillState.lock.lock();
        try {
            long required = requestSlotsToRelease(requestLimit);
            if (!PriorityNormalizer.hasPriority(priority) || required == 0L
                    || required > prefillState.activeIndex.size()) {
                return false;
            }
            for (ScheduledRequest item : prefillState.activeIndex) {
                if (isQueuedPreemptionCandidate(item, priority) && --required == 0L) {
                    return true;
                }
            }
            return false;
        } finally {
            prefillState.lock.unlock();
        }
    }

    /**
     * Select only uncommitted requests; Engine work cannot release a local queue seat.
     */
    public List<ScheduledRequest> queuedPreemptionVictims(int priority, long requestLimit) {
        prefillState.requireLock();
        return findQueuedPreemptionVictims(priority, requestSlotsToRelease(requestLimit));
    }

    public ScheduledRequest oneQueuedPreemptionVictim(int priority) {
        prefillState.requireLock();
        List<ScheduledRequest> victims = findQueuedPreemptionVictims(priority, 1L);
        return victims.isEmpty() ? null : victims.getFirst();
    }

    private List<ScheduledRequest> findQueuedPreemptionVictims(int priority, long victimsNeeded) {
        prefillState.requireLock();
        if (!PriorityNormalizer.hasPriority(priority) || victimsNeeded == 0L
                || victimsNeeded > prefillState.activeIndex.size()) {
            return List.of();
        }
        List<ScheduledRequest> candidates = new ArrayList<>();
        for (ScheduledRequest item : prefillState.activeIndex) {
            if (isQueuedPreemptionCandidate(item, priority)) {
                candidates.add(item);
            }
        }
        if (candidates.size() < victimsNeeded) {
            return List.of();
        }
        candidates.sort(Comparator.comparingInt(ScheduledRequest::priority)
                .thenComparing(Comparator.comparingLong(ScheduledRequest::enqueueSeq).reversed()));
        return candidates.subList(0, (int) victimsNeeded);
    }

    private boolean isQueuedPreemptionCandidate(ScheduledRequest item, int priority) {
        prefillState.requireLock();
        RequestEntry entry = prefillState.requests.get(item.requestId());
        return PriorityNormalizer.hasPriority(item.priority()) && item.priority() < priority
                && !item.future().isDone() && entry != null && entry.activeIdentity(item)
                && entry.reservation instanceof SingleReservation && entry.reservation.state == LeaseState.OPEN;
    }

    private long requestSlotsToRelease(long requestLimit) {
        prefillState.requireLock();
        return requestLimit <= 0L ? 0L
                : Math.max(0L, PrefillState.saturatedAdd(prefillState.requests.size(),
                        prefillState.unknownEngineRequestCount) - requestLimit + 1L);
    }

    /**
     * Replace the selected waiting requests and their single reservations in one transaction.
     */
    public SingleReservation replaceQueuedRequests(List<ScheduledRequest> victims,
                                                   List<SingleReservation> reservations,
                                                   ScheduledRequest incoming) {
        prefillState.requireLock();
        if (victims.isEmpty() || victims.size() != reservations.size()
                || prefillState.requests.containsKey(incoming.requestId())) {
            return null;
        }
        Set<String> ids = new HashSet<>();
        for (int index = 0; index < victims.size(); index++) {
            ScheduledRequest victim = victims.get(index);
            RequestEntry entry = prefillState.requests.get(victim.requestId());
            SingleReservation reservation = reservations.get(index);
            if (!ids.add(victim.requestId()) || entry == null || !entry.activeIdentity(victim)
                    || !prefillState.activeIndex.contains(victim) || reservation == null
                    || reservation.owner != prefillState
                    || entry.reservation != reservation || reservation.state != LeaseState.OPEN
                    || !PriorityNormalizer.hasPriority(victim.priority()) || victim.priority() >= incoming.priority()) {
                return null;
            }
        }
        RequestEntry incomingEntry = new RequestEntry(incoming);
        SingleReservation incomingReservation = new SingleReservation(
                prefillState, incomingEntry, incoming.requestId(), 0L);
        incomingEntry.reservation = incomingReservation;
        prefillState.activeIndex.add(incoming);
        try {
            prefillState.putRequest(incoming.requestId(), incomingEntry);
        } catch (RuntimeException | Error failure) {
            prefillState.activeIndex.remove(incoming);
            throw failure;
        }
        for (int index = 0; index < victims.size(); index++) {
            ScheduledRequest victim = victims.get(index);
            RequestEntry entry = prefillState.requests.get(victim.requestId());
            prefillState.closeOpenLease(reservations.get(index));
            prefillState.activeIndex.remove(victim);
            prefillState.removeRequest(victim.requestId(), entry);
        }
        prefillState.recordMutation();
        return incomingReservation;
    }
}
