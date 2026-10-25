package org.flexlb.balance.endpoint.prefill.single;

import org.flexlb.balance.endpoint.EndpointGenerationLifecycle;
import org.flexlb.balance.endpoint.prefill.LeaseState;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.RequestEntry;
import org.flexlb.balance.endpoint.prefill.ReservationResult;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * Prefill 上只给 NON_BATCH 用的部分：单个请求的预占和整组提交。
 * 请求表和锁仍归 PrefillState，这里的方法和 PrefillState 共用同一把锁。
 */
public final class SinglePrefill {
    private final PrefillState prefillState;

    public SinglePrefill(PrefillState prefillState) {
        this.prefillState = prefillState;
    }

    public ReservationResult<SingleReservation> reserveSingle(ScheduledRequest exactItem, long predictedMs) {
        ScheduledRequest item = exactItem;
        prefillState.lock.lock();
        try {
            RequestEntry entry = prefillState.requests.get(item.requestId());
            if (entry == null || !entry.activeIdentity(item)) {
                return ReservationResult.fail(ReservationResult.Status.REQUEST_LEFT_QUEUE);
            }
            if (entry.reservation != null) {
                return ReservationResult.fail(ReservationResult.Status.REQUEST_ALREADY_RESERVED);
            }
            SingleReservation lease = new SingleReservation(
                    prefillState, entry, item.requestId(), predictedMs);
            entry.reservation = lease;
            prefillState.recordMutationUnderLock();
            return ReservationResult.success(lease);
        } finally {
            prefillState.lock.unlock();
        }
    }

    public PrefillState.CommittedHandoff commitSingleGroup(
            List<ScheduledRequest> items,
            List<SingleReservation> exactReservations,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        if (exactReservations.isEmpty()) {
            throw new IllegalArgumentException(
                    "single commit requires at least one reservation");
        }
        List<SingleReservation> leases = new ArrayList<>(
                exactReservations.size());
        for (SingleReservation reservation : exactReservations) {
            if (reservation == null || reservation.owner != prefillState) {
                throw new IllegalArgumentException(
                        "single reservation belongs to another Prefill ledger");
            }
            leases.add(reservation);
        }
        prefillState.lock.lock();
        try {
            return commitSinglesUnderLock(items, leases, generationHandoff);
        } finally {
            prefillState.lock.unlock();
        }
    }

    private PrefillState.CommittedHandoff commitSinglesUnderLock(
            List<ScheduledRequest> items,
            List<SingleReservation> leases,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        prefillState.requireLock();
        List<ScheduledRequest> members = validateSingleGroup(items);
        if (members.size() != leases.size()) {
            throw new IllegalArgumentException(
                    "single commit requires one exact lease per member");
        }
        Set<SingleReservation> unique = Collections.newSetFromMap(
                new IdentityHashMap<>());
        for (int index = 0; index < members.size(); index++) {
            RequestEntry entry = prefillState.requests.get(members.get(index).requestId());
            SingleReservation lease = leases.get(index);
            if (entry == null || !unique.add(lease)
                    || entry.reservation != lease
                    || lease.requestId != entry.requestId
                    || lease.state != LeaseState.OPEN) {
                throw new IllegalStateException(
                        "single commit does not own exact OPEN lease request_id="
                                + members.get(index).requestId());
            }
        }
        long nowMs = prefillState.clock.getAsLong();
        PrefillState.CommittedHandoff committedHandoff = new PrefillState.CommittedHandoff(generationHandoff,
                prefillState.capturePrecedingWorkUnderLock(members, nowMs));
        for (ScheduledRequest item : members) {
            prefillState.detachAdmissionIndexUnderLock(prefillState.requests.get(item.requestId()), item);
        }
        for (int index = 0; index < members.size(); index++) {
            SingleReservation lease = leases.get(index);
            lease.state = LeaseState.OWNED;
            RequestEntry entry = prefillState.requests.get(members.get(index).requestId());
            entry.commitIndividual(lease, nowMs);
        }
        prefillState.committedWorkCapture = null;
        prefillState.recordMutationUnderLock();
        return committedHandoff;
    }

    private List<ScheduledRequest> validateSingleGroup(List<ScheduledRequest> items) {
        PrefillState.requireState(!items.isEmpty(), "committed group requires members");
        for (ScheduledRequest item : items) {
            RequestEntry entry = prefillState.requests.get(item.requestId());
            PrefillState.requireState(entry != null && entry.activeIdentity(item),
                    "single member is not canonical admission request_id=" + item.requestId());
            prefillState.validateAdmissionIndexUnderLock(entry, item);
        }
        return items;
    }

    /**
     * Remove one NON_BATCH ACTIVE owner and its item-owned OPEN single reservation
     * in the same ownership transaction.
     */
    public boolean removeQueuedSingleRequest(
            ScheduledRequest item,
            SingleReservation exactReservation) {
        prefillState.requireLock();
        RequestEntry entry = prefillState.requests.get(item.requestId());
        if (entry == null || !entry.activeIdentity(item)) {
            return false;
        }
        if (exactReservation == null
                || exactReservation.owner != prefillState
                || entry.reservation != exactReservation
                || exactReservation.state != LeaseState.OPEN) {
            throw new IllegalStateException(
                    "ACTIVE single request lost its exact Prefill reservation request_id="
                            + item.requestId());
        }
        prefillState.validateAdmissionIndexUnderLock(entry, item);

        // Every operation below is allocation-free and runs under the same
        // lock used by generation retirement.
        prefillState.detachAdmissionIndexUnderLock(entry, item);
        prefillState.closeOpenLeaseUnderLock(exactReservation);
        prefillState.removeRequestUnderLock(item.requestId(), entry);
        prefillState.recordMutationUnderLock();
        return true;
    }
}
