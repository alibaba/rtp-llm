package org.flexlb.balance.delivery;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.prefill.CommittedHandoff;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.PrefillAdmissionResources;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

import static org.flexlb.balance.scheduler.PrefillAdmissionResources.accepted;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.createCommittedOwner;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.failed;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.prepareMember;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rejected;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rollbackMember;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rollbackReservation;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.sameIdentitySequence;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.throwRollbackFailure;

/**
 * NON_BATCH 模式的一次投递事务：按顺序持有本次请求的预留，并负责提交、交付和回滚。
 */
public final class NonBatchTransaction implements DeliveryTransaction {
    private final NonBatchDeliveryStrategy owner;
    private final PrefillEndpoint prefill;
    private List<ScheduledRequest> items = List.of();
    private BlockedRequest blocked;
    private ArrayList<SingleReservation> reservations;
    private ArrayList<Boolean> itemOwnedReservations;
    private final IdentityHashMap<ScheduledRequest, Long> predictions = new IdentityHashMap<>();
    private ArrayList<PrefillAdmissionResources.Member> members;
    private PrefillAdmissionResources.CommittedAdmissionOwner committed;

    NonBatchTransaction(NonBatchDeliveryStrategy owner, PrefillEndpoint prefill) {
        this.owner = owner;
        this.prefill = prefill;
        reservations = new ArrayList<>(1);
        itemOwnedReservations = new ArrayList<>(1);
        members = new ArrayList<>(1);
    }

    NonBatchTransaction(NonBatchDeliveryStrategy owner, BlockedRequest blocked) {
        this.owner = owner;
        this.prefill = null;
        this.blocked = blocked;
    }

    static NonBatchTransaction blocked(NonBatchDeliveryStrategy owner, BlockedRequest blocked) {
        return new NonBatchTransaction(owner, blocked);
    }

    public synchronized DeliveryBlockReason.Attempt<ScheduledRequest> append(ScheduledRequest exact,
                                                                             PrefillTimePredictor.Evaluator evaluator) {
        requirePrepared("append");
        if (prefill.isRetired()) {
            return failed(PrefillAdmissionResources.retired("Prefill", exact));
        }
        long predictedMs = NonBatchDeliveryStrategy.predict(evaluator, exact);
        final DeliveryBlockReason.Attempt<ScheduledRequest> acceptedItem;
        try {
            acceptedItem = accepted(exact);
            int capacity = Math.addExact(members.size(), 1);
            reservations.ensureCapacity(capacity);
            members.ensureCapacity(capacity);
        } catch (Throwable failure) {
            return failed(failure);
        }

        SingleReservation published = exact.publishedSingleReservation();
        try {
            if (published == null) {
                return failed(new IllegalStateException(
                        "ACTIVE NON_BATCH request lost its publish-time single reservation: request_id="
                                + exact.requestId()));
            }
            published.updatePrediction(exact, predictedMs);
        } catch (Throwable failure) {
            return failed(failure);
        }
        predictions.put(exact, predictedMs);
        SingleReservation reservation = published;
        final DeliveryBlockReason.Attempt<PrefillAdmissionResources.Member> memberAttempt;
        try {
            memberAttempt = prepareMember(exact);
        } catch (Throwable failure) {
            return failed(failure);
        }
        if (!memberAttempt.accepted()) {
            return rejected(memberAttempt.blockReason());
        }
        reservations.add(reservation);
        itemOwnedReservations.add(true);
        members.add(memberAttempt.value());
        return acceptedItem;
    }

    synchronized void select(List<ScheduledRequest> exactItems, BlockedRequest exactBlocked) {
        requirePrepared("select");
        items = exactItems;
        blocked = exactBlocked;
    }

    long predictedMs(ScheduledRequest item) {
        return predictions.get(item);
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
        if (!sameIdentitySequence(members, items)) {
            throw new IllegalArgumentException("route commit does not match prepared identities");
        }
        if (reservations.isEmpty() || reservations.size() != members.size()
                || itemOwnedReservations.size() != members.size()) {
            throw new IllegalStateException("route admission reservation/member ownership diverged");
        }
        // Validate every item-owned capability before transferring any of
        // them.  commit runs under the endpoint queue lock, so an
        // ACTIVE terminal path cannot remove a validated item between this
        // pass and the transfer pass below.
        for (int index = 0; index < reservations.size(); index++) {
            if (itemOwnedReservations.get(index)
                    && items.get(index).publishedSingleReservation() != reservations.get(index)) {
                throw new IllegalStateException(
                        "ACTIVE NON_BATCH request lost its exact single reservation: request_id="
                                + items.get(index).requestId());
            }
        }
        PrefillEndpoint.RouteCommitAdmission routeCommit = prefill.tryBeginRouteCommitAdmission();
        if (routeCommit == null) {
            throw PrefillAdmissionResources.retired("Prefill", items.get(0));
        }
        try (routeCommit) {
            // Allocate the committed owner before transferring any reservation
            // from its ACTIVE item. After the first successful CAS below,
            // the remainder of this transaction must be allocation-free.
            PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted = createCommittedOwner(members);
            for (int index = 0; index < reservations.size(); index++) {
                if (itemOwnedReservations.get(index)
                        && !items.get(index).takePublishedSingleReservation(reservations.get(index))) {
                    throw new IllegalStateException(
                            "ACTIVE NON_BATCH request lost its exact single reservation: request_id="
                                    + items.get(index).requestId());
                }
                if (itemOwnedReservations.get(index)) {
                    // The transaction is now the sole owner. If a later
                    // commit operation fails, rollbackPrepared closes it.
                    itemOwnedReservations.set(index, false);
                }
            }
            CommittedHandoff handoff = routeCommit.commit(items, reservations);
            exactCommitted.bindPrefillHandoff(handoff);
            committed = exactCommitted;
            reservations = null;
            itemOwnedReservations = null;
            members = null;
            return handoff.precedingWork();
        }
    }

    @Override
    public void deliver(String decisionReason, int remainingQueueDepth, WorkView precedingWork) {
        PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted = takeCommitted();
        owner.deliver(this, exactCommitted, remainingQueueDepth, precedingWork);
    }

    @Override
    public void rollbackCommitted(Throwable cause) {
        PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted;
        synchronized (this) {
            if (reservations == null && committed == null) {
                return;
            }
            if (committed == null) {
                throw new IllegalStateException("route group has not committed its admission");
            }
            exactCommitted = committed;
            committed = null;
        }
        Throwable cleanup = NonBatchDeliveryStrategy.close(exactCommitted);
        if (cleanup != null) {
            throw NonBatchDeliveryStrategy.propagate(cleanup);
        }
    }

    @Override
    public void close() {
        Throwable rollbackFailure = rollbackPrepared(null);
        if (rollbackFailure != null) {
            throwRollbackFailure(rollbackFailure);
        }
    }

    private synchronized PrefillAdmissionResources.CommittedAdmissionOwner takeCommitted() {
        PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted = committed;
        if (exactCommitted == null) {
            throw new IllegalStateException("route group no longer owns a committed admission");
        }
        committed = null;
        return exactCommitted;
    }

    private Throwable rollbackPrepared(Throwable priorFailure) {
        List<SingleReservation> exactReservations;
        List<Boolean> exactItemOwnedReservations;
        List<PrefillAdmissionResources.Member> exactMembers;
        synchronized (this) {
            if (reservations == null) {
                return priorFailure;
            }
            exactReservations = reservations;
            exactItemOwnedReservations = itemOwnedReservations;
            exactMembers = members;
            reservations = null;
            itemOwnedReservations = null;
            members = null;
        }
        Throwable failure = priorFailure;
        for (PrefillAdmissionResources.Member member : exactMembers) {
            failure = rollbackMember(member, failure);
        }
        for (int index = 0; index < exactReservations.size(); index++) {
            if (!exactItemOwnedReservations.get(index)) {
                failure = rollbackReservation(exactReservations.get(index), failure);
            }
        }
        return failure;
    }

    private void requirePrepared(String operation) {
        if (reservations == null || itemOwnedReservations == null || members == null) {
            throw new IllegalStateException(operation + " requires PREPARED route admission");
        }
    }
}
