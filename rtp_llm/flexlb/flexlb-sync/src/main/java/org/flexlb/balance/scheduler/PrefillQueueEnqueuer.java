package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.enums.PrefillQueueEnqueueResult;

import java.util.ArrayList;
import java.util.List;

/**
 * 把请求放进一台 Prefill 的排队队列。
 * <p>
 * 入队时先占引擎排队名额，再登记到队列并预留 Prefill 资源；任一步失败都撤回已做的部分。
 * 队列满时，可以挤掉一个优先级更低的排队请求，让新请求顶替它的位置和预留。
 */
final class PrefillQueueEnqueuer {

    private final QueueState queueState;
    private final PrefillState prefillState;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueCapacity queueCapacity;
    private final EndpointEventHandler endpointEvents;
    private final FlexlbConfig config;

    PrefillQueueEnqueuer(QueueState queueState,
                         PrefillState prefillState,
                         PrefillEndpoint prefillEndpoint,
                         QueueCapacity queueCapacity,
                         EndpointEventHandler endpointEvents,
                         FlexlbConfig config) {
        this.queueState = queueState;
        this.prefillState = prefillState;
        this.prefillEndpoint = prefillEndpoint;
        this.queueCapacity = queueCapacity;
        this.endpointEvents = endpointEvents;
        this.config = config;
    }

    PrefillQueueEnqueueResult enqueue(ScheduledRequest item) {
        EnqueueAttempt attempt;
        List<ScheduledRequest> victims = List.of();
        boolean subQueueCountAdded = false;
        queueState.lock();
        try {
            if (prefillEndpoint.isEngineSubQueueMode()) {
                subQueueCountAdded = prefillEndpoint.tryReserveEngineSubQueueSlot(item.ctx().getConfig());
                if (subQueueCountAdded) {
                    attempt = enqueueUnderLock(item);
                } else {
                    // Shared queue full: replace only a waiting request on this endpoint.
                    attempt = new EnqueueAttempt(false, null, null);
                    if (!queueState.isStopped() && config.allowsPreemption(VictimStage.PREFILL_QUEUED)) {
                        ScheduledRequest victim = prefillState.oneQueuedPreemptionVictimUnderLock(item.priority());
                        if (victim != null) {
                            victims = List.of(victim);
                            attempt = replaceQueuedRequestsUnderLock(item, victims);
                        }
                    }
                }
            } else {
                // BATCH counts waiting requests but does not apply the shared limit.
                prefillEndpoint.increaseEngineSubQueueCount();
                subQueueCountAdded = true;
                attempt = enqueueUnderLock(item);
                if (!attempt.accepted() && attempt.failure() == null && !queueState.isStopped()
                        && config.allowsPreemption(VictimStage.PREFILL_QUEUED)) {
                    long maxOutstandingRequests = maxOutstandingRequests(item);
                    victims = prefillState.queuedPreemptionVictimsUnderLock(item.priority(), maxOutstandingRequests);
                    if (!victims.isEmpty()) {
                        attempt = replaceQueuedRequestsUnderLock(item, victims);
                    }
                }
            }
            queueCapacity.releaseEngineSubQueueSlot(prefillEndpoint.getStatus().getRole(),
                                                    subQueueCountAdded && !attempt.accepted());
        } catch (RuntimeException | Error failure) {
            queueCapacity.releaseEngineSubQueueSlot(prefillEndpoint.getStatus().getRole(), subQueueCountAdded);
            throw failure;
        } finally {
            queueState.unlock();
        }
        Throwable failure = attempt.failure();
        if (attempt.rollback() != null) {
            try {
                attempt.rollback().close();
            } catch (Throwable rollbackFailure) {
                failure = GroupCommitter.appendFailure(failure, rollbackFailure);
            }
        }
        if (failure != null) {
            throw GroupCommitter.propagateCommitFailure(failure);
        }
        if (attempt.accepted()) {
            for (ScheduledRequest victim : victims) {
                prefillEndpoint.reportSubQueueResidence(victim);
                endpointEvents.onQueuedItemPreempted(victim, item);
            }
        }
        if (attempt.accepted()) {
            return PrefillQueueEnqueueResult.ACCEPTED;
        }
        if (!subQueueCountAdded) {
            prefillEndpoint.reportSharedSubQueueBlocked();
            return PrefillQueueEnqueueResult.ENGINE_SUBQUEUE_LIMIT_REACHED;
        }
        return PrefillQueueEnqueueResult.REJECTED;
    }

    private EnqueueAttempt replaceQueuedRequestsUnderLock(ScheduledRequest incoming,
                                                          List<ScheduledRequest> victims) {
        List<PrefillState.RouteReservation> reservations = new ArrayList<>(victims.size());
        PrefillState.RouteReservation replacement = null;
        try {
            for (ScheduledRequest victim : victims) {
                PrefillState.RouteReservation reservation = victim.takePublishedRouteReservation();
                if (reservation == null) {
                    return new EnqueueAttempt(false, null, null);
                }
                reservations.add(reservation);
            }
            replacement = prefillState.replaceQueuedRoutesUnderLock(victims, reservations, incoming);
            if (replacement == null) {
                return new EnqueueAttempt(false, null, null);
            }
            if (!incoming.bindPublishedRouteReservation(replacement)) {
                throw new IllegalStateException("replacement request already owns a Prefill reservation: "
                        + incoming.requestId());
            }
            incoming.markSubQueueEntry();
            queueState.incrementQueueVersion();
            queueState.signalChange();
            return new EnqueueAttempt(true, null, null);
        } finally {
            if (replacement == null) {
                for (int index = 0; index < reservations.size(); index++) {
                    QueuedRequestRemover.restoreRouteReservation(victims.get(index), reservations.get(index));
                }
            }
        }
    }

    /** Caller holds the queue lock. */
    private EnqueueAttempt enqueueUnderLock(ScheduledRequest item) {
        if (queueState.isStopped()) {
            return new EnqueueAttempt(false, null, null);
        }
        if (!publishActiveIndexUnderLock(item)) {
            return new EnqueueAttempt(false, null, null);
        }
        if (!item.requiresRouteReservation()) {
            item.markSubQueueEntry();
            queueState.signalChange();
            return new EnqueueAttempt(true, null, null);
        }

        PrefillState.ReservationResult<PrefillState.RouteReservation> result;
        try {
            // The queue lock is reentrant and is also PrefillState's ownership lock,
            // so ACTIVE publication and exact route-ownership acquisition are
            // one endpoint-local transaction.
            result = prefillEndpoint.reserveRouteOwnership(
                    item,
                    0L);
        } catch (Throwable failure) {
            rollbackFreshActiveUnderLock(item, null);
            return new EnqueueAttempt(false, null, failure);
        }
        if (result.status() != PrefillState.CapacityStatus.ACQUIRED) {
            rollbackFreshActiveUnderLock(item, null);
            return new EnqueueAttempt(false, null, null);
        }

        PrefillState.RouteReservation reservation = result.reservation();
        if (!item.bindPublishedRouteReservation(reservation)) {
            rollbackFreshActiveUnderLock(item, reservation);
            return new EnqueueAttempt(
                    false,
                    null,
                    new IllegalStateException(
                            "ACTIVE request already owns a route reservation: request_id="
                                    + item.requestId()));
        }
        item.markSubQueueEntry();
        queueState.signalChange();
        return new EnqueueAttempt(true, null, null);
    }

    /** Undo an ACTIVE publication which never escaped the queue lock. */
    private void rollbackFreshActiveUnderLock(
            ScheduledRequest item,
            PrefillState.RouteReservation reservation) {
        boolean removed = reservation == null
                ? prefillState.terminalizeActiveUnderLock(item)
                : prefillState.terminalizeActiveRouteUnderLock(
                        item, reservation);
        if (!removed) {
            throw new IllegalStateException(
                    "fresh ACTIVE publication could not roll back: request_id="
                            + item.requestId());
        }
        queueState.incrementQueueVersion();
    }

    /** Caller holds the queue lock. */
    private boolean publishActiveIndexUnderLock(ScheduledRequest item) {
        boolean published = prefillState.enqueueActiveUnderLock(item, 0L);
        if (published) {
            queueState.incrementQueueVersion();
        }
        return published;
    }

    private static long maxOutstandingRequests(ScheduledRequest item) {
        DispatcherConfig dispatcher = item.ctx().getConfig().getDispatcher();
        return dispatcher.getType() == DispatcherConfig.Type.BATCH
                ? 0L : dispatcher.getMaxInflightPerPrefillWorker();
    }
}
