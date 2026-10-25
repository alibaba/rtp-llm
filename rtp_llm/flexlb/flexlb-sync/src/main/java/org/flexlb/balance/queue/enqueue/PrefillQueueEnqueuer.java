package org.flexlb.balance.queue.enqueue;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.ReservationResult;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
import org.flexlb.balance.queue.QueueFailures;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.balance.queue.QueuedRequestRemover;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.ScheduledRequest;
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
public final class PrefillQueueEnqueuer {

    private final QueueState queueState;
    private final PrefillState prefillState;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueCapacity queueCapacity;
    private final EndpointEventHandler endpointEvents;
    private final FlexlbConfig config;

    public PrefillQueueEnqueuer(QueueState queueState,
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

    public PrefillQueueEnqueueResult enqueue(ScheduledRequest item) {
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
                    // 共享排队名额已满：只能替换掉这台 Prefill 上一个还在排队的请求。
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
                // BATCH 模式会统计排队请求数，但不受共享排队名额限制。
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
                failure = QueueFailures.appendFailure(failure, rollbackFailure);
            }
        }
        if (failure != null) {
            throw QueueFailures.propagateCommitFailure(failure);
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
        List<SingleReservation> reservations = new ArrayList<>(victims.size());
        SingleReservation replacement = null;
        try {
            for (ScheduledRequest victim : victims) {
                SingleReservation reservation = victim.takePublishedSingleReservation();
                if (reservation == null) {
                    return new EnqueueAttempt(false, null, null);
                }
                reservations.add(reservation);
            }
            replacement = prefillState.replaceQueuedRequests(victims, reservations, incoming);
            if (replacement == null) {
                return new EnqueueAttempt(false, null, null);
            }
            if (!incoming.bindPublishedSingleReservation(replacement)) {
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
                    QueuedRequestRemover.restoreSingleReservation(victims.get(index), reservations.get(index));
                }
            }
        }
    }

    /**
     * 调用方持有队列锁。
     */
    private EnqueueAttempt enqueueUnderLock(ScheduledRequest item) {
        if (queueState.isStopped()) {
            return new EnqueueAttempt(false, null, null);
        }
        if (!publishActiveIndexUnderLock(item)) {
            return new EnqueueAttempt(false, null, null);
        }
        if (!item.requiresSingleReservation()) {
            item.markSubQueueEntry();
            queueState.signalChange();
            return new EnqueueAttempt(true, null, null);
        }

        ReservationResult<SingleReservation> result;
        try {
            // 队列锁可重入，同时也是 PrefillState 的锁，所以登记进排队索引和预留 Prefill 资源
            // 在同一把锁里完成，其他线程看不到只做了一半的状态。
            result = prefillEndpoint.reserveSingle(
                    item,
                    0L);
        } catch (Throwable failure) {
            rollbackFreshActiveUnderLock(item, null);
            return new EnqueueAttempt(false, null, failure);
        }
        if (!result.isSuccess()) {
            rollbackFreshActiveUnderLock(item, null);
            return new EnqueueAttempt(false, null, null);
        }

        SingleReservation reservation = result.reservation();
        if (!item.bindPublishedSingleReservation(reservation)) {
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

    /**
     * 撤回刚登记进排队索引的请求。这个请求还没在队列锁外被其他线程看到。
     */
    private void rollbackFreshActiveUnderLock(
            ScheduledRequest item,
            SingleReservation reservation) {
        boolean removed = reservation == null
                ? prefillState.removeQueuedRequest(item)
                : prefillState.removeQueuedSingleRequest(
                        item, reservation);
        if (!removed) {
            throw new IllegalStateException(
                    "fresh ACTIVE publication could not roll back: request_id="
                            + item.requestId());
        }
        queueState.incrementQueueVersion();
    }

    /**
     * 调用方持有队列锁。
     */
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
