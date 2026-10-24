package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.BlockedRequest;
import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.delivery.DeliveryGroup;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryTransaction;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.OptionalLong;
import java.util.function.Supplier;

/**
 * 把调度线程选中的一组请求从队列里取出并交付给 Prefill。
 * <p>
 * 先让投递策略按容量从选中的请求里留下能放下的前缀，再在队列锁内确认这些请求还在队列里、没超时，
 * 然后把它们移出队列、归还排队名额，放锁后交付。一个都放不下时，按原因决定：
 * 容量不足就让队首等容量，交付失败就把队首移出队列并通知请求调度器。
 * 队首超时也由这里移出队列。
 */
final class GroupCommitter {

    private final String key;
    private final QueueState queueState;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueCapacity queueCapacity;
    private final EndpointEventHandler endpointEvents;
    private final DeliveryStrategy deliveryStrategy;
    private final CapacityBlockedHead capacityBlockedHead;
    private final QueuedRequestRemover queuedRequestRemover;

    GroupCommitter(String key,
                   QueueState queueState,
                   PrefillEndpoint prefillEndpoint,
                   QueueCapacity queueCapacity,
                   EndpointEventHandler endpointEvents,
                   DeliveryStrategy deliveryStrategy,
                   CapacityBlockedHead capacityBlockedHead,
                   QueuedRequestRemover queuedRequestRemover) {
        this.key = key;
        this.queueState = queueState;
        this.prefillEndpoint = prefillEndpoint;
        this.queueCapacity = queueCapacity;
        this.endpointEvents = endpointEvents;
        this.deliveryStrategy = deliveryStrategy;
        this.capacityBlockedHead = capacityBlockedHead;
        this.queuedRequestRemover = queuedRequestRemover;
    }

    private static long now() {
        return System.currentTimeMillis();
    }

    ProcessQueueResult admitAndDeliverCapacityFeasiblePrefix(List<ScheduledRequest> candidates,
                                                                     String decisionReason,
                                                                     PrefillTimePredictor.Evaluator evaluator,
                                                                     OptionalLong plannedCommittedPredictionMs) {
        if (candidates.isEmpty() || !selectionStillOwned(candidates)) {
            return ProcessQueueResult.NO_ACTION;
        }
        try (DeliveryTransaction transaction =
                deliveryStrategy.prepare(new DeliveryGroup(candidates, evaluator, plannedCommittedPredictionMs))) {
            if (transaction.requests().isEmpty()) {
                return commitBoundary(transaction.blocked());
            }
            ProcessQueueResult admitted = commitPreparedSelection(
                    transaction, decisionReason);
            return admitted == null
                    ? ProcessQueueResult.NO_ACTION : admitted;
        } catch (EndpointRetiredException retired) {
            // The resources are closed, no handoff happened and the head is still queued: requeue it
            // instead of letting the exception stop the loop and fail every queued request.
            return commitBoundary(new BlockedRequest(
                    candidates.get(0), DeliveryBlockReason.failed(retired)));
        }
    }

    private void deliver(DeliveryTransaction transaction,
                         String decisionReason,
                         int remainingQueueDepth,
                         WorkView precedingWork) {
        Throwable deliveryFailure = null;
        try {
            String groupId = java.util.UUID.randomUUID().toString();
            long committedAtMs = System.currentTimeMillis();
            for (ScheduledRequest item : transaction.requests()) {
                var context = item.ctx();
                context.setDecisionGroup(new org.flexlb.dao.pv.DecisionGroup(groupId,
                        context.getConfig().decisionPolicy().getType().name(),
                        context.getConfig().getDispatcher().getType().name(),
                        item.prefillEp().getStatus().getMetricIpPort(), transaction.requests().size(), decisionReason,
                        committedAtMs, Math.max(0L, committedAtMs - item.enqueuedAtMs())));
            }
            transaction.deliver(decisionReason, remainingQueueDepth, precedingWork);
        } catch (Throwable failure) {
            deliveryFailure = failure;
        }
        Throwable unresolved = deliveryFailure != null
                ? deliveryFailure
                : new IllegalStateException(
                        "delivery returned without resolving owner");
        try {
            transaction.rollbackCommitted(unresolved);
        } catch (Throwable cleanupFailure) {
            if (deliveryFailure == null) {
                deliveryFailure = cleanupFailure;
            } else if (deliveryFailure != cleanupFailure) {
                deliveryFailure.addSuppressed(cleanupFailure);
            }
        }
        if (deliveryFailure != null) {
            Logger.error("PrefillQueue[{}] committed delivery failed",
                    key, deliveryFailure);
        }
    }

    ProcessQueueResult runPredictionBound(
            ScheduledRequest exactHead,
            Supplier<ProcessQueueResult> operation) {
        try {
            return operation.get();
        } catch (InvalidPrefillPredictionException failure) {
            return commitBoundary(new BlockedRequest(
                    exactHead, DeliveryBlockReason.failed(failure)));
        }
    }

    private boolean selectionStillOwned(
            List<ScheduledRequest> candidates) {
        queueState.lock();
        try {
            if (queueState.isStopped()) {
                return false;
            }
            long nowMs = now();
            for (ScheduledRequest item : candidates) {
                if (!containsIdentity(item)
                        || item.requestExpired(nowMs)) {
                    return false;
                }
            }
            return true;
        } finally {
            queueState.unlock();
        }
    }

    private ProcessQueueResult commitPreparedSelection(DeliveryTransaction transaction, String decisionReason) {
        List<ScheduledRequest> items = transaction.requests();
        if (items.isEmpty()) {
            throw new IllegalStateException(
                    "prepared selection commit requires a non-empty selection");
        }
        String committedReason = null;
        int remainingQueueDepth = 0;
        boolean removedTerminalBoundary = false;
        Throwable postCommitFailure = null;
        WorkView precedingWork;
        queueState.lock();
        try {
            if (queueState.isStopped()) {
                return null;
            }
            long nowMs = now();
            for (ScheduledRequest item : items) {
                if (!containsIdentity(item)
                        || item.requestExpired(nowMs)) {
                    return null;
                }
            }
            precedingWork = transaction.commit();
            // Committed requests no longer count toward either waiting limit.
            var routingRole = prefillEndpoint.getStatus().getRole();
            for (ScheduledRequest item : items) {
                prefillEndpoint.reportSubQueueResidence(item);
                queueCapacity.releaseEngineSubQueueSlot(routingRole);
                queueCapacity.decreaseTotalQueueCount(item.future());
            }
            try {
                removedTerminalBoundary = removeSelectionBoundaryUnderLock(
                        transaction.blocked(), nowMs);
                queueState.incrementQueueVersion();
                committedReason = transaction.blocked() != null
                        && transaction.blocked().reason().unavailable()
                        ? "delivery_capacity_prefix" : decisionReason;
                remainingQueueDepth = queueState.activeIndex().size();
            } catch (Throwable failure) {
                postCommitFailure = failure;
            }
        } finally {
            queueState.unlock();
        }

        if (postCommitFailure != null) {
            Throwable failure = postCommitFailure;
            if (removedTerminalBoundary) {
                capacityBlockedHead.signalCapacityAvailable();
            }
            try {
                notifyTerminalAdmissionFailure(
                        removedTerminalBoundary, transaction.blocked());
            } catch (Throwable notificationFailure) {
                failure = appendFailure(failure, notificationFailure);
            }
            try {
                transaction.rollbackCommitted(failure);
            } catch (Throwable ownerFailure) {
                failure = appendFailure(failure, ownerFailure);
            }
            throw propagateCommitFailure(failure);
        }
        if (removedTerminalBoundary) {
            capacityBlockedHead.signalCapacityAvailable();
        }
        notifyTerminalAdmissionFailure(
                removedTerminalBoundary, transaction.blocked());
        deliver(transaction, committedReason, remainingQueueDepth, precedingWork);
        return ProcessQueueResult.CAPACITY_CHANGED;
    }

    private ProcessQueueResult commitBoundary(BlockedRequest blocked) {
        ProcessQueueResult result = ProcessQueueResult.NO_ACTION;
        boolean removedTerminalBoundary = false;
        queueState.lock();
        try {
            if (queueState.isStopped()) {
                return result;
            }
            long nowMs = now();
            if (blocked != null && blocked.reason().unavailable()) {
                result = resolveEmptyCapacityUnderLock(
                        blocked.item(), blocked.reason(), nowMs);
            } else {
                removedTerminalBoundary = removeSelectionBoundaryUnderLock(
                        blocked, nowMs);
                if (removedTerminalBoundary) {
                    queueState.incrementQueueVersion();
                    result = ProcessQueueResult.CAPACITY_CHANGED;
                }
            }
        } finally {
            queueState.unlock();
        }
        if (removedTerminalBoundary) {
            capacityBlockedHead.signalCapacityAvailable();
        }
        notifyTerminalAdmissionFailure(removedTerminalBoundary, blocked);
        return result;
    }

    /** Caller holds the queue lock. */
    private ProcessQueueResult resolveEmptyCapacityUnderLock(
            ScheduledRequest item,
            DeliveryBlockReason unavailable,
            long nowMs) {
        if (queueState.activeIndex().peek() != item
                || item.requestExpired(nowMs)
                || !containsIdentity(item)) {
            return ProcessQueueResult.NO_ACTION;
        }
        return ProcessQueueResult.capacityBlocked(item, unavailable);
    }

    /** Caller holds the queue lock. */
    private boolean removeSelectionBoundaryUnderLock(
            BlockedRequest blocked,
            long nowMs) {
        if (blocked == null
                || blocked.reason().unavailable()
                || blocked.reason() == DeliveryBlockReason.OWNERSHIP_LOST) {
            return false;
        }
        ScheduledRequest blockedItem = blocked.item();
        if (!containsIdentity(blockedItem)
                || blockedItem.requestExpired(nowMs)
                || !queuedRequestRemover.removeTerminalActiveUnderLock(blockedItem)) {
            return false;
        }
        return true;
    }

    private void notifyTerminalAdmissionFailure(
            boolean removed,
            BlockedRequest blocked) {
        if (removed && blocked.reason().status() == DeliveryBlockReason.Status.FAILED) {
            notifyAdmissionFailure(blocked.item(), blocked.reason().cause());
        }
    }

    private boolean containsIdentity(ScheduledRequest expected) {
        for (ScheduledRequest item : queueState.activeIndex()) {
            if (item == expected) {
                return true;
            }
        }
        return false;
    }

    private void notifyAdmissionFailure(
            ScheduledRequest item,
            Throwable cause) {
        try {
            endpointEvents.onPreparedDeliveryFailure(item, cause);
        } catch (Throwable callbackFailure) {
            Logger.error("PrefillQueue[{}] delivery-failure callback failed "
                            + "request_id={}",
                    key, item.requestId(), callbackFailure);
        }
    }

    void dropHead(ScheduledRequest head) {
        boolean removed;
        queueState.lock();
        try {
            removed = queuedRequestRemover.removeUnderLock(head);
        } finally {
            queueState.unlock();
        }
        if (!removed) {
            return;
        }
        capacityBlockedHead.signalCapacityAvailable();
        try {
            endpointEvents.onQueuedItemExpired(head);
        } catch (Throwable callbackFailure) {
            Logger.error("PrefillQueue[{}] ACTIVE terminal callback failed "
                            + "request_id={} reason={}",
                    key, head.requestId(), "request_expired",
                    callbackFailure);
        }
    }

    static Throwable appendFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    /** Release the publish-time reservation only while it is still item-owned. */
    static RuntimeException propagateCommitFailure(
            Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(
                "delivery selection failed after ownership commit", failure);
    }
}
