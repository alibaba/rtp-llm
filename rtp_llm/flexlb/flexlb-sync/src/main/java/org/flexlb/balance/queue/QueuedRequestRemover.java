package org.flexlb.balance.queue;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.ScheduledRequest;

/**
 * 把一个还在排队的请求从队列里移除：释放它预留的 Prefill 资源，归还引擎排队名额，并上报它在队列里待了多久。
 * <p>
 * 请求取消、超时、交付失败被丢弃时都走这里。
 */
public final class QueuedRequestRemover {

    private final QueueState queueState;
    private final PrefillState prefillState;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueCapacity queueCapacity;

    QueuedRequestRemover(QueueState queueState,
                         PrefillState prefillState,
                         PrefillEndpoint prefillEndpoint,
                         QueueCapacity queueCapacity) {
        this.queueState = queueState;
        this.prefillState = prefillState;
        this.prefillEndpoint = prefillEndpoint;
        this.queueCapacity = queueCapacity;
    }

    /**
     * 移除请求，但不推进队列版本，由调用方在一批改动做完后统一推进。请求已不在队列里时返回 false。
     * 调用方持有队列锁。
     */
    public boolean removeTerminalActiveUnderLock(ScheduledRequest item) {
        boolean removed;
        if (!item.requiresSingleReservation()) {
            removed = prefillState.terminalizeActiveUnderLock(item);
        } else {
            SingleReservation reservation =
                    item.takePublishedSingleReservation();
            if (reservation == null) {
                return false;
            }
            try {
                removed = prefillState.terminalizeActiveRouteUnderLock(
                        item, reservation);
            } catch (Throwable failure) {
                restoreSingleReservation(item, reservation);
                throw failure;
            }
            if (!removed) {
                restoreSingleReservation(item, reservation);
            }
        }
        if (removed) {
            prefillEndpoint.reportSubQueueResidence(item);
            queueCapacity.releaseEngineSubQueueSlot(prefillEndpoint.getStatus().getRole());
        }
        return removed;
    }

    /**
     * 把取下来的 Prefill 资源预留放回请求上。用于移除没成功时撤回。
     */
    public static void restoreSingleReservation(
            ScheduledRequest item,
            SingleReservation reservation) {
        if (!item.restorePublishedSingleReservation(reservation)) {
            throw new IllegalStateException(
                    "ACTIVE route reservation could not be restored request_id="
                            + item.requestId());
        }
    }

    /**
     * 移除请求并推进队列版本。请求已不在队列里时返回 false。调用方持有队列锁。
     */
    public boolean removeUnderLock(ScheduledRequest item) {
        boolean removed = removeTerminalActiveUnderLock(item);
        if (removed) {
            queueState.incrementQueueVersion();
        }
        return removed;
    }
}
