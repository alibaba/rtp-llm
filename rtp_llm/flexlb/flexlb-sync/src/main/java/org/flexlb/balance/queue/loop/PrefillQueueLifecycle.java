package org.flexlb.balance.queue.loop;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.single.RouteReservation;
import org.flexlb.balance.queue.CapacityBlockedHead;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.util.Logger;

import java.util.concurrent.CancellationException;

/**
 * 一台 Prefill 排队队列的启动和停止。
 * <p>
 * 启动时开出调度线程。停止时先关掉入队，唤醒并中断调度线程，再把还在排队的请求逐个取出、
 * 释放预留的 Prefill 资源，交还给请求调度器处理；最后等调度线程退出。
 * 多个线程同时停止时，只有第一个真正执行清理，其余的等它做完并拿到同一个结果。
 */
public final class PrefillQueueLifecycle {

    private enum RuntimeState {
        NEW,
        STARTING,
        RUNNING,
        STOPPING,
        STOPPED
    }

    /**
     * 这台 Prefill 的调度线程，所有调度都在这一个线程里依次执行。
     */
    private final Thread workerThread;
    /**
     * 正常停止时交给排队请求的失败原因，在这台 Prefill 开始下线之前就创建好。
     */
    private final CancellationException normalStopFailure;
    /**
     * 停止时请求记录已经不在、没法确认移除时用的固定异常，正常不会发生。
     */
    private final IllegalStateException stopAcknowledgementFailure;
    private boolean terminationStarted;
    private boolean terminationFinished;
    private Thread terminationOwner;
    private volatile RuntimeState runtimeState = RuntimeState.NEW;
    private Throwable stopFailure;

    private final String key;
    private final QueueState queueState;
    private final PrefillState prefillState;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueCapacity queueCapacity;
    private final EndpointEventHandler endpointEvents;
    private final CapacityBlockedHead capacityBlockedHead;

    public PrefillQueueLifecycle(String key,
                          QueueState queueState,
                          PrefillState prefillState,
                          PrefillEndpoint prefillEndpoint,
                          QueueCapacity queueCapacity,
                          EndpointEventHandler endpointEvents,
                          CapacityBlockedHead capacityBlockedHead,
                          Runnable schedulingLoop) {
        this.key = key;
        this.queueState = queueState;
        this.prefillState = prefillState;
        this.prefillEndpoint = prefillEndpoint;
        this.queueCapacity = queueCapacity;
        this.endpointEvents = endpointEvents;
        this.capacityBlockedHead = capacityBlockedHead;
        this.normalStopFailure = new CancellationException(
                "FlexLB worker scheduling queue stopped: " + key);
        this.stopAcknowledgementFailure = new IllegalStateException(
                "FlexLB worker stop callback lost its exact retained owner: "
                        + key);
        // 调度线程用虚拟线程，Prefill 节点很多时也不会每台占一个系统线程。
        this.workerThread = Thread.ofVirtual()
                .name("flexlb-batcher-" + key)
                .uncaughtExceptionHandler(
                        (thread, failure) -> Logger.error("PrefillQueue[{}] thread died unexpectedly", key, failure))
                .unstarted(schedulingLoop);
    }

    public synchronized void start() {
        if (runtimeState != RuntimeState.NEW) {
            throw new IllegalStateException(
                    "Worker batcher cannot start from "
                            + runtimeState);
        }
        runtimeState = RuntimeState.STARTING;
        try {
            workerThread.start();
            runtimeState = RuntimeState.RUNNING;
        } catch (RuntimeException | Error startFailure) {
            Throwable cleanupFailure = stopAndDrain(
                    startFailure,
                    false);
            runtimeState = RuntimeState.STOPPED;
            notifyAll();
            addSuppressedNoFail(startFailure, cleanupFailure);
            throw startFailure;
        }
    }

    public boolean isRunning() {
        return runtimeState == RuntimeState.RUNNING;
    }

    /**
     * 调度线程退出时调用。
     */
    synchronized void markLoopFinished() {
        runtimeState = RuntimeState.STOPPED;
        notifyAll();
    }

    public Throwable stopAndAwait() {
        if (Thread.currentThread() == workerThread) {
            throw new IllegalStateException(
                    "Prefill runtime cannot await its own worker thread");
        }
        Throwable failure = stopAndDrain(
                normalStopFailure,
                true);
        boolean interrupted = false;
        while (true) {
            try {
                workerThread.join();
                break;
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        synchronized (this) {
            runtimeState = RuntimeState.STOPPED;
            notifyAll();
            failure = stopFailure;
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        return failure;
    }

    void stopAfterUnexpectedLoopFailure(Throwable loopFailure) {
        try {
            Logger.error("PrefillQueue[{}] stopped after an unexpected loop failure",
                    key, loopFailure);
        } catch (Throwable ignoredLoggingFailure) {
            // 打日志失败也要继续执行停止流程。
        }
        Throwable cleanupFailure = stopAndDrain(
                loopFailure,
                false);
        if (cleanupFailure != null) {
            try {
                Logger.error("PrefillQueue[{}] failure cleanup exposed invariants",
                        key, cleanupFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // 清理已经做完了。
            }
        }
    }

    private Throwable stopAndDrain(Throwable terminalFailure, boolean interruptWorker) {
        boolean interrupted = false;
        synchronized (this) {
            queueState.markStopped();
            if (terminationStarted) {
                if (!terminationFinished
                        && terminationOwner == Thread.currentThread()) {
                    throw new IllegalStateException(
                            "Prefill runtime cannot await its active stop transaction");
                }
                while (!terminationFinished) {
                    try {
                        wait();
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                    }
                }
                Throwable completedFailure = stopFailure;
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                return completedFailure;
            }
            terminationStarted = true;
            terminationOwner = Thread.currentThread();
            runtimeState = runtimeState == RuntimeState.NEW
                    ? RuntimeState.STOPPED : RuntimeState.STOPPING;
        }

        Throwable cleanupFailure = null;
        try {
            boolean queueLocked = false;
            try {
                queueState.lock();
                queueLocked = true;
                stopAcceptingUnderLock();
                try {
                    capacityBlockedHead.setUnderLock(null);
                } catch (Throwable subscriptionFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, subscriptionFailure);
                }
                queueState.signalAllChanges();
            } catch (Throwable wakeFailure) {
                cleanupFailure = appendCleanupFailure(
                        cleanupFailure, wakeFailure);
            } finally {
                if (queueLocked) {
                    try {
                        queueState.unlock();
                    } catch (Throwable unlockFailure) {
                        cleanupFailure = appendCleanupFailure(
                                cleanupFailure, unlockFailure);
                    }
                }
            }

            if (interruptWorker && Thread.currentThread() != workerThread) {
                try {
                    workerThread.interrupt();
                } catch (Throwable interruptFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, interruptFailure);
                }
            }

            boolean workerWasRemoved = terminalFailure == normalStopFailure;
            while (true) {
                ScheduledRequest item;
                try {
                    item = detachNextStoppedItem();
                } catch (Throwable claimFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, claimFailure);
                    break;
                }
                if (item == null) {
                    break;
                }
                try {
                    releaseUnconsumedRouteReservation(item);
                } catch (Throwable reservationFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, reservationFailure);
                }
                try {
                    if (workerWasRemoved) {
                        endpointEvents.onQueuedWorkerStopped(item);
                    } else {
                        endpointEvents.onQueueOfferFailure(item, terminalFailure);
                    }
                } catch (Throwable callbackFailure) {
                    try {
                        Logger.error("PrefillQueue[{}] shutdown callback failed request_id={}",
                                key, item.requestId(), callbackFailure);
                    } catch (Throwable ignoredLoggingFailure) {
                        // 打日志失败不能影响清理。
                    }
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, callbackFailure);
                    // 不确认移除：这台 Prefill 下线时还会再处理这个请求。
                    continue;
                }
                try {
                    if (!acknowledgeStoppedItem(item)) {
                        cleanupFailure = appendCleanupFailure(
                                cleanupFailure,
                                stopAcknowledgementFailure);
                    }
                } catch (Throwable acknowledgementFailure) {
                    // 失败通知已经发出去了。如果请求记录还在，Prefill 下线时会再处理一次，重复处理没有副作用。
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, acknowledgementFailure);
                }
            }
        } catch (Throwable unexpectedCleanupFailure) {
            cleanupFailure = appendCleanupFailure(
                    cleanupFailure, unexpectedCleanupFailure);
        } finally {
            synchronized (this) {
                stopFailure = cleanupFailure;
                terminationFinished = true;
                terminationOwner = null;
                notifyAll();
            }
        }
        return cleanupFailure;
    }

    /**
     * 停止接收新请求后，从排队索引里取出下一个请求，但保留它在 PrefillState 里的记录。
     * 之后按失败通知是否成功，决定确认移除这条记录，还是留给 Prefill 下线时处理。
     */
    private ScheduledRequest detachNextStoppedItem() {
        ScheduledRequest item;
        queueState.lock();
        try {
            item = detachNextStopTerminalUnderLock();
            queueState.signalAllChanges();
        } finally {
            queueState.unlock();
        }
        if (item != null) {
            prefillEndpoint.reportSubQueueResidence(item);
        }
        if (item != null && prefillEndpoint.isEngineSubQueueMode()) {
            capacityBlockedHead.signalCapacityAvailable();
        }
        return item;
    }

    /**
     * 失败通知成功后，确认移除这个请求在 PrefillState 里保留的记录。
     */
    private boolean acknowledgeStoppedItem(ScheduledRequest item) {
        queueState.lock();
        try {
            return acknowledgeStopTerminalUnderLock(item);
        } finally {
            queueState.unlock();
        }
    }

    private static Throwable appendCleanupFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        // 只保留第一个异常，不再挂后面的异常，避免内存不足时连挂异常都失败；后面的请求仍然都会处理到。
        return first;
    }

    private static void addSuppressedNoFail(
            Throwable primary,
            Throwable leaf) {
        if (primary == null || leaf == null || primary == leaf) {
            return;
        }
        try {
            primary.addSuppressed(leaf);
        } catch (Throwable ignoredAggregationFailure) {
            // 挂异常失败也不影响结果：所有请求在这之前都已经处理过了。
        }
    }

    /**
     * 调用方持有队列锁。
     */
    private void stopAcceptingUnderLock() {
        if (!queueState.isLockedByCurrentThread()) {
            throw new IllegalStateException(
                    "queue mutation requires the queue lock");
        }
        queueState.markStopped();
    }

    /**
     * 调用方持有队列锁。
     */
    private ScheduledRequest detachNextStopTerminalUnderLock() {
        ScheduledRequest item =
                prefillState.detachNextActiveForStopUnderLock();
        if (item != null) {
            queueCapacity.releaseEngineSubQueueSlot(prefillEndpoint.getStatus().getRole());
            queueState.incrementQueueVersion();
        }
        return item;
    }

    /**
     * 调用方持有队列锁。
     */
    private boolean acknowledgeStopTerminalUnderLock(
            ScheduledRequest item) {
        return prefillState.acknowledgeStopTerminalUnderLock(item);
    }

    private static void releaseUnconsumedRouteReservation(
            ScheduledRequest item) {
        if (item == null) {
            return;
        }
        RouteReservation reservation =
                item.takePublishedRouteReservation();
        if (reservation != null) {
            reservation.close();
        }
    }
}
