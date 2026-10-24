package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.util.Logger;

import java.util.concurrent.CancellationException;

/**
 * 一台 Prefill 排队队列的启动和停止。
 * <p>
 * 启动时开出调度线程。停止时先关掉入队，唤醒并中断调度线程，再把还在排队的请求逐个取出、
 * 释放预留的 Prefill 资源，交还给请求调度器处理；最后等调度线程退出。
 * 多个线程同时停止时，只有第一个真正执行清理，其余的等它做完并拿到同一个结果。
 */
final class PrefillQueueLifecycle {

    private enum RuntimeState {
        NEW,
        STARTING,
        RUNNING,
        STOPPING,
        STOPPED
    }

    /**
     * Serialized queue runtime for this worker.
     */
    private final Thread workerThread;
    /** Constructed before this endpoint generation can begin retirement. */
    private final CancellationException normalStopFailure;
    /** Fixed diagnostic for an impossible exact stop acknowledgement loss. */
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
    /**
     * 清掉"队首因投递容量不足而阻塞"的记录并取消容量监听。调用方持有队列锁。
     */
    private final Runnable clearCapacityBlockUnderLock;
    /**
     * 通知本队列和全局选路：Prefill 有容量空出来了。
     */
    private final Runnable signalCapacityAvailable;

    PrefillQueueLifecycle(String key,
                          QueueState queueState,
                          PrefillState prefillState,
                          PrefillEndpoint prefillEndpoint,
                          QueueCapacity queueCapacity,
                          EndpointEventHandler endpointEvents,
                          Runnable clearCapacityBlockUnderLock,
                          Runnable signalCapacityAvailable,
                          Runnable schedulingLoop) {
        this.key = key;
        this.queueState = queueState;
        this.prefillState = prefillState;
        this.prefillEndpoint = prefillEndpoint;
        this.queueCapacity = queueCapacity;
        this.endpointEvents = endpointEvents;
        this.clearCapacityBlockUnderLock = clearCapacityBlockUnderLock;
        this.signalCapacityAvailable = signalCapacityAvailable;
        this.normalStopFailure = new CancellationException(
                "FlexLB worker scheduling queue stopped: " + key);
        this.stopAcknowledgementFailure = new IllegalStateException(
                "FlexLB worker stop callback lost its exact retained owner: "
                        + key);
        // Waiting queue runtimes use virtual threads so a large endpoint fleet
        // does not consume one OS thread per worker.
        this.workerThread = Thread.ofVirtual()
                .name("flexlb-batcher-" + key)
                .uncaughtExceptionHandler(
                        (thread, failure) -> Logger.error("PrefillQueue[{}] thread died unexpectedly", key, failure))
                .unstarted(schedulingLoop);
    }

    synchronized void start() {
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

    boolean isRunning() {
        return runtimeState == RuntimeState.RUNNING;
    }

    /**
     * 调度线程退出时调用。
     */
    synchronized void markLoopFinished() {
        runtimeState = RuntimeState.STOPPED;
        notifyAll();
    }

    Throwable stopAndAwait() {
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
            // The exact stop transaction must still run.
        }
        Throwable cleanupFailure = stopAndDrain(
                loopFailure,
                false);
        if (cleanupFailure != null) {
            try {
                Logger.error("PrefillQueue[{}] failure cleanup exposed invariants",
                        key, cleanupFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // Cleanup is already complete.
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
                    clearCapacityBlockUnderLock.run();
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
                        // Cleanup ownership cannot depend on diagnostics.
                    }
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, callbackFailure);
                    // Do not acknowledge: generation retirement still owns
                    // this exact pending identity and will handle it again.
                    continue;
                }
                try {
                    if (!acknowledgeStoppedItem(item)) {
                        cleanupFailure = appendCleanupFailure(
                                cleanupFailure,
                                stopAcknowledgementFailure);
                    }
                } catch (Throwable acknowledgementFailure) {
                    // The terminal fact was already delivered. If the exact
                    // entry remains, generation retirement converges it as a
                    // stale/idempotent replay.
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
     * Detach one exact ACTIVE queue index after the stop gate is closed while
     * retaining its canonical entry. The callback result decides whether that
     * entry is acknowledged or carried into generation retirement.
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
            signalCapacityAvailable.run();
        }
        return item;
    }

    /** Acknowledge only the retained owner whose terminal callback returned. */
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
        // Preserve cleanup totality under allocation failure. The first causal
        // failure is sufficient; later leaves were still all attempted.
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
            // The start transaction already drained every acquired leaf.
        }
    }

    /** Caller holds the queue lock. */
    private void stopAcceptingUnderLock() {
        if (!queueState.isLockedByCurrentThread()) {
            throw new IllegalStateException(
                    "queue mutation requires the queue lock");
        }
        queueState.markStopped();
    }

    /** Caller holds the queue lock. */
    private ScheduledRequest detachNextStopTerminalUnderLock() {
        ScheduledRequest item =
                prefillState.detachNextActiveForStopUnderLock();
        if (item != null) {
            queueCapacity.releaseEngineSubQueueSlot(prefillEndpoint.getStatus().getRole());
            queueState.incrementQueueVersion();
        }
        return item;
    }

    /** Caller holds the queue lock. */
    private boolean acknowledgeStopTerminalUnderLock(
            ScheduledRequest item) {
        return prefillState.acknowledgeStopTerminalUnderLock(item);
    }

    private static void releaseUnconsumedRouteReservation(
            ScheduledRequest item) {
        if (item == null) {
            return;
        }
        PrefillState.RouteReservation reservation =
                item.takePublishedRouteReservation();
        if (reservation != null) {
            reservation.close();
        }
    }
}
