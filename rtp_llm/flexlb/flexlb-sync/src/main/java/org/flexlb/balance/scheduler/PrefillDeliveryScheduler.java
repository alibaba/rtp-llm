package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;

import java.util.ArrayDeque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/**
 * Runs Prefill queue delivery turns and owns their capacity subscriptions and deadlines.
 * Events request a scheduling turn; notifications received during a task request another turn.
 * The shared scheduling driver supplies time and waits for the returned next deadline.
 */
final class PrefillDeliveryScheduler {
    private final QueueExecutionSettings settings;
    private final DeliveryCoordinator dispatcher;
    private final ExecutorService planners;
    private final int plannerCount;
    private final Runnable wakeDriver;
    private final Consumer<Throwable> recordFailure;
    private final ReentrantLock lock = new ReentrantLock();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final Map<PrefillEndpoint, QueueTask> prefillQueueSchedulers = new IdentityHashMap<>();
    private final ArrayDeque<QueueTask> readyPrefillQueues = new ArrayDeque<>();
    private final java.util.PriorityQueue<QueueTask> timedPrefillQueues =
            new java.util.PriorityQueue<>(
                    java.util.Comparator.comparingLong(queueScheduler -> queueScheduler.nextCheck));

    private int queueSchedulingInFlight;

    PrefillDeliveryScheduler(
            QueueExecutionSettings settings,
            DeliveryCoordinator dispatcher,
            ExecutorService planners,
            int plannerCount,
            Runnable wakeDriver,
            Consumer<Throwable> recordFailure) {
        this.settings = settings;
        this.dispatcher = dispatcher;
        this.planners = planners;
        this.plannerCount = plannerCount;
        this.wakeDriver = wakeDriver;
        this.recordFailure = recordFailure;
    }

    void endpointChanged(PrefillEndpoint endpoint) {
        lock.lock();
        try {
            if (closed.get()) return;
            var queue = prefillQueueSchedulers.get(endpoint);
            if (queue == null && endpoint.queueEnabled() && !endpoint.queueStopped()) {
                queue =
                        new QueueTask(
                                endpoint,
                                new PrefillQueueScheduler(
                                        endpoint.ipPort(), endpoint, settings, dispatcher));
                prefillQueueSchedulers.put(endpoint, queue);
            }
            if (queue != null) requestScheduling(queue);
        } finally {
            lock.unlock();
        }
        wakeDriver.run();
    }

    Map<String, Object> getLatestQueueWaitSnapshot(PrefillEndpoint endpoint) {
        QueueTask queue;
        lock.lock();
        try {
            queue = prefillQueueSchedulers.get(endpoint);
        } finally {
            lock.unlock();
        }
        return queue == null
                ? Map.of("cause", "waiting for Prefill decision")
                : queue.scheduler.getLatestQueueWaitSnapshot();
    }

    void stop() {
        lock.lock();
        try {
            closed.set(true);
            readyPrefillQueues.clear();
            timedPrefillQueues.clear();
        } finally {
            lock.unlock();
        }
    }

    void clearAfterTasksStopped() {
        List<QueueTask> queues;
        lock.lock();
        try {
            queues = List.copyOf(prefillQueueSchedulers.values());
            prefillQueueSchedulers.clear();
            readyPrefillQueues.clear();
            timedPrefillQueues.clear();
        } finally {
            lock.unlock();
        }
        for (var queue : queues) queue.scheduler.observeWait(null);
    }

    long scheduleAt(long nowNanos) {
        QueueTask queueScheduler;
        lock.lock();
        try {
            if (closed.get()) return Long.MAX_VALUE;
            while (!timedPrefillQueues.isEmpty()
                    && timedPrefillQueues.peek().nextCheck <= nowNanos) {
                var dueQueueScheduler = timedPrefillQueues.remove();
                requestScheduling(dueQueueScheduler);
            }
            queueScheduler =
                    queueSchedulingInFlight < plannerCount ? readyPrefillQueues.pollFirst() : null;
            if (queueScheduler != null) {
                // Only PENDING queues are enqueued. Claim before accepting later notifications.
                queueScheduler.schedulingState = SchedulingState.SCHEDULED;
                queueSchedulingInFlight++;
            }
        } finally {
            lock.unlock();
        }
        if (queueScheduler != null) {
            try {
                planners.execute(() -> runQueueScheduling(queueScheduler));
            } catch (Throwable failure) {
                finishQueueScheduling(
                        queueScheduler,
                        Long.MAX_VALUE,
                        rejectedDuringStop(failure) ? null : failure);
            }
        }
        lock.lock();
        try {
            return !readyPrefillQueues.isEmpty() && queueSchedulingInFlight < plannerCount
                    ? nowNanos
                    : timedPrefillQueues.isEmpty()
                            ? Long.MAX_VALUE
                            : timedPrefillQueues.peek().nextCheck;
        } finally {
            lock.unlock();
        }
    }

    /**
     * One task owns decision, wait subscription and next check; no completed-result queue.
     */
    private void runQueueScheduling(QueueTask task) {

        PrefillQueueScheduler queueScheduler = task.scheduler;
        long next = Long.MAX_VALUE;
        Throwable failure = null;
        try {
            if (queueScheduler.stopping()) queueScheduler.observeWait(null);
            if (!closed.get() && !queueScheduler.stopping()) {
                var wait = queueScheduler.scheduleOnce();
                if (wait.waitReason() != null)
                    queueScheduler.recordQueueWait(wait.request(), wait.waitReason());
                queueScheduler.observeWait(wait);
                long nowNanos = System.nanoTime();
                long nowMs = System.currentTimeMillis();
                if (queueScheduler.needsImmediateScheduling(wait, nowMs)) {
                    next = nowNanos;
                } else if (wait.request() != null) {
                    long at = Math.min(wait.wakeAtMs(), wait.request().expiresAtMs());
                    if (at != Long.MAX_VALUE)
                        next = nowNanos + TimeUnit.MILLISECONDS.toNanos(Math.max(0L, at - nowMs));
                }
            }
        } catch (Throwable problem) {
            failure = problem;
        } finally {
            finishQueueScheduling(task, next, failure);
        }
    }

    private void finishQueueScheduling(QueueTask task, long next, Throwable failure) {
        try {
            if (failure != null) {
                recordFailure.accept(failure);
                recordFailure.accept(task.scheduler.failQueuedRequests(failure));
            }
        } finally {
            lock.lock();
            try {
                queueSchedulingInFlight--;
                if (failure != null || task.scheduler.stopping()) {
                    prefillQueueSchedulers.remove(task.endpoint);
                } else if (!closed.get()) {
                    boolean pending = task.schedulingState == SchedulingState.SCHEDULED_PENDING;
                    task.schedulingState = SchedulingState.IDLE;
                    task.nextCheck = next;
                    if (pending || next <= System.nanoTime()) {
                        requestScheduling(task);
                    } else if (next != Long.MAX_VALUE) {
                        timedPrefillQueues.add(task);
                    }
                }
            } finally {
                lock.unlock();
                wakeDriver.run();
            }
        }
    }

    /**
     * Called under lock. PENDING has exactly one ready-queue entry; scheduled tasks never do.
     */
    private void requestScheduling(QueueTask queue) {
        switch (queue.schedulingState) {
            case IDLE -> {
                timedPrefillQueues.remove(queue);
                queue.schedulingState = SchedulingState.PENDING;
                readyPrefillQueues.addLast(queue);
            }
            case SCHEDULED -> queue.schedulingState = SchedulingState.SCHEDULED_PENDING;
            case PENDING, SCHEDULED_PENDING -> {}
        }
    }

    private boolean rejectedDuringStop(Throwable failure) {
        return closed.get()
                && planners.isShutdown()
                && failure instanceof java.util.concurrent.RejectedExecutionException;
    }

    private enum SchedulingState {
        IDLE,
        PENDING,
        // Claimed, including time waiting for an executor thread.
        SCHEDULED,
        // Claimed, with another check requested during this task.
        SCHEDULED_PENDING
    }

    /**
     * One endpoint's task and deadline registration, accessed under the delivery scheduler lock.
     * PENDING has one ready-queue entry; scheduled states have one unfinished task instead.
     */
    private static final class QueueTask {
        final PrefillEndpoint endpoint;
        final PrefillQueueScheduler scheduler;

        SchedulingState schedulingState = SchedulingState.IDLE;
        long nextCheck = Long.MAX_VALUE;

        QueueTask(PrefillEndpoint endpoint, PrefillQueueScheduler scheduler) {
            this.endpoint = endpoint;
            this.scheduler = scheduler;
        }
    }
}
