package org.flexlb.balance.scheduler;

import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static com.google.common.base.Preconditions.checkState;

/**
 * Executes submitted tasks and waits for their completion before closing its workers.
 * Request state changes and resource cleanup are handled by the submitting component.
 */
public final class RequestTaskPool implements RequestTaskExecutor {
    private final ExecutorService requestWorkers =
            Executors.newFixedThreadPool(
                    Math.clamp(Runtime.getRuntime().availableProcessors(), 2, 8),
                    Thread.ofPlatform().daemon().name("request-work-", 1).factory());
    private final ThreadLocal<Boolean> executing = new ThreadLocal<>();
    private final Object tasksLock = new Object();

    private long submittedTasks;
    private boolean closed;

    @Override
    public void execute(RequestContext context, Runnable work) {
        Objects.requireNonNull(work);
        synchronized (tasksLock) {
            if (closed)
                throw new java.util.concurrent.RejectedExecutionException(
                        "request executor closed");
            submittedTasks++;
        }
        var submitted = new SubmittedTask(work);
        try {
            requestWorkers.execute(submitted);
        } catch (RuntimeException | Error failure) {
            submitted.rejectBeforeStart();
            throw failure;
        }
    }

    public void awaitIdle() {
        checkState(
                !isExecutingOnCurrentThread(), "request task cannot wait for its own completion");
        boolean interrupted = false;
        synchronized (tasksLock) {
            while (submittedTasks != 0) {
                try {
                    tasksLock.wait();
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }

    public boolean isExecutingOnCurrentThread() {
        return Boolean.TRUE.equals(executing.get());
    }

    public void close() {
        checkState(!isExecutingOnCurrentThread(), "request task cannot close its own executor");
        synchronized (tasksLock) {
            closed = true;
        }
        requestWorkers.shutdown();
        boolean interrupted = false;
        while (!requestWorkers.isTerminated()) {
            try {
                requestWorkers.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) {
                interrupted = true;
            }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }

    private void completed() {
        synchronized (tasksLock) {
            if (--submittedTasks == 0) tasksLock.notifyAll();
        }
    }

    /**
     * Releases the executor's task count once, even when submission throws after execution starts.
     */
    private final class SubmittedTask implements Runnable {
        private final Runnable action;
        private final java.util.concurrent.atomic.AtomicReference<TaskState> state =
                new java.util.concurrent.atomic.AtomicReference<>(TaskState.QUEUED);

        private SubmittedTask(Runnable action) {
            this.action = action;
        }

        @Override
        public void run() {
            if (!state.compareAndSet(TaskState.QUEUED, TaskState.RUNNING)) return;
            executing.set(true);
            try {
                action.run();
            } catch (Throwable failure) {
                Logger.error("Request task failed", failure);
            } finally {
                executing.remove();
                state.set(TaskState.FINISHED);
                completed();
            }
        }

        private void rejectBeforeStart() {
            if (state.compareAndSet(TaskState.QUEUED, TaskState.REJECTED)) completed();
        }
    }

    private enum TaskState {
        QUEUED,
        RUNNING,
        FINISHED,
        REJECTED
    }
}
