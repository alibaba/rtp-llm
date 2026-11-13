package org.flexlb.balance.schedule;

import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static com.google.common.base.Preconditions.checkState;

/**
 * Executes submitted tasks and waits for their completion before closing its workers.
 * Request state changes and resource cleanup are handled by the submitting component.
 */
public final class RequestTaskPool implements Executor, AutoCloseable {
    private final ExecutorService requestWorkers =
            Executors.newFixedThreadPool(
                    Math.clamp(Runtime.getRuntime().availableProcessors(), 2, 8),
                    Thread.ofPlatform().daemon().name("request-work-", 1).factory());
    private final ThreadLocal<Boolean> executing = new ThreadLocal<>();
    private final Object tasksLock = new Object();

    private long submittedTasks;
    private boolean closed;

    @Override
    public void execute(Runnable work) {
        Objects.requireNonNull(work);
        synchronized (tasksLock) {
            if (closed)
                throw new java.util.concurrent.RejectedExecutionException(
                        "request executor closed");
            submittedTasks++;
        }
        var startedOrRejected = new java.util.concurrent.atomic.AtomicBoolean();
        Runnable submitted =
                () -> {
                    if (!startedOrRejected.compareAndSet(false, true)) return;
                    executing.set(true);
                    try {
                        work.run();
                    } catch (Throwable failure) {
                        Logger.error("Request task failed", failure);
                    } finally {
                        executing.remove();
                        completed();
                    }
                };
        try {
            requestWorkers.execute(submitted);
        } catch (RuntimeException | Error failure) {
            if (startedOrRejected.compareAndSet(false, true)) completed();
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

    boolean isExecutingOnCurrentThread() {
        return Boolean.TRUE.equals(executing.get());
    }

    private void completed() {
        synchronized (tasksLock) {
            if (--submittedTasks == 0) tasksLock.notifyAll();
        }
    }
}
