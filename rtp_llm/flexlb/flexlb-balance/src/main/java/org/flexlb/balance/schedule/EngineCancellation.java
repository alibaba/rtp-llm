package org.flexlb.balance.schedule;

import com.google.common.collect.MapMaker;
import org.flexlb.util.Failures;
import org.flexlb.worker.rpc.EngineGrpcClient;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import java.util.function.BooleanSupplier;

/**
 * Runs bounded cancellation retries and records cleanup replies against the exact published route.
 * Late replies are applied to the request scheduler before shutdown drains the callbacks.
 */
@org.springframework.stereotype.Component
public final class EngineCancellation {
    static final long CLEANUP_TIMEOUT_MS = 5_000L;
    private static final int MAX_ATTEMPTS = 8;

    private static final ThreadLocal<Boolean> executingStop = new ThreadLocal<>();

    private final EngineGrpcClient grpcClient;
    private final java.util.concurrent.Executor stopWorkers =
            task -> Thread.ofVirtual().name("delivery-stop-work").start(task);
    // MapMaker weak keys use identity: equal-valued claims must not share an operation.
    private final ConcurrentMap<RequestResult, StopTask> sessions =
            new MapMaker().weakKeys().makeMap();
    private final java.util.Set<StopTask> activeStops =
            java.util.concurrent.ConcurrentHashMap.newKeySet();

    public EngineCancellation(EngineGrpcClient grpcClient) {
        this.grpcClient = grpcClient;
    }

    boolean isStoppingOnCurrentThread() {
        return Boolean.TRUE.equals(executingStop.get());
    }

    void awaitStops() {
        Throwable problem = null;
        long deadline =
                System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS + 1_000L);
        for (var task : activeStops) {
            try {
                synchronized (task) {
                    while (task.taskScheduled
                            || task.applyingReplies != 0
                            || !task.closed && task.pending != 0) {
                        long remaining = deadline - System.nanoTime();
                        if (remaining <= 0) {
                            throw new java.util.concurrent.TimeoutException(
                                    "stop callbacks pending");
                        }
                        TimeUnit.NANOSECONDS.timedWait(task, remaining);
                    }
                }
            } catch (Exception failure) {
                if (failure instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                }
                task.finished.completeExceptionally(failure);
                task.close();
                problem = Failures.append(problem, failure);
            }
        }
        Failures.rethrow(problem, "delivery stop callbacks did not drain");
    }

    /**
     * Cancels the committed route with bounded retries and applies cleanup replies to its scheduler.
     * Completion ends the cancellation attempts; the scheduler separately tracks resource cleanup.
     */
    CompletableFuture<Void> stop(RequestResult claim) {
        for (; ; ) {
            var delivery = claim.ctx().snapshot().committedRoute();
            if (delivery == null || delivery != claim) {
                return CompletableFuture.completedFuture(null);
            }
            var task = record(claim);
            synchronized (task) {
                if (task.closed) {
                    sessions.remove(claim, task);
                    continue;
                }
                if (task.stopRequested) {
                    return task.finished;
                }
                task.stopRequested = true;
            }
            task.start();
            return task.finished;
        }
    }

    void forget(RequestResult route) {
        var task = sessions.remove(route);
        if (task != null) task.close();
    }

    private StopTask record(RequestResult claim) {
        var existing = sessions.get(claim);
        if (existing != null) {
            return existing;
        }
        var created = new StopTask(claim);
        var previous = sessions.putIfAbsent(claim, created);
        if (previous != null) {
            activeStops.remove(created);
            return previous;
        }
        return created;
    }

    /**
     * Serializes cancellation attempts and applies their replies to the owning request before completing.
     */
    private final class StopTask implements Runnable {
        private final RequestResult claim;
        private final RequestContext context;
        private final AbstractRequestScheduler owner;
        private final CompletableFuture<Void> finished = new CompletableFuture<>();

        private volatile boolean remoteCleanupConfirmed;
        private volatile boolean closed, stopRequested;

        private volatile Thread runner;

        // One scheduled task serializes retries; callbacks are counted separately for shutdown.
        private int pending, applyingReplies;
        private boolean taskScheduled;

        // Only the running workflow changes its retry state.
        // Completion means the Scheduler has applied the receipt, including late receipts.
        private CompletableFuture<Long> rpc;

        private int attempts;
        private long deadline, rpcDeadline;

        StopTask(RequestResult claim) {
            this.claim = claim;
            this.context = claim.ctx();
            this.owner = context.getRequestScheduler();
            activeStops.add(this);
        }

        @Override
        public void run() {
            executingStop.set(true);
            runner = Thread.currentThread();
            try {
                if (shouldStopSendingCancel()) {
                    finished.complete(null);
                } else {
                    stopRemotely();
                }
            } catch (RuntimeException | Error failure) {
                finished.completeExceptionally(failure);
                owner.recordRequestFailure(failure);
            } finally {
                synchronized (this) {
                    executingStop.remove();
                    runner = null;
                    taskScheduled = false;
                    notifyProgress();
                }
            }
        }

        private void start() {
            synchronized (this) {
                if (closed) {
                    return;
                }
                if (taskScheduled) {
                    signal();
                    return;
                }
                taskScheduled = true;
            }
            signal();
            try {
                stopWorkers.execute(this);
            } catch (RuntimeException | Error failure) {
                try {
                    finished.completeExceptionally(failure);
                    close();
                    owner.recordRequestFailure(failure);
                } finally {
                    synchronized (this) {
                        taskScheduled = false;
                        notifyProgress();
                    }
                }
            }
        }

        private void signal() {
            LockSupport.unpark(runner);
        }

        private void close() {
            synchronized (this) {
                closed = true;
                sessions.remove(claim, this);
                notifyProgress();
            }
            signal();
            finished.complete(null);
        }

        /**
         * Called under this task's monitor after task or reply progress changes.
         */
        private void notifyProgress() {
            if (closed && !taskScheduled && applyingReplies == 0) activeStops.remove(this);
            notifyAll();
        }

        private void stopRemotely() {
            if (deadline == 0) {
                deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CLEANUP_TIMEOUT_MS);
            }
            while (!shouldStopSendingCancel()) {
                if (System.nanoTime() >= deadline) break;
                if (rpc != null) {
                    await(Math.min(deadline, rpcDeadline), rpc::isDone);
                    long completedAt = rpc.getNow(System.nanoTime());
                    rpc = null;
                    await(
                            Math.min(
                                    deadline,
                                    completedAt
                                            + TimeUnit.MILLISECONDS.toNanos(
                                                    Math.min(400L, 25L << (attempts - 1)))),
                            () -> false);
                }
                if (shouldStopSendingCancel()
                        || System.nanoTime() >= deadline
                        || attempts >= MAX_ATTEMPTS) {
                    break;
                }
                sendCancel();
            }
            if (shouldStopSendingCancel()) finished.complete(null);
            else finishUnresolved();
        }

        private boolean shouldStopSendingCancel() {
            if (closed) {
                return true;
            }
            var state = context.snapshot();
            var delivery = state.committedRoute();
            if (delivery != claim) {
                close();
                return true;
            }
            if (state.prefillWorkerReleaseConfirmed() && state.decodeWorkerReleaseConfirmed()) {
                return true;
            }
            return remoteCleanupConfirmed;
        }

        private void await(long until, BooleanSupplier ready) {
            while (!shouldStopSendingCancel() && !ready.getAsBoolean()) {
                long remaining = until - System.nanoTime();
                if (remaining <= 0) {
                    return;
                }
                LockSupport.parkNanos(this, remaining);
                if (Thread.currentThread().isInterrupted()) {
                    throw new IllegalStateException("delivery stop interrupted");
                }
            }
        }

        private void sendCancel() {
            var state = context.snapshot();
            var reason = state.cancellationReason();
            long timeout = 500L;
            attempts++;
            synchronized (this) {
                pending++;
            }
            rpcDeadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeout);
            try {
                var remote =
                        Objects.requireNonNull(
                                grpcClient.cancelAsync(
                                        claim.prefillEndpoint().getIp(),
                                        claim.prefillEndpoint().getGrpcPort(),
                                        reason.toRequest(context.getRequestId()),
                                        timeout),
                                "cancel future");
                rpc = remote.handle(this::onCancelReply);
                rpc.whenComplete((ignored, failure) -> signal());
            } catch (RuntimeException | Error failed) {
                if (failed instanceof Error) {
                    owner.recordRequestFailure(failed);
                }
                rpc =
                        CompletableFuture.completedFuture(
                                applyReply(EngineGrpcClient.CancelReply.FAILED));
            }
        }

        private long applyReply(EngineGrpcClient.CancelReply reply) {
            synchronized (this) {
                applyingReplies++;
            }
            var previous = executingStop.get();
            executingStop.set(true);
            var evidence = EngineGrpcClient.CancelReply.FAILED;
            try {
                owner.onEngineCancelReply(claim, reply);
                evidence = reply;
            } catch (RuntimeException | Error failed) {
                owner.recordRequestFailure(failed);
            } finally {
                if (evidence == EngineGrpcClient.CancelReply.REQUEST_CLEANED) {
                    remoteCleanupConfirmed = true;
                }
                synchronized (this) {
                    pending--;
                    applyingReplies--;
                    notifyProgress();
                }
                if (previous == null) executingStop.remove();
                else executingStop.set(previous);
            }
            return System.nanoTime();
        }

        private void finishUnresolved() {
            if (finished.complete(null)) {
                owner.recordRequestFailure(
                        new IllegalStateException(
                                "delivery stop ended without release proof: request_id="
                                        + context.getRequestId()));
            }
        }

        private long onCancelReply(EngineGrpcClient.CancelReply reply, Throwable failure) {
            if (failure != null || reply == null) {
                reply = EngineGrpcClient.CancelReply.FAILED;
            }
            return applyReply(reply);
        }
    }
}
