package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkState;

/**
 * Runs asynchronous frontend completion tasks on dedicated workers. External Future
 * mutations use completeNow on their caller to preserve synchronous return semantics.
 * Owns execution registration and drain, including callback-reentrant close; the
 * scheduler owns request arbitration, lock checks, deadlines and ACK reporting.
 */
final class ResponseCompletionExecutor implements AutoCloseable {

    /**
     * Reserves a close obligation before submission. The caller closes unused registrations;
     * submit or completeNow releases accepted registrations after execution, including failure.
     * Contains no request state or response policy.
     */
    static final class CompletionRegistration {
        private final ResponseCompletionExecutor owner;
        private final AtomicBoolean closed = new AtomicBoolean();

        CompletionRegistration(ResponseCompletionExecutor owner) {
            this.owner = Objects.requireNonNull(owner);
        }

        void close() {
            if (closed.compareAndSet(false, true)) { owner.exitCompletion(this); }
        }
    }

    private final ThreadPoolExecutor completionWorkers;
    private final ExecutorService rejectedTaskWorker = Executors.newSingleThreadExecutor(
            Thread.ofPlatform().daemon().name("response-completion-recovery-", 1).factory());

    private final Object lifecycleMonitor = new Object();

    private final ThreadLocal<Boolean> insideCompletion =
            new ThreadLocal<>();

    /** Null while open; otherwise the shared, uninterruptible close result. */
    private CompletableFuture<Throwable> closeResult;

    private final Set<CompletionRegistration> registrations =
            Collections.newSetFromMap(new IdentityHashMap<>());

    ResponseCompletionExecutor(int configuredWorkers) {
        completionWorkers = new ThreadPoolExecutor(
                configuredWorkers,
                configuredWorkers,
                0L,
                TimeUnit.MILLISECONDS,
                // Queue completions so a busy completion executor never runs client callbacks
                // inline on a decision thread. RequestContext owns request lifetime; the
                // executor owns only these in-flight frontend completions.
                new LinkedBlockingQueue<>(),
                Thread.ofPlatform().daemon().name("response-completion-", 0).factory(),
                new ThreadPoolExecutor.AbortPolicy());
        completionWorkers.prestartAllCoreThreads();
    }

    CompletionRegistration tryRegister() {
        synchronized (lifecycleMonitor) {
            if (closeResult != null) { return null; }
            var registration = new CompletionRegistration(this);
            registrations.add(registration);
            return registration;
        }
    }

    private void exitCompletion(CompletionRegistration registration) {
        synchronized (lifecycleMonitor) {
            registrations.remove(registration);
            if (registrations.isEmpty()) { lifecycleMonitor.notifyAll(); }
        }
    }

    private void requireOwnedRegistration(CompletionRegistration registration) {
        checkState(registration.owner == this, "completion registration belongs to another executor");
    }

    /** Runs a scheduler-owned completion operation, without interpreting request facts. */
    void submit(CompletionRegistration registration, BooleanSupplier operation) {
        try {
            requireOwnedRegistration(registration);
            try {
                completionWorkers.execute(() -> completeNow(registration, operation));
            } catch (RejectedExecutionException rejection) {
                // An accepted operation must not run client callbacks on its submitter.
                rejectedTaskWorker.execute(() -> completeNow(registration, operation));
            }
        } catch (RuntimeException | Error failure) {
            registration.close();
            throw failure;
        }
    }

    /** External Future mutations stay synchronous and share callback-reentrant drain. */
    boolean completeNow(CompletionRegistration registration, BooleanSupplier operation) {
        boolean outermost = false;
        try {
            requireOwnedRegistration(registration);
            outermost = insideCompletion.get() == null;
            if (outermost) { insideCompletion.set(Boolean.TRUE); }
            return operation.getAsBoolean();
        } finally {
            if (outermost) { insideCompletion.remove(); }
            registration.close();
        }
    }

    // ── 关闭：停止接收、等待在途发布、关闭线程池 ──
    @Override
    public void close() {
        boolean reentrant = insideCompletion.get() != null;
        boolean alreadyClosing;
        CompletableFuture<Throwable> sharedCloseResult;
        synchronized (lifecycleMonitor) {
            alreadyClosing = closeResult != null;
            if (!alreadyClosing) {
                closeResult = new CompletableFuture<>();
            }
            sharedCloseResult = closeResult;
        }
        if (alreadyClosing) {
            // A callback cannot wait for itself; external callers join the same result.
            if (!reentrant || sharedCloseResult.isDone()) {
                Failures.rethrow(sharedCloseResult.join(), "response completion executor close failed");
            }
            return;
        }

        if (reentrant) {
            try {
                Thread closer = new Thread(
                        this::finishClose,
                        "response-completion-close");
                closer.setDaemon(false);
                closer.start();
            } catch (RuntimeException | Error startFailure) {
                Failures.run(startFailure, completionWorkers::shutdown);
                Failures.run(startFailure, rejectedTaskWorker::shutdown);
                sharedCloseResult.complete(startFailure);
                throw startFailure;
            }
            return;
        }
        finishClose();
        Failures.rethrow(sharedCloseResult.join(), "response completion executor close failed");
    }

    private void finishClose() {
        boolean interrupted = false;
        synchronized (lifecycleMonitor) {
            while (!registrations.isEmpty()) {
                try {
                    lifecycleMonitor.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }

        Throwable failure = null;
        try {
            failure = Failures.run(null, completionWorkers::shutdown);
            failure = Failures.run(failure, rejectedTaskWorker::shutdown);
            while (failure == null && (!completionWorkers.isTerminated() || !rejectedTaskWorker.isTerminated())) {
                try {
                    completionWorkers.awaitTermination(1, TimeUnit.DAYS);
                    rejectedTaskWorker.awaitTermination(1, TimeUnit.DAYS);
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        } catch (Throwable shutdownFailure) {
            failure = Failures.append(failure, shutdownFailure);
        } finally {
            closeResult.complete(failure);
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

}
