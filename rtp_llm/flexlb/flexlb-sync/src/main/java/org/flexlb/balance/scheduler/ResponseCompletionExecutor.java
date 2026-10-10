package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.ThreadFactory;
import java.util.function.BooleanSupplier;

/**
 * Runs asynchronous frontend completion tasks on dedicated workers. External Future
 * mutations use completeNow on their caller to preserve synchronous return semantics.
 * Owns execution registration and drain. After request cleanup, the external lifecycle
 * owner closes it. Completion callbacks do not own executor shutdown. The scheduler
 * owns request arbitration, lock checks, deadlines and ACK reporting.
 */
final class ResponseCompletionExecutor implements AutoCloseable {

    /**
     * Reserves a close obligation before submission. The caller closes unused registrations;
     * submit or completeNow releases accepted registrations after execution, including failure.
     * Contains no request state or response policy.
     */
    private enum Phase { RESERVED, SUBMITTED, RUNNING, COMPLETED, REJECTED, CLOSED_COMPLETED, CLOSED_REJECTED }

    final class CompletionRegistration {
        private final AtomicReference<Phase> phase = new AtomicReference<>(Phase.RESERVED);

        private CompletionRegistration() { }

        /** Every accepted response runs asynchronously, including executor rejection recovery. */
        void submit(BooleanSupplier operation) {
            if (!phase.compareAndSet(Phase.RESERVED, Phase.SUBMITTED)) {
                throw new IllegalStateException("response registration already resolved");
            }
            Runnable task = () -> completeNow(operation);
            try { completionWorkers.execute(task); }
            catch (RuntimeException | Error rejected) {
                if (phase.get() != Phase.SUBMITTED) { throw rejected; }
                try { rejectedTaskWorker.execute(task); }
                catch (RuntimeException | Error recoveryRejected) {
                    Throwable failure = Failures.append(rejected, recoveryRejected);
                    if (phase.get() != Phase.SUBMITTED) { throw Failures.propagate(failure, "response submission failed"); }
                    try { emergencyWorkers.newThread(task).start(); }
                    catch (RuntimeException | Error startFailure) {
                        if (phase.compareAndSet(Phase.SUBMITTED, Phase.REJECTED)) { close(); }
                        throw Failures.propagate(Failures.append(failure, startFailure), "response submission failed");
                    }
                }
            }
        }

        /** External mutations stay synchronous; queued/recovered tasks share one execution claim. */
        boolean completeNow(BooleanSupplier operation) {
            Phase observed;
            do {
                observed = phase.get();
                if (observed != Phase.RESERVED && observed != Phase.SUBMITTED) { return false; }
            } while (!phase.compareAndSet(observed, Phase.RUNNING));
            Boolean previous = completing.get();
            completing.set(Boolean.TRUE);
            try { return operation.getAsBoolean(); }
            finally {
                completing.set(previous);
                phase.set(Phase.COMPLETED);
                close();
            }
        }

        boolean hasCompletionOwner() {
            return switch (phase.get()) {
                case SUBMITTED, RUNNING, COMPLETED, CLOSED_COMPLETED -> true;
                default -> false;
            };
        }

        void close() {
            Phase observed;
            Phase closed;
            do {
                observed = phase.get();
                closed = switch (observed) {
                    case RESERVED, REJECTED -> Phase.CLOSED_REJECTED;
                    case COMPLETED -> Phase.CLOSED_COMPLETED;
                    default -> null;
                };
                if (closed == null) { return; }
            } while (!phase.compareAndSet(observed, closed));
            synchronized (registrations) {
                registrations.remove(this);
                if (registrations.isEmpty()) { registrations.notifyAll(); }
            }
        }
    }

    private final ThreadFactory emergencyWorkers = Thread.ofPlatform().daemon().name("response-completion-emergency-", 1).factory();
    private final ThreadLocal<Boolean> completing = new ThreadLocal<>();
    private final ThreadPoolExecutor completionWorkers;
    private final ExecutorService rejectedTaskWorker = Executors.newSingleThreadExecutor(
            Thread.ofPlatform().daemon().name("response-completion-recovery-", 1).factory());

    /** Null while accepting registrations; otherwise the shared final close result. */
    private CompletableFuture<Throwable> closeResult;

    /** Also serves as the monitor protecting registrations and closeResult. */
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
        synchronized (registrations) {
            if (closeResult != null) { return null; }
            var registration = new CompletionRegistration();
            registrations.add(registration);
            return registration;
        }
    }

    boolean isCompletingOnCurrentThread() { return Boolean.TRUE.equals(completing.get()); }

    /** Stops registration, drains accepted completions and closes workers. */
    @Override
    public void close() {
        if (isCompletingOnCurrentThread()) {
            throw new IllegalStateException("response completion cannot wait for its own callback");
        }
        boolean interrupted = false;
        try {
            synchronized (registrations) {
                if (closeResult == null) {
                    closeResult = new CompletableFuture<>();
                }
                while (!registrations.isEmpty()) {
                    try {
                        registrations.wait();
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                    }
                }
                // Drained workers no longer need this monitor. Serialize final
                // closure so concurrent callers receive the same result.
                if (!closeResult.isDone()) {
                    finishClose();
                }
                Failures.rethrow(closeResult.join(), "response completion executor close failed");
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void finishClose() {
        boolean interrupted = false;
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
