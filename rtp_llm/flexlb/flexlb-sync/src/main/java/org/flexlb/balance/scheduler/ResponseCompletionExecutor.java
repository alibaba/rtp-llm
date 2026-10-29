package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

/**
 * Runs asynchronous frontend completion tasks on dedicated workers. External Future
 * mutations use completeNow on their caller to preserve synchronous return semantics.
 * Owns execution registration and drain. After request cleanup, the external lifecycle
 * owner closes it. Completion callbacks do not own executor shutdown. The scheduler
 * owns request arbitration, lock checks, deadlines and ACK reporting.
 */
final class ResponseCompletionExecutor implements AutoCloseable {

    private final ThreadLocal<Boolean> completing = new ThreadLocal<>();
    private final ThreadPoolExecutor completionWorkers;

    /** Null while accepting registrations; otherwise the shared final close result. */
    private CompletableFuture<Throwable> closeResult;

    /** Also serves as the monitor protecting registrations and closeResult. */
    private final Set<ResponsePublication> registrations =
            Collections.newSetFromMap(new IdentityHashMap<>());

    ResponseCompletionExecutor(int configuredWorkers) {
        completionWorkers =
                new ThreadPoolExecutor(
                        configuredWorkers,
                        configuredWorkers,
                        0L,
                        TimeUnit.MILLISECONDS,
                        // Queue completions so a busy completion executor never runs client
                        // callbacks
                        // inline on a decision thread. RequestContext owns request lifetime; the
                        // executor owns only these in-flight frontend completions.
                        new LinkedBlockingQueue<>(),
                        Thread.ofPlatform().daemon().name("response-completion-", 0).factory(),
                        new ThreadPoolExecutor.AbortPolicy());
        completionWorkers.prestartAllCoreThreads();
    }

    boolean register(ResponsePublication publication) {
        synchronized (registrations) {
            if (closeResult != null) {
                return false;
            }
            registrations.add(publication);
            return true;
        }
    }

    void release(ResponsePublication publication) {
        synchronized (registrations) {
            registrations.remove(publication);
            if (registrations.isEmpty()) {
                registrations.notifyAll();
            }
        }
    }

    boolean invoke(BooleanSupplier operation) {
        Boolean previous = completing.get();
        completing.set(Boolean.TRUE);
        try {
            return operation.getAsBoolean();
        } finally {
            completing.set(previous);
        }
    }

    void submit(ResponsePublication publication, Runnable task) {
        try {
            completionWorkers.execute(task);
        } catch (RuntimeException | Error rejected) {
            if (publication.isSubmitted()) {
                publication.reject(rejected);
            }
            throw Failures.propagate(rejected, "response submission failed");
        }
    }

    boolean isCompletingOnCurrentThread() {
        return Boolean.TRUE.equals(completing.get());
    }

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
            while (failure == null && !completionWorkers.isTerminated()) {
                try {
                    completionWorkers.awaitTermination(1, TimeUnit.DAYS);
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
