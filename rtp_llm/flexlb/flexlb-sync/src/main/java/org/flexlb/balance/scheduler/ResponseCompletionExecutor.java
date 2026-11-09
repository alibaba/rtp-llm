package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.util.Failures;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.util.Failures.recover;

/**
 * Runs asynchronous frontend completion tasks on dedicated workers. External Future
 * mutations use completeNow on their caller to preserve synchronous return semantics.
 * Owns execution registration and drain. After request cleanup, the request scheduler
 * owner closes it. Completion callbacks do not own executor shutdown. The scheduler
 * owns request arbitration, lock checks, deadlines and ACK reporting.
 */
public final class ResponseCompletionExecutor implements AutoCloseable {
    private final ThreadLocal<Boolean> completing = new ThreadLocal<>();
    private final ThreadPoolExecutor completionWorkers;

    /** Null while accepting registrations; otherwise the shared final close result. */
    private CompletableFuture<Throwable> closeResult;

    // Registration's atomic phase owns exact release; the executor only needs a drain count.
    private final Object completionLock = new Object();

    private int pendingCompletions;

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

    /** Stops registration, drains accepted completions and closes workers. */
    @Override
    public void close() {
        if (isCompletingOnCurrentThread()) {
            throw new IllegalStateException("response completion cannot wait for its own callback");
        }
        boolean interrupted = false;
        try {
            synchronized (completionLock) {
                if (closeResult == null) {
                    closeResult = new CompletableFuture<>();
                }
                while (pendingCompletions != 0) {
                    try {
                        completionLock.wait();
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

    public boolean isCompletingOnCurrentThread() {
        return Boolean.TRUE.equals(completing.get());
    }

    boolean register() {
        synchronized (completionLock) {
            if (closeResult != null) {
                return false;
            }
            pendingCompletions++;
            return true;
        }
    }

    void release() {
        synchronized (completionLock) {
            checkState(pendingCompletions > 0, "response completion already released");
            if (--pendingCompletions == 0) completionLock.notifyAll();
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

    void submit(Registration publication, Runnable task) {
        try {
            completionWorkers.execute(task);
        } catch (RuntimeException | Error rejected) {
            publication.reject(rejected);
            throw Failures.propagate(rejected, "response submission failed");
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

    /**
     * One response execution owner, registered before Context freezes the selected outcome.
     */
    static final class Registration implements AutoCloseable {
        private final AtomicReference<Phase> phase = new AtomicReference<>(Phase.RESERVED);
        private final ResponseCompletionExecutor executor;
        final RequestContext owner;

        Registration(ResponseCompletionExecutor executor, RequestContext owner) {
            this.owner = Objects.requireNonNull(owner);
            this.executor = executor;
            checkState(executor.register(), "frontend publication is closed");
        }

        @Override
        public void close() {
            if (phase.compareAndSet(Phase.RESERVED, Phase.CLOSED)) {
                executor.release();
            }
        }

        static Registration reserve(
                ResponseCompletionExecutor executor,
                RequestContext ctx,
                java.util.function.Consumer<Throwable> failures) {
            var publication = recover(failures, () -> new Registration(executor, ctx));
            if (publication == null)
                throw new IllegalStateException("frontend publication is closed");
            return publication;
        }

        boolean hasCompletionOwner() {
            return switch (phase.get()) {
                case SUBMITTED, RUNNING, COMPLETED -> true;
                default -> false;
            };
        }

        void submit(BooleanSupplier completion) {
            if (Thread.holdsLock(owner)) {
                close();
                throw new IllegalStateException("response submission under Context lock");
            }
            checkState(
                    phase.compareAndSet(Phase.RESERVED, Phase.SUBMITTED),
                    "response task already consumed");
            executor.submit(this, () -> run(Phase.SUBMITTED, completion));
        }

        boolean completeNow(BooleanSupplier completion) {
            if (Thread.holdsLock(owner)) {
                close();
                throw new IllegalStateException("response completion under Context lock");
            }
            return run(Phase.RESERVED, completion);
        }

        void reject(Throwable failure) {
            if (phase.compareAndSet(Phase.SUBMITTED, Phase.CLOSED)) {
                try {
                    var data = owner.data();
                    if (data != null && data.getFuture() instanceof RequestFuture future) {
                        future.completeExceptionallyOwned(failure);
                    }
                } finally {
                    executor.release();
                }
            }
        }

        boolean complete(RequestFuture.Completion kind, Response response, Throwable failure) {
            checkState(
                    !Thread.holdsLock(owner),
                    "response callbacks require the Context lock to be released");
            RequestFuture future = (RequestFuture) owner.data().getFuture();
            return switch (kind) {
                case RESPONSE -> future.completeOwned(response);
                case FAILURE -> future.completeExceptionallyOwned(failure);
                case CANCELLATION -> future.cancelOwned();
            };
        }

        private boolean run(Phase expected, BooleanSupplier completion) {
            if (!phase.compareAndSet(expected, Phase.RUNNING)) {
                return false;
            }
            try {
                return executor.invoke(completion);
            } finally {
                phase.set(Phase.COMPLETED);
                executor.release();
            }
        }

        private enum Phase {
            RESERVED,
            SUBMITTED,
            RUNNING,
            COMPLETED,
            CLOSED
        }
    }

    /**
     * External writes remain synchronous and arbitrate through Scheduler; owned writes run outside Context.
     */
    public static final class RequestFuture extends CompletableFuture<Response> {
        private volatile CompletionTarget target;

        public RequestFuture(CompletionTarget target) {
            this.target = Objects.requireNonNull(target);
        }

        @Override
        public boolean complete(Response response) {
            var current = target;
            return current != null && current.complete(Completion.RESPONSE, response, null);
        }

        @Override
        public boolean completeExceptionally(Throwable failure) {
            Objects.requireNonNull(failure);
            var current = target;
            return current != null && current.complete(Completion.FAILURE, null, failure);
        }

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            var current = target;
            return current != null && current.complete(Completion.CANCELLATION, null, null);
        }

        boolean completeOwned(Response response) {
            try {
                return super.complete(response);
            } finally {
                target = null;
            }
        }

        boolean completeExceptionallyOwned(Throwable failure) {
            try {
                return super.completeExceptionally(failure);
            } finally {
                target = null;
            }
        }

        boolean cancelOwned() {
            try {
                return super.cancel(false);
            } finally {
                target = null;
            }
        }

        public enum Completion {
            RESPONSE,
            FAILURE,
            CANCELLATION
        }

        /**
         * Routes a future completion attempt through the request completion owner.
         */
        @FunctionalInterface
        public interface CompletionTarget {
            boolean complete(Completion completion, Response response, Throwable failure);
        }
    }
}
