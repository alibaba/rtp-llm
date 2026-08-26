package org.flexlb.balance.scheduler;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.BiPredicate;
import java.util.function.LongPredicate;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

/**
 * Semantic owner of request deadlines and lifecycle-retention maintenance.
 *
 * <p>The timer never keeps a request map. Exact request generations remain in
 * the supplied {@link SlotDirectory}; a slot stores the opaque registration
 * returned by this class. Scheduled callbacks and maintenance scans can only
 * ask the narrow {@link Sink} to reduce that exact generation.
 *
 * <p>Maintenance always completes its three phases in this order:
 * stale-request reduction, exact tombstone removal, then endpoint-orphan
 * sweeping. A failure in one slot is surfaced after the remaining slots and
 * later phases have run, so a single bad generation cannot stop retention
 * progress for every other generation.
 */
final class ExpirationTimer<S> implements AutoCloseable {

    /** Canonical exact-generation directory; implementations must return a snapshot. */
    interface SlotDirectory<S> {
        List<S> snapshot();

        /** Remove only {@code exactSlot} when it is an eligible older tombstone. */
        boolean removeExactTombstone(S exactSlot, long updatedBeforeMs);

        /** Whether one non-tombstoned scheduler generation still owns this id. */
        boolean ownsRequestGeneration(long requestId);
    }

    /** Lifecycle mutations implemented by the exact slot aggregate boundary. */
    interface Sink<S> {
        /** Install the exact capability in the slot before returning true. */
        boolean installRequestDeadline(
                S exactSlot, RequestDeadline exactDeadline);

        /** Install the exact capability in the slot before returning true. */
        boolean installAcceptanceDeadline(
                S exactSlot, AcceptanceDeadline exactDeadline);

        /** Reduce a consumed request deadline after checking exact slot identity. */
        void requestDeadlineExpired(
                S exactSlot, RequestDeadline exactDeadline);

        /** Reduce a consumed acceptance deadline after checking exact slot identity. */
        void acceptanceDeadlineExpired(
                S exactSlot, AcceptanceDeadline exactDeadline);

        /**
         * Atomically detach both exact deadline capabilities during timer
         * shutdown. The implementation must linearize this operation at the
         * exact slot aggregate boundary.
         */
        DetachedDeadlines detachDeadlinesForClose(S exactSlot);

        /**
         * Reduce one stale exact generation if eligible.
         *
         * @return true only when this call claimed its stale transition
         */
        boolean reduceStale(S exactSlot, long nowMs, long staleTtlMs);
    }

    /** Synchronous endpoint-ledger sweep; the ownership predicate must not be retained. */
    @FunctionalInterface
    interface OrphanSweeper {
        void sweep(long staleTtlMs, LongPredicate ownsRequest);
    }

    /** Dynamically supplied retention policy captured once per maintenance pass. */
    record RetentionPolicy(long staleTtlMs, long tombstoneRetentionMs) {
        RetentionPolicy {
            if (staleTtlMs < 0L) {
                throw new IllegalArgumentException(
                        "staleTtlMs must be non-negative");
            }
            if (tombstoneRetentionMs < 0L) {
                throw new IllegalArgumentException(
                        "tombstoneRetentionMs must be non-negative");
            }
        }
    }

    /** Observable result of one complete, ordered maintenance pass. */
    record MaintenanceResult(
            int scannedSlots,
            int staleReduced,
            int tombstonesRemoved) {
    }

    private enum DeadlineState {
        PREPARED,
        FIRED_BEFORE_INSTALL,
        ARMED,
        CONSUMED,
        CANCELED
    }

    /** Exact capabilities detached together from one slot during shutdown. */
    record DetachedDeadlines(
            RequestDeadline requestDeadline,
            AcceptanceDeadline acceptanceDeadline) {
    }

    private enum CloseState {
        OPEN,
        CLOSING,
        CLOSED
    }

    private abstract static class DeadlineRegistration {
        private final ExpirationTimer<?> owner;
        private DeadlineState state = DeadlineState.PREPARED;
        private ScheduledFuture<?> scheduled;

        private DeadlineRegistration(ExpirationTimer<?> owner) {
            this.owner = Objects.requireNonNull(owner, "owner");
        }

        private synchronized void installScheduled(
                ScheduledFuture<?> exactScheduled) {
            if (scheduled != null) {
                throw new IllegalStateException(
                        "deadline already owns a scheduled task");
            }
            scheduled = Objects.requireNonNull(exactScheduled, "exactScheduled");
            if (state == DeadlineState.CANCELED) {
                exactScheduled.cancel(false);
            }
        }

        /** Publish only after the exact slot has stored this capability. */
        final synchronized boolean publishAfterInstall() {
            return switch (state) {
                case PREPARED -> {
                    state = DeadlineState.ARMED;
                    yield false;
                }
                case FIRED_BEFORE_INSTALL -> {
                    state = DeadlineState.CONSUMED;
                    yield true;
                }
                case CANCELED, CONSUMED -> false;
                case ARMED -> throw new IllegalStateException(
                        "deadline capability was published twice");
            };
        }

        final synchronized boolean consume() {
            return switch (state) {
                case PREPARED -> {
                    state = DeadlineState.FIRED_BEFORE_INSTALL;
                    yield false;
                }
                case ARMED -> {
                    state = DeadlineState.CONSUMED;
                    yield true;
                }
                case FIRED_BEFORE_INSTALL, CONSUMED, CANCELED -> false;
            };
        }

        final boolean cancel() {
            ScheduledFuture<?> exactScheduled;
            synchronized (this) {
                if (state == DeadlineState.CONSUMED
                        || state == DeadlineState.CANCELED) {
                    return false;
                }
                state = DeadlineState.CANCELED;
                exactScheduled = scheduled;
            }
            if (exactScheduled != null) {
                exactScheduled.cancel(false);
            }
            return true;
        }
    }

    /** Exact one-shot capability for one request's absolute scheduling deadline. */
    static final class RequestDeadline extends DeadlineRegistration {
        private RequestDeadline(ExpirationTimer<?> owner) {
            super(owner);
        }
    }

    /** Exact one-shot capability for one delivered request's acceptance deadline. */
    static final class AcceptanceDeadline extends DeadlineRegistration {
        private AcceptanceDeadline(ExpirationTimer<?> owner) {
            super(owner);
        }
    }

    private final SlotDirectory<S> slotDirectory;
    private final Sink<S> sink;
    private final Supplier<RetentionPolicy> retentionPolicy;
    private final LongSupplier clock;
    private final ScheduledThreadPoolExecutor executor;
    private final Object acceptanceGate = new Object();
    private CloseState closeState = CloseState.OPEN;
    private int inflightRegistrations;
    private Throwable closeFailure;

    ExpirationTimer(
            SlotDirectory<S> slotDirectory,
            Sink<S> sink,
            Supplier<RetentionPolicy> retentionPolicy) {
        this(slotDirectory, sink, retentionPolicy,
                System::currentTimeMillis);
    }

    ExpirationTimer(
            SlotDirectory<S> slotDirectory,
            Sink<S> sink,
            Supplier<RetentionPolicy> retentionPolicy,
            LongSupplier clock) {
        this.slotDirectory = Objects.requireNonNull(
                slotDirectory, "slotDirectory");
        this.sink = Objects.requireNonNull(sink, "sink");
        this.retentionPolicy = Objects.requireNonNull(
                retentionPolicy, "retentionPolicy");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.executor = new ScheduledThreadPoolExecutor(1, runnable -> {
            Thread thread = new Thread(
                    runnable, "request-scheduler-expiration");
            thread.setDaemon(true);
            return thread;
        }, new ThreadPoolExecutor.AbortPolicy());
        executor.setRemoveOnCancelPolicy(true);
        executor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
    }

    /**
     * Register one absolute request deadline.
     *
     * @return its exact slot-owned capability, or null when the slot rejected
     *         installation because another lifecycle transition already won
     */
    RequestDeadline registerRequestDeadlineAt(
            S exactSlot, long deadlineAtMs) {
        return register(
                exactSlot,
                new RequestDeadline(this),
                delayUntil(deadlineAtMs),
                sink::installRequestDeadline,
                sink::requestDeadlineExpired);
    }

    /**
     * Register one acceptance deadline relative to the current clock value.
     *
     * @return its exact slot-owned capability, or null when the slot rejected
     *         installation because another lifecycle transition already won
     */
    AcceptanceDeadline registerAcceptanceDeadlineAfter(
            S exactSlot, long timeoutMs) {
        if (timeoutMs < 0L) {
            throw new IllegalArgumentException(
                    "timeoutMs must be non-negative");
        }
        return register(
                exactSlot,
                new AcceptanceDeadline(this),
                timeoutMs,
                sink::installAcceptanceDeadline,
                sink::acceptanceDeadlineExpired);
    }

    boolean cancel(RequestDeadline exactDeadline) {
        return requireOwner(exactDeadline).cancel();
    }

    boolean cancel(AcceptanceDeadline exactDeadline) {
        return requireOwner(exactDeadline).cancel();
    }

    /** Run one complete maintenance pass using one dynamic policy snapshot. */
    MaintenanceResult maintainNow(OrphanSweeper exactSweeper) {
        RetentionPolicy policy = retentionPolicy.get();
        long nowMs = clock.getAsLong();
        List<S> exactSlots = List.of();
        Throwable failure = null;
        try {
            exactSlots = List.copyOf(slotDirectory.snapshot());
        } catch (RuntimeException | Error snapshotFailure) {
            failure = snapshotFailure;
        }

        int staleReduced = 0;
        for (S exactSlot : exactSlots) {
            try {
                if (sink.reduceStale(
                        exactSlot, nowMs, policy.staleTtlMs())) {
                    staleReduced++;
                }
            } catch (RuntimeException | Error reductionFailure) {
                failure = append(failure, reductionFailure);
            }
        }

        int tombstonesRemoved = 0;
        long tombstoneCutoff = subtractSaturated(
                nowMs, policy.tombstoneRetentionMs());
        for (S exactSlot : exactSlots) {
            try {
                if (slotDirectory.removeExactTombstone(
                        exactSlot, tombstoneCutoff)) {
                    tombstonesRemoved++;
                }
            } catch (RuntimeException | Error removalFailure) {
                failure = append(failure, removalFailure);
            }
        }

        try {
            exactSweeper.sweep(
                    policy.staleTtlMs(),
                    slotDirectory::ownsRequestGeneration);
        } catch (RuntimeException | Error sweepFailure) {
            failure = append(failure, sweepFailure);
        }
        rethrow(failure);
        return new MaintenanceResult(
                exactSlots.size(), staleReduced, tombstonesRemoved);
    }

    private <D extends DeadlineRegistration> D register(
            S exactSlot,
            D exact,
            long delayMs,
            BiPredicate<S, D> install,
            BiConsumer<S, D> expire) {
        beginRegistration();
        try {
            schedule(exact, () -> {
                if (exact.consume()) {
                    expire.accept(exactSlot, exact);
                }
            }, delayMs);
            boolean installed;
            try {
                installed = install.test(exactSlot, exact);
            } catch (RuntimeException | Error installationFailure) {
                exact.cancel();
                throw installationFailure;
            }
            if (!installed) {
                exact.cancel();
                return null;
            }
            if (exact.publishAfterInstall()) {
                expire.accept(exactSlot, exact);
            }
            return exact;
        } finally {
            endRegistration();
        }
    }

    private void beginRegistration() {
        synchronized (acceptanceGate) {
            if (closeState != CloseState.OPEN) {
                throw new RejectedExecutionException(
                        "ExpirationTimer is closing");
            }
            inflightRegistrations++;
        }
    }

    private void endRegistration() {
        synchronized (acceptanceGate) {
            if (inflightRegistrations <= 0) {
                throw new IllegalStateException(
                        "ExpirationTimer registration count underflow");
            }
            inflightRegistrations--;
            if (inflightRegistrations == 0) {
                acceptanceGate.notifyAll();
            }
        }
    }

    private void schedule(
            DeadlineRegistration exact,
            Runnable callback,
            long delayMs) {
        ScheduledFuture<?> scheduled;
        try {
            scheduled = executor.schedule(
                    callback, delayMs, TimeUnit.MILLISECONDS);
        } catch (RejectedExecutionException rejected) {
            exact.cancel();
            throw rejected;
        }
        exact.installScheduled(scheduled);
    }

    private long delayUntil(long deadlineAtMs) {
        long nowMs = clock.getAsLong();
        if (deadlineAtMs <= nowMs) {
            return 0L;
        }
        long delayMs = deadlineAtMs - nowMs;
        return delayMs < 0L ? Long.MAX_VALUE : delayMs;
    }

    private DeadlineRegistration requireOwner(DeadlineRegistration exact) {
        if (exact.owner != this) {
            throw new IllegalArgumentException(
                    "deadline belongs to another ExpirationTimer");
        }
        return exact;
    }

    private static long subtractSaturated(long value, long decrement) {
        try {
            return Math.subtractExact(value, decrement);
        } catch (ArithmeticException underflow) {
            return Long.MIN_VALUE;
        }
    }

    private static Throwable append(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
    }

    @Override
    public void close() {
        boolean interrupted = false;
        boolean closeOwner = false;
        synchronized (acceptanceGate) {
            if (closeState == CloseState.OPEN) {
                closeState = CloseState.CLOSING;
                closeOwner = true;
            }
            while (closeState == CloseState.CLOSING
                    && (!closeOwner || inflightRegistrations != 0)) {
                try {
                    acceptanceGate.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
            if (!closeOwner) {
                Throwable completedFailure = closeFailure;
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                rethrow(completedFailure);
                return;
            }
        }

        Throwable failure = detachAllDeadlines();
        try {
            executor.shutdownNow();
        } catch (RuntimeException | Error shutdownFailure) {
            failure = append(failure, shutdownFailure);
        } finally {
            synchronized (acceptanceGate) {
                closeFailure = failure;
                closeState = CloseState.CLOSED;
                acceptanceGate.notifyAll();
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        rethrow(failure);
    }

    private Throwable detachAllDeadlines() {
        List<S> exactSlots;
        try {
            exactSlots = List.copyOf(slotDirectory.snapshot());
        } catch (RuntimeException | Error snapshotFailure) {
            return snapshotFailure;
        }

        Throwable failure = null;
        for (S exactSlot : exactSlots) {
            DetachedDeadlines detached;
            try {
                detached = Objects.requireNonNull(
                        sink.detachDeadlinesForClose(exactSlot),
                        "sink.detachDeadlinesForClose()");
            } catch (RuntimeException | Error detachFailure) {
                failure = append(failure, detachFailure);
                continue;
            }
            failure = cancelDetached(
                    detached.requestDeadline(), failure);
            failure = cancelDetached(
                    detached.acceptanceDeadline(), failure);
        }
        return failure;
    }

    private Throwable cancelDetached(
            DeadlineRegistration exact,
            Throwable failure) {
        if (exact == null) {
            return failure;
        }
        try {
            requireOwner(exact).cancel();
        } catch (RuntimeException | Error cancelFailure) {
            return append(failure, cancelFailure);
        }
        return failure;
    }
}
