package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

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

/**
 * Semantic owner of request deadlines and lifecycle-retention maintenance.
 *
 * <p>The timer never keeps a request map. Exact request generations remain in
 * the lifecycle coordinator; a slot stores the opaque registration returned
 * by this class.
 *
 * <p>Maintenance always completes its three phases in this order:
 * stale-request reduction, exact tombstone removal, then endpoint-orphan
 * sweeping. A failure in one slot is surfaced after the remaining slots and
 * later phases have run, so a single bad generation cannot stop retention
 * progress for every other generation.
 */
final class ExpirationTimer implements AutoCloseable {

    private static final MaintenanceResult NO_MAINTENANCE =
            new MaintenanceResult(0, 0, 0);

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
        private final ExpirationTimer owner;
        private DeadlineState state = DeadlineState.PREPARED;
        private ScheduledFuture<?> scheduled;

        private DeadlineRegistration(ExpirationTimer owner) {
            this.owner = owner;
        }

        private synchronized void installScheduled(
                ScheduledFuture<?> exactScheduled) {
            if (scheduled != null) {
                throw new IllegalStateException(
                        "deadline already owns a scheduled task");
            }
            scheduled = exactScheduled;
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
        private RequestDeadline(ExpirationTimer owner) {
            super(owner);
        }
    }

    /** Exact one-shot capability for one delivered request's acceptance deadline. */
    static final class AcceptanceDeadline extends DeadlineRegistration {
        private AcceptanceDeadline(ExpirationTimer owner) {
            super(owner);
        }
    }

    private final RequestRegistry lifecycle;
    private final ConfigService config;
    private final BatchSchedulerReporter reporter;
    private final LongSupplier clock;
    private final ScheduledThreadPoolExecutor executor;
    private final Object acceptanceGate = new Object();
    private CloseState closeState = CloseState.OPEN;
    private int inflightRegistrations;
    private Throwable closeFailure;

    ExpirationTimer(
            RequestRegistry lifecycle,
            ConfigService config,
            BatchSchedulerReporter reporter) {
        this(lifecycle, config, reporter, System::currentTimeMillis);
    }

    ExpirationTimer(
            RequestRegistry lifecycle,
            ConfigService config,
            BatchSchedulerReporter reporter,
            LongSupplier clock) {
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.config = Objects.requireNonNull(config, "config");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
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
    RequestDeadline attachRequestDeadline(
            RequestSlot exactSlot,
            long deadlineAtMs) {
        if (lifecycle.isShuttingDown()) {
            return null;
        }
        try {
            return register(
                    exactSlot,
                    new RequestDeadline(this),
                    delayUntil(deadlineAtMs),
                    this::installRequestDeadline,
                    this::requestDeadlineExpired);
        } catch (RuntimeException timerStopped) {
            if (lifecycle.isShuttingDown()) {
                return null;
            }
            throw timerStopped;
        }
    }

    /**
     * Register one acceptance deadline relative to the current clock value.
     *
     * @return its exact slot-owned capability, or null when the slot rejected
     *         installation because another lifecycle transition already won
     */
    AcceptanceDeadline registerAcceptanceDeadline(
            RequestSlot exactSlot,
            long timeoutMs) {
        if (timeoutMs < 0L) {
            throw new IllegalArgumentException(
                    "timeoutMs must be non-negative");
        }
        if (lifecycle.isShuttingDown()) {
            return null;
        }
        try {
            return register(
                    exactSlot,
                    new AcceptanceDeadline(this),
                    timeoutMs,
                    this::installAcceptanceDeadline,
                    this::acceptanceDeadlineExpired);
        } catch (RuntimeException timerStopped) {
            if (lifecycle.isShuttingDown()) {
                return null;
            }
            throw timerStopped;
        }
    }

    boolean cancel(RequestDeadline exactDeadline) {
        return requireOwner(exactDeadline).cancel();
    }

    boolean cancel(AcceptanceDeadline exactDeadline) {
        return requireOwner(exactDeadline).cancel();
    }

    void release(RequestSlot.AdmissionCleanup cleanup) {
        if (cleanup == null) {
            return;
        }
        try {
            cleanup.release(this);
        } catch (Throwable failure) {
            Logger.error("Admission cleanup isolated", failure);
        }
    }

    /** Release exact terminal resources; the terminal reducer aggregates failure. */
    void release(RequestSlot.TerminalResources resources) {
        if (resources != null) {
            resources.release(this);
        }
    }

    /** Run one complete maintenance pass using one dynamic policy snapshot. */
    MaintenanceResult maintain(
            BiConsumer<Long, LongPredicate> exactSweeper) {
        if (lifecycle.isShuttingDown()
                || !config.loadBalanceConfig().isQueue()) {
            return NO_MAINTENANCE;
        }
        long ttlMs = config.loadBalanceConfig()
                .queueScheduler()
                .getLifecycle()
                .getStaleInflightTimeoutMs();
        if (ttlMs < 0L) {
            throw new IllegalArgumentException(
                    "stale inflight timeout must be non-negative");
        }
        long nowMs = clock.getAsLong();
        List<RequestSlot> exactSlots = List.of();
        Throwable failure = null;
        try {
            exactSlots = lifecycle.snapshotSlots();
        } catch (RuntimeException | Error snapshotFailure) {
            failure = snapshotFailure;
        }

        int staleReduced = 0;
        for (RequestSlot exactSlot : exactSlots) {
            try {
                if (lifecycle.reduceStale(exactSlot, nowMs, ttlMs)) {
                    staleReduced++;
                }
            } catch (RuntimeException | Error reductionFailure) {
                failure = append(failure, reductionFailure);
            }
        }

        int tombstonesRemoved = 0;
        long tombstoneCutoff = subtractSaturated(nowMs, ttlMs);
        for (RequestSlot exactSlot : exactSlots) {
            try {
                if (lifecycle.removeExactTombstone(
                        exactSlot, tombstoneCutoff)) {
                    tombstonesRemoved++;
                }
            } catch (RuntimeException | Error removalFailure) {
                failure = append(failure, removalFailure);
            }
        }

        try {
            exactSweeper.accept(ttlMs, lifecycle::ownsRequestGeneration);
        } catch (RuntimeException | Error sweepFailure) {
            failure = append(failure, sweepFailure);
        }
        rethrow(failure);
        MaintenanceResult result = new MaintenanceResult(
                exactSlots.size(), staleReduced, tombstonesRemoved);
        if (staleReduced > 0) {
            // Scheduler-ledger eviction: report through the split-by-ledger
            // series (role=SCHEDULER + engineIp="scheduler" + reason) so it
            // is no longer mislabelled as a PREFILL endpoint series. This
            // architecture has a single stale-inflight exit, so the reason
            // bucket is always "ttl".
            reporter.reportSchedulerInflightTtlExpired(
                    "ttl", staleReduced);
            Logger.info(
                    "event=scheduler_inflight_ttl_eviction evicted={} scanned={}",
                    staleReduced, exactSlots.size());
        }
        return result;
    }

    private <D extends DeadlineRegistration> D register(
            RequestSlot exactSlot,
            D exact,
            long delayMs,
            BiPredicate<RequestSlot, D> install,
            BiConsumer<RequestSlot, D> expire) {
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

    private boolean installRequestDeadline(
            RequestSlot exactSlot,
            RequestDeadline exactDeadline) {
        synchronized (exactSlot) {
            return lifecycle.isCurrent(exactSlot)
                    && exactSlot.installRequestDeadline(exactDeadline);
        }
    }

    private boolean installAcceptanceDeadline(
            RequestSlot exactSlot,
            AcceptanceDeadline exactDeadline) {
        synchronized (exactSlot) {
            return lifecycle.isCurrent(exactSlot)
                    && exactSlot.installAcceptanceDeadline(exactDeadline);
        }
    }

    private void requestDeadlineExpired(
            RequestSlot exactSlot,
            RequestDeadline exactDeadline) {
        RequestSlot.RequestDeadlineExpiry expiry;
        synchronized (exactSlot) {
            expiry = exactSlot.expireRequestDeadline(exactDeadline);
        }
        if (expiry == RequestSlot.RequestDeadlineExpiry.CANCEL_REQUEST) {
            lifecycle.cancelForDeadline(exactSlot);
        }
    }

    private void acceptanceDeadlineExpired(
            RequestSlot exactSlot,
            AcceptanceDeadline exactDeadline) {
        RequestSlot.AcceptanceExpiry expiry;
        synchronized (exactSlot) {
            expiry = exactSlot.expireAcceptanceDeadline(exactDeadline);
        }
        lifecycle.acceptanceExpired(expiry);
    }

    private DetachedDeadlines detachDeadlinesForClose(
            RequestSlot exactSlot) {
        synchronized (exactSlot) {
            return exactSlot.detachDeadlinesForTimerClose();
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
        List<RequestSlot> exactSlots;
        try {
            exactSlots = lifecycle.snapshotSlots();
        } catch (RuntimeException | Error snapshotFailure) {
            return snapshotFailure;
        }

        Throwable failure = null;
        for (RequestSlot exactSlot : exactSlots) {
            DetachedDeadlines detached;
            try {
                detached = detachDeadlinesForClose(exactSlot);
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
