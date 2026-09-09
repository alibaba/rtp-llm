package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
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
 * the request registry; a slot stores the opaque registration returned
 * by this class.
 *
 * <p>Maintenance removes settled tombstones before sweeping endpoint orphans.
 * A request's inactivity deadline bounds local ownership independently of
 * Engine cancellation acknowledgements or terminal status delivery.
 */
final class ExpirationTimer implements AutoCloseable {

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
            DecisionDeadline decisionDeadline, InactivityDeadline inactivityDeadline) {
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

    /** Wake-up to recheck the latest request activity, not proof of expiration. */
    static final class InactivityDeadline extends DeadlineRegistration {
        private InactivityDeadline(ExpirationTimer owner) { super(owner); }
    }

    /** Exact one-shot capability for request visibility and PD handoff detection. */
    static final class DecisionDeadline extends DeadlineRegistration {
        private final long deadlineAtMs;
        private DecisionDeadline(ExpirationTimer owner, long deadlineAtMs) {
            super(owner);
            this.deadlineAtMs = deadlineAtMs;
        }
        long deadlineAtMs() { return deadlineAtMs; }
    }

    private final RequestRegistry lifecycle;
    private final ConfigService config;
    private final BatchSchedulerReporter reporter;
    private final LongSupplier clock;
    private final ScheduledThreadPoolExecutor executor;
    private final Object registrationMonitor = new Object();
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
     * Register one exact stage deadline from the current delivery evidence.
     *
     * @return its exact slot-owned capability, or null when the slot rejected
     *         installation because another lifecycle transition already won
     */
    DecisionDeadline registerDecisionDeadline(
            RequestSlot exactSlot,
            long deadlineAtMs) {
        if (lifecycle.isShuttingDown()) {
            return null;
        }
        try {
            return register(
                    exactSlot,
                    new DecisionDeadline(this, deadlineAtMs),
                    delayUntil(deadlineAtMs),
                    this::installDecisionDeadline,
                    this::decisionDeadlineExpired);
        } catch (RuntimeException timerStopped) {
            if (lifecycle.isShuttingDown()) {
                return null;
            }
            throw timerStopped;
        }
    }

    InactivityDeadline attachInactivityDeadline(RequestSlot slot) {
        if (lifecycle.isShuttingDown()) {
            return null;
        }
        OptionalLong deadlineAtMs;
        synchronized (slot) {
            if (!lifecycle.isCurrentSlot(slot)) {
                return null;
            }
            deadlineAtMs = slot.inactivityDeadlineAtMs();
        }
        if (deadlineAtMs.isEmpty()) {
            return null;
        }
        try {
            return register(slot, new InactivityDeadline(this), delayUntil(deadlineAtMs.getAsLong()),
                    (owner, exact) -> {
                        synchronized (owner) {
                            return lifecycle.isCurrentSlot(owner) && owner.installInactivityDeadline(exact);
                        }
                    }, this::inactivityDeadlineExpired);
        } catch (RuntimeException timerStopped) {
            if (lifecycle.isShuttingDown()) {
                return null;
            }
            throw timerStopped;
        }
    }

    private void inactivityDeadlineExpired(RequestSlot slot, InactivityDeadline exact) {
        synchronized (slot) {
            if (!slot.expireInactivityDeadline(exact)) {
                return;
            }
        }
        try {
            lifecycle.expireInactiveRequest(slot, clock.getAsLong());
        } finally {
            // Engine facts only renew the timestamp. Rearm when the old wake-up
            // fires, so frequent status reports do not create new timer tasks.
            attachInactivityDeadline(slot);
        }
    }

    boolean cancel(InactivityDeadline exactDeadline) {
        return requireOwner(exactDeadline).cancel();
    }

    boolean cancel(RequestDeadline exactDeadline) {
        return requireOwner(exactDeadline).cancel();
    }

    boolean cancel(DecisionDeadline exactDeadline) {
        return requireOwner(exactDeadline).cancel();
    }

    void release(DecisionDeadline cleanup) {
        if (cleanup == null) {
            return;
        }
        try {
            cancel(cleanup);
        } catch (Throwable failure) {
            Logger.error("Decision timer cleanup failed", failure);
        }
    }

    /** Release exact terminal resources; the terminal reducer aggregates failure. */
    void release(RequestSlot.TerminalResources resources) {
        if (resources != null) {
            resources.release(this);
        }
    }

    /** Run one complete maintenance pass using one dynamic policy snapshot. */
    void maintain(
            BiConsumer<Long, LongPredicate> exactSweeper) {
        if (lifecycle.isShuttingDown()) {
            return;
        }
        long ttlMs = config.loadBalanceConfig().getWorkerRegistry().getHealth().getStatusStaleAfterMs();
        long nowMs = clock.getAsLong();
        List<RequestSlot> exactSlots = List.of();
        Throwable failure = null;
        try {
            exactSlots = lifecycle.snapshotSlots();
        } catch (RuntimeException | Error snapshotFailure) {
            failure = snapshotFailure;
        }

        long tombstoneCutoff = subtractSaturated(nowMs, ttlMs);
        for (RequestSlot exactSlot : exactSlots) {
            try {
                lifecycle.removeExactTombstone(exactSlot, tombstoneCutoff);
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
            return lifecycle.isCurrentSlot(exactSlot)
                    && exactSlot.installRequestDeadline(exactDeadline);
        }
    }

    private boolean installDecisionDeadline(
            RequestSlot exactSlot,
            DecisionDeadline exactDeadline) {
        synchronized (exactSlot) {
            return lifecycle.isCurrentSlot(exactSlot)
                    && exactSlot.installDecisionDeadline(exactDeadline);
        }
    }

    private void requestDeadlineExpired(
            RequestSlot exactSlot,
            RequestDeadline exactDeadline) {
        boolean cancelRequest;
        synchronized (exactSlot) {
            cancelRequest = exactSlot.expireRequestDeadline(exactDeadline);
        }
        if (cancelRequest) {
            lifecycle.cancelForDeadline(exactSlot);
        }
    }

    private void decisionDeadlineExpired(
            RequestSlot exactSlot,
            DecisionDeadline exactDeadline) {
        RequestSlot.DecisionExpiry expiry;
        synchronized (exactSlot) {
            expiry = exactSlot.expireDecisionDeadline(exactDeadline);
        }
        if (expiry != null) {
            lifecycle.decisionExpired(expiry);
        }
    }

    private DetachedDeadlines detachDeadlinesForClose(
            RequestSlot exactSlot) {
        synchronized (exactSlot) {
            return exactSlot.detachDeadlinesForTimerClose();
        }
    }

    private void beginRegistration() {
        synchronized (registrationMonitor) {
            if (closeState != CloseState.OPEN) {
                throw new RejectedExecutionException(
                        "ExpirationTimer is closing");
            }
            inflightRegistrations++;
        }
    }

    private void endRegistration() {
        synchronized (registrationMonitor) {
            if (inflightRegistrations <= 0) {
                throw new IllegalStateException(
                        "ExpirationTimer registration count underflow");
            }
            inflightRegistrations--;
            if (inflightRegistrations == 0) {
                registrationMonitor.notifyAll();
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
        synchronized (registrationMonitor) {
            if (closeState == CloseState.OPEN) {
                closeState = CloseState.CLOSING;
                closeOwner = true;
            }
            while (closeState == CloseState.CLOSING
                    && (!closeOwner || inflightRegistrations != 0)) {
                try {
                    registrationMonitor.wait();
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
            synchronized (registrationMonitor) {
                closeFailure = failure;
                closeState = CloseState.CLOSED;
                registrationMonitor.notifyAll();
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
                    detached.decisionDeadline(), failure);
            failure = cancelDetached(detached.inactivityDeadline(), failure);
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
