package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;

/**
 * Request-lifecycle integration boundary for semantic deadlines and retention.
 *
 * <p>This controller owns the one {@link ExpirationTimer} instance, but never a
 * request directory. Exact generation lookup, removal, and lifecycle reduction
 * remain behind {@link Lifecycle}; timer callbacks cannot reach the scheduler
 * facade or its request map directly.
 */
final class RequestExpirationController implements AutoCloseable {

    /**
     * Exact request-generation operations required by expiration. Implementations
     * own all slot locking and canonical-directory mutations not performed under
     * the explicit slot locks in this controller's timer sink.
     */
    interface Lifecycle {
        boolean isShuttingDown();

        boolean isCurrent(RequestSlot exactSlot);

        List<RequestSlot> snapshotSlots();

        boolean ownsRequestGeneration(long requestId);

        boolean removeExactTombstone(
                RequestSlot exactSlot, long updatedBeforeMs);

        void cancelForDeadline(RequestSlot exactSlot);

        void acceptanceExpired(RequestSlot.AcceptanceExpiry expiry);

        boolean reduceStale(
                RequestSlot exactSlot, long nowMs, long staleTtlMs);
    }

    private static final ExpirationTimer.MaintenanceResult NO_MAINTENANCE =
            new ExpirationTimer.MaintenanceResult(0, 0, 0);

    private final Lifecycle lifecycle;
    private final ConfigService config;
    private final BatchSchedulerReporter reporter;
    private final ExpirationTimer<RequestSlot> timer;

    RequestExpirationController(
            Lifecycle lifecycle,
            ConfigService config,
            BatchSchedulerReporter reporter) {
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.config = Objects.requireNonNull(config, "config");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.timer = new ExpirationTimer<>(
                new SlotDirectory(),
                new DeadlineSink(),
                this::retentionPolicy);
    }

    ExpirationTimer.RequestDeadline attachRequestDeadline(
            RequestSlot exactSlot, long deadlineAtMs) {
        if (lifecycle.isShuttingDown()) {
            return null;
        }
        try {
            return timer.registerRequestDeadlineAt(exactSlot, deadlineAtMs);
        } catch (RuntimeException timerStopped) {
            if (lifecycle.isShuttingDown()) {
                return null;
            }
            throw timerStopped;
        }
    }

    ExpirationTimer.AcceptanceDeadline registerAcceptanceDeadline(
            RequestSlot exactSlot, long delayMs) {
        if (lifecycle.isShuttingDown()) {
            return null;
        }
        try {
            return timer.registerAcceptanceDeadlineAfter(exactSlot, delayMs);
        } catch (RuntimeException timerStopped) {
            if (lifecycle.isShuttingDown()) {
                return null;
            }
            throw timerStopped;
        }
    }

    boolean cancel(ExpirationTimer.RequestDeadline exactDeadline) {
        return timer.cancel(exactDeadline);
    }

    void release(RequestSlot.AdmissionCleanup cleanup) {
        if (cleanup == null) {
            return;
        }
        try {
            cleanup.release(timer);
        } catch (Throwable failure) {
            Logger.error("Admission cleanup isolated", failure);
        }
    }

    /** Release exact terminal resources; the terminal reducer aggregates failure. */
    void release(RequestSlot.TerminalResources resources) {
        if (resources != null) {
            resources.release(timer);
        }
    }

    ExpirationTimer.MaintenanceResult maintain(
            ExpirationTimer.OrphanSweeper exactSweeper) {
        if (lifecycle.isShuttingDown()
                || !config.loadBalanceConfig().isQueue()) {
            return NO_MAINTENANCE;
        }
        ExpirationTimer.MaintenanceResult result =
                timer.maintainNow(exactSweeper);
        if (result.staleReduced() > 0) {
            // Scheduler-ledger eviction: report through the split-by-ledger
            // series (role=SCHEDULER + engineIp="scheduler" + reason) so it
            // is no longer mislabelled as a PREFILL endpoint series. This
            // architecture has a single stale-inflight exit, so the reason
            // bucket is always "ttl".
            reporter.reportSchedulerInflightTtlExpired(
                    "ttl", result.staleReduced());
            Logger.info(
                    "event=scheduler_inflight_ttl_eviction evicted={} scanned={}",
                    result.staleReduced(), result.scannedSlots());
        }
        return result;
    }

    @Override
    public void close() {
        timer.close();
    }

    private ExpirationTimer.RetentionPolicy retentionPolicy() {
        long ttlMs = config.loadBalanceConfig()
                .queueScheduler()
                .getLifecycle()
                .getStaleInflightTimeoutMs();
        return new ExpirationTimer.RetentionPolicy(ttlMs, ttlMs);
    }

    private final class SlotDirectory
            implements ExpirationTimer.SlotDirectory<RequestSlot> {
        @Override
        public List<RequestSlot> snapshot() {
            return lifecycle.snapshotSlots();
        }

        @Override
        public boolean removeExactTombstone(
                RequestSlot exactSlot, long updatedBeforeMs) {
            return lifecycle.removeExactTombstone(
                    exactSlot, updatedBeforeMs);
        }

        @Override
        public boolean ownsRequestGeneration(long requestId) {
            return lifecycle.ownsRequestGeneration(requestId);
        }
    }

    private final class DeadlineSink
            implements ExpirationTimer.Sink<RequestSlot> {
        @Override
        public boolean installRequestDeadline(
                RequestSlot exactSlot,
                ExpirationTimer.RequestDeadline exactDeadline) {
            synchronized (exactSlot) {
                return lifecycle.isCurrent(exactSlot)
                        && exactSlot.installRequestDeadline(exactDeadline);
            }
        }

        @Override
        public boolean installAcceptanceDeadline(
                RequestSlot exactSlot,
                ExpirationTimer.AcceptanceDeadline exactDeadline) {
            synchronized (exactSlot) {
                return lifecycle.isCurrent(exactSlot)
                        && exactSlot.installAcceptanceDeadline(exactDeadline);
            }
        }

        @Override
        public void requestDeadlineExpired(
                RequestSlot exactSlot,
                ExpirationTimer.RequestDeadline exactDeadline) {
            RequestSlot.RequestDeadlineExpiry expiry;
            synchronized (exactSlot) {
                expiry = exactSlot.expireRequestDeadline(exactDeadline);
            }
            if (expiry == RequestSlot.RequestDeadlineExpiry.CANCEL_REQUEST) {
                lifecycle.cancelForDeadline(exactSlot);
            }
        }

        @Override
        public void acceptanceDeadlineExpired(
                RequestSlot exactSlot,
                ExpirationTimer.AcceptanceDeadline exactDeadline) {
            RequestSlot.AcceptanceExpiry expiry;
            synchronized (exactSlot) {
                expiry = exactSlot.expireAcceptanceDeadline(exactDeadline);
            }
            lifecycle.acceptanceExpired(expiry);
        }

        @Override
        public ExpirationTimer.DetachedDeadlines detachDeadlinesForClose(
                RequestSlot exactSlot) {
            synchronized (exactSlot) {
                return exactSlot.detachDeadlinesForTimerClose();
            }
        }

        @Override
        public boolean reduceStale(
                RequestSlot exactSlot,
                long nowMs,
                long staleTtlMs) {
            return lifecycle.reduceStale(exactSlot, nowMs, staleTtlMs);
        }
    }
}
