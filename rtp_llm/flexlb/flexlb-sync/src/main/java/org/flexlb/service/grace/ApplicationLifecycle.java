package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.util.Arrays;

/** Owns the fixed application online, health and graceful-shutdown workflow. */
@Slf4j
@Component
public class ApplicationLifecycle {

    private static final long DEFAULT_WARM_UP_WAIT_MS = 3_000L;
    private static final long DEFAULT_SHUTDOWN_TIMEOUT_MS = 300_000L;
    private static final long DEFAULT_QUIET_PERIOD_MS = 5_000L;
    private static final long DRAIN_POLL_MS = 500L;

    private final LBStatusConsistencyService consistency;
    private final ActiveRequestCounter activeRequests;
    private final GracefulLifecycleReporter reporter;
    private final Environment environment;
    private final long warmUpWaitMs;
    private final long shutdownTimeoutMs;
    private final long quietPeriodMs;

    private volatile boolean warmUpFinished;
    private volatile boolean shutdownReceived;
    private volatile boolean shutdownCompletedSuccessfully;

    @Autowired
    public ApplicationLifecycle(
            LBStatusConsistencyService consistency,
            ActiveRequestCounter activeRequests,
            GracefulLifecycleReporter reporter,
            Environment environment) {
        this(consistency, activeRequests, reporter, environment,
                DEFAULT_WARM_UP_WAIT_MS,
                DEFAULT_SHUTDOWN_TIMEOUT_MS,
                DEFAULT_QUIET_PERIOD_MS);
    }

    ApplicationLifecycle(
            LBStatusConsistencyService consistency,
            ActiveRequestCounter activeRequests,
            GracefulLifecycleReporter reporter,
            Environment environment,
            long warmUpWaitMs,
            long shutdownTimeoutMs,
            long quietPeriodMs) {
        if (warmUpWaitMs < 0L || shutdownTimeoutMs < 0L
                || quietPeriodMs < 0L) {
            throw new IllegalArgumentException(
                    "lifecycle durations must not be negative");
        }
        this.consistency = consistency;
        this.activeRequests = activeRequests;
        this.reporter = reporter;
        this.environment = environment;
        this.warmUpWaitMs = warmUpWaitMs;
        this.shutdownTimeoutMs = shutdownTimeoutMs;
        this.quietPeriodMs = quietPeriodMs;
    }

    public synchronized void online() {
        if (Arrays.stream(environment.getActiveProfiles())
                .anyMatch("test"::equals)) {
            log.info("test env, skip online lifecycle");
            return;
        }
        shutdownReceived = false;
        shutdownCompletedSuccessfully = false;
        warmUpFinished = false;

        long consistencyStartedAt = System.currentTimeMillis();
        try {
            consistency.start();
            reporter.reportZkNodeOnline(
                    System.currentTimeMillis() - consistencyStartedAt);
        } catch (Exception e) {
            Logger.error("application online registration failed", e);
        }

        log.info("waiting {} ms for initial worker synchronization", warmUpWaitMs);
        long warmUpStartedAt = System.currentTimeMillis();
        try {
            Thread.sleep(warmUpWaitMs);
            reporter.reportWarmerComplete(
                    System.currentTimeMillis() - warmUpStartedAt);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("application warm up interrupted", e);
        } finally {
            warmUpFinished = true;
        }
    }

    public synchronized boolean offline() {
        shutdownReceived = true;
        shutdownCompletedSuccessfully = false;
        reporter.reportHealthCheckOffline(0L);

        long consistencyStartedAt = System.currentTimeMillis();
        try {
            consistency.offline();
            reporter.reportZkNodeOffline(
                    System.currentTimeMillis() - consistencyStartedAt);
        } catch (Throwable failure) {
            Logger.error("application offline deregistration failed", failure);
        }

        long drainStartedAt = System.currentTimeMillis();
        try {
            shutdownCompletedSuccessfully = awaitQuiet(
                    drainStartedAt + shutdownTimeoutMs);
            long duration = System.currentTimeMillis() - drainStartedAt;
            if (shutdownCompletedSuccessfully) {
                reporter.reportShutdownComplete(duration);
            } else {
                reporter.reportShutdownTimeout(duration);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("application shutdown drain interrupted", e);
        }
        return shutdownCompletedSuccessfully;
    }

    public boolean isHealthy() {
        return warmUpFinished && !shutdownReceived;
    }

    public boolean shutdownCompletedSuccessfully() {
        return shutdownCompletedSuccessfully;
    }

    private boolean awaitQuiet(long hardDeadline) throws InterruptedException {
        long quietDeadline = System.currentTimeMillis() + quietPeriodMs;
        while (System.currentTimeMillis() < quietDeadline) {
            long now = System.currentTimeMillis();
            if (now >= hardDeadline) {
                return false;
            }
            if (activeRequests.getCount() > 0L) {
                quietDeadline = now + quietPeriodMs;
            }
            Thread.sleep(Math.min(DRAIN_POLL_MS,
                    Math.max(1L, Math.min(quietDeadline, hardDeadline) - now)));
        }
        return true;
    }
}
