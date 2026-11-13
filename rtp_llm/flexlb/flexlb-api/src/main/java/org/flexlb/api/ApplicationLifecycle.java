package org.flexlb.api;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.api.server.FlexlbGrpcServer;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.master.MasterStatusService;
import org.flexlb.util.ApplicationWarmupState;
import org.flexlb.util.Logger;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.event.ContextClosedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Owns the fixed application online, health and graceful-shutdown workflow.
 */
@Slf4j
@Component
public class ApplicationLifecycle implements ApplicationContextAware {
    private static final long DEFAULT_WARM_UP_WAIT_MS = 3_000L;

    private final AbstractRequestScheduler scheduling;
    private final MasterStatusService consistency;
    private final FlexlbGrpcServer grpcServer;
    private final ServiceMetricsReporter reporter;
    private final Environment environment;
    private final ApplicationWarmupState warmupState;

    private final long warmUpWaitMs;

    private ApplicationContext owningContext;

    private volatile boolean shutdownReceived;
    private volatile boolean shutdownCompletedSuccessfully;

    @Autowired
    public ApplicationLifecycle(
            MasterStatusService consistency,
            FlexlbGrpcServer grpcServer,
            ServiceMetricsReporter reporter,
            Environment environment,
            ApplicationWarmupState warmupState,
            AbstractRequestScheduler scheduling) {
        this(
                consistency,
                grpcServer,
                reporter,
                environment,
                warmupState,
                scheduling,
                DEFAULT_WARM_UP_WAIT_MS);
    }

    ApplicationLifecycle(
            MasterStatusService consistency,
            FlexlbGrpcServer grpcServer,
            ServiceMetricsReporter reporter,
            Environment environment,
            ApplicationWarmupState warmupState,
            AbstractRequestScheduler scheduling,
            long warmUpWaitMs) {
        this.scheduling = java.util.Objects.requireNonNull(scheduling);
        this.consistency = consistency;
        this.grpcServer = grpcServer;
        this.reporter = reporter;
        this.environment = environment;
        this.warmupState = warmupState;
        this.warmUpWaitMs = warmUpWaitMs;
    }

    @Override
    public void setApplicationContext(ApplicationContext context) {
        owningContext = context;
    }

    public synchronized void online() {
        if (Arrays.stream(environment.getActiveProfiles()).anyMatch("test"::equals)) {
            log.info("test env, skip online lifecycle");
            return;
        }
        shutdownReceived = false;
        shutdownCompletedSuccessfully = false;
        warmupState.setWarmupFinished(false);

        long consistencyStartedAt = System.currentTimeMillis();
        try {
            consistency.start();
            reporter.reportDuration(
                    ServiceMetricsReporter.Event.ZK_NODE_ONLINE,
                    System.currentTimeMillis() - consistencyStartedAt);
        } catch (Exception e) {
            Logger.error("application online registration failed", e);
        }

        log.info("waiting {} ms for initial worker synchronization", warmUpWaitMs);
        long warmUpStartedAt = System.currentTimeMillis();
        try {
            Thread.sleep(warmUpWaitMs);
            reporter.reportDuration(
                    ServiceMetricsReporter.Event.WARMER_COMPLETE,
                    System.currentTimeMillis() - warmUpStartedAt);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("application warm up interrupted", e);
        } finally {
            warmupState.setWarmupFinished(true);
        }
    }

    @EventListener(ContextClosedEvent.class)
    @Order(Ordered.HIGHEST_PRECEDENCE)
    public void onContextClosed(ContextClosedEvent event) {
        if (event.getApplicationContext() != owningContext) {
            return;
        }
        offline();
        log.info("context closing: requests drained; destroying serving resources");
    }

    public boolean offline() {
        scheduling.checkShutdownThread();
        return closeApplication();
    }

    public boolean isHealthy() {
        return warmupState.isWarmupFinished() && !shutdownReceived;
    }

    private synchronized boolean closeApplication() {
        if (shutdownCompletedSuccessfully) {
            return true;
        }
        if (!shutdownReceived) {
            shutdownReceived = true;
            reporter.reportDuration(ServiceMetricsReporter.Event.HEALTH_CHECK_OFFLINE, 0L);
            long consistencyStartedAt = System.currentTimeMillis();
            try {
                consistency.offline();
                reporter.reportDuration(
                        ServiceMetricsReporter.Event.ZK_NODE_OFFLINE,
                        System.currentTimeMillis() - consistencyStartedAt);
            } catch (Throwable failure) {
                Logger.error("application offline deregistration failed", failure);
            }
        }

        long drainStartedAt = System.nanoTime();
        // Keep scheduler, forwarder and transport dependencies alive until all
        // accepted RPCs finish. The platform owns the forced-kill deadline.
        grpcServer.drain();
        scheduling.shutdown(System.nanoTime() + TimeUnit.SECONDS.toNanos(30));
        shutdownCompletedSuccessfully = true;
        reporter.reportDuration(
                ServiceMetricsReporter.Event.SHUTDOWN_COMPLETE,
                TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - drainStartedAt));
        return true;
    }
}
