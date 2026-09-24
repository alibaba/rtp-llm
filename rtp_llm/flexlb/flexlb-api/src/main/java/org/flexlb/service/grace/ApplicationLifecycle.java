package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.httpserver.FlexlbGrpcServer;
import org.flexlb.listener.ApplicationWarmupState;
import org.flexlb.util.Logger;
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

/** Owns the fixed application online, health and graceful-shutdown workflow. */
@Slf4j
@Component
public class ApplicationLifecycle implements ApplicationContextAware {

    private static final long DEFAULT_WARM_UP_WAIT_MS = 3_000L;

    private final LBStatusConsistencyService consistency;
    private final FlexlbGrpcServer grpcServer;
    private final GracefulLifecycleReporter reporter;
    private final Environment environment;
    private final ApplicationWarmupState warmupState;
    private final long warmUpWaitMs;

    private ApplicationContext owningContext;

    @Override
    public void setApplicationContext(ApplicationContext context) {
        owningContext = context;
    }

    private volatile boolean shutdownReceived;
    private volatile boolean shutdownCompletedSuccessfully;

    @Autowired
    public ApplicationLifecycle(LBStatusConsistencyService consistency,
                                FlexlbGrpcServer grpcServer,
                                GracefulLifecycleReporter reporter,
                                Environment environment,
                                ApplicationWarmupState warmupState) {
        this(consistency, grpcServer, reporter, environment, warmupState, DEFAULT_WARM_UP_WAIT_MS);
    }

    ApplicationLifecycle(LBStatusConsistencyService consistency,
                         FlexlbGrpcServer grpcServer,
                         GracefulLifecycleReporter reporter,
                         Environment environment,
                         ApplicationWarmupState warmupState,
                         long warmUpWaitMs) {
        this.consistency = consistency;
        this.grpcServer = grpcServer;
        this.reporter = reporter;
        this.environment = environment;
        this.warmupState = warmupState;
        this.warmUpWaitMs = warmUpWaitMs;
    }

    public synchronized void online() {
        if (Arrays.stream(environment.getActiveProfiles())
                .anyMatch("test"::equals)) {
            log.info("test env, skip online lifecycle");
            return;
        }
        shutdownReceived = false;
        shutdownCompletedSuccessfully = false;
        warmupState.setWarmupFinished(false);

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
            warmupState.setWarmupFinished(true);
        }
    }

    public synchronized boolean offline() {
        if (shutdownCompletedSuccessfully) {
            return true;
        }
        if (!shutdownReceived) {
            shutdownReceived = true;
            reporter.reportHealthCheckOffline(0L);
            long consistencyStartedAt = System.currentTimeMillis();
            try {
                consistency.offline();
                reporter.reportZkNodeOffline(
                        System.currentTimeMillis() - consistencyStartedAt);
            } catch (Throwable failure) {
                Logger.error("application offline deregistration failed", failure);
            }
        }

        long drainStartedAt = System.nanoTime();
        // Keep scheduler, forwarder and transport dependencies alive until all
        // accepted RPCs finish. The platform owns the forced-kill deadline.
        grpcServer.drain();
        shutdownCompletedSuccessfully = true;
        reporter.reportShutdownComplete(
                TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - drainStartedAt));
        return true;
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

    public boolean isHealthy() {
        return warmupState.isWarmupFinished() && !shutdownReceived;
    }

    public boolean shutdownCompletedSuccessfully() {
        return shutdownCompletedSuccessfully;
    }

}
