package org.flexlb.api;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.api.server.FlexlbGrpcServer;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.EngineCancellation;
import org.flexlb.balance.schedule.ExpirationTimer;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.balance.schedule.RequestTaskPool;
import org.flexlb.balance.schedule.ResponseCompletionExecutor;
import org.flexlb.master.MasterStatusService;
import org.flexlb.util.ApplicationWarmupState;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.EndpointRegistry;
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
    private final RequestRepository requests;
    private final RequestTaskPool tasks;
    private final ExpirationTimer deadlines;
    private final EngineCancellation engineCancellation;
    private final ResponseCompletionExecutor responses;
    private final EndpointRegistry endpoints;

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
            AbstractRequestScheduler scheduling,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            EngineCancellation engineCancellation,
            ResponseCompletionExecutor responses,
            EndpointRegistry endpoints) {
        this(
                consistency,
                grpcServer,
                reporter,
                environment,
                warmupState,
                scheduling,
                requests,
                tasks,
                deadlines,
                engineCancellation,
                responses,
                endpoints,
                DEFAULT_WARM_UP_WAIT_MS);
    }

    ApplicationLifecycle(
            MasterStatusService consistency,
            FlexlbGrpcServer grpcServer,
            ServiceMetricsReporter reporter,
            Environment environment,
            ApplicationWarmupState warmupState,
            AbstractRequestScheduler scheduling,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer deadlines,
            EngineCancellation engineCancellation,
            ResponseCompletionExecutor responses,
            EndpointRegistry endpoints,
            long warmUpWaitMs) {
        this.scheduling = java.util.Objects.requireNonNull(scheduling);
        this.requests = requests;
        this.tasks = tasks;
        this.deadlines = deadlines;
        this.engineCancellation = engineCancellation;
        this.responses = responses;
        this.endpoints = endpoints;

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
        closeScheduling(System.nanoTime() + TimeUnit.SECONDS.toNanos(30));
        shutdownCompletedSuccessfully = true;
        reporter.reportDuration(
                ServiceMetricsReporter.Event.SHUTDOWN_COMPLETE,
                TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - drainStartedAt));
        return true;
    }

    private void closeScheduling(long deadlineNanos) {
        requests.stopAccepting();
        Throwable failure = Failures.run(null, () -> requests.awaitResponses(deadlineNanos));
        requests.closeRegistration();
        failure = Failures.run(failure, scheduling::stopAndCancelRequests);
        failure = Failures.run(failure, engineCancellation::awaitStops);
        failure = Failures.run(failure, tasks::awaitIdle);
        failure = Failures.run(failure, scheduling::awaitRequestCleanup);
        failure = Failures.run(failure, endpoints::close);
        failure = Failures.run(failure, engineCancellation::awaitStops);
        failure = Failures.run(failure, deadlines::close);
        failure = Failures.run(failure, tasks::awaitIdle);
        failure = Failures.run(failure, scheduling::finishRequestCleanup);
        failure = Failures.run(failure, scheduling::awaitRequestCleanup);
        failure = Failures.run(failure, tasks::awaitIdle);
        failure = Failures.run(failure, engineCancellation::awaitStops);
        failure = Failures.run(failure, tasks::close);
        failure = Failures.run(failure, responses::close);
        failure = Failures.run(failure, scheduling::throwIfRequestFailed);
        if (requests.liveRequestCount() != 0)
            failure =
                    Failures.append(
                            failure,
                            new IllegalStateException(
                                    "unsettled requests at shutdown: "
                                            + requests.liveRequestCount()));
        Failures.rethrow(failure, "application scheduling shutdown failed");
    }
}
