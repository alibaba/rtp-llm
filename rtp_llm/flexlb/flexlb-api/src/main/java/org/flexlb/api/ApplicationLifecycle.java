package org.flexlb.api;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.api.metrics.ApiMetricsReporter;
import org.flexlb.api.server.FlexlbGrpcServer;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.EngineRequestCanceller;
import org.flexlb.balance.schedule.ExpirationTimer;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.balance.schedule.ResponseCompletionExecutor;
import org.flexlb.master.MasterStatusService;
import org.flexlb.util.ApplicationWarmupState;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.event.ContextClosedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.core.env.Environment;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Coordinates application startup, health, request maintenance and graceful shutdown.
 */
@Slf4j
@Component
public class ApplicationLifecycle implements ApplicationContextAware {
    private static final long DEFAULT_WARM_UP_WAIT_MS = 3_000L;

    private final AbstractRequestScheduler scheduling;
    private final RequestRepository requests;
    private final ExecutorService requestWork;
    private final ExpirationTimer deadlines;
    private final EngineRequestCanceller engineRequestCanceller;
    private final ResponseCompletionExecutor responses;
    private final EndpointRegistry endpoints;
    private final MasterStatusService consistency;
    private final FlexlbGrpcServer grpcServer;
    private final ApiMetricsReporter reporter;
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
            ApiMetricsReporter reporter,
            Environment environment,
            ApplicationWarmupState warmupState,
            AbstractRequestScheduler scheduling,
            RequestRepository requests,
            @Qualifier("requestWorkExecutor") ExecutorService requestWork,
            ExpirationTimer deadlines,
            EngineRequestCanceller engineRequestCanceller,
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
                requestWork,
                deadlines,
                engineRequestCanceller,
                responses,
                endpoints,
                DEFAULT_WARM_UP_WAIT_MS);
    }

    ApplicationLifecycle(
            MasterStatusService consistency,
            FlexlbGrpcServer grpcServer,
            ApiMetricsReporter reporter,
            Environment environment,
            ApplicationWarmupState warmupState,
            AbstractRequestScheduler scheduling,
            RequestRepository requests,
            ExecutorService requestWork,
            ExpirationTimer deadlines,
            EngineRequestCanceller engineRequestCanceller,
            ResponseCompletionExecutor responses,
            EndpointRegistry endpoints,
            long warmUpWaitMs) {
        this.scheduling = java.util.Objects.requireNonNull(scheduling);
        this.requests = requests;
        this.requestWork = requestWork;
        this.deadlines = deadlines;
        this.engineRequestCanceller = engineRequestCanceller;
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
                    ApiMetricsReporter.Event.ZK_NODE_ONLINE,
                    System.currentTimeMillis() - consistencyStartedAt);
        } catch (Exception e) {
            Logger.error("application online registration failed", e);
        }

        log.info("waiting {} ms for initial worker synchronization", warmUpWaitMs);
        long warmUpStartedAt = System.currentTimeMillis();
        try {
            Thread.sleep(warmUpWaitMs);
            reporter.reportDuration(
                    ApiMetricsReporter.Event.WARMER_COMPLETE,
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

    /**
     * Sweeps expired endpoint records once the request repository no longer retains their identity.
     */
    @Scheduled(fixedRate = 60_000L)
    void evictExpiredEndpointOrphans() {
        if (!requests.isClosed()) endpoints.evictExpiredOrphans(10_000L, requests::retainsIdentity);
    }

    private synchronized boolean closeApplication() {
        if (shutdownCompletedSuccessfully) {
            return true;
        }
        if (!shutdownReceived) {
            shutdownReceived = true;
            reporter.reportDuration(ApiMetricsReporter.Event.HEALTH_CHECK_OFFLINE, 0L);
            long consistencyStartedAt = System.currentTimeMillis();
            try {
                consistency.offline();
                reporter.reportDuration(
                        ApiMetricsReporter.Event.ZK_NODE_OFFLINE,
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
                ApiMetricsReporter.Event.SHUTDOWN_COMPLETE,
                TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - drainStartedAt));
        return true;
    }

    private void closeScheduling(long deadlineNanos) {
        requests.stopAccepting();
        Throwable failure = Failures.run(null, () -> requests.awaitResponses(deadlineNanos));
        requests.closeRegistration();
        failure = Failures.run(failure, scheduling::stopAndCancelRequests);
        failure = Failures.run(failure, engineRequestCanceller::awaitStops);
        failure = Failures.run(failure, scheduling::awaitRequestWork);
        failure = Failures.run(failure, endpoints::close);
        failure = Failures.run(failure, engineRequestCanceller::awaitStops);
        failure = Failures.run(failure, deadlines::close);
        failure = Failures.run(failure, scheduling::awaitRequestWork);
        failure = Failures.run(failure, scheduling::closeRemainingRequestsOnShutdown);
        failure = Failures.run(failure, scheduling::awaitRequestWork);
        failure = Failures.run(failure, engineRequestCanceller::awaitStops);
        failure = Failures.run(failure, requestWork::close);
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
