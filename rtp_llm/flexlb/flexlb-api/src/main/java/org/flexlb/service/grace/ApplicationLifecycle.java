package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.DeliveryExecution;
import org.flexlb.balance.scheduler.ExpirationTimer;
import org.flexlb.balance.scheduler.RequestLifecycle;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.RequestTaskPool;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor;
import org.flexlb.balance.scheduler.ScheduledTaskTimer;
import org.flexlb.consistency.MasterStatusService;
import org.flexlb.httpserver.FlexlbGrpcServer;
import org.flexlb.listener.ApplicationWarmupState;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.event.ContextClosedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.concurrent.TimeUnit;

/** Owns the fixed application online, health and graceful-shutdown workflow. */
@Slf4j
@Component
public class ApplicationLifecycle implements ApplicationContextAware {

    private RequestScheduler scheduler;
    private DecodePreemption preemption;
    private RequestLifecycle lifecycle;
    private RequestRepository requests;
    private EndpointRegistry endpoints;
    private DeliveryExecution deliveries;
    private RequestTaskPool tasks;
    private ExpirationTimer deadlines;
    private ResponseCompletionExecutor completions;
    private ScheduledTaskTimer taskTimer;
    private ScheduledTaskTimer deadlineTaskTimer;
    private DeliveryDispatcher dispatcher;

    @Autowired
    public void configureScheduling(
            RequestScheduler scheduler,
            RequestLifecycle lifecycle,
            RequestRepository requests,
            EndpointRegistry endpoints,
            DecodePreemption preemption) {
        this.scheduler = scheduler;
        this.lifecycle = lifecycle;
        this.requests = requests;
        this.endpoints = endpoints;
        this.preemption = preemption;
    }

    @Autowired
    public void configureExecution(
            DeliveryExecution deliveries,
            RequestTaskPool tasks,
            ResponseCompletionExecutor completions,
            DeliveryDispatcher dispatcher) {
        this.deliveries = deliveries;
        this.tasks = tasks;
        this.completions = completions;
        this.dispatcher = dispatcher;
    }

    @Autowired
    public void configureTimers(
            ExpirationTimer deadlines,
            @Qualifier("deadlineTaskTimer") ScheduledTaskTimer deadlineTaskTimer,
            ScheduledTaskTimer taskTimer) {
        this.deadlines = deadlines;
        this.deadlineTaskTimer = deadlineTaskTimer;
        this.taskTimer = taskTimer;
    }

    @Autowired
    public void startScheduling(RequestScheduler scheduler, RequestLifecycle lifecycle) {
        lifecycle.bindScheduler(scheduler);
        scheduler.start();
    }

    private void closeScheduling() {
        if (scheduler == null || requests.isClosed()) return;
        requests.stopAccepting();
        Throwable failure =
                Failures.run(
                        null,
                        () ->
                                requests.awaitResponses(
                                        System.nanoTime() + TimeUnit.SECONDS.toNanos(30L)));
        if (!requests.closeRegistration()) return;
        failure = Failures.run(failure, preemption::stopAdmission);
        failure = Failures.run(failure, scheduler::stop);
        failure = Failures.run(failure, scheduler::close);
        failure = Failures.run(failure, preemption::close);
        var owners = new LinkedHashSet<RequestLifecycle>();
        owners.add(lifecycle);
        for (var context : requests.snapshotActive()) owners.add(context.data().getLifecycle());
        for (var owner : owners)
            failure = Failures.run(failure, owner::stopAdmissionsAndAwaitCompletion);
        for (var context : requests.snapshotActive()) {
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    context.data()
                                            .getLifecycle()
                                            .cancel(context, 0L, CancelReason.SHUTDOWN));
        }
        failure = Failures.run(failure, deliveries::awaitDeliveries);
        for (var owner : owners)
            failure = Failures.run(failure, () -> owner.awaitDeliveryCleanup(true));
        failure = Failures.run(failure, endpoints::close);
        for (var owner : owners)
            failure = Failures.run(failure, () -> owner.awaitDeliveryCleanup(false));
        failure = Failures.run(failure, deadlines::close);
        failure = Failures.run(failure, deadlineTaskTimer::close);
        failure = Failures.run(failure, tasks::awaitIdle);
        for (var owner : owners)
            failure = Failures.run(failure, owner::closeOutstandingAndTerminalize);
        failure = Failures.run(failure, dispatcher::awaitStops);
        failure = Failures.run(failure, tasks::close);
        failure = Failures.run(failure, completions::close);
        failure = Failures.run(failure, taskTimer::close);
        failure = Failures.run(failure, tasks::throwIfFailed);
        if (requests.liveRequestCount() != 0)
            failure =
                    Failures.append(
                            failure,
                            new IllegalStateException(
                                    "unsettled requests at shutdown: "
                                            + requests.liveRequestCount()));
        Failures.rethrow(failure, "application scheduling shutdown failed");
    }

    private static final long DEFAULT_WARM_UP_WAIT_MS = 3_000L;

    private final MasterStatusService consistency;
    private final FlexlbGrpcServer grpcServer;
    private final EngineHealthReporter reporter;
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
    public ApplicationLifecycle(
            MasterStatusService consistency,
            FlexlbGrpcServer grpcServer,
            EngineHealthReporter reporter,
            Environment environment,
            ApplicationWarmupState warmupState) {
        this(consistency, grpcServer, reporter, environment, warmupState, DEFAULT_WARM_UP_WAIT_MS);
    }

    ApplicationLifecycle(
            MasterStatusService consistency,
            FlexlbGrpcServer grpcServer,
            EngineHealthReporter reporter,
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
                    EngineHealthReporter.Event.ZK_NODE_ONLINE,
                    System.currentTimeMillis() - consistencyStartedAt);
        } catch (Exception e) {
            Logger.error("application online registration failed", e);
        }

        log.info("waiting {} ms for initial worker synchronization", warmUpWaitMs);
        long warmUpStartedAt = System.currentTimeMillis();
        try {
            Thread.sleep(warmUpWaitMs);
            reporter.reportDuration(
                    EngineHealthReporter.Event.WARMER_COMPLETE,
                    System.currentTimeMillis() - warmUpStartedAt);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("application warm up interrupted", e);
        } finally {
            warmupState.setWarmupFinished(true);
        }
    }

    public synchronized boolean offline() {
        if (tasks != null
                && (tasks.isExecutingOnCurrentThread()
                        || completions.isCompletingOnCurrentThread())) {
            throw new IllegalStateException("shutdown cannot wait for its own request callback");
        }
        if (shutdownCompletedSuccessfully) {
            return true;
        }
        if (!shutdownReceived) {
            shutdownReceived = true;
            reporter.reportDuration(EngineHealthReporter.Event.HEALTH_CHECK_OFFLINE, 0L);
            long consistencyStartedAt = System.currentTimeMillis();
            try {
                consistency.offline();
                reporter.reportDuration(
                        EngineHealthReporter.Event.ZK_NODE_OFFLINE,
                        System.currentTimeMillis() - consistencyStartedAt);
            } catch (Throwable failure) {
                Logger.error("application offline deregistration failed", failure);
            }
        }

        long drainStartedAt = System.nanoTime();
        // Keep scheduler, forwarder and transport dependencies alive until all
        // accepted RPCs finish. The platform owns the forced-kill deadline.
        grpcServer.drain();
        closeScheduling();
        shutdownCompletedSuccessfully = true;
        reporter.reportDuration(
                EngineHealthReporter.Event.SHUTDOWN_COMPLETE,
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
}
