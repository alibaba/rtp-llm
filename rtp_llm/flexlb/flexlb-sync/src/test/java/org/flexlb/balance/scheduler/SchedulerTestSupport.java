package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.Registration;
import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Duration;
import java.util.Map;
import java.util.WeakHashMap;

import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/** Builds real request owners and shared facilities; contains no request state machine. */
public final class SchedulerTestSupport {
    public static AbstractRequestScheduler createStartedScheduler(
            AbstractRequestScheduler lifecycle,
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.eviction.DecodePreemption preemption,
            PlacementAvailability availability,
            org.flexlb.balance.planner.GroupPlanner grouping,
            PlacementQueue queue) {
        var scheduler =
                SchedulerTestServices.create(
                        lifecycle, config, router, preemption, availability, grouping, queue);
        scheduler.start();
        return scheduler;
    }

    static org.flexlb.balance.PlacementResult<RouteExecution, PlacementKey> enqueueRoute(
            QueuedRequestScheduler scheduler, QueuedRequestScheduler.PlacementPlan plan) {
        var route = RequestProtocolTestSupport.planRouting(plan);
        var settings =
                (QueueExecutionSettings) ReflectionTestUtils.getField(scheduler, "queueSettings");
        return lifecycle(scheduler).enqueueRoute(route, settings);
    }

    static void startScheduling(RequestScheduler scheduler) {
        scheduler.start();
    }

    public static AbstractRequestScheduler lifecycle(RequestScheduler scheduler) {
        return (AbstractRequestScheduler) scheduler;
    }

    static AbstractRequestScheduler spyLifecycle(RequestScheduler scheduler) {
        return (AbstractRequestScheduler) scheduler;
    }

    public static DeliveryExecution queueDependencies(SchedulerTestServices runtime) {
        return runtime.deliveries();
    }

    public static DeliveryExecution queueDependencies(
            AbstractRequestScheduler owner, org.flexlb.config.FlexlbConfig config) {
        if (org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner) == null) {
            var runtime = mock(SchedulerTestServices.class);
            var deliveries =
                    new DeliveryExecution(DeliveryTestSupport.service(config), ignored -> {});
            when(runtime.deliveries()).thenReturn(deliveries);

            org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                    runtime, "deliveryReporter", mock(DeliveryMetricsReporter.class));
            org.flexlb.balance.scheduler.SchedulerTestServices.bind(owner, runtime);
        }
        return queueDependencies(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner));
    }

    static ShutdownResult shutdown(SchedulerTestServices runtime) {
        return runtime.shutdown(ShutdownMode.IMMEDIATE, Duration.ZERO);
    }

    /** Empty Decode ownership uses typed domain results, including in tests without a real ledger. */
    public static org.flexlb.balance.endpoint.DecodeEndpoint mockDecodeEndpoint() {
        return RequestProtocolTestSupport.decodeEndpoint();
    }

    public static DeliveryDispatcher dispatcher() {
        return spy(
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.newConfig()),
                        new DeliveryMetricsReporter(null),
                        org.mockito.Mockito.mock(org.flexlb.balance.scheduler.TaskTimer.class),
                        new org.flexlb.balance.scheduler.PlacementConfiguration()
                                .decodeAdmissionCapacity(
                                        DeliveryTestSupport.service(
                                                SchedulingTestConfig.newConfig())),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.scheduler.DeliveryExecution.class)));
    }

    static DeliveryDispatcher dispatcher(SchedulerTestServices runtime) {
        return (DeliveryDispatcher) ReflectionTestUtils.getField(runtime, "dispatcher");
    }

    private static final Map<AbstractRequestScheduler, RequestRepository> mockRepositories =
            new WeakHashMap<>();

    public static synchronized RequestRepository repository(AbstractRequestScheduler owner) {
        if (owner == null) {
            return null;
        }
        return owner.requests != null
                ? owner.requests
                : mockRepositories.computeIfAbsent(owner, SchedulerTestSupport::mockRepository);
    }

    private static RequestRepository mockRepository(AbstractRequestScheduler owner) {
        var repository = mock(RequestRepository.class);
        when(repository.ownerOf(org.mockito.ArgumentMatchers.anyString())).thenReturn(owner);
        when(repository.findActive(org.mockito.ArgumentMatchers.anyString()))
                .thenAnswer(
                        call -> {
                            var context = mock(RequestContext.class);
                            var data = mock(RequestData.class);
                            when(context.data()).thenReturn(data);
                            when(data.getRequestScheduler()).thenReturn(owner);
                            return context;
                        });
        return repository;
    }

    public static void bindOwner(RequestContext context, AbstractRequestScheduler owner) {
        if (org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner) == null) {
            var runtime = mock(SchedulerTestServices.class);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
            var deliveries = new DeliveryExecution(service, ignored -> {});
            when(runtime.deliveries()).thenReturn(deliveries);

            org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                    runtime, "deliveryReporter", mock(DeliveryMetricsReporter.class));
            org.flexlb.balance.scheduler.SchedulerTestServices.bind(owner, runtime);
        }
        if (context.data().getRequestScheduler() == null) {
            // Minimal fixtures may bind an owner without registering a runnable request.
            ReflectionTestUtils.setField(context.data(), "requestScheduler", owner);
        }
    }

    private static final Map<org.flexlb.balance.endpoint.PrefillEndpoint, RequestRepository>
            endpointRepositories = java.util.Collections.synchronizedMap(new WeakHashMap<>());

    public static void associateEndpoint(
            org.flexlb.balance.endpoint.PrefillEndpoint endpoint, RequestRepository repository) {
        endpointRepositories.put(endpoint, repository);
    }

    public static void bindEndpointOwner(
            org.flexlb.balance.endpoint.PrefillEndpoint endpoint, RouteExecution route) {
        if (route.ctx() == null) {
            var context = mock(RequestContext.class);
            when(route.ctx()).thenReturn(context);
            var repository = endpointRepositories.get(endpoint);
            var owner = repository == null ? null : repository.ownerOf(route.requestId());
            var data = mock(RequestData.class);
            when(context.data()).thenReturn(data);
            when(data.getRequestScheduler())
                    .thenReturn(owner == null ? mock(AbstractRequestScheduler.class) : owner);
        } else if (route.ctx().data().getRequestScheduler() == null) {
            var repository = endpointRepositories.get(endpoint);
            var owner = repository == null ? null : repository.ownerOf(route.requestId());
            bindOwner(route.ctx(), owner == null ? mock(AbstractRequestScheduler.class) : owner);
        }
        var context = route.ctx();
        var owner = context.data().getRequestScheduler();
        if (owner.requests == null) {
            var snapshot = context.snapshot();
            if (snapshot == null || org.mockito.Mockito.mockingDetails(snapshot).isMock()) {
                var facts = snapshot == null ? mock(ContextResult.Snapshot.class) : snapshot;
                when(context.snapshot()).thenReturn(facts);
                when(facts.route()).thenReturn(route);
                when(facts.preparedDelivery()).thenReturn(true);
                when(context.data().getConfig()).thenReturn(SchedulingTestConfig.newConfig());
            } else if (snapshot.key() == null) {
                long now = System.currentTimeMillis();
                context.register(
                        route.requestId(),
                        now,
                        route.expiresAtMs(),
                        context.data()
                                .getConfig()
                                .getRequestLifecycle()
                                .getRequest()
                                .getTimeoutMs());
                var placement = (PlacementOperation) context.beginPlacement(null);
                context.bindPlacement(placement, route, ContextResult.PublicationResult.PREPARED);
                context.bindPlacement(placement, route, ContextResult.PublicationResult.PUBLISHED);
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.completePlacement(
                                        placement, ContextResult.PublicationResult.PUBLISHED, now));
            }
        }
    }

    static AbstractRequestScheduler initializedScheduler(SchedulerTestServices runtime) {
        synchronized (ReflectionTestUtils.getField(runtime, "schedulerLock")) {
            return (AbstractRequestScheduler) ReflectionTestUtils.getField(runtime, "scheduler");
        }
    }

    public static void awaitRequestWork(SchedulerTestServices runtime) {
        var tasks = (RequestTaskPool) runtime.tasks();
        if (!tasks.isExecutingOnCurrentThread()) {
            tasks.awaitIdle();
            return;
        }
        // Race hooks run inside asynchronous DIRECT work. Drain the other tasks, not this hook.
        var monitor = ReflectionTestUtils.getField(tasks, "workLifecycle");
        try {
            var pending = RequestTaskPool.class.getDeclaredField("pendingWork");
            pending.setAccessible(true);
            long until = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(5);
            while (System.nanoTime() < until) {
                synchronized (monitor) {
                    if (pending.getLong(tasks) == 1L) return;
                }
                java.util.concurrent.locks.LockSupport.parkNanos(1_000_000L);
            }
            throw new AssertionError("other request work did not drain while running a race hook");
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    static void closeRequestWork(SchedulerTestServices runtime) {
        ReflectionTestUtils.invokeMethod(runtime.tasks(), "close");
    }

    static java.util.concurrent.ExecutorService requestWorkers(SchedulerTestServices runtime) {
        return (java.util.concurrent.ExecutorService)
                ReflectionTestUtils.getField(runtime.tasks(), "requestWorkers");
    }

    static ResponseCompletionExecutor responseCompletions(SchedulerTestServices runtime) {
        return (ResponseCompletionExecutor)
                ReflectionTestUtils.getField(runtime, "responseCompletions");
    }

    static ExpirationTimer timer(SchedulerTestServices runtime) {
        return (ExpirationTimer) ReflectionTestUtils.getField(runtime, "timer");
    }

    public static java.util.concurrent.ScheduledThreadPoolExecutor cleanupExecutor(
            SchedulerTestServices runtime) {
        return (java.util.concurrent.ScheduledThreadPoolExecutor)
                ReflectionTestUtils.getField(runtime.taskTimer(), "executor");
    }

    static void closeRequestExecutors(SchedulerTestServices runtime) {
        ReflectionTestUtils.invokeMethod(runtime, "closeRequestExecutors");
    }

    /** Response/timer fixtures explicitly supply resource proof; production has no manual close path. */
    static RequestState settleTerminalFixture(RequestContext context, ContextResult.End terminal) {
        org.junit.jupiter.api.Assertions.assertSame(terminal, context.snapshot().outcome());
        var delivery = context.snapshot().delivery();
        if (delivery != null) {
            if (delivery.send() == ContextResult.SendState.NOT_STARTED
                    || delivery.send() == ContextResult.SendState.SENDING) {
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new ContextResult.TransportResult(
                                                delivery.identity(),
                                                org.flexlb.balance.scheduler.Delivery.Result
                                                        .notSent(
                                                                new IllegalStateException(
                                                                        "fixture sender exited before send")),
                                                System.currentTimeMillis())));
            }
            org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                    context,
                    observedRequest ->
                            observedRequest.recordDeliveryResult(
                                    new ContextResult.CleanupAcknowledgement(
                                            delivery.identity(),
                                            ContextResult.CancelEvidenceKind.CLEANED,
                                            System.currentTimeMillis())));
        }
        context.completeClose(
                new ContextResult.CloseProgress(true, false, false, true),
                System.currentTimeMillis());
        org.junit.jupiter.api.Assertions.assertNotNull(context.releaseReason());
        context.completeClose(
                new ContextResult.CloseProgress(false, true, true, false),
                System.currentTimeMillis());
        org.junit.jupiter.api.Assertions.assertNotNull(context.snapshot().terminalRecord());
        return context.snapshot().terminalRecord();
    }

    enum Stage {
        QUEUED,
        ROUTING,
        READY_TO_DELIVER,
        DELIVERING,
        RESULT_PENDING,
        FINALIZING,
        FINISHED
    }

    static Stage stage(RequestContext context) {
        var state = context.snapshot();
        if (state.phase() == ContextResult.Phase.CLOSED) {
            return Stage.FINISHED;
        }
        if (state.phase() == ContextResult.Phase.ENDING) {
            return Stage.FINALIZING;
        }
        if (state.delivery() != null) {
            return state.delivery().acknowledged() || state.engine().decodeAccepted()
                    ? Stage.RESULT_PENDING
                    : Stage.DELIVERING;
        }
        if (state.placement() != null
                && (!state.placement().published()
                        || state.placement().operation().withdrawingRoute() != null)) {
            return Stage.ROUTING;
        }
        return state.route() == null ? Stage.QUEUED : Stage.READY_TO_DELIVER;
    }

    static CancelReason cancellationReason(RequestContext context) {
        var facts = context.snapshot().cancellation();
        return facts == null ? null : facts.reason();
    }

    static PreemptionRegistration preemption(RequestContext context) {
        var facts = context.snapshot().preemption();
        return facts == null ? null : facts.identity();
    }

    static Delivery.Member delivery(RequestContext context) {
        var facts = context.snapshot().delivery();
        return facts == null ? null : facts.identity();
    }

    static boolean isOpen(RequestContext context) {
        var facts = context.snapshot();
        return facts.phase() == ContextResult.Phase.OPEN
                && facts.cancellation() == null
                && !context.data().getFuture().isDone();
    }

    public static SchedulerTestServices runtime(Object owner) {
        return SchedulerTestServices.forOwner(owner);
    }

    static EngineGrpcClient engineClient(Object owner) {
        return (EngineGrpcClient)
                ReflectionTestUtils.getField(runtime(owner).dispatcher, "grpcClient");
    }

    static Throwable failure(Object owner) {
        return (Throwable)
                org.flexlb.balance.scheduler.SchedulerTestServices.failure(runtime(owner));
    }

    static RequestRepository.TerminalRecord terminalRecord(
            AbstractRequestScheduler owner, RequestState state) {
        var record = repository(owner).findTerminal(state.requestId());
        return record != null && record.state() == state
                ? record
                : new RequestRepository.TerminalRecord(state, owner);
    }

    /** State/protocol fixtures have no running placement loop. */
    public static AbstractRequestScheduler create(
            ConfigService config,
            DeliveryMetricsReporter batches,
            RequestSchedulerReporter requests,
            TheoryCacheHitReporter trace) {
        return create(
                config,
                batches,
                requests,
                trace,
                mock(RequestRouteSelector.class),
                mock(DecodePreemption.class),
                new PlacementAvailability(),
                false);
    }

    /** Construct all placement dependencies together and let the runtime own their shutdown. */
    public static AbstractRequestScheduler create(
            ConfigService config,
            DeliveryMetricsReporter batches,
            RequestSchedulerReporter requests,
            TheoryCacheHitReporter trace,
            RequestRouteSelector router,
            DecodePreemption eviction,
            PlacementAvailability availability) {
        return create(config, batches, requests, trace, router, eviction, availability, true);
    }

    static AbstractRequestScheduler create(
            ConfigService config,
            DeliveryMetricsReporter batches,
            RequestSchedulerReporter requests,
            TheoryCacheHitReporter trace,
            RequestRouteSelector router,
            DecodePreemption eviction,
            PlacementAvailability availability,
            boolean startPlacement) {
        var runtime =
                spy(
                        new SchedulerTestServices(
                                new RequestRepository(),
                                mock(EndpointRegistry.class),
                                batches,
                                requests,
                                SchedulerTestSupport.dispatcher(),
                                config,
                                mock(EngineGrpcClient.class)));
        return create(
                runtime, config, batches, trace, router, eviction, availability, startPlacement);
    }

    public static AbstractRequestScheduler create(
            SchedulerTestServices runtime,
            ConfigService config,
            DeliveryMetricsReporter batches,
            TheoryCacheHitReporter trace,
            RequestRouteSelector router,
            DecodePreemption eviction,
            PlacementAvailability availability) {
        return create(runtime, config, batches, trace, router, eviction, availability, true);
    }

    private static AbstractRequestScheduler create(
            SchedulerTestServices runtime,
            ConfigService config,
            DeliveryMetricsReporter batches,
            TheoryCacheHitReporter trace,
            RequestRouteSelector router,
            DecodePreemption eviction,
            PlacementAvailability availability,
            boolean startPlacement) {
        var snapshot = config.loadBalanceConfig();
        var capacity = new PlacementConfiguration().decodeAdmissionCapacity(config);
        RequestScheduler owner =
                snapshot.isQueue()
                        ? mock(
                                QueuedRequestScheduler.class,
                                withSettings()
                                        .useConstructor(
                                                SchedulerTestServices.constructorArguments(
                                                        snapshot,
                                                        router,
                                                        eviction,
                                                        runtime.newLifecycle(
                                                                snapshot, capacity, trace),
                                                        availability,
                                                        new OrderedRequestQueue(
                                                                snapshot.isPriorityOrdering()),
                                                        org.flexlb.balance.planner
                                                                .BuiltinGroupPlanner.from(
                                                                snapshot)))
                                        .defaultAnswer(CALLS_REAL_METHODS))
                        : mock(
                                DirectRequestScheduler.class,
                                withSettings()
                                        .useConstructor(
                                                SchedulerTestServices.constructorArguments(
                                                        router,
                                                        runtime.newLifecycle(
                                                                snapshot, capacity, trace)))
                                        .defaultAnswer(CALLS_REAL_METHODS));
        runtime.registerScheduler(owner);
        if (startPlacement || owner instanceof DirectRequestScheduler) {
            owner.start();
        }
        return (AbstractRequestScheduler) owner;
    }

    static Registration responsePublication(ResponseCompletionExecutor executor) {
        return new Registration(executor, mock(RequestContext.class));
    }

    static DeliveryDispatcher createDispatcher(
            EngineGrpcClient client,
            org.flexlb.config.ConfigService service,
            int threads,
            int queueCapacity) {
        var config = org.mockito.Mockito.spy(service.loadBalanceConfig());
        var sizing = org.mockito.Mockito.spy(config.getInternalRuntime());
        org.mockito.Mockito.doReturn(threads).when(sizing).getBatchDispatchThreads();
        org.mockito.Mockito.doReturn(queueCapacity).when(sizing).getBatchDispatchQueueCapacity();
        org.mockito.Mockito.doReturn(sizing).when(config).getInternalRuntime();
        var dispatcherService =
                org.mockito.Mockito.mock(
                        org.flexlb.config.ConfigService.class,
                        org.mockito.AdditionalAnswers.delegatesTo(service));
        org.mockito.Mockito.doReturn(config).when(dispatcherService).loadBalanceConfig();
        return new DeliveryDispatcher(
                client,
                dispatcherService,
                new DeliveryMetricsReporter(null),
                org.mockito.Mockito.mock(org.flexlb.balance.scheduler.TaskTimer.class),
                new org.flexlb.balance.scheduler.PlacementConfiguration()
                        .decodeAdmissionCapacity(dispatcherService),
                org.mockito.Mockito.mock(org.flexlb.balance.scheduler.DeliveryExecution.class));
    }

    public static PlacementOperation claim(
            AbstractRequestScheduler scheduler,
            String id,
            java.util.concurrent.CompletableFuture<?> future) {
        var context = scheduler.requests.findActive(id);
        return context != null && context.data().getFuture() == future
                ? scheduler.claimPlacement(context)
                : null;
    }
}
