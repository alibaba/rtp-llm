package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.preemption.DecodePreemption;
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

/**
 * Builds real request owners and shared facilities; contains no request state machine.
 */
public final class SchedulerTestSupport {
    private static final Map<AbstractRequestScheduler, RequestRepository> mockRepositories =
            new WeakHashMap<>();
    private static final Map<org.flexlb.balance.endpoint.PrefillEndpoint, RequestRepository>
            endpointRepositories = java.util.Collections.synchronizedMap(new WeakHashMap<>());
    private static final java.util.Map<
                    AbstractRequestScheduler, org.flexlb.balance.preemption.PreemptionCompletion>
            preemptionHandlers =
                    java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());
    private static final java.util.Map<RequestContext, RequestCloseQueue> testCloseQueues =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    public static AbstractRequestScheduler createStartedScheduler(
            AbstractRequestScheduler lifecycle,
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.preemption.DecodePreemption preemption,
            CapacityNotifier availability,
            org.flexlb.balance.planner.GroupPlanner grouping,
            SchedulingQueue queue) {
        var scheduler =
                SchedulerTestServices.create(
                        lifecycle, config, router, preemption, availability, grouping, queue);
        scheduler.start();
        return scheduler;
    }

    public static AbstractRequestScheduler lifecycle(RequestScheduler scheduler) {
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

    /** Empty Decode ownership uses typed domain results, including in tests without a real ledger. */
    public static org.flexlb.balance.endpoint.DecodeEndpoint mockDecodeEndpoint() {
        return RequestProtocolTestSupport.decodeEndpoint();
    }

    public static void associateEndpoint(
            org.flexlb.balance.endpoint.PrefillEndpoint endpoint, RequestRepository repository) {
        endpointRepositories.put(endpoint, repository);
    }

    public static void bindEndpointOwner(
            org.flexlb.balance.endpoint.PrefillEndpoint endpoint, RouteResult route) {
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
        if (owner.repository == null) {
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
                                .getTimeoutMs(),
                        org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
                var placement = (RoutingContext) context.beginRouteSelection();
                context.publishRoute(placement, route);
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.completeRouting(
                                        placement, ContextResult.RoutingCompletion.FINISHED, now));
            }
        }
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

    public static org.flexlb.balance.preemption.PreemptionCompletion preemption(
            AbstractRequestScheduler scheduler) {
        return preemptionHandlers.computeIfAbsent(
                scheduler,
                ignored -> {
                    var details = org.mockito.Mockito.mockingDetails(scheduler);
                    if (!details.isMock() || details.isSpy())
                        return org.mockito.Mockito.spy(
                                new org.flexlb.balance.preemption.PreemptionCompletion());
                    var handler =
                            org.mockito.Mockito.mock(
                                    org.flexlb.balance.preemption.PreemptionCompletion.class);
                    org.mockito.Mockito.doCallRealMethod()
                            .when(handler)
                            .cancelRequest(
                                    org.mockito.ArgumentMatchers.any(),
                                            org.mockito.ArgumentMatchers.any(),
                                    org.mockito.ArgumentMatchers.any(),
                                            org.mockito.ArgumentMatchers.any(),
                                    org.mockito.ArgumentMatchers.any(),
                                            org.mockito.ArgumentMatchers.anyLong());
                    return handler;
                });
    }

    public static void awaitRequestWork(SchedulerTestServices runtime) {
        var tasks = (RequestTaskPool) runtime.tasks();
        if (!tasks.isExecutingOnCurrentThread()) {
            tasks.awaitIdle();
            return;
        }
        // Race hooks run inside asynchronous DIRECT work. Drain the other tasks, not this hook.
        var monitor = ReflectionTestUtils.getField(tasks, "tasksLock");
        try {
            var pending = RequestTaskPool.class.getDeclaredField("submittedTasks");
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

    public static java.util.concurrent.ScheduledThreadPoolExecutor cleanupExecutor(
            SchedulerTestServices runtime) {
        return (java.util.concurrent.ScheduledThreadPoolExecutor)
                ReflectionTestUtils.getField(runtime.taskTimer(), "executor");
    }

    public static org.flexlb.balance.preemption.DefaultDecodePreemption routePreemption(
            AbstractRequestScheduler owner) {
        var runtime = runtime(owner);
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.batchConfig());
        return new org.flexlb.balance.preemption.DefaultDecodePreemption(
                dispatcher(runtime),
                repository(owner),
                runtime.tasks(),
                runtime.taskTimer(),
                mock(RequestSchedulerReporter.class),
                config,
                (org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity)
                        org.flexlb.balance.scheduler.DeliveryTestSupport.getField(
                                owner.dispatcher, "decodeAdmissionCapacity"),
                new org.flexlb.balance.preemption.PriorityDecodePreemptionPolicy());
    }

    public static SchedulerTestServices runtime(Object owner) {
        return SchedulerTestServices.forOwner(owner);
    }

    public static DeliveryCoordinator dispatcher() {
        return spy(
                new DeliveryCoordinator(
                        new org.flexlb.balance.scheduler.RequestRepository(),
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.newConfig()),
                        new DeliveryMetricsReporter(null),
                        new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                .decodeAdmissionCapacity(
                                        DeliveryTestSupport.service(
                                                SchedulingTestConfig.newConfig())),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.scheduler.DeliveryExecution.class)));
    }

    public static synchronized RequestRepository repository(AbstractRequestScheduler owner) {
        if (owner == null) {
            return null;
        }
        return owner.repository != null
                ? owner.repository
                : mockRepositories.computeIfAbsent(owner, SchedulerTestSupport::mockRepository);
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
                new CapacityNotifier(),
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
            CapacityNotifier availability) {
        return create(config, batches, requests, trace, router, eviction, availability, true);
    }

    public static AbstractRequestScheduler create(
            SchedulerTestServices runtime,
            ConfigService config,
            DeliveryMetricsReporter batches,
            TheoryCacheHitReporter trace,
            RequestRouteSelector router,
            DecodePreemption eviction,
            CapacityNotifier availability) {
        return create(runtime, config, batches, trace, router, eviction, availability, true);
    }

    public static RoutingContext claim(
            AbstractRequestScheduler scheduler,
            String id,
            java.util.concurrent.CompletableFuture<?> future) {
        var context = scheduler.repository.findActive(id);
        return context != null && context.data().getFuture() == future
                ? scheduler.tryBeginRouteSelection(context)
                : null;
    }

    static void submitClose(
            RequestTaskPool pool,
            RequestContext context,
            Runnable work,
            boolean inline,
            java.util.function.Consumer<Throwable> rejected,
            boolean requiredResponse) {
        closeQueue(context)
                .submit(
                        context,
                        work,
                        inline ? Runnable::run : task -> pool.execute(context, task),
                        !requiredResponse,
                        rejected);
    }

    static RequestCloseQueue closeQueue(RequestContext context) {
        var scheduler = context.data().getRequestScheduler();
        if (scheduler != null) {
            var queue = (RequestCloseQueue) ReflectionTestUtils.getField(scheduler, "closeQueue");
            if (queue != null) return queue;
        }
        return testCloseQueues.computeIfAbsent(context, ignored -> new RequestCloseQueue());
    }

    static org.flexlb.balance.RoutingResult<RouteResult, CapacityKey> enqueueRoute(
            QueuedRequestScheduler scheduler, RequestProtocolTestSupport.SelectedRoute plan) {
        var route = RequestProtocolTestSupport.planRouting(plan);
        var result = scheduler.enqueueRoute(route);
        if (result.status() == RoutingResult.Status.SUCCESS) scheduler.notifyQueuedRoute(route);
        return result;
    }

    static RoutingResult<RouteResult, CapacityKey> selectAndPublish(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            java.util.function.Supplier<RoutingResult<RouteResult, CapacityKey>> selection) {
        return scheduler instanceof DirectRequestScheduler direct
                ? direct.selectAndPublish(context, selection)
                : ((QueuedRequestScheduler) scheduler).selectAndPublish(context, selection);
    }

    static RoutingResult<RouteResult, CapacityKey> selectAndPublish(
            AbstractRequestScheduler scheduler,
            RoutingContext routing,
            java.util.function.Supplier<RoutingResult<RouteResult, CapacityKey>> selection) {
        return scheduler instanceof DirectRequestScheduler direct
                ? direct.selectAndPublish(routing, selection)
                : ((QueuedRequestScheduler) scheduler).selectAndPublish(routing, selection);
    }

    static void startScheduling(RequestScheduler scheduler) {
        scheduler.start();
    }

    static AbstractRequestScheduler spyLifecycle(RequestScheduler scheduler) {
        return (AbstractRequestScheduler) scheduler;
    }

    static ShutdownResult shutdown(SchedulerTestServices runtime) {
        return runtime.shutdown(ShutdownMode.IMMEDIATE, Duration.ZERO);
    }

    static DeliveryCoordinator dispatcher(SchedulerTestServices runtime) {
        return (DeliveryCoordinator) ReflectionTestUtils.getField(runtime, "dispatcher");
    }

    static AbstractRequestScheduler initializedScheduler(SchedulerTestServices runtime) {
        synchronized (ReflectionTestUtils.getField(runtime, "schedulerLock")) {
            return (AbstractRequestScheduler) ReflectionTestUtils.getField(runtime, "scheduler");
        }
    }

    static void closeRequestWork(SchedulerTestServices runtime) {
        ((RequestTaskPool) runtime.tasks()).awaitIdle();
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

    static void closeRequestExecutors(SchedulerTestServices runtime) {
        Throwable failure =
                org.flexlb.util.Failures.run(null, runtime.scheduling()::closeExecutors);
        failure = org.flexlb.util.Failures.run(failure, runtime::closeTimers);
        org.flexlb.util.Failures.rethrow(failure, "fixture execution resources failed to close");
    }

    /** Response/timer fixtures explicitly supply resource proof; production has no manual close path. */
    static RequestState settleTerminalFixture(RequestContext context, ContextResult.End terminal) {
        org.junit.jupiter.api.Assertions.assertSame(terminal, context.snapshot().outcome());
        var delivery = context.snapshot().delivery();
        if (delivery != null) {
            org.junit.jupiter.api.Assertions.assertEquals(
                    Delivery.SendMode.BATCH_ENQUEUE,
                    delivery.identity().kind(),
                    "this fixture supplies batch transport cleanup evidence");
            if (delivery.send() == ContextResult.SendState.SENDING) {
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
                            observedRequest.recordRemoteCleanup(
                                    new ContextResult.CleanupAcknowledgement(
                                            delivery.identity(),
                                            ContextResult.CancelEvidenceKind.CLEANED,
                                            System.currentTimeMillis())));
        }
        context.recordCloseNotifications(true, true, System.currentTimeMillis());
        org.junit.jupiter.api.Assertions.assertNotNull(context.releaseReason());
        context.recordResourcesReleased(
                context.snapshot().route(), true, true, System.currentTimeMillis());
        org.junit.jupiter.api.Assertions.assertNotNull(context.snapshot().terminalRecord());
        return context.snapshot().terminalRecord();
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
        if (state.routingContext() != null
                && (state.route() == null || state.routingContext().routeToWithdraw() != null)) {
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

    static EngineGrpcClient engineClient(Object owner) {
        return (EngineGrpcClient)
                org.flexlb.balance.scheduler.DeliveryTestSupport.getField(
                        runtime(owner).dispatcher, "grpcClient");
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

    static AbstractRequestScheduler create(
            ConfigService config,
            DeliveryMetricsReporter batches,
            RequestSchedulerReporter requests,
            TheoryCacheHitReporter trace,
            RequestRouteSelector router,
            DecodePreemption eviction,
            CapacityNotifier availability,
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

    static Registration responsePublication(ResponseCompletionExecutor executor) {
        return new Registration(
                executor,
                new Object(),
                new ResponseCompletionExecutor.RequestFuture((kind, response, failure) -> false),
                () -> true);
    }

    static DeliveryCoordinator createDispatcher(
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
        return new DeliveryCoordinator(
                new org.flexlb.balance.scheduler.RequestRepository(),
                client,
                dispatcherService,
                new DeliveryMetricsReporter(null),
                new org.flexlb.balance.scheduler.SchedulerConfiguration()
                        .decodeAdmissionCapacity(dispatcherService),
                org.mockito.Mockito.mock(org.flexlb.balance.scheduler.DeliveryExecution.class));
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

    private static AbstractRequestScheduler create(
            SchedulerTestServices runtime,
            ConfigService config,
            DeliveryMetricsReporter batches,
            TheoryCacheHitReporter trace,
            RequestRouteSelector router,
            DecodePreemption eviction,
            CapacityNotifier availability,
            boolean startPlacement) {
        var snapshot = config.loadBalanceConfig();
        var capacity = new SchedulerConfiguration().decodeAdmissionCapacity(config);
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

    enum Stage {
        QUEUED,
        ROUTING,
        READY_TO_DELIVER,
        DELIVERING,
        RESULT_PENDING,
        FINALIZING,
        FINISHED
    }
}
