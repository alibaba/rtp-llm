package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.Registration;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.rpc.EngineGrpcClient;
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
    private static final Map<PrefillEndpoint, RequestRepository> endpointRepositories =
            java.util.Collections.synchronizedMap(new WeakHashMap<>());
    private static final java.util.Map<RequestContext, RequestCloseQueue> testCloseQueues =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    public static AbstractRequestScheduler createStartedScheduler(
            AbstractRequestScheduler lifecycle,
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.schedule.DecodePreemption preemption,
            CapacityNotifier availability,
            SchedulingQueue queue) {
        var scheduler =
                SchedulerTestServices.create(
                        lifecycle, config, router, preemption, availability, queue);
        scheduler.start();
        return scheduler;
    }

    public static AbstractRequestScheduler lifecycle(RequestScheduler scheduler) {
        return (AbstractRequestScheduler) scheduler;
    }

    /**
     * Empty Decode ownership uses typed domain results, including in tests without a real ledger.
     */
    public static DecodeEndpoint mockDecodeEndpoint() {
        return RequestProtocolTestSupport.decodeEndpoint();
    }

    public static void associateEndpoint(PrefillEndpoint endpoint, RequestRepository repository) {
        endpointRepositories.put(endpoint, repository);
    }

    public static void bindEndpointOwner(PrefillEndpoint endpoint, RequestResult route) {
        if (route.ctx() == null) {
            var context = mock(RequestContext.class);
            when(route.ctx()).thenReturn(context);
            var repository = endpointRepositories.get(endpoint);
            var owner = repository == null ? null : repository.ownerOf(route.requestId());

            when(context.getRequestScheduler())
                    .thenReturn(owner == null ? mock(AbstractRequestScheduler.class) : owner);
        } else if (route.ctx().getRequestScheduler() == null) {
            var repository = endpointRepositories.get(endpoint);
            var owner = repository == null ? null : repository.ownerOf(route.requestId());
            bindOwner(route.ctx(), owner == null ? mock(AbstractRequestScheduler.class) : owner);
        }
        var context = route.ctx();
        var owner = context.getRequestScheduler();
        if (owner.repository == null) {
            var snapshot = context.snapshot();
            if (snapshot == null || org.mockito.Mockito.mockingDetails(snapshot).isMock()) {
                var facts = snapshot == null ? mock(RequestContext.Snapshot.class) : snapshot;
                when(context.snapshot()).thenReturn(facts);
                when(facts.route()).thenReturn(route);
                when(context.getConfig()).thenReturn(SchedulingTestConfig.newConfig());
            } else if (snapshot.requestId() == null) {
                long now = System.currentTimeMillis();
                RequestProtocolTestSupport.register(
                        context,
                        route.requestId(),
                        now,
                        route.expiresAtMs(),
                        context.getConfig().getRequestLifecycle().getRequest().getTimeoutMs(),
                        org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
                var placement = (SelectionAttempt) context.beginRouteSelection();
                synchronized (context) {
                    context.registerSelectedRoute(placement, route);
                    context.tryCommitRoute(
                            placement,
                            route,
                            new RequestResult(context, RequestContext.ResponseKind.ACK, now),
                            now);
                }
                org.flexlb.balance.schedule.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.finishSelectionWithoutCommit(placement, null, now));
            }
        }
    }

    public static void bindOwner(RequestContext context, AbstractRequestScheduler owner) {
        if (org.flexlb.balance.schedule.SchedulerTestSupport.runtime(owner) == null) {
            var runtime = mock(SchedulerTestServices.class);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());

            org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                    runtime, "deliveryReporter", mock(DeliveryMetricsReporter.class));
            org.flexlb.balance.schedule.SchedulerTestServices.bind(owner, runtime);
        }
        if (context.getRequestScheduler() == null) {
            // Minimal fixtures may bind an owner without registering a runnable request.
            ReflectionTestUtils.setField(context, "requestScheduler", owner);
        }
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
                ReflectionTestUtils.getField(runtime, "cleanupTimerExecutor");
    }

    public static org.flexlb.balance.schedule.DefaultDecodePreemption routePreemption(
            AbstractRequestScheduler owner) {
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        return new org.flexlb.balance.schedule.DefaultDecodePreemption(
                repository(owner),
                mock(RequestSchedulerReporter.class),
                config,
                (org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity)
                        org.flexlb.balance.schedule.DeliveryTestSupport.getField(
                                runtime(owner).dispatcher, "decodeAdmissionCapacity"),
                new org.flexlb.balance.schedule.PriorityDecodePreemptionPolicy());
    }

    public static synchronized RequestRepository repository(AbstractRequestScheduler owner) {
        if (owner == null) {
            return null;
        }
        return owner.repository != null
                ? owner.repository
                : mockRepositories.computeIfAbsent(owner, SchedulerTestSupport::mockRepository);
    }

    public static SchedulerTestServices runtime(Object owner) {
        return SchedulerTestServices.forOwner(owner);
    }

    public static EngineCancellation dispatcher() {
        return spy(new EngineCancellation(mock(EngineGrpcClient.class)));
    }

    public static EngineCancellation dispatcher(SchedulerTestServices runtime) {
        return runtime.dispatcher;
    }

    /**
     * State/protocol fixtures have no running placement loop.
     */
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

    /**
     * Construct all placement dependencies together and let the runtime own their shutdown.
     */
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

    public static SelectionAttempt claim(
            AbstractRequestScheduler scheduler,
            String id,
            java.util.concurrent.CompletableFuture<?> future) {
        var context = scheduler.repository.findActive(id);
        return context != null && context.getFuture() == future
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
                        inline ? Runnable::run : task -> pool.execute(task),
                        !requiredResponse,
                        rejected);
    }

    static RequestCloseQueue closeQueue(RequestContext context) {
        var scheduler = context.getRequestScheduler();
        if (scheduler != null) {
            var queue = (RequestCloseQueue) ReflectionTestUtils.getField(scheduler, "closeQueue");
            if (queue != null) return queue;
        }
        return testCloseQueues.computeIfAbsent(context, ignored -> new RequestCloseQueue());
    }

    static RequestResult selectAndPublish(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            java.util.function.Supplier<RequestResult> selection) {
        return scheduler.planRoute(context, selection, false);
    }

    static RequestResult selectAndPublish(
            AbstractRequestScheduler scheduler,
            SelectionAttempt routing,
            java.util.function.Supplier<RequestResult> selection) {
        return scheduler.executeRouteSelection(routing, selection, false);
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

    static Stage stage(RequestContext context) {
        var state = context.snapshot();
        if (state.phase() == RequestContext.Phase.CLOSED) {
            return Stage.FINISHED;
        }
        if (state.phase() == RequestContext.Phase.ENDING) {
            return Stage.FINALIZING;
        }
        if (state.committedRoute() != null) {
            return state.responseForFuture() != null || state.decodeAccepted()
                    ? Stage.RESULT_PENDING
                    : Stage.DELIVERING;
        }
        if (state.selectionId() != 0L) {
            return Stage.ROUTING;
        }
        return state.route() == null ? Stage.QUEUED : Stage.READY_TO_DELIVER;
    }

    static CancelReason cancellationReason(RequestContext context) {
        var facts = context.snapshot().cancellationReason();
        return facts;
    }

    static RequestResult delivery(RequestContext context) {
        return context.snapshot().committedRoute();
    }

    static boolean isOpen(RequestContext context) {
        var facts = context.snapshot();
        return facts.phase() == RequestContext.Phase.OPEN
                && facts.cancellationReason() == null
                && !context.getFuture().isDone();
    }

    static EngineGrpcClient engineClient(Object owner) {
        return (EngineGrpcClient)
                org.flexlb.balance.schedule.DeliveryTestSupport.getField(
                        runtime(owner).dispatcher, "grpcClient");
    }

    static Throwable failure(Object owner) {
        return (Throwable)
                org.flexlb.balance.schedule.SchedulerTestServices.failure(runtime(owner));
    }

    static RequestResult requestRecordForArchive(
            AbstractRequestScheduler owner, RequestResult state) {
        var record = repository(owner).findTerminal(state.requestId());
        return record == state
                ? record
                : new RequestResult(
                        state.requestId(),
                        state.originalNumericRequestId(),
                        state.requestPhase(),
                        state.state(),
                        state.createdAtMs(),
                        state.updatedAtMs(),
                        state.detail(),
                        owner);
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

    private static RequestRepository mockRepository(AbstractRequestScheduler owner) {
        var repository = mock(RequestRepository.class);
        when(repository.ownerOf(org.mockito.ArgumentMatchers.anyString())).thenReturn(owner);
        when(repository.findActive(org.mockito.ArgumentMatchers.anyString()))
                .thenAnswer(
                        call -> {
                            var context = mock(RequestContext.class);

                            when(context.getRequestScheduler()).thenReturn(owner);
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
        var dependencies = runtime.newLifecycle(snapshot, capacity, trace);
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
                                                        dependencies,
                                                        availability,
                                                        new OrderedRequestQueue(
                                                                snapshot.isPriorityOrdering())))
                                        .defaultAnswer(CALLS_REAL_METHODS))
                        : mock(
                                DirectRequestScheduler.class,
                                withSettings()
                                        .useConstructor(
                                                SchedulerTestServices.constructorArguments(
                                                        router, dependencies))
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
