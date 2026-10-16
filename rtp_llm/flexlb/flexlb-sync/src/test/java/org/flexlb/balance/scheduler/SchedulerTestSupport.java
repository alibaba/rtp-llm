package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.WeakHashMap;

import static org.mockito.Mockito.*;

/** Builds real request owners and shared facilities; contains no request state machine. */
public final class SchedulerTestSupport {
    /** Empty Decode ownership uses typed domain results, including in tests without a real ledger. */
    public static org.flexlb.balance.endpoint.DecodeEndpoint mockDecodeEndpoint() {
        return RequestProtocolTestSupport.decodeEndpoint();
    }

    public static DeliveryDispatcher dispatcher() {
        return spy(new DeliveryDispatcher(null, DeliveryTestSupport.service(SchedulingTestConfig.newConfig()),
                null, new DeliveryMetricsReporter(null), null));
    }
    static DeliveryDispatcher dispatcher(SchedulerRuntime runtime) {
        return (DeliveryDispatcher) ReflectionTestUtils.getField(runtime, "dispatcher");
    }

    private static final Map<AbstractRequestScheduler, RequestRepository> mockRepositories = new WeakHashMap<>();
    public static synchronized RequestRepository repository(AbstractRequestScheduler owner) {
        if (owner == null) { return null; }
        return owner.requests != null ? owner.requests : mockRepositories.computeIfAbsent(owner, SchedulerTestSupport::mockRepository);
    }
    private static RequestRepository mockRepository(AbstractRequestScheduler owner) {
        var repository = mock(RequestRepository.class);
        when(repository.ownerOf(anyLong())).thenReturn(owner);
        when(repository.findActive(anyLong())).thenAnswer(call -> {
            var context = mock(RequestContext.class);
            var data = mock(RequestData.class);
            when(context.data()).thenReturn(data);
            when(data.getScheduler()).thenReturn(owner);
            return context;
        });
        return repository;
    }
    public static void bindOwner(RequestContext context, AbstractRequestScheduler owner) {
        if (owner.runtime == null) {
            var runtime = mock(SchedulerRuntime.class);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
            var deliveries = new DeliveryExecution(service, ignored -> { });
            when(runtime.deliveries()).thenReturn(deliveries);
            when(runtime.signals()).thenReturn(new RequestSignals());
            when(runtime.deliveryReporter()).thenReturn(mock(DeliveryMetricsReporter.class));
            ReflectionTestUtils.setField(owner, "runtime", runtime);
        }
        if (context.data().getScheduler() == null) {
            // Minimal fixtures may bind an owner without registering a runnable request.
            ReflectionTestUtils.setField(context.data(), "scheduler", owner);
        }
    }
    private static final Map<org.flexlb.balance.endpoint.PrefillEndpoint, RequestRepository> endpointRepositories = java.util.Collections.synchronizedMap(new WeakHashMap<>());
    public static void associateEndpoint(org.flexlb.balance.endpoint.PrefillEndpoint endpoint, RequestRepository repository) {
        endpointRepositories.put(endpoint, repository);
    }
    public static void bindEndpointOwner(org.flexlb.balance.endpoint.PrefillEndpoint endpoint, RequestRoute route) {
        if (route.ctx() == null) {
            var context = mock(RequestContext.class);
            when(route.ctx()).thenReturn(context);
            var repository = endpointRepositories.get(endpoint);
            var owner = repository == null ? null : repository.ownerOf(route.requestId());
            var data = mock(RequestData.class);
            when(context.data()).thenReturn(data);
            when(data.getScheduler()).thenReturn(owner == null ? mock(AbstractRequestScheduler.class) : owner);
        } else if (route.ctx().data().getScheduler() == null) {
            var repository = endpointRepositories.get(endpoint);
            var owner = repository == null ? null : repository.ownerOf(route.requestId());
            bindOwner(route.ctx(), owner == null ? mock(AbstractRequestScheduler.class) : owner);
        }
        var context = route.ctx();
        var owner = context.data().getScheduler();
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
                context.register(route.requestId(), now, route.expiresAtMs(),
                        context.data().getConfig().getRequestLifecycle().getRequest().getTimeoutMs());
                var placement = (ContextResult.PlacementToken) ((ContextResult.Granted<?>) context.beginPlacement(null)).value();
                context.bindPlacement(placement, route, ContextResult.PublicationResult.PREPARED);
                context.bindPlacement(placement, route, ContextResult.PublicationResult.PUBLISHED);
                context.completePlacement(placement, ContextResult.PublicationResult.PUBLISHED, now);
            }
        }

    }
    static RequestScheduler initializedScheduler(SchedulerRuntime runtime) {
        synchronized (ReflectionTestUtils.getField(runtime, "schedulerLock")) {
            return (RequestScheduler) ReflectionTestUtils.getField(runtime, "scheduler");
        }
    }
    public static void awaitRequestWork(SchedulerRuntime runtime) {
        ReflectionTestUtils.invokeMethod(runtime, "awaitRequestWork");
    }
    static void closeRequestWork(SchedulerRuntime runtime) {
        ReflectionTestUtils.invokeMethod(runtime, "closeRequestWork");
    }
    static java.util.concurrent.ExecutorService requestWorkers(SchedulerRuntime runtime) {
        return (java.util.concurrent.ExecutorService) ReflectionTestUtils.getField(runtime, "requestWorkers");
    }
    static ResponseCompletionExecutor responseCompletions(SchedulerRuntime runtime) {
        return (ResponseCompletionExecutor) ReflectionTestUtils.getField(runtime, "responseCompletions");
    }
    static ExpirationTimer timer(SchedulerRuntime runtime) {
        return (ExpirationTimer) ReflectionTestUtils.getField(runtime, "timer");
    }
    public static java.util.concurrent.ScheduledThreadPoolExecutor cleanupExecutor(SchedulerRuntime runtime) {
        return (java.util.concurrent.ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(runtime, "cleanupExecutor");
    }
    static void closeRequestExecutors(SchedulerRuntime runtime) {
        ReflectionTestUtils.invokeMethod(runtime, "closeRequestExecutors");
    }
    /** Response/timer fixtures explicitly supply resource proof; production has no manual close path. */
    static FixtureSettlement settleTerminalFixture(RequestContext context, ContextResult.EndToken terminal) {
        var delivery = context.snapshot().delivery();
        if (delivery != null) {
            if (delivery.send() == ContextResult.SendState.NOT_STARTED || delivery.send() == ContextResult.SendState.SENDING) {
                context.recordDeliveryResult(new ContextResult.TransportResult(delivery.identity(),
                        org.flexlb.balance.delivery.DeliveryResult.notSent(new IllegalStateException("fixture sender exited before send")), System.currentTimeMillis()));
            }
            context.recordDeliveryResult(new ContextResult.CleanupAcknowledgement(delivery.identity(),
                    ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
        }
        context.completeClose(terminal, new ContextResult.CloseProgress(ContextResult.ProgressKind.NOTIFICATIONS,
                false, false, true, true), System.currentTimeMillis());
        var claim = org.junit.jupiter.api.Assertions.assertInstanceOf(ContextResult.Granted.class, context.beginSettlement());
        var permit = (ContextResult.ClosePermit) claim.value();
        var update = context.completeClose(permit, new ContextResult.CloseProgress(ContextResult.ProgressKind.RELEASE,
                true, true, false, false), System.currentTimeMillis());
        org.junit.jupiter.api.Assertions.assertNotNull(update.after().terminalRecord());
        return new FixtureSettlement(permit, update);
    }

    record FixtureSettlement(ContextResult.ClosePermit permit, ContextResult.Update update) {
        void releaseAndArchive(AbstractRequestScheduler scheduler, RequestContext context) {
            org.junit.jupiter.api.Assertions.assertFalse(Thread.holdsLock(context));
            scheduler.requests.archive(context, update.after().terminalRecord());
            scheduler.runtime.deadlines().remove(context);
        }
    }

    enum Stage { QUEUED, ROUTING, READY_TO_DELIVER, DELIVERING, RESULT_PENDING, FINALIZING, FINISHED }

    static Stage stage(RequestContext context) {
        var state = context.snapshot();
        if (state.phase() == ContextResult.Phase.CLOSED) { return Stage.FINISHED; }
        if (state.phase() == ContextResult.Phase.ENDING) { return Stage.FINALIZING; }
        if (state.delivery() != null && state.delivery().transferred()) {
            return state.delivery().acknowledged() || state.engine().decodeAccepted() ? Stage.RESULT_PENDING : Stage.DELIVERING;
        }
        if (state.placement() != null && (!state.placement().published() || state.placement().operation().withdrawing() != null)) {
            return Stage.ROUTING;
        }
        return state.route() == null ? Stage.QUEUED : Stage.READY_TO_DELIVER;
    }

    static CancelReason cancellationReason(RequestContext context) {
        var facts = context.snapshot().cancellation(); return facts == null ? null : facts.reason();
    }
    static PreemptionRegistration preemption(RequestContext context) {
        var facts = context.snapshot().preemption(); return facts == null ? null : facts.identity();
    }
    static DeliveryClaim delivery(RequestContext context) {
        var facts = context.snapshot().delivery(); return facts == null ? null : facts.identity();
    }
    static boolean isOpen(RequestContext context) {
        var facts = context.snapshot();
        return facts.phase() == ContextResult.Phase.OPEN && facts.cancellation() == null
                && !context.data().getFuture().isDone();
    }

    public static SchedulerRuntime runtime(RequestScheduler owner) { return ((AbstractRequestScheduler) owner).runtime; }
    static EngineGrpcClient engineClient(RequestScheduler owner) {
        return (EngineGrpcClient) ReflectionTestUtils.getField(runtime(owner), "engineClient");
    }
    static Throwable failure(RequestScheduler owner) {
        return (Throwable) ReflectionTestUtils.getField(runtime(owner), "failure");
    }
    static RequestRepository.TerminalRecord terminalRecord(AbstractRequestScheduler owner, RequestState state) {
        var record = repository(owner).findTerminal(state.requestId());
        return record != null && record.state() == state ? record : new RequestRepository.TerminalRecord(state, owner);
    }
    public static AbstractRequestScheduler create(ConfigService config, DeliveryMetricsReporter batches,
            RequestSchedulerReporter requests, RecentCacheKeyTraceReporter trace) {
        var runtime = spy(new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class), batches, requests,
                SchedulerTestSupport.dispatcher(), config, mock(EngineGrpcClient.class)));
        var snapshot = config.loadBalanceConfig();
        AbstractRequestScheduler owner = snapshot.isQueue()
                ? mock(QueuedRequestScheduler.class, withSettings().useConstructor(snapshot, mock(RequestWorkerSelector.class), batches, mock(DecodeCapacityAcquirer.class), runtime, new PlacementAvailability(), new PlacementConfiguration().decodeAdmissionCapacity(config), trace).defaultAnswer(CALLS_REAL_METHODS))
                : mock(DirectRequestScheduler.class, withSettings().useConstructor(mock(RequestWorkerSelector.class), runtime, snapshot, new PlacementConfiguration().decodeAdmissionCapacity(config), trace).defaultAnswer(CALLS_REAL_METHODS));
        if (owner instanceof QueuedRequestScheduler queue) {
            doAnswer(call -> {
                queue.resolveControl(call.getArgument(0), false);
                return null;
            })
                    .when(queue).queueControl(org.mockito.ArgumentMatchers.any());
        }
        runtime.registerScheduler(owner);
        return owner;
    }
    public static RequestScheduler configure(AbstractRequestScheduler owner, FlexlbConfig config,
            RequestWorkerSelector router, DeliveryMetricsReporter reporter, DecodeCapacityAcquirer eviction, PlacementAvailability availability) {
        ReflectionTestUtils.setField(owner, "router", router);
        if (owner instanceof QueuedRequestScheduler queue) {
            ReflectionTestUtils.setField(queue, "decodeCapacity", eviction);
            ReflectionTestUtils.setField(queue, "reporter", reporter);
            ReflectionTestUtils.setField(queue, "plannerCount", Math.max(2, config.getInternalRuntime().getQueuePlannerThreads()));
            ReflectionTestUtils.setField(queue, "queueSettings", QueueExecutionSettings.capture(config));
            ReflectionTestUtils.setField(queue, "orderedQueue", new OrderedRequestQueue(config.isPriorityOrdering()));
            ReflectionTestUtils.setField(queue, "scanBudgetMultiplier", config.queueScheduler().getScanBudgetMultiplier());
            var oldAvailability = (PlacementAvailability) ReflectionTestUtils.getField(queue, "availability");
            var listener = (PlacementAvailability.Listener) ReflectionTestUtils.getField(queue, "availabilityListener");
            oldAvailability.removeListener(listener);
            ReflectionTestUtils.setField(queue, "availability", availability);
            ReflectionTestUtils.setField(queue, "waitingRequests", new PlacementCapacityWaitQueue(config.isPriorityOrdering(), availability));
            availability.addListener(listener);
            doCallRealMethod().when(queue).queueControl(org.mockito.ArgumentMatchers.any());
            if (!((Thread)ReflectionTestUtils.getField(queue, "decisionThread")).isAlive()) { queue.start(); }
        }
        return owner;
    }

    static DeliveryDispatcher createDispatcher(EngineGrpcClient client,
                                                   org.flexlb.config.ConfigService service,
                                                   int threads, int queueCapacity) {
        var config = org.mockito.Mockito.spy(service.loadBalanceConfig());
        var sizing = org.mockito.Mockito.spy(config.getInternalRuntime());
        org.mockito.Mockito.doReturn(threads).when(sizing).getBatchDispatchThreads();
        org.mockito.Mockito.doReturn(queueCapacity).when(sizing).getBatchDispatchQueueCapacity();
        org.mockito.Mockito.doReturn(sizing).when(config).getInternalRuntime();
        var dispatcherService = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class,
                org.mockito.AdditionalAnswers.delegatesTo(service));
        org.mockito.Mockito.doReturn(config).when(dispatcherService).loadBalanceConfig();
        return new DeliveryDispatcher(client, dispatcherService, null, new DeliveryMetricsReporter(null), null);
    }
}
