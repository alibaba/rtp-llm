package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.WeakHashMap;

import static org.mockito.Mockito.*;

/** Builds real request owners and shared facilities; contains no request state machine. */
public final class SchedulerTestSupport {
    /** Build a real local reservation without publishing it to the master queue. */
    public static DecodeResources.ReservationHandle reserveUnqueuedDecode(
            DecodeEndpoint endpoint, WorkerEndpoint.GenerationPin pin, long requestId,
            long hardKv, long expectedKv, int priority) {
        endpoint.requirePinnedGeneration(pin);
        Object state = ReflectionTestUtils.getField(endpoint, "state");
        var lock = (java.util.concurrent.locks.ReentrantLock) ReflectionTestUtils.getField(state, "admissionLock");
        lock.lock();
        try {
            if (!Boolean.TRUE.equals(ReflectionTestUtils.invokeMethod(
                    state, "requestIdAvailableForReservationLocked", requestId))) {
                throw new IllegalStateException("Decode request id is already owned: " + requestId);
            }
            return ReflectionTestUtils.invokeMethod(state, "createReservationLocked", requestId,
                    hardKv, expectedKv, priority, org.flexlb.enums.DecodeTaskPhase.LOCAL_RESERVED);
        } finally {
            lock.unlock();
        }
    }

    /** Read shadow ownership for assertions, without granting a production lookup capability. */
    public static DecodeResources.ReservationHandle decodeReservation(DecodeEndpoint endpoint, long requestId) {
        return endpoint.isRetired() ? null : decodeReservation(endpoint.resourceSnapshot(), requestId);
    }

    private static DecodeResources.ReservationHandle decodeReservation(
            DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(requestId);
        if (request == null || request.reservationToken() <= 0L) { return null; }
        return switch (request.phase()) {
            case LOCAL_RESERVED, MASTER_QUEUED_NOT_DISPATCHED, ENGINE_MAY_HAVE_SEEN ->
                    new DecodeResources.ReservationHandle(
                            snapshot.routing().generationId(), requestId, request.reservationToken());
            default -> null;
        };
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
            when(context.scheduler()).thenReturn(owner);
            return context;
        });
        return repository;
    }
    public static void bindOwner(RequestContext context, AbstractRequestScheduler owner) {
        if (context.scheduler() == null) { context.bindScheduler(owner); }
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
            when(context.scheduler()).thenReturn(owner == null ? mock(AbstractRequestScheduler.class) : owner);
        } else if (route.ctx().scheduler() == null) {
            var repository = endpointRepositories.get(endpoint);
            var owner = repository == null ? null : repository.ownerOf(route.requestId());
            bindOwner(route.ctx(), owner == null ? mock(AbstractRequestScheduler.class) : owner);
        }
    }
    public static SchedulerRuntime runtime(RequestScheduler owner) { return ((AbstractRequestScheduler) owner).runtime; }
    public static FlexlbConfig config(AbstractRequestScheduler owner) { return owner.config; }
    public static DecodeCapacityAcquirer eviction(AbstractRequestScheduler owner) {
        return new DecodeCapacityAcquirer(mock(EngineCancelChannel.class), repository(owner), runtime(owner), mock(RequestSchedulerReporter.class));
    }
    static RequestRepository.TerminalRecord terminalRecord(AbstractRequestScheduler owner, RequestState state) {
        var record = repository(owner).findTerminal(state.requestId());
        return record != null && record.state() == state ? record : new RequestRepository.TerminalRecord(state, owner);
    }
    public static AbstractRequestScheduler create(ConfigService config, DeliveryMetricsReporter batches,
            RequestSchedulerReporter requests, RecentCacheKeyTraceReporter trace) {
        var runtime = new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class), batches, requests,
                mock(DefaultBatchDispatcher.class), config, trace, mock(EngineCancelChannel.class));
        var snapshot = config.loadBalanceConfig();
        var settings = snapshot;
        AbstractRequestScheduler owner = snapshot.isQueue()
                ? mock(QueuedRequestScheduler.class, withSettings().useConstructor(settings, mock(RequestWorkerSelector.class), batches, mock(DecodeCapacityAcquirer.class), runtime, new PlacementAvailability()).defaultAnswer(CALLS_REAL_METHODS))
                : mock(DirectRequestScheduler.class, withSettings().useConstructor(mock(RequestWorkerSelector.class), runtime, settings).defaultAnswer(CALLS_REAL_METHODS));
        if (owner instanceof QueuedRequestScheduler queue) {
            doAnswer(call -> {
                queue.onGlobalControl(call.getArgument(0));
                return null;
            })
                    .when(queue).signalControl(org.mockito.ArgumentMatchers.any());
        }
        runtime.initializeScheduler(owner);
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
            ReflectionTestUtils.setField(queue, "waitingRequests", new PlacementWaitQueue(config.isPriorityOrdering(), availability));
            availability.addListener(listener);
            doCallRealMethod().when(queue).signalControl(org.mockito.ArgumentMatchers.any());
            if (!((Thread)ReflectionTestUtils.getField(queue, "decisionThread")).isAlive()) { queue.start(); }
        }
        return owner;
    }
}
