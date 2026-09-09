package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** DIRECT uses real admission ledgers and lifecycle while bypassing waiting queues. */
class DirectAdmissionContractTest {

    @ParameterizedTest
    @ValueSource(ints = {1, 2})
    void concurrentDirectAdmissionsUseCurrentCapacityWithoutRetryingSelection(int prefillCapacity) throws Exception {
        try (Fixture fixture = new Fixture(prefillCapacity, 2);
             var executor = Executors.newFixedThreadPool(2)) {
            var readyToPublish = new CyclicBarrier(2);
            doAnswer(call -> {
                // Both requests have selected this generation and acquired Decode capacity.
                // Their Prefill publications must arbitrate against current ownership.
                readyToPublish.await(2L, TimeUnit.SECONDS);
                return call.callRealMethod();
            }).when(fixture.requests).commitRoute(any(), any());
            var first = executor.submit(() -> fixture.scheduler.submit(fixture.context(201L))
                    .get(2L, TimeUnit.SECONDS));
            var second = executor.submit(() -> fixture.scheduler.submit(fixture.context(202L))
                    .get(2L, TimeUnit.SECONDS));
            var responses = List.of(first.get(3L, TimeUnit.SECONDS), second.get(3L, TimeUnit.SECONDS));

            assertEquals(prefillCapacity, responses.stream().filter(Response::isSuccess).count());
            responses.stream().filter(response -> !response.isSuccess()).forEach(response ->
                    assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode()));
            assertEquals(prefillCapacity, fixture.prefill.observedRequestCount());
            assertEquals(prefillCapacity, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(48L * prefillCapacity, fixture.decode.routingView().inflightExpectedKv());
            fixture.assertNoWaitingQueue();
            verify(fixture.requests, times(2)).commitRoute(any(), any());
            verify(fixture.prefillSelector, times(2)).select(any(), eq(RoleType.PREFILL), any());
        }
    }

    @Test
    void directRouteOwnsPrefillAndDecodeUntilEngineTerminalWithoutEnteringWaitingQueues() throws Exception {
        try (Fixture fixture = new Fixture()) {
            Response response = fixture.scheduler.submit(fixture.context(101L)).get(2L, TimeUnit.SECONDS);

            assertTrue(response.isSuccess());
            assertFalse(response.isEnqueuedByMaster());
            fixture.assertNoWaitingQueue();
            assertEquals(1, fixture.prefill.observedRequestCount());
            assertEquals(0, fixture.prefill.getInflightBatchCount());
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(48L, fixture.decode.routingView().inflightExpectedKv());
            assertEquals(0, fixture.decode.layeredAdmissionView().queuedCount());
            assertEquals(1, fixture.scheduler.getInflightSize());
            assertEquals(RequestState.Phase.ACKNOWLEDGED, fixture.scheduler.getRequestState(101L, 0L).state());
            var reservation = fixture.decode.reservationHandle(101L);
            assertNotNull(reservation);
            assertThrows(IllegalStateException.class, () -> fixture.decode.releaseReservationExact(reservation));

            fixture.observe(fixture.prefill, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(0L, fixture.prefill.observedRequestCount());
            assertEquals(48L, fixture.decode.routingView().inflightExpectedKv(),
                    "Prefill completion must retain Decode ownership until its own observation");
            fixture.observe(fixture.decode, Map.of("101", task(101L, TaskPhase.RUNNING)), Map.of());
            assertTrue(fixture.decode.isReservationAccepted(reservation));
            fixture.observe(fixture.decode, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(0, fixture.scheduler.getInflightSize());
            fixture.assertNoWaitingQueue();
        }
    }

    @Test
    void fullDecodeRejectsImmediatelyAndRollsBackTheProvisionalRoute() throws Exception {
        try (Fixture fixture = new Fixture()) {
            DecodeEndpoint.ReservationHandle occupant;
            try (var pin = fixture.decode.tryPinGeneration()) {
                assertNotNull(pin);
                occupant = fixture.decode.tryReservePlacementPinned(pin, 999L, 32L, 48L, 50);
            }
            assertNotNull(occupant);
            var acquired = fixture.decode.acquireEngineDispatchPermit(occupant,
                    new DecodeEndpoint.AdmissionCapacity(1L, 90L));
            assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
            assertEquals(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquired.permit().transferToEngineLifecycle());

            Response response = assertTimeoutPreemptively(Duration.ofSeconds(2),
                    () -> fixture.scheduler.submit(fixture.context(102L)).get(2L, TimeUnit.SECONDS));

            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
            fixture.assertNoPrefillOwnership();
            assertNull(fixture.decode.reservationHandle(102L));
            assertEquals(occupant, fixture.decode.reservationHandle(999L));
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(48L, fixture.decode.routingView().inflightExpectedKv());
            assertEquals(0, fixture.decode.layeredAdmissionView().queuedCount());
            assertEquals(0, fixture.scheduler.getInflightSize());
        }
    }

    @Test
    void decodeTerminalBetweenAcceptedPreparationAndSlotBindingCannotPublishASuccessfulRoute() throws Exception {
        try (Fixture fixture = new Fixture()) {
            AtomicBoolean raced = new AtomicBoolean();
            doAnswer(call -> {
                DecodeEndpoint.ReservationHandle reservation = call.getArgument(0);
                assertEquals(103L, reservation.requestId());
                fixture.assertItemNotBound(103L);
                fixture.observe(fixture.decode, Map.of("103", task(103L, TaskPhase.KV_ALLOCATED)), Map.of());
                var acquired = (DecodeEndpoint.EngineDispatchPermitAcquisition) call.callRealMethod();
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED, acquired.status());
                assertNotNull(acquired.permit(), "accepted preparation must retain an exact handoff capability");
                fixture.observe(fixture.decode, Map.of(), Map.of("103", task(103L, TaskPhase.RUNNING)));
                fixture.assertItemNotBound(103L);
                raced.set(true);
                return acquired;
            }).when(fixture.decode).acquireEngineDispatchPermit(any(), any());

            Response response = fixture.scheduler.submit(fixture.context(103L)).get(2L, TimeUnit.SECONDS);

            assertTrue(raced.get());
            assertFalse(response.isSuccess(), "an already ended request must not receive a new successful route");
            fixture.assertNoPrefillOwnership();
            assertNull(fixture.decode.reservationHandle(103L));
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(0L, fixture.decode.routingView().inflightExpectedKv());
            assertEquals(0, fixture.scheduler.getInflightSize());
        }
    }

    private static TaskInfo task(long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(32L);
        task.setPhase(phase);
        return task;
    }

    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config = SchedulingTestConfig.newConfig();
        private final RequestRegistry requests;
        private final EndpointRegistry endpoints;
        private final RouteDeliveryStrategy queuedDelivery;
        private final PrefillEndpoint prefill;
        private final DecodeEndpoint decode;
        private final CostBasedPrefillStrategy prefillSelector;
        private final RequestScheduler scheduler;
        private final SchedulerRuntime runtime;

        private Fixture() {
            this(16, 1);
        }

        private Fixture(int prefillCapacity, int decodeCapacity) {
            config.setScheduler(SchedulerConfig.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getDispatcher().setMaxInflightPerPrefillWorker(prefillCapacity);
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests((long) decodeCapacity);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(BatchSchedulerReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);
            requests = spy(new RequestRegistry(service, reporter, requestReporter));
            var projector = new EndpointEventProjector(requests);
            var placement = new PlacementAvailability();
            queuedDelivery = spy(new RouteDeliveryStrategy(requests, new DeliveryMetrics(reporter)));
            endpoints = new EndpointRegistry(service, projector, reporter, queuedDelivery, placement);
            WorkerStatus worker = worker(RoleType.PREFILL, "127.0.0.1");
            worker.lock.lock();
            try {
                var response = status(worker, Map.of(), Map.of());
                prefill = (PrefillEndpoint) endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(worker.freezeStatusResponse(response))).endpoint();
            } finally {
                worker.lock.unlock();
            }
            decode = spy(new DecodeEndpoint(worker(RoleType.DECODE, "127.0.0.2"), projector));
            observe(decode, Map.of(), Map.of());
            prefillSelector = mock(CostBasedPrefillStrategy.class);
            var decodeSelector = mock(DecodeSelector.class);
            when(prefillSelector.select(any(), eq(RoleType.PREFILL), any())).thenAnswer(call -> {
                var context = call.getArgument(0, BalanceContext.class);
                var pin = prefill.tryPinGeneration();
                assertNotNull(pin);
                return PlacementResult.success(SelectedRole.prefill(pin,
                        metadata(prefill, context.getRequestId()), 30_000L, prefill.placementVersion()));
            });
            when(decodeSelector.select(any(), any())).thenAnswer(call -> {
                var request = call.getArgument(0, ScheduledRequest.DecodeBinding.class);
                var pin = decode.tryPinGeneration();
                assertNotNull(pin);
                return PlacementResult.success(SelectedRole.decode(pin,
                        metadata(decode, request.requestId()), decode.placementVersion()));
            });
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
            var router = new DefaultRouter(prefillSelector, decodeSelector, mock(RandomStrategy.class), service, model);
            scheduler = new RequestScheduler(service, router, endpoints, reporter, mock(EvictionManager.class),
                    requests, placement);
            runtime = new SchedulerRuntime(requests, endpoints, reporter, requestReporter, scheduler);
        }

        private BalanceContext context(long requestId) {
            var context = RequestLifecycleTestSupport.context(config, requestId);
            context.getRequest().setSeqLen(32L);
            context.getRequest().setMaxNewTokens(16);
            return context;
        }

        private void assertNoWaitingQueue() {
            assertEquals(0, scheduler.getQueuedRequestCount());
            assertEquals(0, prefill.queuedRequestCount());
            verify(queuedDelivery, never()).prepare(anyList(), any(), any());
        }

        private void assertNoPrefillOwnership() {
            assertNoWaitingQueue();
            assertEquals(0L, prefill.observedRequestCount());
            assertEquals(0, prefill.getInflightBatchCount());
        }

        private void assertItemNotBound(long requestId) {
            RequestSlot slot = requests.requestSlot(requestId);
            assertNotNull(slot);
            synchronized (slot) {
                assertNull(slot.activeItem(), "this Engine observation must precede item binding");
            }
        }

        private void observe(WorkerEndpoint endpoint, Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
            WorkerStatus worker = endpoint.getStatus();
            Runnable projection;
            worker.lock.lock();
            try {
                var response = status(worker, running, finished);
                projection = endpoint.applyPreparedStatus(worker,
                        worker.prepareNewStatus(worker.freezeStatusResponse(response)));
            } finally {
                worker.lock.unlock();
            }
            projection.run();
        }

        @Override
        public void close() {
            try {
                decode.close();
            } finally {
                runtime.shutdown();
            }
        }

        private static WorkerStatus worker(RoleType role, String ip) {
            return WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
        }

        private static WorkerStatusResponse status(WorkerStatus worker,
                                                  Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
            var response = new WorkerStatusResponse();
            response.setRole(worker.getRole());
            response.setAlive(true);
            response.setStatusVersion(Math.max(1L, worker.appliedStatusCursor().statusVersion() + 1L));
            response.setLatestFinishedVersion(Math.max(0L, worker.appliedStatusCursor().latestFinishedTaskVersion())
                    + (finished.isEmpty() ? 0L : 1L));
            response.setRunningTaskInfo(running);
            response.setRunningQueryLen((long) running.size());
            response.setFinishedTaskInfo(finished);
            if (worker.getRole() == RoleType.DECODE) {
                response.setTotalKvCacheTokens(10_000L);
                response.setAvailableKvCacheTokens(10_000L - running.size() * 32L);
            }
            return response;
        }

        private static ServerStatus metadata(WorkerEndpoint endpoint, long requestId) {
            var result = new ServerStatus();
            result.setSuccess(true);
            result.setRequestId(requestId);
            result.setRole(endpoint.getStatus().getRole());
            result.setGroup("g1");
            result.setServerIp(endpoint.getIp());
            result.setHttpPort(endpoint.getHttpPort());
            result.setGrpcPort(8081);
            if (endpoint instanceof PrefillEndpoint) {
                DebugInfo debug = new DebugInfo();
                debug.setHitCacheLen(8L);
                result.setDebugInfo(debug);
                result.setPrefillTime(30_000L);
            }
            return result;
        }
    }
}
