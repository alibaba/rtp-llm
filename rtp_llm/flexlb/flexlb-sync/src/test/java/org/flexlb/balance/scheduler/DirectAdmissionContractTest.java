package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.balance.strategy.WorkerAssignment;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
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
            assertEquals(prefillCapacity, fixture.prefill.admissionSummary(0).occupiedRequests());
            assertEquals(prefillCapacity, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(48L * prefillCapacity, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            fixture.assertNoWaitingQueue();
            verify(fixture.requests, times(2)).commitRoute(any(), any());
            verify(fixture.prefillSelector, times(2)).select(any(), any(), eq(RoleType.PREFILL), any());
        }
    }

    @Test
    void directRouteOwnsPrefillAndDecodeUntilEngineTerminalWithoutEnteringWaitingQueues() throws Exception {
        try (Fixture fixture = new Fixture()) {
            Response response = fixture.scheduler.submit(fixture.context(101L)).get(2L, TimeUnit.SECONDS);

            assertTrue(response.isSuccess());
            assertFalse(response.isEnqueuedByMaster());
            fixture.assertNoWaitingQueue();
            assertEquals(1, fixture.prefill.admissionSummary(0).occupiedRequests());
            assertEquals(0, fixture.prefill.ownershipStats().batchCount());
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(1, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            assertEquals(RequestState.Phase.ACKNOWLEDGED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests).getRequestState(101L, 0L).state());
            var reservation = EndpointTestSupport.decodeReservation(fixture.decode, 101L);
            assertNotNull(reservation);
            assertThrows(IllegalStateException.class, () -> fixture.decode.release(
                    reservation,
                    DecodeResources.ReleaseReason.LOCAL_ROLLBACK));

            fixture.applyWorkerStatus(fixture.prefill, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(0L, fixture.prefill.admissionSummary(0).occupiedRequests());
            assertEquals(48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()),
                    "Prefill completion must retain Decode ownership until its own observation");
            fixture.applyWorkerStatus(fixture.decode, Map.of("101", task(101L, TaskPhase.RUNNING)), Map.of());
            assertTrue(fixture.decode.isAcceptedByEngine(reservation));
            fixture.applyWorkerStatus(fixture.decode, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            fixture.assertNoWaitingQueue();
        }
    }

    @Test
    void fullDecodeRejectsImmediatelyAndReleasesUncommittedResources() throws Exception {
        try (Fixture fixture = new Fixture()) {
            DecodeResources.ReservationHandle occupant;
            try (var pin = fixture.decode.tryPinGeneration()) {
                assertNotNull(pin);
                occupant = fixture.decode.tryReserveQueuedRequest(pin, 999L, 32L, 48L, 50, null);
            }
            assertNotNull(occupant);
            var acquired = fixture.decode.acquireDispatchPermit(occupant, new DecodeResources.AdmissionCapacity(1L, 90L));
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
            assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquired.permit().dispatch());

            Response response = assertTimeoutPreemptively(Duration.ofSeconds(2),
                    () -> fixture.scheduler.submit(fixture.context(102L)).get(2L, TimeUnit.SECONDS));

            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 102L));
            assertEquals(occupant, EndpointTestSupport.decodeReservation(fixture.decode, 999L));
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void decodeTerminalBetweenAcceptedPreparationAndContextBindingCannotPublishASuccessfulRoute() throws Exception {
        try (Fixture fixture = new Fixture()) {
            AtomicBoolean raced = new AtomicBoolean();
            doAnswer(call -> {
                DecodeResources.ReservationHandle reservation = call.getArgument(0);
                assertEquals(103L, reservation.requestId());
                fixture.assertItemNotBound(103L);
                fixture.applyWorkerStatus(fixture.decode, Map.of("103", task(103L, TaskPhase.KV_ALLOCATED)), Map.of());
                var acquired = (DecodeEndpoint.EngineDispatchPermitAcquisition) call.callRealMethod();
                assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED, acquired.status());
                assertNotNull(acquired.permit(), "accepted preparation must retain an exact handoff capability");
                fixture.applyWorkerStatus(fixture.decode, Map.of(), Map.of("103", task(103L, TaskPhase.RUNNING)));
                fixture.assertItemNotBound(103L);
                raced.set(true);
                return acquired;
            }).when(fixture.decode).acquireDispatchPermit(any(), any());

            Response response = fixture.scheduler.submit(fixture.context(103L)).get(2L, TimeUnit.SECONDS);

            assertTrue(raced.get());
            assertFalse(response.isSuccess(), "an already ended request must not receive a new successful route");
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 103L));
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(0L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void decodeTerminalAfterRouteClaimSettlesResourcesBeforeLateAddressPublication() throws Exception {
        try (Fixture fixture = new Fixture()) {
            AtomicBoolean raced = new AtomicBoolean();
            AtomicReference<RequestContext.DeliveryClaim> claimed = new AtomicReference<>();
            doAnswer(call -> {
                RequestContext.DeliveryClaim claim = call.getArgument(0);
                claimed.set(claim);
                RequestContext context = claim.item.ctx();
                assertEquals(105L, context.getRequestId());
                assertEquals(DeliveryClaimKind.ROUTE_DECISION, context.deliveryClaimKind());
                assertEquals(RequestContext.DeliveryClaim.SendOutcome.NOT_STARTED,
                        org.springframework.test.util.ReflectionTestUtils.getField(claim, "sendOutcome"));
                assertFalse(claim.settlement().toCompletableFuture().isDone());

                fixture.applyWorkerStatus(fixture.decode, Map.of(), Map.of("105", task(105L, TaskPhase.RUNNING)));

                Response terminal = context.getFuture().get(2L, TimeUnit.SECONDS);
                assertTrue(terminal.isSuccess());
                assertEquals(RequestContext.RequestStage.FINISHED, context.stage());
                assertEquals(RequestState.Phase.COMPLETED, context.snapshot().state());
                assertTrue(claim.settlement().toCompletableFuture().isDone());
                assertEquals(RequestContext.DeliveryClaim.SendOutcome.NOT_STARTED,
                        org.springframework.test.util.ReflectionTestUtils.getField(claim, "sendOutcome"),
                        "a worker terminal must not fabricate address publication");
                fixture.assertNoPrefillOwnership();
                assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 105L));
                assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
                assertEquals(0L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
                assertEquals(0, fixture.requests.requests.liveRequestCount());

                call.callRealMethod();

                assertSame(terminal, context.getFuture().join());
                assertEquals(RequestContext.RequestStage.FINISHED, context.stage());
                assertEquals(RequestContext.DeliveryClaim.SendOutcome.NOT_STARTED,
                        org.springframework.test.util.ReflectionTestUtils.getField(claim, "sendOutcome"));
                raced.set(true);
                return null;
            }).when(fixture.requests).publishRoute(any(), any(), org.mockito.ArgumentMatchers.anyLong());

            try {
                Response response = fixture.scheduler.submit(fixture.context(105L)).get(2L, TimeUnit.SECONDS);

                assertTrue(raced.get());
                assertTrue(response.isSuccess());
                fixture.assertNoPrefillOwnership();
                assertEquals(0, fixture.requests.requests.liveRequestCount());
            } finally {
                // A broken sender gate must fail this test without hanging runtime shutdown.
                RequestContext.DeliveryClaim claim = claimed.get();
                if (claim != null && !claim.settlement().toCompletableFuture().isDone()) {
                    claim.item.ctx().scheduler().abandonDelivery(claim, CancelReason.CLIENT_CANCELLED);
                    fixture.requests.resumeCleanup(claim.item.ctx());
                }
            }
        }
    }

    @Test
    void directCommitKeepsPrimaryFailureWhenPermitCleanupAlsoFails() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var prefill = spy(fixture.prefill);
            var context = SchedulingTestConfig.freezeInputs(fixture.context(104L));
            var primary = new IllegalStateException("Prefill commit unavailable");
            var cleanup = new IllegalStateException("Decode cleanup observer failed");
            org.mockito.Mockito.doThrow(primary).when(prefill).tryBeginRouteCommitAdmission();
            doAnswer(call -> {
                call.callRealMethod();
                throw cleanup;
            }).when(fixture.decode).dispatch(any(), eq(DecodeResources.DispatchOutcome.ABANDONED));
            try (var admission = RequestRoute.prepare(context, List.of(
                    WorkerAssignment.prefill(prefill.tryPinGeneration(), Fixture.metadata(prefill, 104L),
                            30_000L, prefill.placementVersion()),
                    WorkerAssignment.decode(fixture.decode.tryPinGeneration(), Fixture.metadata(fixture.decode, 104L),
                            fixture.decode.placementVersion())))) {
                var thrown = assertThrows(IllegalStateException.class,
                        () -> fixture.scheduler.commitDirectRoute(context, admission));
                org.junit.jupiter.api.Assertions.assertSame(primary, thrown);
                assertEquals(List.of(cleanup), List.of(thrown.getSuppressed()));
                verify(fixture.decode).dispatch(any(), eq(DecodeResources.DispatchOutcome.ABANDONED));
                assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            }
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 104L));
            fixture.assertNoPrefillOwnership();
        }
    }

    @Test
    void cancellationAfterPrefillCommitRejectsClaimAndSettlesBothLedgers() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(108L);
            var prefill = spy(fixture.prefill);
            doAnswer(call -> {
                var registration = (PrefillEndpoint.RouteCommitAdmission) call.callRealMethod();
                var commit = spy(registration);
                doAnswer(committing -> {
                    var handoff = committing.callRealMethod();
                    assertEquals(1, fixture.decode.resourceSnapshot().activeDispatchPermits());
                    fixture.scheduler.cancel(108L, 0L, CancelReason.CLIENT_CANCELLED);
                    return handoff;
                }).when(commit).commit(anyList(), anyList());
                return commit;
            }).when(prefill).tryBeginRouteCommitAdmission();
            when(fixture.prefillSelector.select(eq(RequestRequirements.capture(context)), eq(context.getConfig()), eq(RoleType.PREFILL), any())).thenAnswer(call ->
                    PlacementResult.success(WorkerAssignment.prefill(prefill.tryPinGeneration(),
                            Fixture.metadata(prefill, 108L), 30_000L, prefill.placementVersion())));
            assertFalse(fixture.scheduler.submit(context).get(2, TimeUnit.SECONDS).isSuccess());
            fixture.runtime.continuations().awaitIdle();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 108L));
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            fixture.assertNoPrefillOwnership();
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false", "true,false", "false,true", "true,true"})
    void queuePublicationRetainsCapacityAcrossWakeFailureAndRepeatedConsumption(
            boolean separateDecode, boolean wakeFails) throws Exception {
        try (Fixture fixture = new Fixture()) {
            var config = SchedulingTestConfig.newConfig();
            SchedulingTestConfig.useFifoQueue(config);
            SchedulingTestConfig.useNonBatchDispatcher(config);
            var prefill = spy(fixture.prefill);
            var wakeFailure = new IllegalStateException("wake failed after queue publication");
            if (wakeFails) { org.mockito.Mockito.doThrow(wakeFailure).when(prefill).signalRouteReady(); }
            try (var queue = new QueuedRequestScheduler(config, fixture.router, mock(DeliveryMetricsReporter.class),
                    mock(DecodeCapacityAcquirer.class), fixture.runtime, new PlacementAvailability())) {
                var context = RequestProtocolTestSupport.context(config, 109L);
                var future = queue.register(context, StrategyErrorType.BATCH_SLO_EXPIRED);
                try (var handle = queue.claimAdmissionHandle(109L, future);
                     var finish = RequestProtocolTestSupport.finishOnExit(handle)) {
                    assertNotNull(handle);
                    var selections = new java.util.ArrayList<WorkerAssignment>();
                    selections.add(WorkerAssignment.prefill(prefill.tryPinGeneration(), Fixture.metadata(prefill, 109L),
                            30_000L, prefill.placementVersion()));
                    if (separateDecode) {
                        selections.add(WorkerAssignment.decode(fixture.decode.tryPinGeneration(),
                                Fixture.metadata(fixture.decode, 109L), fixture.decode.placementVersion()));
                    }
                    var plan = RequestProtocolTestSupport.plan(context, RequestRoute.prepare(context, selections));
                    try (plan) {
                        if (wakeFails) {
                            org.junit.jupiter.api.Assertions.assertSame(wakeFailure,
                                    assertThrows(IllegalStateException.class, () -> queue.enqueueRoute(plan)));
                        } else {
                            assertEquals(PlacementResult.Status.SUCCESS, queue.enqueueRoute(plan).status());
                        }
                        var exact = context.activeRoute();
                        assertNotNull(exact);
                        assertEquals(RequestContext.RequestStage.READY_TO_DELIVER, context.stage());
                        assertEquals(separateDecode ? PlacementResult.Status.BLOCKED : PlacementResult.Status.CLOSED,
                                queue.enqueueRoute(plan).status());
                        org.junit.jupiter.api.Assertions.assertSame(exact, context.activeRoute());
                        verify(prefill).offerPinned(any(), eq(exact), any());
                        verify(fixture.decode, never()).release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
                        assertEquals(1L, prefill.queuedRequestCount());
                    }
                    assertEquals(separateDecode ? 1 : 0, fixture.decode.resourceSnapshot().queuedCount());
                }
            }
        }
    }

    @Test
    void queueActivationPreservesExistingDirectReservations() throws Exception {
        try (Fixture fixture = new Fixture(4, 2)) {
            assertTrue(fixture.scheduler.submit(fixture.context(101L)).get(2, TimeUnit.SECONDS).isSuccess());

            var config = SchedulingTestConfig.newConfig();
            SchedulingTestConfig.useFifoQueue(config);
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getDispatcher().setMaxInflightPerPrefillWorker(4);
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
            try (var queue = (QueuedRequestScheduler) PlacementConfiguration.create(fixture.requests.runtime, config,
                    fixture.router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class),
                    new PlacementAvailability())) {
                var request = RequestProtocolTestSupport.context(config, 102L);
                request.getRequest().setSeqLen(32L);
                request.getRequest().setMaxNewTokens(16);
                assertTrue(queue.submit(request).get(3, TimeUnit.SECONDS).isSuccess());
                assertEquals(2, fixture.prefill.admissionSummary(0).occupiedRequests());
                assertEquals(2, fixture.decode.routingView().engineCapacityUsed());
                verify(fixture.queuedDelivery).prepare(anyList(), any(), any());

                fixture.applyWorkerStatus(fixture.prefill, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
                fixture.applyWorkerStatus(fixture.decode, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
                RequestProtocolTestSupport.awaitCondition(() -> fixture.requests.requests.findActive(101L) == null);
                assertEquals(1, fixture.prefill.admissionSummary(0).occupiedRequests());
                assertNotNull(EndpointTestSupport.decodeReservation(fixture.decode, 102L));

            }
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
        private final AbstractRequestScheduler requests;
        private final EndpointRegistry endpoints;
        private final RouteDeliveryStrategy queuedDelivery;
        private final PrefillEndpoint prefill;
        private final DecodeEndpoint decode;
        private final CostBasedPrefillStrategy prefillSelector;
        private final DirectRequestScheduler scheduler;
        private final RequestWorkerSelector router;
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
            var reporter = mock(DeliveryMetricsReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);
            requests = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, reporter, requestReporter,
                mock(RecentCacheKeyTraceReporter.class));
            var projector = requests;
            var placement = new PlacementAvailability();
            queuedDelivery = spy(new RouteDeliveryStrategy(reporter));
            endpoints = new EndpointRegistry(service, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(projector), reporter, queuedDelivery, placement);
            WorkerStatus worker = worker(RoleType.PREFILL, "127.0.0.1");
            worker.lock.lock();
            try {
                var response = status(worker, Map.of(), Map.of());
                prefill = (PrefillEndpoint) endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(worker.freezeStatusResponse(response)));
            } finally {
                worker.lock.unlock();
            }
            decode = spy(EndpointTestSupport.decode(worker(RoleType.DECODE, "127.0.0.2"), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(projector)));
            applyWorkerStatus(decode, Map.of(), Map.of());
            prefillSelector = mock(CostBasedPrefillStrategy.class);
            var decodeSelector = mock(DecodeSelector.class);
            when(prefillSelector.select(any(), any(), eq(RoleType.PREFILL), any())).thenAnswer(call -> {
                var request = call.getArgument(0, RequestRequirements.class);
                var pin = prefill.tryPinGeneration();
                assertNotNull(pin);
                return PlacementResult.success(WorkerAssignment.prefill(pin,
                        metadata(prefill, request.requestId()), 30_000L, prefill.placementVersion()));
            });
            when(decodeSelector.select(any(), any())).thenAnswer(call -> {
                var request = call.getArgument(0, RequestRequirements.class);
                var pin = decode.tryPinGeneration();
                assertNotNull(pin);
                return PlacementResult.success(WorkerAssignment.decode(pin,
                        metadata(decode, request.requestId()), decode.placementVersion()));
            });
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
            router = new RequestWorkerSelector(prefillSelector, decodeSelector, mock(VitWorkerSelector.class), model);
            scheduler = (DirectRequestScheduler) org.flexlb.balance.scheduler.SchedulerTestSupport.configure(requests, service.loadBalanceConfig(), router, reporter, mock(DecodeCapacityAcquirer.class), placement);
            runtime = requests.runtime;
            ReflectionTestUtils.setField(runtime, "endpoints", endpoints);
        }

        private RequestContext context(long requestId) {
            var context = RequestProtocolTestSupport.context(config, requestId);
            context.getRequest().setSeqLen(32L);
            context.getRequest().setMaxNewTokens(16);
            return context;
        }

        private void assertNoWaitingQueue() {
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests).pendingDeliveryRequestCount());
            assertEquals(0, prefill.queuedRequestCount());
            verify(queuedDelivery, never()).prepare(anyList(), any(), any());
        }

        private void assertNoPrefillOwnership() {
            assertNoWaitingQueue();
            assertEquals(0L, prefill.admissionSummary(0).occupiedRequests());
            assertEquals(0, prefill.ownershipStats().batchCount());
        }

        private void assertItemNotBound(long requestId) {
            RequestContext requestContext = requests.findRequestContext(requestId);
            assertNotNull(requestContext);
            synchronized (requestContext) {
                assertNull(requestContext.activeRoute(), "this Engine observation must precede item binding");
            }
        }

        private void applyWorkerStatus(WorkerEndpoint endpoint, Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
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
            requests.runtime.continuations().awaitIdle();
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
