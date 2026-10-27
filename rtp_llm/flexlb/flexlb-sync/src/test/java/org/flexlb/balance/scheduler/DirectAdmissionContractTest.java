package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeAdmission;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

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
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * DIRECT uses real admission ledgers and lifecycle while bypassing waiting queues.
 */
class DirectAdmissionContractTest {

    @ParameterizedTest
    @ValueSource(ints = {1, 2})
    void concurrentDirectAdmissionsUseCurrentCapacityWithoutRetryingSelection(int prefillCapacity)
            throws Exception {
        try (Fixture fixture = new Fixture(prefillCapacity, 2);
                var executor = Executors.newFixedThreadPool(2)) {
            var readyToPublish = new CyclicBarrier(2);
            doAnswer(
                            call -> {
                                // Both requests have selected this generation and acquired Decode
                                // capacity.
                                // Their Prefill publications must arbitrate against current
                                // ownership.
                                readyToPublish.await(2L, TimeUnit.SECONDS);
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .commitRoute(any(), any());
            var first =
                    executor.submit(
                            () ->
                                    fixture.scheduler
                                            .submit(fixture.context("201"))
                                            .get(2L, TimeUnit.SECONDS));
            var second =
                    executor.submit(
                            () ->
                                    fixture.scheduler
                                            .submit(fixture.context("202"))
                                            .get(2L, TimeUnit.SECONDS));
            var responses =
                    List.of(first.get(3L, TimeUnit.SECONDS), second.get(3L, TimeUnit.SECONDS));

            assertEquals(prefillCapacity, responses.stream().filter(Response::isSuccess).count());
            for (var response : responses) {
                if (!response.isSuccess()) {
                    assertEquals(
                            StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                            response.getCode());
                    assertTrue(
                            response.getErrorMessage()
                                    .endsWith("trigger=Prefill request capacity exhausted"));
                }
            }
            assertEquals(
                    prefillCapacity,
                    fixture.prefill
                            .ownership
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(prefillCapacity, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    48L * prefillCapacity,
                    EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            fixture.assertNoWaitingQueue();
            verify(fixture.requests, times(2)).commitRoute(any(), any());
            verify(fixture.prefillSelector, times(2))
                    .select(
                            any(),
                            any(),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.any(RequestRoute.class));
        }
    }

    @Test
    void directRouteOwnsPrefillAndDecodeUntilEngineTerminalWithoutEnteringWaitingQueues()
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            Response response =
                    fixture.scheduler.submit(fixture.context("101")).get(2L, TimeUnit.SECONDS);

            assertTrue(response.isSuccess());
            assertFalse(response.isEnqueuedByMaster());
            fixture.assertNoWaitingQueue();
            assertEquals(
                    1,
                    fixture.prefill
                            .ownership
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(0, fixture.prefill.ownership.stats().batchCount());
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(1, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            assertEquals(
                    RequestState.Phase.ACKNOWLEDGED,
                    SchedulerTestSupport.repository(fixture.requests)
                            .getRequestState("101", 0L)
                            .state());
            var reservation =
                    EndpointTestSupport.decodeReservation(fixture.decode, String.valueOf(101L));
            assertNotNull(reservation);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            fixture.decode
                                    .release(
                                            reservation,
                                            DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                                    .status());

            fixture.applyWorkerStatus(
                    fixture.prefill, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(
                    0L,
                    fixture.prefill
                            .ownership
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(
                    48L,
                    EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()),
                    "Prefill completion must retain Decode ownership until its own observation");
            fixture.applyWorkerStatus(
                    fixture.decode, Map.of("101", task(101L, TaskPhase.RUNNING)), Map.of());
            assertTrue(fixture.decode.ownership(reservation).confirmed());
            fixture.applyWorkerStatus(
                    fixture.decode, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            fixture.assertNoWaitingQueue();
        }
    }

    @Test
    void fullDecodeRejectsImmediatelyAndReleasesUncommittedResources() throws Exception {
        try (Fixture fixture = new Fixture()) {
            DecodeResources.ReservationHandle occupant;
            try (var pin = fixture.decode.tryPinGeneration()) {
                assertNotNull(pin);
                occupant =
                        fixture.decode
                                .tryReserveQueuedRequest(pin, "999", 32L, 48L, 50, null)
                                .handle();
            }
            assertNotNull(occupant);
            var acquired =
                    fixture.decode.acquireDispatchPermit(
                            occupant, new DecodeResources.AdmissionCapacity(1L, 90L));
            assertEquals(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
            assertEquals(
                    DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquired.handle().dispatch().status());

            Response response =
                    assertTimeoutPreemptively(
                            Duration.ofSeconds(2),
                            () ->
                                    fixture.scheduler
                                            .submit(fixture.context("102"))
                                            .get(2L, TimeUnit.SECONDS));

            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, String.valueOf(102L)));
            assertEquals(
                    occupant,
                    EndpointTestSupport.decodeReservation(fixture.decode, String.valueOf(999L)));
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void decodeTerminalBetweenAcceptedPreparationAndContextBindingCannotPublishASuccessfulRoute()
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            AtomicBoolean raced = new AtomicBoolean();
            doAnswer(
                            call -> {
                                DecodeResources.ReservationHandle reservation = call.getArgument(0);
                                assertEquals("103", reservation.requestId());
                                fixture.assertItemNotBound("103");
                                fixture.applyWorkerStatus(
                                        fixture.decode,
                                        Map.of("103", task(103L, TaskPhase.KV_ALLOCATED)),
                                        Map.of());
                                var acquired =
                                        (org.flexlb.balance.endpoint.DecodeResources.Acquisition<
                                                        org.flexlb.balance.endpoint.DecodeResources
                                                                .EngineDispatchPermitAcquireStatus,
                                                        DecodeEndpoint.EngineDispatchPermit>)
                                                call.callRealMethod();
                                assertEquals(
                                        DecodeResources.EngineDispatchPermitAcquireStatus
                                                .ALREADY_ACCEPTED,
                                        acquired.status());
                                assertNotNull(
                                        acquired.handle(),
                                        "accepted preparation must retain an exact handoff capability");
                                fixture.applyWorkerStatus(
                                        fixture.decode,
                                        Map.of(),
                                        Map.of("103", task(103L, TaskPhase.RUNNING)));
                                fixture.assertItemNotBound("103");
                                raced.set(true);
                                return acquired;
                            })
                    .when(fixture.decode)
                    .acquireDispatchPermit(any(), any());

            Response response =
                    fixture.scheduler.submit(fixture.context("103")).get(2L, TimeUnit.SECONDS);

            assertTrue(raced.get());
            assertFalse(
                    response.isSuccess(),
                    "an already ended request must not receive a new successful route");
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, String.valueOf(103L)));
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    0L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void decodeTerminalAfterRouteClaimSettlesResourcesBeforeLateAddressPublication()
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            AtomicBoolean raced = new AtomicBoolean();
            AtomicReference<DeliveryClaim> claimed = new AtomicReference<>();
            doAnswer(
                            call -> {
                                DeliveryClaim claim = call.getArgument(0);
                                claimed.set(claim);
                                RequestContext context = claim.item().ctx();
                                assertEquals("105", context.data().getRequestId());
                                assertEquals(
                                        DeliveryClaimKind.ROUTE_DECISION,
                                        context.snapshot().delivery().identity().kind());
                                synchronized (context) {
                                    assertFalse(
                                            context.snapshot().delivery().send()
                                                    == ContextResult.SendState.DELIVERED);
                                }
                                assertFalse(
                                        fixture.requests
                                                .runtime
                                                .signals()
                                                .delivery(claim)
                                                .toCompletableFuture()
                                                .isDone());

                                fixture.applyWorkerStatus(
                                        fixture.decode,
                                        Map.of(),
                                        Map.of("105", task(105L, TaskPhase.RUNNING)));

                                Response terminal =
                                        context.data().getFuture().get(2L, TimeUnit.SECONDS);
                                assertTrue(terminal.isSuccess());
                                assertEquals(
                                        ContextResult.Phase.CLOSED, context.snapshot().phase());
                                assertEquals(
                                        RequestState.Phase.COMPLETED,
                                        context.snapshot().terminalRecord().state());
                                assertTrue(
                                        fixture.requests
                                                .runtime
                                                .signals()
                                                .delivery(claim)
                                                .toCompletableFuture()
                                                .isDone());
                                synchronized (context) {
                                    assertFalse(
                                            context.snapshot().delivery().send()
                                                    == ContextResult.SendState.DELIVERED,
                                            "a worker terminal must not fabricate address publication");
                                }
                                fixture.assertNoPrefillOwnership();
                                assertNull(
                                        EndpointTestSupport.decodeReservation(
                                                fixture.decode, String.valueOf(105L)));
                                assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
                                assertEquals(
                                        0L,
                                        EndpointTestSupport.expectedReservedKv(
                                                fixture.decode.resourceSnapshot()));
                                assertEquals(0, fixture.requests.requests.liveRequestCount());

                                call.callRealMethod();

                                assertSame(terminal, context.data().getFuture().join());
                                assertEquals(
                                        ContextResult.Phase.CLOSED, context.snapshot().phase());
                                synchronized (context) {
                                    assertFalse(
                                            context.snapshot().delivery().send()
                                                    == ContextResult.SendState.DELIVERED);
                                }
                                raced.set(true);
                                return null;
                            })
                    .when(fixture.requests)
                    .onDeliveryEvidence(
                            any(),
                            org.mockito.ArgumentMatchers.isA(ContextResult.RouteDelivered.class));

            try {
                Response response =
                        fixture.scheduler.submit(fixture.context("105")).get(2L, TimeUnit.SECONDS);

                assertTrue(raced.get());
                assertTrue(response.isSuccess());
                fixture.assertNoPrefillOwnership();
                assertEquals(0, fixture.requests.requests.liveRequestCount());
            } finally {
                // A broken sender gate must fail this test without hanging runtime shutdown.
                DeliveryClaim claim = claimed.get();
                if (claim != null
                        && !fixture.requests
                                .runtime
                                .signals()
                                .delivery(claim)
                                .toCompletableFuture()
                                .isDone()) {
                    claim.item()
                            .ctx()
                            .data()
                            .getScheduler()
                            .onDeliveryEvidence(
                                    claim,
                                    new ContextResult.Abandonment(
                                            claim,
                                            CancelReason.CLIENT_CANCELLED,
                                            System.currentTimeMillis()));
                    fixture.requests.requestCleanup(claim.item().ctx());
                }
            }
        }
    }

    @Test
    void directCommitKeepsPrimaryFailureWhenPermitCleanupAlsoFails() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var prefill = spy(fixture.prefill);
            var context = fixture.context("104");
            var future =
                    fixture.scheduler.register(
                            context, () -> {}, StrategyErrorType.BATCH_SLO_EXPIRED);
            var primary = new IllegalStateException("Prefill commit unavailable");
            var cleanup = new IllegalStateException("Decode cleanup observer failed");
            org.mockito.Mockito.doThrow(primary)
                    .when(prefill)
                    .commit(any(org.flexlb.balance.endpoint.PrefillState.CommitPlan.class));
            doAnswer(
                            call -> {
                                call.callRealMethod();
                                throw cleanup;
                            })
                    .when(fixture.decode)
                    .dispatch(any(), eq(DecodeResources.DispatchOutcome.ABANDONED));
            try (var mutation = fixture.scheduler.claimAdmissionHandle("104", future);
                    var completion = RequestProtocolTestSupport.finishOnExit(mutation);
                    var admission =
                            SchedulingTestConfig.prepareRoute(
                                            context,
                                            route -> {
                                                route.add(
                                                        prefill.tryPinGeneration(),
                                                        Fixture.metadata(prefill, "104"),
                                                        30_000L,
                                                        prefill.ownership
                                                                .progress()
                                                                .ownershipVersion());
                                                route.add(
                                                        fixture.decode.tryPinGeneration(),
                                                        Fixture.metadata(fixture.decode, "104"),
                                                        -1L,
                                                        fixture.decode.placementVersion());
                                                return PlacementResult.success(route);
                                            })
                                    .value()) {
                var thrown =
                        assertThrows(
                                IllegalStateException.class,
                                () -> fixture.scheduler.commitDirectRoute(context, admission));
                org.junit.jupiter.api.Assertions.assertSame(primary, thrown);
                assertEquals(List.of(cleanup), List.of(thrown.getSuppressed()));
                verify(fixture.decode)
                        .dispatch(any(), eq(DecodeResources.DispatchOutcome.ABANDONED));
                assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            } finally {
                fixture.scheduler.cancel("104", 0L, CancelReason.CLIENT_CANCELLED);
            }
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, String.valueOf(104L)));
            fixture.assertNoPrefillOwnership();
        }
    }

    @Test
    void cancellationAfterPrefillCommitRejectsClaimAndSettlesBothLedgers() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context("108");
            var prefill = spy(fixture.prefill);
            doAnswer(
                            call -> {
                                var handoff = call.callRealMethod();
                                assertEquals(
                                        1,
                                        fixture.decode.resourceSnapshot().activeDispatchPermits());
                                fixture.scheduler.cancel("108", 0L, CancelReason.CLIENT_CANCELLED);
                                return handoff;
                            })
                    .when(prefill)
                    .commit(any(org.flexlb.balance.endpoint.PrefillState.CommitPlan.class));
            when(fixture.prefillSelector.select(
                            eq(context.data().getRequest()),
                            eq(context.data().getSchedulingMetadata()),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.any(RequestRoute.class)))
                    .thenAnswer(
                            call ->
                                    PlacementResult.success(
                                            call.getArgument(4, RequestRoute.class)
                                                    .add(
                                                            prefill.tryPinGeneration(),
                                                            Fixture.metadata(prefill, "108"),
                                                            30_000L,
                                                            prefill.ownership
                                                                    .progress()
                                                                    .ownershipVersion())));
            assertFalse(fixture.scheduler.submit(context).get(2, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, String.valueOf(108L)));
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
            if (wakeFails) {
                org.mockito.Mockito.doThrow(wakeFailure)
                        .doCallRealMethod()
                        .when(prefill)
                        .signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.INPUT);
            }
            try (var queue =
                    new QueuedRequestScheduler(
                            config,
                            fixture.router,
                            mock(DeliveryMetricsReporter.class),
                            mock(DecodeAdmission.class),
                            fixture.runtime,
                            new PlacementAvailability(),
                            SchedulingTestConfig.capacity(config),
                            org.mockito.Mockito.mock(
                                    org.flexlb.service.TheoryCacheHitReporter.class))) {
                var context = RequestProtocolTestSupport.context(config, "109");
                var future = queue.register(context, () -> {}, StrategyErrorType.BATCH_SLO_EXPIRED);
                try (var handle = queue.claimAdmissionHandle("109", future);
                        var finish = RequestProtocolTestSupport.finishOnExit(handle)) {
                    assertNotNull(handle);
                    var selected =
                            SchedulingTestConfig.prepareRoute(
                                            context,
                                            route -> {
                                                route.add(
                                                        prefill.tryPinGeneration(),
                                                        Fixture.metadata(prefill, "109"),
                                                        30_000L,
                                                        prefill.ownership
                                                                .progress()
                                                                .ownershipVersion());
                                                if (separateDecode) {
                                                    route.add(
                                                            fixture.decode.tryPinGeneration(),
                                                            Fixture.metadata(fixture.decode, "109"),
                                                            -1L,
                                                            fixture.decode.placementVersion());
                                                }
                                                return PlacementResult.success(route);
                                            })
                                    .value();
                    var plan = RequestProtocolTestSupport.plan(context, selected);
                    try (plan) {
                        if (wakeFails) {
                            org.junit.jupiter.api.Assertions.assertSame(
                                    wakeFailure,
                                    assertThrows(
                                            IllegalStateException.class,
                                            () -> queue.enqueueRoute(plan)));
                        } else {
                            assertEquals(
                                    PlacementResult.Status.SUCCESS,
                                    queue.enqueueRoute(plan).status());
                        }
                        var exact = context.snapshot().route();
                        assertNotNull(exact);
                        assertEquals(ContextResult.Phase.OPEN, context.snapshot().phase());
                        assertNotNull(context.snapshot().route());
                        assertNull(context.snapshot().delivery());
                        assertEquals(
                                separateDecode
                                        ? PlacementResult.Status.BLOCKED
                                        : PlacementResult.Status.CLOSED,
                                queue.enqueueRoute(plan).status());
                        org.junit.jupiter.api.Assertions.assertSame(
                                exact, context.snapshot().route());
                        verify(prefill).offerPinned(any(), eq(exact), any());
                        verify(fixture.decode, never())
                                .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
                        assertEquals(1L, prefill.ownership.progress().queueDepth());
                    }
                    assertEquals(
                            separateDecode ? 1 : 0,
                            fixture.decode.resourceSnapshot().queuedCount());
                }
            }
        }
    }

    @Test
    void queueActivationPreservesExistingDirectReservations() throws Exception {
        try (Fixture fixture = new Fixture(4, 2)) {
            assertTrue(
                    fixture.scheduler
                            .submit(fixture.context("101"))
                            .get(2, TimeUnit.SECONDS)
                            .isSuccess());

            var config = SchedulingTestConfig.newConfig();
            SchedulingTestConfig.useFifoQueue(config);
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getDispatcher().setMaxInflightPerPrefillWorker(4);
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
            try (var queue =
                    (QueuedRequestScheduler)
                            PlacementConfiguration.create(
                                    fixture.requests.runtime,
                                    config,
                                    fixture.router,
                                    mock(DeliveryMetricsReporter.class),
                                    mock(DecodeAdmission.class),
                                    new PlacementAvailability(),
                                    SchedulingTestConfig.capacity(config),
                                    org.mockito.Mockito.mock(
                                            org.flexlb.service.TheoryCacheHitReporter.class))) {
                var request = RequestProtocolTestSupport.context(config, "102");
                request.data()
                        .setRequest(request.data().getRequest().toBuilder().seqLen(32L).build());
                request.data()
                        .setRequest(
                                request.data().getRequest().toBuilder().maxNewTokens(16).build());
                assertTrue(queue.submit(request).get(3, TimeUnit.SECONDS).isSuccess());
                assertEquals(
                        2,
                        fixture.prefill
                                .ownership
                                .admissionSummary(0, fixture.prefill.requestLimit)
                                .occupiedRequests());
                assertEquals(2, fixture.decode.routingView().engineCapacityUsed());
                assertTrue(fixture.prefill.ownership.progress().queueDepth() == 0);

                fixture.applyWorkerStatus(
                        fixture.prefill, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
                fixture.applyWorkerStatus(
                        fixture.decode, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
                RequestProtocolTestSupport.awaitCondition(
                        () -> fixture.requests.requests.findActive("101") == null);
                assertEquals(
                        1,
                        fixture.prefill
                                .ownership
                                .admissionSummary(0, fixture.prefill.requestLimit)
                                .occupiedRequests());
                assertNotNull(
                        EndpointTestSupport.decodeReservation(
                                fixture.decode, String.valueOf(102L)));
            }
        }
    }

    private static TaskInfo task(long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(Long.toString(requestId));
        task.setInputLength(32L);
        task.setPhase(phase);
        return task;
    }

    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config = SchedulingTestConfig.newConfig();
        private final AbstractRequestScheduler requests;
        private final EndpointRegistry endpoints;
        private final DeliveryDispatcher queuedDelivery;
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
            config.setScheduler(ConfigTestSupport.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getDispatcher().setMaxInflightPerPrefillWorker(prefillCapacity);
            config.getRouter()
                    .getRoles()
                    .getDecode()
                    .getAvailability()
                    .setMaxEngineRequests((long) decodeCapacity);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);
            var placement = new PlacementAvailability();
            queuedDelivery = spy(DeliveryTestSupport.route(reporter));
            var repository = new RequestRepository();
            endpoints =
                    new EndpointRegistry(service, repository, reporter, queuedDelivery, placement);
            runtime =
                    spy(
                            new SchedulerRuntime(
                                    repository,
                                    endpoints,
                                    reporter,
                                    requestReporter,
                                    queuedDelivery,
                                    service,
                                    mock(org.flexlb.engine.grpc.client.EngineGrpcClient.class)));
            prefillSelector = mock(CostBasedPrefillStrategy.class);
            var decodeSelector = mock(DecodeSelector.class);
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
            router =
                    new RequestWorkerSelector(
                            prefillSelector, decodeSelector, mock(VitWorkerSelector.class), model);
            requests =
                    SchedulerTestSupport.create(
                            runtime,
                            service,
                            reporter,
                            mock(TheoryCacheHitReporter.class),
                            router,
                            mock(DecodeAdmission.class),
                            placement);
            scheduler = (DirectRequestScheduler) requests;
            WorkerStatus worker = worker(RoleType.PREFILL, "127.0.0.1");
            worker.lock.lock();
            try {
                var response = status(worker, Map.of(), Map.of());
                prefill =
                        (PrefillEndpoint)
                                endpoints.publishPreparedEndpoint(
                                        worker.getIpPort(),
                                        worker,
                                        worker.prepareNewStatus(
                                                WorkerStatusTestSupport.freezeStatusResponse(
                                                        worker, response)));
            } finally {
                worker.lock.unlock();
            }
            decode =
                    spy(
                            EndpointTestSupport.decode(
                                    worker(RoleType.DECODE, "127.0.0.2"), repository));
            applyWorkerStatus(decode, Map.of(), Map.of());
            when(prefillSelector.select(
                            any(),
                            any(),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.any(RequestRoute.class)))
                    .thenAnswer(
                            call -> {
                                var request =
                                        call.getArgument(
                                                0, org.flexlb.dao.loadbalance.Request.class);
                                var pin = prefill.tryPinGeneration();
                                assertNotNull(pin);
                                return PlacementResult.success(
                                        call.getArgument(4, RequestRoute.class)
                                                .add(
                                                        pin,
                                                        metadata(prefill, request.getRequestId()),
                                                        30_000L,
                                                        prefill.ownership
                                                                .progress()
                                                                .ownershipVersion()));
                            });
            when(decodeSelector.select(any(RequestData.class), any(), any(RequestRoute.class)))
                    .thenAnswer(
                            call -> {
                                var request = call.getArgument(0, RequestData.class).getRequest();
                                var pin = decode.tryPinGeneration();
                                assertNotNull(pin);
                                return PlacementResult.success(
                                        call.getArgument(2, RequestRoute.class)
                                                .add(
                                                        pin,
                                                        metadata(decode, request.getRequestId()),
                                                        -1L,
                                                        decode.placementVersion()));
                            });
        }

        private RequestContext context(String requestId) {
            var context = RequestProtocolTestSupport.context(config, requestId);
            context.data().setRequest(context.data().getRequest().toBuilder().seqLen(32L).build());
            context.data()
                    .setRequest(context.data().getRequest().toBuilder().maxNewTokens(16).build());
            return context;
        }

        private void assertNoWaitingQueue() {
            assertEquals(
                    0, SchedulerTestSupport.repository(requests).pendingDeliveryRequestCount());
            assertEquals(0, prefill.ownership.progress().queueDepth());
            assertEquals(0, prefill.ownership.progress().queueDepth());
        }

        private void assertNoPrefillOwnership() {
            assertNoWaitingQueue();
            assertEquals(
                    0L,
                    prefill.ownership.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            assertEquals(0, prefill.ownership.stats().batchCount());
        }

        private void assertItemNotBound(String requestId) {
            RequestContext requestContext =
                    SchedulerTestSupport.repository(requests).findActive(requestId);
            assertNotNull(requestContext);
            synchronized (requestContext) {
                assertNull(
                        requestContext.snapshot().route(),
                        "this Engine observation must precede item binding");
            }
        }

        private void applyWorkerStatus(
                WorkerEndpoint endpoint,
                Map<String, TaskInfo> running,
                Map<String, TaskInfo> finished) {
            WorkerStatus worker = endpoint.getStatus();
            org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection;
            worker.lock.lock();
            try {
                var response = status(worker, running, finished);
                projection =
                        endpoint.applyPreparedStatus(
                                worker,
                                worker.prepareNewStatus(
                                        WorkerStatusTestSupport.freezeStatusResponse(
                                                worker, response)));
            } finally {
                worker.lock.unlock();
            }
            endpoint.publishStatus(projection);
            SchedulerTestSupport.awaitRequestWork(requests.runtime);
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

        private static WorkerStatusResponse status(
                WorkerStatus worker,
                Map<String, TaskInfo> running,
                Map<String, TaskInfo> finished) {
            var response = new WorkerStatusResponse();
            response.setRole(worker.getRole());
            response.setAlive(true);
            response.setStatusVersion(
                    Math.max(1L, worker.appliedStatusCursor().statusVersion() + 1L));
            response.setLatestFinishedVersion(
                    Math.max(0L, worker.appliedStatusCursor().latestFinishedTaskVersion())
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

        private static ServerStatus metadata(WorkerEndpoint endpoint, String requestId) {
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
