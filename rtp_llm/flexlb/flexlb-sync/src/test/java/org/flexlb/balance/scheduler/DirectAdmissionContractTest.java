package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemption;
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

/** DIRECT uses real admission ledgers and lifecycle while bypassing waiting queues. */
class DirectAdmissionContractTest {

    @Test
    void schedulerRetriesOneGenerationHandoffFailureBeforeCommitting() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var captures = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(
                            call ->
                                    captures.getAndIncrement() == 0
                                            ? null
                                            : fixture.decode.tryPinGeneration())
                    .when(fixture.endpoints)
                    .captureDecodeGeneration(any());
            var response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(9010L))
                            .get(2L, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals(2, captures.get());
            assertEquals(1L, fixture.prefill.admissionSummary(0, 0).occupiedRequests());
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            fixture.assertNoWaitingQueue();
        }
    }

    @Test
    void completeSubmissionRejectsAnotherAdmissionBeforeResourcePreparation() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(9001L);
            var future =
                    fixture.requests.register(
                            context, () -> {}, StrategyErrorType.BATCH_SLO_EXPIRED);
            var admission = fixture.requests.claimAdmissionHandle("9001", future);
            var foreign = mock(AdmissionHandle.class);
            var route = mock(RequestRoute.class);
            when(route.ctx()).thenReturn(context);
            try {
                assertThrows(
                        IllegalStateException.class,
                        () -> fixture.requests.dispatcher.submitRoute(route, foreign));
                assertSame(admission, context.snapshot().placement().operation());
                org.mockito.Mockito.verifyNoInteractions(foreign);
                fixture.assertNoPrefillOwnership();
                assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            } finally {
                admission.finish();
                fixture.requests.cancel("9001", 0L, CancelReason.CLIENT_CANCELLED);
            }
        }
    }

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
                    .beginRoute(any());
            var first =
                    executor.submit(
                            () ->
                                    SchedulerTestServices.entry(fixture.scheduler)
                                            .submit(fixture.context(201L))
                                            .get(2L, TimeUnit.SECONDS));
            var second =
                    executor.submit(
                            () ->
                                    SchedulerTestServices.entry(fixture.scheduler)
                                            .submit(fixture.context(202L))
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
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(prefillCapacity, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    48L * prefillCapacity,
                    EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            fixture.assertNoWaitingQueue();
            verify(fixture.requests, times(2)).beginRoute(any());
            verify(fixture.prefillSelector, times(2))
                    .select(
                            any(),
                            any(),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(
                                    java.util.function.LongSupplier.class));
        }
    }

    @Test
    void directTemporaryCleanupFailureCannotUndoAnAcceptedRoute() {
        var failure = new IllegalStateException("generation cleanup failed after release");
        var shutdownFailure =
                assertThrows(
                        IllegalStateException.class,
                        () -> {
                            try (Fixture fixture = new Fixture()) {
                                doAnswer(
                                                call -> {
                                                    var pin =
                                                            org.mockito.Mockito.spy(
                                                                    fixture.prefill
                                                                            .tryPinGeneration());
                                                    doAnswer(
                                                                    close -> {
                                                                        close.callRealMethod();
                                                                        throw failure;
                                                                    })
                                                            .when(pin)
                                                            .close();
                                                    return pin;
                                                })
                                        .when(fixture.endpoints)
                                        .capture(RoleType.PREFILL, fixture.prefill.ipPort());
                                var context = fixture.context(100L);
                                var response =
                                        SchedulerTestServices.entry(fixture.scheduler)
                                                .submit(context)
                                                .get(2, TimeUnit.SECONDS);
                                assertTrue(response.isSuccess());
                                assertNotNull(context.snapshot().route().prefillReservation());
                                assertEquals(1, fixture.prefill.stats().locallyOwnedRequests());
                                assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
                                fixture.assertNoWaitingQueue();
                                assertSame(
                                        failure,
                                        assertThrows(
                                                IllegalStateException.class,
                                                fixture.requests.tasks::throwIfFailed));
                            }
                        });
        assertSame(failure, shutdownFailure);
    }

    @Test
    void directRouteOwnsPrefillAndDecodeUntilEngineTerminalWithoutEnteringWaitingQueues()
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            Response response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(101L))
                            .get(2L, TimeUnit.SECONDS);

            assertTrue(response.isSuccess());
            assertFalse(response.isEnqueuedByMaster());
            fixture.assertNoWaitingQueue();
            assertEquals(
                    1,
                    fixture.prefill
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(0, fixture.prefill.stats().batchCount());
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
            var reservation = EndpointTestSupport.decodeReservation(fixture.decode, 101L);
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
                                .tryReserveQueuedRequest(
                                        pin, String.valueOf(999L), 32L, 48L, 50, null)
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
                                    SchedulerTestServices.entry(fixture.scheduler)
                                            .submit(fixture.context(102L))
                                            .get(2L, TimeUnit.SECONDS));

            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 102L));
            assertEquals(occupant, EndpointTestSupport.decodeReservation(fixture.decode, 999L));
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
                                fixture.assertItemNotBound(103L);
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
                                fixture.assertItemNotBound(103L);
                                raced.set(true);
                                return acquired;
                            })
                    .when(fixture.decode)
                    .acquireDispatchPermit(any(), any());

            Response response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(103L))
                            .get(2L, TimeUnit.SECONDS);

            assertTrue(raced.get());
            assertFalse(
                    response.isSuccess(),
                    "an already ended request must not receive a new successful route");
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 103L));
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
            AtomicReference<Delivery.Member> claimed = new AtomicReference<>();
            doAnswer(
                            call -> {
                                var fact =
                                        call.getArgument(0, ContextResult.DeliveryEvidence.class);
                                Delivery.Member claim = fact.delivery();
                                claimed.set(claim);
                                RequestContext context = claim.route().ctx();
                                assertEquals("105", context.data().getRequestId());
                                assertEquals(
                                        Delivery.SendMode.ROUTE_DECISION,
                                        context.snapshot().delivery().identity().kind());
                                synchronized (context) {
                                    assertFalse(
                                            context.snapshot().delivery().send()
                                                    == ContextResult.SendState.DELIVERED);
                                }
                                assertFalse(claim.completion().toCompletableFuture().isDone());

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
                                assertTrue(claim.completion().toCompletableFuture().isDone());
                                synchronized (context) {
                                    assertFalse(
                                            context.snapshot().delivery().send()
                                                    == ContextResult.SendState.DELIVERED,
                                            "a worker terminal must not fabricate address publication");
                                }
                                fixture.assertNoPrefillOwnership();
                                assertNull(
                                        EndpointTestSupport.decodeReservation(
                                                fixture.decode, 105L));
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
                    .onDeliveryResult(
                            org.mockito.ArgumentMatchers.any(ContextResult.DeliveryEvidence.class));

            try {
                Response response =
                        SchedulerTestServices.entry(fixture.scheduler)
                                .submit(fixture.context(105L))
                                .get(2L, TimeUnit.SECONDS);

                SchedulerTestSupport.awaitRequestWork(fixture.runtime);
                assertTrue(raced.get());
                assertTrue(response.isSuccess());
                fixture.assertNoPrefillOwnership();
                assertEquals(0, fixture.requests.requests.liveRequestCount());
            } finally {
                // A broken sender gate must fail this test without hanging runtime shutdown.
                Delivery.Member claim = claimed.get();
                if (claim != null && !claim.completion().toCompletableFuture().isDone()) {
                    claim.route()
                            .ctx()
                            .data()
                            .getLifecycle()
                            .onDeliveryResult(
                                    new ContextResult.Abandonment(
                                            claim,
                                            CancelReason.CLIENT_CANCELLED,
                                            System.currentTimeMillis()));
                    fixture.requests.scheduleClose(claim.route().ctx());
                }
            }
        }
    }

    @Test
    void directCommitKeepsPrimaryFailureWhenPermitCleanupAlsoFails() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var prefill = spy(fixture.prefill);
            var context = fixture.context(104L);
            var future =
                    SchedulerTestServices.lifecycle(fixture.scheduler)
                            .register(context, () -> {}, StrategyErrorType.BATCH_SLO_EXPIRED);
            var primary = new IllegalStateException("Prefill commit unavailable");
            var cleanup = new IllegalStateException("Decode cleanup observer failed");
            org.mockito.Mockito.doThrow(primary).when(prefill).commitDirect(any(), any(), any());
            doAnswer(
                            call -> {
                                call.callRealMethod();
                                throw cleanup;
                            })
                    .when(fixture.decode)
                    .dispatch(any(), eq(DecodeResources.DispatchOutcome.ABANDONED));
            try (var mutation =
                            SchedulerTestServices.lifecycle(fixture.scheduler)
                                    .claimAdmissionHandle("104", future);
                    var completion = RequestProtocolTestSupport.finishOnExit(mutation);
                    var admission =
                            SchedulingTestConfig.prepareRoute(
                                            context,
                                            route -> {
                                                route.add(
                                                        prefill.tryPinGeneration(),
                                                        Fixture.metadata(prefill, 104L),
                                                        30_000L,
                                                        prefill.progress().ownershipVersion());
                                                route.add(
                                                        fixture.decode.tryPinGeneration(),
                                                        Fixture.metadata(fixture.decode, 104L),
                                                        -1L,
                                                        fixture.decode.placementVersion());
                                                return PlacementResult.success(route);
                                            })
                                    .value()) {
                var thrown =
                        assertThrows(
                                IllegalStateException.class,
                                () ->
                                        SchedulerTestServices.lifecycle(fixture.scheduler)
                                                .dispatcher
                                                .commitDirectRoute(admission));
                org.junit.jupiter.api.Assertions.assertSame(primary, thrown);
                assertEquals(List.of(cleanup), List.of(thrown.getSuppressed()));
                verify(fixture.decode)
                        .dispatch(any(), eq(DecodeResources.DispatchOutcome.ABANDONED));
                assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            } finally {
                SchedulerTestServices.lifecycle(fixture.scheduler)
                        .cancel("104", 0L, CancelReason.CLIENT_CANCELLED);
            }
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 104L));
            fixture.assertNoPrefillOwnership();
        }
    }

    @Test
    void predictionFailureAfterDirectCommitSettlesBothLedgers() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(109L);
            var prefill = spy(fixture.prefill);
            doAnswer(
                            call -> {
                                var work =
                                        spy(
                                                (org.flexlb.balance.projection.RequestTimeEstimation
                                                                .WorkSnapshot)
                                                        call.callRealMethod());
                                org.mockito.Mockito.doThrow(
                                                new IllegalStateException("prediction failed"))
                                        .when(work)
                                        .calculate();
                                return work;
                            })
                    .when(prefill)
                    .commitDirect(any(), any(), any());
            fixture.bindPrefill(prefill);
            assertFalse(
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(2, TimeUnit.SECONDS)
                            .isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 109L));
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            fixture.assertNoPrefillOwnership();
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void cancellationAfterPrefillCommitRejectsClaimAndSettlesBothLedgers() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(108L);
            var prefill = spy(fixture.prefill);
            doAnswer(
                            call -> {
                                var handoff = call.callRealMethod();
                                assertEquals(
                                        1,
                                        fixture.decode.resourceSnapshot().activeDispatchPermits());
                                SchedulerTestServices.lifecycle(fixture.scheduler)
                                        .cancel("108", 0L, CancelReason.CLIENT_CANCELLED);
                                return handoff;
                            })
                    .when(prefill)
                    .commitDirect(any(), any(), any());
            fixture.bindPrefill(prefill);
            var selected = fixture.prefillSelection(prefill);
            when(fixture.prefillSelector.select(
                            eq(context.data().getRequest()),
                            eq(context.data().getSchedulingMetadata()),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(
                                    java.util.function.LongSupplier.class)))
                    .thenReturn(selected);
            assertFalse(
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(2, TimeUnit.SECONDS)
                            .isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
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
                            mock(DecodePreemption.class),
                            fixture.runtime.newLifecycle(
                                    config,
                                    SchedulingTestConfig.capacity(config),
                                    org.mockito.Mockito.mock(
                                            org.flexlb.service.TheoryCacheHitReporter.class)),
                            new PlacementAvailability(),
                            new OrderedRequestQueue(config.isPriorityOrdering()),
                            org.flexlb.balance.planner.BuiltinGroupPlanner.from(config))) {
                var context = RequestProtocolTestSupport.context(config, 109L);
                var future =
                        SchedulerTestServices.lifecycle(queue)
                                .register(context, () -> {}, StrategyErrorType.BATCH_SLO_EXPIRED);
                try (var handle =
                                SchedulerTestServices.lifecycle(queue)
                                        .claimAdmissionHandle("109", future);
                        var finish = RequestProtocolTestSupport.finishOnExit(handle)) {
                    assertNotNull(handle);
                    var selected =
                            SchedulingTestConfig.prepareRoute(
                                            context,
                                            route -> {
                                                route.add(
                                                        prefill.tryPinGeneration(),
                                                        Fixture.metadata(prefill, 109L),
                                                        30_000L,
                                                        prefill.progress().ownershipVersion());
                                                if (separateDecode) {
                                                    route.add(
                                                            fixture.decode.tryPinGeneration(),
                                                            Fixture.metadata(fixture.decode, 109L),
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
                                            () -> SchedulerTestSupport.enqueueRoute(queue, plan)));
                        } else {
                            assertEquals(
                                    PlacementResult.Status.SUCCESS,
                                    SchedulerTestSupport.enqueueRoute(queue, plan).status());
                        }
                        var exact = context.snapshot().route();
                        assertNotNull(exact);
                        assertEquals(ContextResult.Phase.OPEN, context.snapshot().phase());
                        assertNotNull(context.snapshot().route());
                        assertNull(context.snapshot().delivery());
                        assertEquals(
                                PlacementResult.Status.CLOSED,
                                SchedulerTestSupport.enqueueRoute(queue, plan).status());
                        org.junit.jupiter.api.Assertions.assertSame(
                                exact, context.snapshot().route());
                        verify(prefill).offerPinned(any(), eq(exact), any());
                        verify(fixture.decode, never())
                                .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
                        assertEquals(1L, prefill.progress().queueDepth());
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
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(101L))
                            .get(2, TimeUnit.SECONDS)
                            .isSuccess());

            var config = SchedulingTestConfig.newConfig();
            SchedulingTestConfig.useFifoQueue(config);
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getDispatcher().setMaxInflightPerPrefillWorker(4);
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
            try (var queue =
                    (QueuedRequestScheduler)
                            org.flexlb.balance.scheduler.SchedulerTestSupport
                                    .createStartedScheduler(
                                            org.flexlb.balance.scheduler.SchedulerTestSupport
                                                    .runtime(fixture.requests)
                                                    .newLifecycle(
                                                            config,
                                                            SchedulingTestConfig.capacity(config),
                                                            org.mockito.Mockito.mock(
                                                                    org.flexlb.service
                                                                            .TheoryCacheHitReporter
                                                                            .class)),
                                            config,
                                            fixture.router,
                                            mock(DecodePreemption.class),
                                            fixture.placement,
                                            org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                    config),
                                            new OrderedRequestQueue(config.isPriorityOrdering()))) {
                var request = RequestProtocolTestSupport.context(config, 102L);
                request.data()
                        .setRequest(request.data().getRequest().toBuilder().seqLen(32L).build());
                request.data()
                        .setRequest(
                                request.data().getRequest().toBuilder().maxNewTokens(16).build());
                assertTrue(
                        SchedulerTestServices.entry(queue)
                                .submit(request)
                                .get(3, TimeUnit.SECONDS)
                                .isSuccess());
                assertEquals(
                        2,
                        fixture.prefill
                                .admissionSummary(0, fixture.prefill.requestLimit)
                                .occupiedRequests());
                assertEquals(2, fixture.decode.routingView().engineCapacityUsed());
                assertTrue(fixture.prefill.progress().queueDepth() == 0);

                fixture.applyWorkerStatus(
                        fixture.prefill, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
                fixture.applyWorkerStatus(
                        fixture.decode, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
                RequestProtocolTestSupport.awaitCondition(
                        () -> fixture.requests.requests.findActive("101") == null);
                assertEquals(
                        1,
                        fixture.prefill
                                .admissionSummary(0, fixture.prefill.requestLimit)
                                .occupiedRequests());
                assertNotNull(EndpointTestSupport.decodeReservation(fixture.decode, 102L));
            }
        }
    }

    private static TaskInfo task(long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        task.setInputLength(32L);
        task.setPhase(phase);
        return task;
    }

    private static final class Fixture implements AutoCloseable {
        final PlacementAvailability placement = new PlacementAvailability();
        private final FlexlbConfig config = SchedulingTestConfig.newConfig();
        private final RequestLifecycle requests;
        private final EndpointRegistry endpoints;
        private final DeliveryDispatcher queuedDelivery;
        private final PrefillEndpoint prefill;
        private final DecodeEndpoint decode;
        private final CostBasedPrefillStrategy prefillSelector;
        private final DirectRequestScheduler scheduler;
        private final RequestWorkerSelector router;
        private final SchedulerTestServices runtime;

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

            queuedDelivery = spy(DeliveryTestSupport.route(reporter));
            var repository = new RequestRepository();
            endpoints =
                    spy(
                            new EndpointRegistry(
                                    service,
                                    repository,
                                    reporter,
                                    queuedDelivery,
                                    placement,
                                    org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                            service.loadBalanceConfig()),
                                    order ->
                                            org.flexlb.balance.endpoint.PrefillActiveIndex.ordered(
                                                    16, order)));
            runtime =
                    spy(
                            new SchedulerTestServices(
                                    repository,
                                    endpoints,
                                    reporter,
                                    requestReporter,
                                    queuedDelivery,
                                    service,
                                    mock(org.flexlb.engine.grpc.EngineGrpcClient.class)));
            prefillSelector = mock(CostBasedPrefillStrategy.class);
            var decodeSelector = mock(DecodeSelector.class);
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
            router =
                    new RequestWorkerSelector(
                            endpoints,
                            prefillSelector,
                            decodeSelector,
                            mock(VitWorkerSelector.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.cache.match.CacheAwareService.class),
                            null,
                            mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                            mock(org.flexlb.cache.telemetry.CacheMetricsReporter.class),
                            service,
                            model);
            requests =
                    SchedulerTestSupport.create(
                            runtime,
                            service,
                            reporter,
                            mock(TheoryCacheHitReporter.class),
                            router,
                            mock(DecodePreemption.class),
                            placement);
            scheduler =
                    (DirectRequestScheduler)
                            org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(requests);
            WorkerStatus worker = worker(RoleType.PREFILL, "127.0.0.1");
            worker.lock.lock();
            try {
                var response = status(worker, Map.of(), Map.of());
                prefill =
                        (PrefillEndpoint)
                                endpoints.publishPreparedEndpoint(
                                        worker.getLogicalIpPort(),
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
            bindPrefill(prefill);
            when(prefillSelector.select(
                            any(),
                            any(),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(
                                    java.util.function.LongSupplier.class)))
                    .thenAnswer(call -> prefillSelection(prefill));
            when(decodeSelector.select(
                            any(),
                            org.mockito.ArgumentMatchers.any(),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any()))
                    .thenAnswer(call -> PlacementResult.success(decode.routingView()));
            doAnswer(
                            call -> {
                                call.getArgument(1, List.class).add(decode.routingView());
                                return null;
                            })
                    .when(endpoints)
                    .decodeRoutingSnapshot(any(), any());
            doAnswer(call -> decode.tryPinGeneration())
                    .when(endpoints)
                    .captureDecodeGeneration(any());
        }

        private void bindPrefill(PrefillEndpoint selected) {
            String address = selected.ipPort();
            org.mockito.Mockito.doReturn(
                            List.of(new EndpointRegistry.PrefillRoutingEntry(address, selected)))
                    .when(endpoints)
                    .prefillRoutingSnapshot(RoleType.PREFILL);
            doAnswer(call -> selected.tryPinGeneration())
                    .when(endpoints)
                    .capture(RoleType.PREFILL, address);
        }

        private PlacementResult<
                        org.flexlb.balance.strategy.PrefillSelectionStrategy.Selection<
                                org.flexlb.balance.strategy.PrefillSelectionStrategy
                                        .CandidateInput>,
                        PlacementKey>
                prefillSelection(PrefillEndpoint selected) {
            return PlacementResult.success(
                    new org.flexlb.balance.strategy.PrefillSelectionStrategy.Selection<>(
                            selected.ipPort(),
                            selected.getStatus().getGenerationId(),
                            selected.progress().ownershipVersion(),
                            30_000L,
                            30_000L,
                            0L,
                            0L,
                            0L,
                            null,
                            1,
                            new EndpointRegistry.PrefillRoutingEntry(selected.ipPort(), selected),
                            null));
        }

        private RequestContext context(long requestId) {
            var context = RequestProtocolTestSupport.context(config, requestId);
            context.data().setRequest(context.data().getRequest().toBuilder().seqLen(32L).build());
            context.data()
                    .setRequest(context.data().getRequest().toBuilder().maxNewTokens(16).build());
            return context;
        }

        private void assertNoWaitingQueue() {
            assertEquals(
                    0, SchedulerTestSupport.repository(requests).pendingDeliveryRequestCount());
            assertEquals(0, prefill.progress().queueDepth());
            assertEquals(0, prefill.progress().queueDepth());
        }

        private void assertNoPrefillOwnership() {
            assertNoWaitingQueue();
            assertEquals(0L, prefill.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            assertEquals(0, prefill.stats().batchCount());
        }

        private void assertItemNotBound(long requestId) {
            RequestContext requestContext =
                    SchedulerTestSupport.repository(requests).findActive(String.valueOf(requestId));
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
            SchedulerTestSupport.awaitRequestWork(runtime);
        }

        @Override
        public void close() {
            try {
                decode.close();
            } finally {
                SchedulerTestSupport.shutdown(runtime);
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

        private static ServerStatus metadata(WorkerEndpoint endpoint, long requestId) {
            var result = new ServerStatus();
            result.setSuccess(true);
            result.setRequestId(String.valueOf(requestId));
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
