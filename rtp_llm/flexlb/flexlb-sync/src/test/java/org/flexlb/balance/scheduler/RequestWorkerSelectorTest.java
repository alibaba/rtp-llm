package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.AdmissionHandle;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.balance.strategy.WorkerAssignment;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Final selector/pin ownership contracts for {@link RequestWorkerSelector}.
 */
class RequestWorkerSelectorTest {

    private DeliveryClaim lastDelivery;

    private CostBasedPrefillStrategy prefillSelector;

    private DecodeSelector decodeSelector;

    private VitWorkerSelector vitSelector;

    private ConfigService configService;

    private ModelMetaConfig modelMeta;

    private AbstractRequestScheduler requests;

    @BeforeEach
    void setUp() {
        prefillSelector = mock(CostBasedPrefillStrategy.class);
        decodeSelector = mock(DecodeSelector.class);
        vitSelector = mock(VitWorkerSelector.class);
        configService = mock(ConfigService.class);
        modelMeta = mock(ModelMetaConfig.class);
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        when(configService.loadBalanceConfig()).thenReturn(config);
        var runtime = SchedulerTestSupport.runtime(SchedulerTestSupport.create(configService,
                mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                mock(org.flexlb.service.RecentCacheKeyTraceReporter.class)));
        requests = org.mockito.Mockito.mock(DirectRequestScheduler.class, org.mockito.Mockito.withSettings()
                .useConstructor(mock(RequestWorkerSelector.class), runtime, config, SchedulingTestConfig.capacity(config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class))
                .defaultAnswer(invocation -> {
                    if (invocation.getMethod().getDeclaringClass() == DirectRequestScheduler.class) { return invocation.callRealMethod(); }
                    return org.mockito.Answers.RETURNS_DEFAULTS.answer(invocation);
                }));
        when(requests.claimAdmissionHandle(anyLong(), any())).thenReturn(mock(AdmissionHandle.class));
        when(requests.register(any(), any(Runnable.class), org.mockito.ArgumentMatchers.any())).thenAnswer(call -> {
            var context = call.getArgument(0, RequestContext.class);
            var future = new CompletableFuture<Response>();
            context.data().setFuture(future);
            SchedulerTestSupport.bindOwner(context, requests);
            return future;
        });
        when(requests.commitRoute(any(), any())).thenAnswer(call -> RequestProtocolTestSupport.publish(call.getArgument(1)) ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED);
        when(requests.claimDelivery(any(), eq(DeliveryClaimKind.ROUTE_DECISION), eq(0L), any())).thenAnswer(call -> {
            var permit = call.getArgument(3, DecodeEndpoint.EngineDispatchPermit.class);
            if (permit != null) { permit.dispatch().status(); }
            RequestRoute item = call.getArgument(0);
            DeliveryClaim claim = new DeliveryClaim(item.ctx(), 1L, item, DeliveryClaimKind.ROUTE_DECISION, 0L);
            lastDelivery = claim;
            doAnswer(inv -> {
                item.future().complete(item.successResponse(false));
                return null;
            }).when(requests).onDeliveryEvidence(eq(claim), any(ContextResult.RouteDelivered.class));
            return claim;
        });
        when(requests.publish(any(), any(), eq(false))).thenAnswer(call -> call.getArgument(0, RequestContext.class).data().getFuture().complete(call.getArgument(1)));
    }

    @Test
    void invalidRequestFailsBeforePolicyOrEndpointSelection() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestWorkerSelector router = router();
        org.mockito.Mockito.clearInvocations(configService);

        PlacementResult<RequestRoute, PlacementKey> result = router.select(new RequestContext(SchedulingTestConfig.newConfig()), null);

        assertEquals(PlacementResult.Status.REJECTED, result.status());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                result.failure().getCode());
        verify(configService, never()).loadBalanceConfig();
        verifyNoInteractions(prefillSelector, decodeSelector, vitSelector);
    }

    @ParameterizedTest
    @CsvSource({"false,false", "true,false", "false,true"})
    void directRouteCommitsRolesAndReleasesGenerationPins(boolean pinCloseFails, boolean materializationFails) {
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestWorkerSelector router = router();
        RequestContext context = directContext(7L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 7L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 7L, "d", 8002, "g1");
        PrefillState.RouteReservation registration = mock(PrefillState.RouteReservation.class);
        DecodeResources.ReservationHandle reservation = new DecodeResources.ReservationHandle(1L, 7L, 2L);
        when(prefillSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), "g1"))
                .thenReturn(PlacementResult.success(decode.selection));
        var capture = stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
        when(((PrefillEndpoint) prefill.endpoint).reserveUnqueuedRoute(eq(prefill.pin), any(RequestRoute.class), eq(1L)))
                .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, registration));
        when(((DecodeEndpoint) decode.endpoint).tryReserveQueuedRequest(eq(decode.pin), eq(7L), eq(32L), eq(48L), eq(50), isNull()))
                .thenReturn(new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED, reservation));
        var permit = stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);
        if (pinCloseFails) {
            doThrow(new IllegalStateException("pin close failed after direct handoff"))
                    .when(prefill.pin).close();
        }

        if (materializationFails) {
            when(capture.materialize()).thenThrow(new IllegalStateException("snapshot materialization failed"));
        }
        Response response = scheduler(router, context).submit(context).join();
        if (materializationFails) {
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), response.getCode());
            verify(permit, never()).dispatch();
            verify(permit).release();
            verify((PrefillEndpoint) prefill.endpoint).release(registration, null);
            verify(prefill.pin).close();
            verify(decode.pin).close();
            verify(requests, never()).onDeliveryEvidence(any(), any(ContextResult.RouteDelivered.class));
            return;
        }

        assertTrue(response.isSuccess());
        assertEquals(List.of(prefill.status, decode.status),
                response.getServerStatus());
        verify((PrefillEndpoint) prefill.endpoint).release(registration, null);
        verify(permit).dispatch();
        verify((DecodeEndpoint) decode.endpoint, never())
                .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(prefill.pin).close();
        verify(decode.pin).close();
        var order = org.mockito.Mockito.inOrder(decode.pin, prefill.pin, requests);
        order.verify(decode.pin).close();
        order.verify(prefill.pin).close();
        order.verify(requests).onDeliveryEvidence(any(), any(ContextResult.RouteDelivered.class));
    }

    @Test
    void directRouteRollsBackDecodeWhenPrefillAdmissionIsFull() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestWorkerSelector router = router();
        RequestContext context = directContext(8L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 8L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 8L, "d", 8002, "g1");
        DecodeResources.ReservationHandle reservation = new DecodeResources.ReservationHandle(1L, 8L, 2L);
        when(prefillSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), "g1"))
                .thenReturn(PlacementResult.success(decode.selection));
        stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
        when(((DecodeEndpoint) decode.endpoint).tryReserveQueuedRequest(eq(decode.pin), eq(8L), eq(32L), eq(48L), eq(50), isNull()))
                .thenReturn(new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED, reservation));
        stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);
        when(((PrefillEndpoint) prefill.endpoint).reserveUnqueuedRoute(eq(prefill.pin), any(RequestRoute.class), eq(1L)))
                .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.CAPACITY_FULL, null));

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                scheduler(router, context).submit(context).join().getCode());
        verify(prefillSelector).select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);
        verify((DecodeEndpoint) decode.endpoint).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(prefill.pin).close();
        verify(decode.pin).close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directAdmissionPassesZeroAndUnknownTimeToLifecycle(boolean unknown) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestContext context = directContext(81L);
        SelectionFixture selected = selection(RoleType.PREFILL, 81L, "p", 8001, "g1");
        when(selected.selection.prefillWorkMs()).thenReturn(0L);
        when(prefillSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(selected.selection));
        PrefillEndpoint endpoint = (PrefillEndpoint) selected.endpoint;
        long committedAtMs = System.currentTimeMillis();
        stubRouteCommit(endpoint, new org.flexlb.balance.projection.WorkSnapshot(
                committedAtMs, List.of(), List.of(), unknown ? 1L : 0L));
        var registration = mock(PrefillState.RouteReservation.class);
        when(endpoint.reserveUnqueuedRoute(eq(selected.pin), any(), eq(0L)))
                .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, registration));

        assertTrue(scheduler(router(), context).submit(context).join().isSuccess());

        var evidence = org.mockito.ArgumentCaptor.forClass(ContextResult.RouteDelivered.class);
        verify(requests).onDeliveryEvidence(eq(lastDelivery), evidence.capture());
        assertEquals(unknown ? 0L : evidence.getValue().nowMs() + 10_000L, evidence.getValue().predictedAtMs());
        verify(endpoint).release(registration, null);
    }

    @Test
    void directRequestRemainsInCanonicalLifecycleAfterResponseAndReconcilesEarlyDecodeEvidence() throws Exception {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestContext context = directContext(9L);
        when(configService.loadBalanceConfig()).thenReturn(context.data().getConfig());
        requests = SchedulerTestSupport.create(configService, mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class), mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        try {
            SelectionFixture prefill = selection(RoleType.PREFILL, 9L, "p", 8001, "g1");
            SelectionFixture decode = selection(RoleType.DECODE, 9L, "d", 8002, "g1");
            var reservation = new DecodeResources.ReservationHandle(1L, 9L, 2L);
            when(prefillSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null)).thenReturn(PlacementResult.success(prefill.selection));
            when(decodeSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), "g1")).thenReturn(PlacementResult.success(decode.selection));
            stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
            when(((PrefillEndpoint) prefill.endpoint).reserveUnqueuedRoute(eq(prefill.pin), any(RequestRoute.class), eq(1L))).thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, mock(PrefillState.RouteReservation.class)));
            when(((DecodeEndpoint) decode.endpoint).tryReserveQueuedRequest(eq(decode.pin), eq(9L), eq(32L), eq(48L), eq(50), isNull())).thenReturn(new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED, reservation));
            stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);
            when(((DecodeEndpoint) decode.endpoint).ownership(reservation)).thenReturn(org.flexlb.balance.endpoint.DecodeResources.Ownership.CONFIRMED);
            assertTrue(scheduler(router(), context).submit(context).get(2L, TimeUnit.SECONDS).isSuccess());
            assertTrue(context.data().getFuture().get(2L, TimeUnit.SECONDS).isSuccess());
            RequestContext requestContext = requests.requests.findActive(9L);
            synchronized (requestContext) {
                assertTrue(requestContext.snapshot().engine().decodeAccepted());
                assertTrue(requestContext.snapshot().phase() == ContextResult.Phase.OPEN);
                assertEquals(RequestState.Phase.ACKNOWLEDGED, requests.requests.getRequestState(requestContext.data().getRequestId(), 0L).state());
            }
            RequestProtocolTestSupport.expireInactiveRequest(requests, requestContext, System.currentTimeMillis() + context.data().getConfig().getRequestLifecycle().getRequest().getTimeoutMs());
            assertEquals(RequestState.Phase.TIMED_OUT, SchedulerTestSupport.repository(requests).getRequestState(9L, 0L).state());
            assertEquals(0, SchedulerTestSupport.repository(requests).liveRequestCount());
            verify((DecodeEndpoint) decode.endpoint).release(reservation, DecodeResources.ReleaseReason.EXPIRED);
            verify((PrefillEndpoint) prefill.endpoint).release(isNull(), any(RequestRoute.class));
        } finally {
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(requests);
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(requests)).close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(requests));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void directSelectionExceptionPublishesFailureAndClosesLifecycle(boolean fatal) throws Exception {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestContext context = directContext(10L);
        when(configService.loadBalanceConfig()).thenReturn(context.data().getConfig());
        requests = SchedulerTestSupport.create(configService, mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class), mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        try {
            Throwable failure = fatal ? new AssertionError("selection failed") : new IllegalStateException("selection failed");
            when(prefillSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null)).thenThrow(failure);
            if (fatal) {
                assertEquals(failure, assertThrows(AssertionError.class, () -> scheduler(router(), context).submit(context)));
            } else {
                var returned = scheduler(router(), context).submit(context);
                assertEquals(context.data().getFuture(), returned);
            }
            Response response = context.data().getFuture().get(2L, TimeUnit.SECONDS);
            assertEquals(8510, response.getCode());
            assertEquals("DISPATCH_FAILED", response.getErrorMessage());
            assertEquals(RequestState.Phase.FAILED, SchedulerTestSupport.repository(requests).getRequestState(10L, 0L).state());
            verifyNoInteractions(decodeSelector, vitSelector);
        } finally {
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(requests);
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(requests)).close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(requests));
        }
    }

    @Test
    void requestDemandUsesImmutablePromptAndOutputInputs() {
        var request = org.flexlb.dao.loadbalance.Request.builder().seqLen(32L).maxNewTokens(16).build();
        var changed = request.toBuilder().seqLen(4096L).build();
        assertEquals(32L, request.getSeqLen());
        assertEquals(32L, request.getHardKvTokens());
        assertEquals(48L, request.getExpectedKvTokens());
        assertEquals(4112L, changed.getExpectedKvTokens());
    }

    @Test
    void sharedCapacityRetainsItsStartupLimit() {
        var config = SchedulingTestConfig.newConfig();
        var availability = config.getRouter().getRoles().getDecode().getAvailability();
        availability.setMaxEngineRequests(7L);
        var capacity = SchedulingTestConfig.capacity(config);
        availability.setMaxEngineRequests(null);
        assertEquals(7L, capacity.maxEngineRequests());
    }

    @ParameterizedTest
    @CsvSource({"32,16,32,48", "500,10000,500,10500",
            "9223372036854775806,100,9223372036854775806,9223372036854775807",
            "-1,100,0,100", "500,-1,500,500"})
    void reservationAndDispatchUseImmutableDemandAndSharedStartupCapacity(
            long prompt, int output, long hardKv, long expectedKv) {
        var config = SchedulingTestConfig.newConfig();
        var limits = config.getRouter().getRoles().getDecode().getAvailability();
        limits.setMaxEngineRequests(1L);
        limits.setMaxKvUsagePercent(90L);
        var estimator = config.getRouter().getRoles().getDecode().getCostEstimator();
        estimator.setExpression("2 * running_size / max_running_size + 3 * kvcache_used_ratio");
        var capacity = SchedulingTestConfig.capacity(config);
        var selector = new DecodeSelector(mock(org.flexlb.balance.endpoint.EndpointRegistry.class), SchedulingTestConfig.configService(config), capacity);
        var formula = (org.flexlb.balance.prediction.DecodeCostFormula)
                org.springframework.test.util.ReflectionTestUtils.getField(selector, "costFormula");
        var context = RequestProtocolTestSupport.context(config, 701L);
        context.data().setRequest(context.data().getRequest().toBuilder().seqLen(prompt).maxNewTokens(output).build());
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(73, Long.MAX_VALUE));
        var scheduler = queueScheduler(context);
        org.springframework.test.util.ReflectionTestUtils.setField(scheduler, "decodeAdmissionCapacity", capacity);
        var prefill = selection(RoleType.PREFILL, 701L, "10.0.0.1", 8080, "g1");
        var selectedDecode = selection(RoleType.DECODE, 701L, "10.0.0.2", 8080, "g1");
        var decode = (DecodeEndpoint) selectedDecode.endpoint();
        var reservation = new DecodeResources.ReservationHandle(1L, 701L, 1L);
        when(decode.tryReserveQueuedRequest(any(), anyLong(), anyLong(), anyLong(), anyInt(), eq(capacity)))
                .thenReturn(new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED, reservation));
        when(decode.acquireDispatchPermit(reservation, capacity)).thenReturn(
                new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                        DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL, null));
        limits.setMaxEngineRequests(99L);
        limits.setMaxKvUsagePercent(1L);
        estimator.setExpression("7 * running_size / max_running_size + 11 * kvcache_used_ratio");
        assertNotSame(estimator.compiledFormula(), formula);
        assertEquals("2 * running_size / max_running_size + 3 * kvcache_used_ratio", formula.expression());
        org.mockito.Mockito.doReturn(PlacementResult.Status.SUCCESS).when(scheduler).commitRoute(any(), any());
        try (var route = RequestRoute.prepare(context, List.of(prefill.selection(), selectedDecode.selection()));
             var plan = RequestProtocolTestSupport.plan(context, route)) {
            var placement = scheduler.enqueueRoute(plan);
            assertEquals(PlacementResult.Status.SUCCESS, placement.status());
            var item = placement.value();
            assertSame(reservation, item.decodeReservation());
            {
                assertSame(context.data().getRequest(), item.ctx().data().getRequest());
                assertSame(reservation, item.decodeReservation());
                assertSame(decode, item.decodeEp());
                assertEquals(hardKv, item.seqLen());
                var delivery = DeliveryDispatcher.Member.prepare(item, capacity);
                assertFalse(delivery.accepted());
                assertTrue(delivery.boundary().unavailable());
                assertFalse(item.decodeEp().shouldRetryDispatch(item.decodeReservation(), capacity));
                verify(decode).shouldRetryDispatch(reservation, capacity);
                verify(decode).acquireDispatchPermit(reservation, capacity);
                verify(decode).tryReserveQueuedRequest(any(), eq(701L), eq(hardKv), eq(expectedKv), eq(73), eq(capacity));
                verify(decode, never()).tryReserveQueuedRequest(any(), anyLong(), anyLong(), anyLong(), anyInt(), isNull());
            }
        }
        verify(decode).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedDecodeAdoptionReleasesOnceAndPreservesOriginalFailure(boolean adoptionThrows) {
        var context = context(702L);
        var prefill = selection(RoleType.PREFILL, 702L, "p", 8001, "g1");
        var selectedDecode = selection(RoleType.DECODE, 702L, "d", 8002, "g1");
        var decode = (DecodeEndpoint) selectedDecode.endpoint();
        var reservation = new DecodeResources.ReservationHandle(1L, 702L, 1L);
        var adoptionFailure = new IllegalStateException("adoption failed");
        var cleanupFailure = new IllegalStateException("cleanup failed");
        if (adoptionThrows) {
            when(decode.markQueued(selectedDecode.pin(), reservation)).thenThrow(adoptionFailure);
        }
        doThrow(cleanupFailure).when(decode)
                .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);

        try (var route = RequestRoute.prepare(context,
                List.of(prefill.selection(), selectedDecode.selection()))) {
            var actual = assertThrows(IllegalStateException.class,
                    () -> RequestProtocolTestSupport.adopt(queueScheduler(context),
                            RequestProtocolTestSupport.plan(context, route),
                            new org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult(reservation, false, "test")));
            assertSame(adoptionThrows ? adoptionFailure : cleanupFailure, actual);
            assertEquals(adoptionThrows ? List.of(cleanupFailure) : List.of(),
                    List.of(actual.getSuppressed()));
        }
        verify(decode).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(prefill.pin()).close();
        verify(selectedDecode.pin()).close();
    }

    @Test
    void committedQueueOwnsCapacityEvenWhenSelectionPinCleanupFails() {
        var context = context(703L);
        var prefill = selection(RoleType.PREFILL, 703L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 703L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var reservation = new DecodeResources.ReservationHandle(1L, 703L, 1L);
        var cleanupFailure = new IllegalStateException("pin cleanup failed after publication");
        var route = RequestRoute.prepare(context,
                List.of(prefill.selection(), decode.selection()));
        var scheduler = queueScheduler(context);
        var plan = RequestProtocolTestSupport.plan(context, route);
        when(endpoint.markQueued(decode.pin(), reservation)).thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED, false));
        assertTrue(RequestProtocolTestSupport.adopt(scheduler, plan,
                new org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult(reservation, false, "test")));
        when(route.prefillEp().offerPinned(eq(prefill.pin()), any(), any())).thenReturn(true);
        doThrow(cleanupFailure).when(prefill.pin()).close();
        assertEquals(PlacementResult.Status.SUCCESS, scheduler.enqueueRoute(plan).status());
        verify(prefill.pin(), never()).close();
        verify(decode.pin(), never()).close();
        assertSame(cleanupFailure, assertThrows(IllegalStateException.class, plan::close));
        plan.close();
        verify(prefill.pin()).close();
        verify(decode.pin()).close();
        verify(endpoint, never()).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
    }

    @Test
    void attachingCapacityDoesNotAssignRequestFifoIdentity() {
        var context = context(708L);
        var prefill = selection(RoleType.PREFILL, 708L, "p", 8001, "g1");
        try (var selected = RequestRoute.prepare(context, List.of(prefill.selection()))) {
            var route = RequestRoute.create(context, selected, null);
            assertSame(context, route.ctx());
            assertEquals(0L, context.data().getWorkerEnqueueSequence());
            assertEquals(0L, context.data().getFirstWorkerEnqueueTime());
        }
    }

    @Test
    void callerAndDeliveredResponseChangesCannotMutateTheRouteTemplate() {
        var context = context(707L);
        var prefill = selection(RoleType.PREFILL, 707L, "p", 8001, "g1");
        try (var route = RequestRoute.prepare(context, List.of(prefill.selection()))) {
            prefill.status().setGroup("caller changed");
            var exported = route.successResponse(true);
            exported.setCode(503);
            exported.getServerStatus().getFirst().setGroup("frontend changed");
            var subsequent = route.successResponse(false);
            assertTrue(subsequent.isSuccess());
            assertEquals(200, subsequent.getCode());
            assertEquals("g1", subsequent.getServerStatus().getFirst().getGroup());
            assertFalse(subsequent.isEnqueuedByMaster());
        }
        verify(prefill.pin()).close();
    }

    @Test
    void invalidResultConstructionLeavesPinOwnershipWithTheSelector() {
        var context = context(706L);
        var prefill = selection(RoleType.PREFILL, 706L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 706L, "d", 8002, "g1");
        try (var prefillSelection = prefill.selection(); var decodeSelection = decode.selection()) {
            when(decodeSelection.requestId()).thenReturn(999L);
            assertThrows(IllegalStateException.class, () -> RequestRoute.prepare(context,
                    List.of(prefillSelection, decodeSelection)));
            verify(prefillSelection, never()).assignToRequest();
            verify(decodeSelection, never()).assignToRequest();
        }
        verify(prefill.pin()).close();
        verify(decode.pin()).close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void routeConstructionFailureRollsBackCapacityAndClosesPins(boolean direct) {
        var context = org.mockito.Mockito.spy(direct ? directContext(704L) : context(704L));
        var prefill = selection(RoleType.PREFILL, 704L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 704L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var reservation = new DecodeResources.ReservationHandle(1L, 704L, 1L);
        stubDecodePermit(endpoint, reservation);
        stubPrefillCommit((PrefillEndpoint) prefill.endpoint());
        when(endpoint.tryReserveQueuedRequest(any(), anyLong(), anyLong(), anyLong(), anyInt(), any()))
                .thenReturn(new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED, reservation));
        var primary = new IllegalStateException("route construction failed");
        var cleanup = new IllegalStateException("rollback failed");

        doThrow(cleanup).when(endpoint).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        var routing = RequestRoute.prepare(context, List.of(prefill.selection(), decode.selection()));
        var actual = assertThrows(IllegalStateException.class, () -> {
            if (direct) {
                doThrow(primary).when(requests).commitRoute(any(), any());
                try (routing) { ((DirectRequestScheduler) requests).commitDirectRoute(context, routing); }
            } else {
                var queued = queueScheduler(context);
                doThrow(primary).when(queued).commitRoute(any(), any());
                try (var plan = RequestProtocolTestSupport.plan(context, routing)) { queued.enqueueRoute(plan); }
            }
        });
        assertSame(primary, actual);
        assertEquals(List.of(cleanup), List.of(actual.getSuppressed()));
        verify(endpoint).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(prefill.pin()).close();
        verify(decode.pin()).close();
    }

    @Test
    void conflictingPreemptionResultDoesNotOverwriteOrReleaseExistingCapacityTwice() {
        var context = context(705L);
        var prefill = selection(RoleType.PREFILL, 705L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 705L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var existing = new DecodeResources.ReservationHandle(1L, 705L, 1L);
        var incoming = new DecodeResources.ReservationHandle(1L, 705L, 2L);
        when(endpoint.markQueued(decode.pin(), existing)).thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED, false));
        var scheduler = queueScheduler(context);
        var plan = RequestProtocolTestSupport.plan(context, RequestRoute.prepare(context,
                List.of(prefill.selection(), decode.selection())));
        try (plan) {
            assertTrue(RequestProtocolTestSupport.adopt(scheduler, plan,
                    new org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult(existing, false, "first")));
            assertFalse(RequestProtocolTestSupport.adopt(scheduler, plan,
                    new org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult(incoming, false, "conflict")));
            assertEquals(existing, org.springframework.test.util.ReflectionTestUtils.getField(plan, "pendingDecodeRollback"));
            assertEquals(false, org.springframework.test.util.ReflectionTestUtils.invokeMethod(scheduler, "settlePreemptionResult", plan, true));
            verify(endpoint, never()).release(existing, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
        plan.close();
        verify(endpoint).release(incoming, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(endpoint).release(existing, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(endpoint, never()).markQueued(decode.pin(), incoming);
        verify(prefill.pin()).close();
        verify(decode.pin()).close();
    }

    private QueuedRequestScheduler queueScheduler(RequestContext context) {

        var config = SchedulingTestConfig.newConfig();
        var scheduler = mock(QueuedRequestScheduler.class, org.mockito.Mockito.withSettings()
                .useConstructor(config, mock(RequestWorkerSelector.class),
                        mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        mock(org.flexlb.balance.eviction.DecodeCapacityAcquirer.class), SchedulerTestSupport.runtime(requests), new PlacementAvailability(), SchedulingTestConfig.capacity(config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class))
                .defaultAnswer(invocation -> invocation.getMethod().getDeclaringClass() == QueuedRequestScheduler.class
                        ? invocation.callRealMethod() : org.mockito.Answers.RETURNS_DEFAULTS.answer(invocation)));
        when(scheduler.commitRoute(any(), any())).thenAnswer(call ->
                RequestProtocolTestSupport.publish(call.getArgument(1)) ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED);
        return scheduler;
    }

    private static PrefillState.WorkCapture stubPrefillCommit(PrefillEndpoint prefill) {
        return stubRouteCommit(prefill, new org.flexlb.balance.projection.WorkSnapshot(System.currentTimeMillis(), List.of(), List.of(), 0L));
    }

    private static PrefillState.WorkCapture stubRouteCommit(PrefillEndpoint endpoint,
            org.flexlb.balance.projection.WorkSnapshot precedingWork) {
        var handoff = mock(PrefillState.CommittedHandoff.class);
        var capture = mock(PrefillState.WorkCapture.class);
        when(capture.materialize()).thenReturn(precedingWork);
        when(handoff.precedingWork()).thenReturn(capture);
        when(endpoint.commit(any(PrefillState.CommitPlan.class))).thenReturn(handoff);
        return capture;
    }

    private static DecodeEndpoint.EngineDispatchPermit stubDecodePermit(
            DecodeEndpoint endpoint, DecodeResources.ReservationHandle reservation) {
        var permit = mock(DecodeEndpoint.EngineDispatchPermit.class);
        when(endpoint.acquireDispatchPermit(eq(reservation), any()))
                .thenReturn(new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                        DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, permit));
        when(permit.belongsTo(endpoint, reservation)).thenReturn(true);
        when(permit.dispatch())
                .thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED, false));
        return permit;
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {
            "PREFILL", "DECODE", "PDFUSION", "VIT"})
    void missingRequiredRoleReturnsItsExactWaitDomain(RoleType role) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(role));
        RequestWorkerSelector router = router();
        RequestContext context = context(11L);
        stubQueueSelection(
                context, role, null, PlacementResult.blocked(role));

        PlacementResult<RequestRoute, PlacementKey> blocked = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(role, null, null), blocked.blocker());
    }

    @Test
    void projectedDecodeBlockUsesDecodeWaitDomain() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestWorkerSelector router = router();
        RequestContext context = context(111L);
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.blocked(RoleType.DECODE));

        PlacementResult<RequestRoute, PlacementKey> blocked = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.DECODE, null, null),
                blocked.blocker());
    }

    @Test
    void staticCapacityFailureIsTerminalAndReleasesEarlierSelections() {
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestWorkerSelector router = router();
        RequestContext context = context(12L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 12L, "p", 8001, "g1");
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), "g1"))
                .thenReturn(PlacementResult.rejected(
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED)));

        PlacementResult<RequestRoute, PlacementKey> rejected = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.REJECTED, rejected.status());

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                rejected.failure().getCode());
        verify(prefill.selection).close();
    }

    @Test
    void selectionFailureClosesAllPinsInReverseOrderAndKeepsPrimaryCause() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.VIT));
        RequestContext context = context(120L);
        var prefill = selection(RoleType.PREFILL, 120L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 120L, "d", 8002, "g1");
        when(prefillSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), "g1"))
                .thenReturn(PlacementResult.success(decode.selection));
        var primary = new IllegalStateException("selection failed");
        var decodeClose = new IllegalStateException("Decode pin close failed");
        var prefillClose = new IllegalStateException("Prefill pin close failed");
        when(vitSelector.select(context.data().getRequestId(), RoleType.VIT, "g1")).thenThrow(primary);
        doThrow(decodeClose).when(decode.selection).close();
        doThrow(prefillClose).when(prefill.selection).close();

        assertSame(primary, assertThrows(IllegalStateException.class, () -> router().select(context, null)));

        var order = org.mockito.Mockito.inOrder(decode.selection, prefill.selection);
        order.verify(decode.selection).close();
        order.verify(prefill.selection).close();
        assertEquals(List.of(decodeClose), List.of(primary.getSuppressed()));
        assertEquals(List.of(prefillClose), List.of(decodeClose.getSuppressed()));
    }

    @Test
    void queueRouteTransfersTheExactPrefillPinIntoAdmissionOwnership() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestWorkerSelector router = router();
        RequestContext context = context(21L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, context.data().getRequestId(), "p", 8001, "g1");
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));

        PlacementResult<RequestRoute, PlacementKey> admitted = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());

        assertTrue(admitted.value().successResponse(false).isSuccess());
        assertEquals(List.of(prefill.status),
                admitted.value().successResponse(false).getServerStatus());
        verify(prefill.selection).assignToRequest();
        verify(prefill.pin, never()).close();

        admitted.value().close();
        verify(prefill.pin).close();
    }

    @Test
    void prefillSelectionUsesRegisteredInputsAndOriginalConfiguration() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        var context = context(30L);
        var inputs = context.data().getRequest();
        var metadata = context.data().getSchedulingMetadata();
        var changed = inputs.toBuilder().seqLen(4_096L).build();
        assertEquals(4_096L, changed.getSeqLen());
        var prefill = selection(RoleType.PREFILL, 30L, "p", 8001, "g1");
        when(prefillSelector.select(same(inputs), same(metadata), eq(RoleType.PREFILL), isNull()))
                .thenReturn(PlacementResult.success(prefill.selection));

        var result = router().select(context, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        try (var route = result.value()) {
            assertSame(inputs, route.ctx().data().getRequest());
            assertEquals(32L, route.seqLen());
            assertEquals(50, route.priority());
        }
        verify(prefillSelector).select(same(inputs), same(metadata), eq(RoleType.PREFILL), isNull());
    }

    @Test
    void firstSelectedGroupChainsToLaterRolesWhenPolicyDidNotForceOne() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestWorkerSelector router = router();
        RequestContext context = context(31L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 31L, "p", 8001, "selected-group");
        SelectionFixture vit = selection(
                RoleType.VIT, 31L, "v", 8002, "selected-group");
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context.data().getRequestId(), RoleType.VIT, "selected-group"))
                .thenReturn(vit.selection);

        PlacementResult<RequestRoute, PlacementKey> admitted = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);
        verify(vitSelector).select(
                context.data().getRequestId(), RoleType.VIT, "selected-group");
        verify(vit.pin).close();
        verify(prefill.pin).close();
    }

    @Test
    void policyGroupRemainsAuthoritativeAcrossEveryRole() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestWorkerSelector router = router();
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        TrafficPolicyConfig groupSelector = new TrafficPolicyConfig();
        var target = new TrafficPolicyConfig.Target();
        target.setGroup("forced");
        groupSelector.setDefaultTargets(List.of(target));
        config.getRouter().setGroupSelector(groupSelector);
        RequestContext context = context(41L, config);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 41L, "p", 8001, "other");
        SelectionFixture vit = selection(
                RoleType.VIT, 41L, "v", 8002, "other");
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, "forced"))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context.data().getRequestId(), RoleType.VIT, "forced"))
                .thenReturn(vit.selection);

        PlacementResult<RequestRoute, PlacementKey> admitted = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, "forced");
        verify(vitSelector).select(context.data().getRequestId(), RoleType.VIT, "forced");
    }

    @Test
    void laterSelectionFailureClosesEveryEarlierExactPinOwner() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestWorkerSelector router = router();
        RequestContext context = context(51L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 51L, "p", 8001, "g1");
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context.data().getRequestId(), RoleType.VIT, "g1"))
                .thenReturn(null);

        PlacementResult<RequestRoute, PlacementKey> blocked = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.VIT, "g1", null),
                blocked.blocker());
        verify(prefill.selection).close();
    }

    @Test
    void decodeMissAfterPrefillSuccessReleasesThePrefillPinOwner() {
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestWorkerSelector router = router();
        RequestContext context = context(52L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 52L, "p", 8001, "g1");
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), "g1"))
                .thenReturn(PlacementResult.blocked(RoleType.DECODE));

        PlacementResult<RequestRoute, PlacementKey> blocked = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.DECODE, "g1", null),
                blocked.blocker());
        verify(prefill.selection).close();
        verify(prefill.selection, never()).assignToRequest();
    }

    @Test
    void mismatchedSelectedRequestFailsClosedAndReleasesPins() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestWorkerSelector router = router();
        RequestContext context = context(61L);
        SelectionFixture foreign = selection(
                RoleType.PREFILL, 999L, "p", 8001, "g1");
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(foreign.selection));

        assertThrows(IllegalStateException.class,
                () -> router.select(context, router.resolvePolicyGroup(context)));

        verify(foreign.selection).close();
        verify(foreign.selection, never()).assignToRequest();
    }

    @Test
    void requiredTopologyIsSnapshottedAtConstruction() {
        List<RoleType> mutable = new ArrayList<>();
        mutable.add(RoleType.PREFILL);
        when(modelMeta.requiredRoles()).thenReturn(mutable);
        RequestWorkerSelector router = router();
        mutable.clear();
        RequestContext context = context(71L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 71L, "p", 8001, "g1");
        when(prefillSelector.select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));

        PlacementResult<RequestRoute, PlacementKey> admitted = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).select(
                context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);
    }

    private RequestScheduler scheduler(RequestWorkerSelector router, RequestContext context) {

        when(configService.loadBalanceConfig()).thenReturn(context.data().getConfig());
        return SchedulerTestSupport.configure(requests, configService.loadBalanceConfig(), router, mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class), mock(org.flexlb.balance.eviction.DecodeCapacityAcquirer.class), new PlacementAvailability());
    }

    private RequestWorkerSelector router() {
        return new RequestWorkerSelector(
                prefillSelector,
                decodeSelector,
                vitSelector,
                modelMeta);
    }

    private void stubQueueSelection(
            RequestContext context,
            RoleType role,
            String group,
            PlacementResult<WorkerAssignment, RoleType> result) {
        switch (role) {
            case PREFILL, PDFUSION -> when(prefillSelector.select(
                    context.data().getRequest(), context.data().getSchedulingMetadata(), role, group)).thenReturn(result);
            case DECODE -> when(decodeSelector.select(
                    context.data().getRequest(), context.data().getSchedulingMetadata(), group)).thenReturn(result);
            case VIT -> when(vitSelector.select(context.data().getRequestId(), role, group))
                    .thenReturn(result.value());
            case FRONTEND -> throw new IllegalArgumentException();
        }
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "PDFUSION"})
    void reportedBlockerResolvesExactRoleAndCapturedCapacityVersion(RoleType prefillRole) {
        var context = context(990L);
        var prefill = selection(prefillRole, 990L, "shared", 8080, "g1");
        var decode = selection(RoleType.DECODE, 990L, "shared", 8080, "g1");
        when(prefill.endpoint().ipPort()).thenReturn("shared:8080");
        when(decode.endpoint().ipPort()).thenReturn("shared:8080");
        when(prefill.selection().placementVersion()).thenReturn(11L);
        when(decode.selection().placementVersion()).thenReturn(23L);
        var prefillEndpoint = (PrefillEndpoint) prefill.endpoint();
        var decodeEndpoint = (DecodeEndpoint) decode.endpoint();
        when(prefillEndpoint.ownership.progress()).thenReturn(new PrefillState.Progress(0L, 0L, 11L, 0, 0, 0L));
        when(decodeEndpoint.placementVersion()).thenReturn(23L);
        try (var route = RequestRoute.prepare(context,
                List.of(prefill.selection(), decode.selection()))) {
            var prefillKey = route.prefillPlacementKey();
            var decodeKey = route.decodePlacementKey();
            assertSame(prefillEndpoint, route.blockedEndpointIfCurrent(prefillKey));
            assertSame(decodeEndpoint, route.blockedEndpointIfCurrent(decodeKey));
            // Changing one role does not invalidate the other role at the same address.
            when(decodeEndpoint.placementVersion()).thenReturn(24L);
            org.junit.jupiter.api.Assertions.assertNull(route.blockedEndpointIfCurrent(decodeKey));
            assertSame(prefillEndpoint, route.blockedEndpointIfCurrent(prefillKey));
            when(prefillEndpoint.ownership.progress()).thenReturn(new PrefillState.Progress(0L, 0L, 12L, 0, 0, 0L));
            org.junit.jupiter.api.Assertions.assertNull(route.blockedEndpointIfCurrent(prefillKey));
            assertThrows(IllegalArgumentException.class, () -> route.blockedEndpointIfCurrent(
                    PlacementKey.exact(RoleType.DECODE, "g1", "other:8080")));
            assertThrows(IllegalArgumentException.class, () -> route.blockedEndpointIfCurrent(
                    PlacementKey.exact(RoleType.DECODE, "other-group", "shared:8080")));
        }
        verify(prefill.pin()).close();
        verify(decode.pin()).close();
    }

    private static RequestContext directContext(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        SchedulingTestConfig.useNonBatchDispatcher(config);
        return context(requestId, config);
    }

    private static RequestContext context(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        return context(requestId, config);
    }

    private static RequestContext context(long requestId, FlexlbConfig config) {
        Request request = Request.builder()
                .requestId(requestId)
                .seqLen(32L)
                .maxNewTokens(16).build();
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    private static SelectionFixture selection(
            RoleType role,
            long requestId,
            String ip,
            int httpPort,
            String group) {
        WorkerAssignment selection = mock(WorkerAssignment.class);
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        WorkerEndpoint endpoint = switch (role) {
            case PREFILL, PDFUSION -> mock(PrefillEndpoint.class);
            case DECODE -> RequestProtocolTestSupport.decodeEndpoint();
            default -> mock(WorkerEndpoint.class);
        };
        if (endpoint instanceof PrefillEndpoint prefill) {
            var ledger = mock(PrefillState.class);
            when(ledger.progress()).thenReturn(new PrefillState.Progress(0L, 0L, 0L, 0, 0, 0L));
            org.springframework.test.util.ReflectionTestUtils.setField(prefill, "ownership", ledger);
        }
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setRequestId(requestId);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGroup(group);
        when(selection.serverStatus()).thenAnswer(ignored -> ServerStatus.copyOf(status));
        when(selection.requestId()).thenReturn(requestId);
        when(selection.role()).thenReturn(role);
        when(selection.group()).thenReturn(group);
        when(selection.prefillWorkMs()).thenReturn(1L);
        when(selection.generationPin()).thenReturn(pin);
        when(selection.endpoint()).thenReturn(endpoint);
        var selectionOpen = new java.util.concurrent.atomic.AtomicBoolean(true);
        org.mockito.Mockito.doAnswer(call -> {
            if (selectionOpen.compareAndSet(true, false)) { pin.close(); }
            return null;
        }).when(selection).close();
        when(pin.endpoint()).thenReturn(endpoint);
        return new SelectionFixture(selection, pin, endpoint, status);
    }

    private record SelectionFixture(WorkerAssignment selection, WorkerEndpoint.GenerationPin pin, WorkerEndpoint endpoint, ServerStatus status) {
    }
}
