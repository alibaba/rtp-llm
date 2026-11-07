package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.DecodeSelectionStrategy;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.balance.strategy.VitWorkerSelector;
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
import org.flexlb.service.TheoryCacheHitReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

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
 * Final selector/Worker selection ownership contracts for {@link RequestRouteSelector}.
 */
class RequestRouteSelectorTest {

    private Delivery.Member lastDelivery;

    private PrefillSelectionStrategy prefillSelector;
    private org.flexlb.balance.endpoint.EndpointRegistry endpoints;

    private DecodeSelectionStrategy decodeSelector;

    private VitWorkerSelector vitSelector;

    private ConfigService configService;

    private ModelMetaConfig modelMeta;

    private AbstractRequestScheduler requests;

    @BeforeEach
    void setUp() {
        endpoints = mock(org.flexlb.balance.endpoint.EndpointRegistry.class);
        when(endpoints.endpointAddressSnapshot(RoleType.VIT)).thenReturn(List.of("vit:8080"));
        prefillSelector = mock(PrefillSelectionStrategy.class);
        decodeSelector = mock(DecodeSelectionStrategy.class);
        vitSelector = mock(VitWorkerSelector.class);
        configService = mock(ConfigService.class);
        modelMeta = mock(ModelMetaConfig.class);
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        when(configService.loadBalanceConfig()).thenReturn(config);
        var runtime =
                SchedulerTestSupport.runtime(
                        SchedulerTestSupport.create(
                                configService,
                                mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                                mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                                mock(org.flexlb.service.TheoryCacheHitReporter.class)));
        var algorithm =
                org.mockito.Mockito.mock(
                        DirectRequestScheduler.class,
                        org.mockito.Mockito.withSettings()
                                .useConstructor(
                                        SchedulerTestServices.constructorArguments(
                                                mock(RequestRouteSelector.class),
                                                runtime.newLifecycle(
                                                        config,
                                                        SchedulingTestConfig.capacity(config),
                                                        org.mockito.Mockito.mock(
                                                                org.flexlb.service
                                                                        .TheoryCacheHitReporter
                                                                        .class))))
                                .defaultAnswer(
                                        invocation -> {
                                            if (AbstractRequestScheduler.class.isAssignableFrom(
                                                    invocation.getMethod().getDeclaringClass())) {
                                                return invocation.callRealMethod();
                                            }
                                            return org.mockito.Answers.RETURNS_DEFAULTS.answer(
                                                    invocation);
                                        }));
        requests = algorithm;
        org.mockito.Mockito.doAnswer(
                        call ->
                                RequestProtocolTestSupport.admissionHandle(
                                        call.getArgument(0, RequestContext.class)))
                .when(requests)
                .claimPlacement(any(RequestContext.class));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var context = call.getArgument(0, RequestContext.class);
                            var future = new CompletableFuture<Response>();
                            context.data().setFuture(future);
                            SchedulerTestSupport.bindOwner(context, requests);
                            requests.onRequest(context);
                            return future;
                        })
                .when(requests)
                .submit(any(), any(Runnable.class));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var route = call.getArgument(0, RouteExecution.class);
                            var handle = RequestProtocolTestSupport.admissionHandle(route.ctx());
                            route.ctx()
                                    .bindPlacement(
                                            handle,
                                            route,
                                            ContextResult.PublicationResult.PREPARED);
                            return handle;
                        })
                .when(requests)
                .beginRoute(any());
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var claim = call.getArgument(0, Delivery.Member.class);
                            var item = claim.route();
                            var prepared =
                                    DeliveryTestSupport.prepared(
                                            claim, Delivery.SendMode.ROUTE_DECISION, 0L);
                            lastDelivery = claim;
                            doAnswer(
                                            inv -> {
                                                item.ctx()
                                                        .data()
                                                        .getFuture()
                                                        .complete(item.successResponse(false));
                                                return null;
                                            })
                                    .when(requests)
                                    .onDeliveryResult(any(ContextResult.RouteDelivered.class));
                            return prepared;
                        })
                .when(requests)
                .prepareDelivery(
                        any(Delivery.Member.class), eq(Delivery.SendMode.ROUTE_DECISION), eq(0L));
        org.mockito.Mockito.doAnswer(
                        call ->
                                call.getArgument(0, RequestContext.class)
                                        .data()
                                        .getFuture()
                                        .complete(call.getArgument(1)))
                .when(requests)
                .publish(any(), any(), eq(false));
    }

    @Test
    void invalidRequestFailsBeforePolicyOrEndpointSelection() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestRouteSelector router = router();
        org.mockito.Mockito.clearInvocations(configService);

        PlacementResult<RouteResult, PlacementKey> result =
                router.select(new RequestContext(SchedulingTestConfig.newConfig()), null);

        assertEquals(PlacementResult.Status.REJECTED, result.status());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), result.failure().getCode());
        verify(configService, never()).loadBalanceConfig();
        verifyNoInteractions(prefillSelector, decodeSelector, vitSelector);
    }

    @ParameterizedTest
    @CsvSource({"false,false", "true,false", "false,true"})
    void directRouteCommitsRolesAndReleasesSelectedWorkers(
            boolean pinCloseFails, boolean materializationFails) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestRouteSelector router = router();
        RequestContext context = directContext(7L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 7L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 7L, "d", 8002, "g1");
        PrefillEndpoint.RequestRecord registration = mock(PrefillEndpoint.RequestRecord.class);
        DecodeResources.RequestReservation reservation =
                new DecodeResources.RequestReservation(1L, String.valueOf(7L), 2L);
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));
        when(decodeSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any()))
                .thenAnswer(call -> selected(call, decode));
        var capture = stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
        when(((PrefillEndpoint) prefill.endpoint)
                        .reserveUnqueuedRoute(any(RouteExecution.class), eq(1L)))
                .thenReturn(CapacityBoundary.Attempt.accepted(registration));
        when(((DecodeEndpoint) decode.endpoint)
                        .tryReserveQueuedRequest(eq("7"), eq(32L), eq(48L), eq(50), isNull()))
                .thenReturn(
                        new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED,
                                reservation));
        var permit = stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);
        if (pinCloseFails) {
            doThrow(new IllegalStateException("selectedEndpoint close failed after direct handoff"))
                    .when(prefill.selectedEndpoint)
                    .close();
        }

        if (materializationFails) {
            when(capture.calculate())
                    .thenThrow(new IllegalStateException("snapshot materialization failed"));
        }
        Response response =
                SchedulerTestServices.entry(scheduler(router, context)).submit(context).join();
        if (materializationFails) {
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), response.getCode());
            verify(permit, never()).dispatch();
            verify(permit).release();
            verify(requests).publish(eq(context), any(), eq(false));
            verify(prefill.selectedEndpoint, never()).close();
            verify(decode.selectedEndpoint, never()).close();
            verify(requests, never()).onDeliveryResult(any(ContextResult.RouteDelivered.class));
            return;
        }

        assertTrue(response.isSuccess());
        assertEquals(List.of(prefill.status, decode.status), response.getServerStatus());
        verify((PrefillEndpoint) prefill.endpoint, never())
                .releaseRequest(any(RouteExecution.class));
        verify(permit).dispatch();
        verify((DecodeEndpoint) decode.endpoint, never())
                .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(prefill.selectedEndpoint, never()).close();
        verify(decode.selectedEndpoint, never()).close();
        var order =
                org.mockito.Mockito.inOrder(
                        decode.selectedEndpoint, prefill.selectedEndpoint, requests);
        order.verify(decode.selectedEndpoint, never()).close();
        order.verify(prefill.selectedEndpoint, never()).close();
        order.verify(requests).onDeliveryResult(any(ContextResult.RouteDelivered.class));
    }

    @Test
    void directRouteRollsBackDecodeWhenPrefillAdmissionIsFull() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestRouteSelector router = router();
        RequestContext context = directContext(8L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 8L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 8L, "d", 8002, "g1");
        DecodeResources.RequestReservation reservation =
                new DecodeResources.RequestReservation(1L, String.valueOf(8L), 2L);
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));
        when(decodeSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any()))
                .thenAnswer(call -> selected(call, decode));
        stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
        when(((DecodeEndpoint) decode.endpoint)
                        .tryReserveQueuedRequest(eq("8"), eq(32L), eq(48L), eq(50), isNull()))
                .thenReturn(
                        new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED,
                                reservation));
        stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);
        when(((PrefillEndpoint) prefill.endpoint)
                        .reserveUnqueuedRoute(any(RouteExecution.class), eq(1L)))
                .thenReturn(
                        CapacityBoundary.Attempt.rejected(
                                CapacityBoundary.unavailable(
                                        CapacityBoundary.Resource.PREFILL_REQUEST)));

        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                SchedulerTestServices.entry(scheduler(router, context))
                        .submit(context)
                        .join()
                        .getCode());
        verify(prefillSelector)
                .select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class));
        verify((DecodeEndpoint) decode.endpoint)
                .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(prefill.selectedEndpoint, never()).close();
        verify(decode.selectedEndpoint, never()).close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directAdmissionPassesZeroAndUnknownTimeToLifecycle(boolean unknown) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestContext context = directContext(81L);
        SelectionFixture selected = selection(RoleType.PREFILL, 81L, "p", 8001, "g1");
        selected.status.setPrefillTime(0L);
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, selected));
        PrefillEndpoint endpoint = (PrefillEndpoint) selected.endpoint;
        long committedAtMs = System.currentTimeMillis();
        stubRouteCommit(
                endpoint,
                new org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot(
                        committedAtMs, List.of(), List.of(), unknown ? 1L : 0L));
        var registration = mock(PrefillEndpoint.RequestRecord.class);
        when(endpoint.reserveUnqueuedRoute(any(), eq(0L)))
                .thenReturn(CapacityBoundary.Attempt.accepted(registration));

        assertTrue(
                SchedulerTestServices.entry(scheduler(router(), context))
                        .submit(context)
                        .join()
                        .isSuccess());

        var evidence = org.mockito.ArgumentCaptor.forClass(ContextResult.DeliveryEvidence.class);
        verify(requests).onDeliveryResult(evidence.capture());
        assertTrue(evidence.getValue() instanceof ContextResult.RouteDelivered);
        var routeDelivered = (ContextResult.RouteDelivered) evidence.getValue();
        assertEquals(
                unknown ? 0L : routeDelivered.nowMs() + 10_000L, routeDelivered.predictedAtMs());
        verify(endpoint, never()).releaseRequest(any(RouteExecution.class));
    }

    @Test
    void directRequestRemainsInCanonicalLifecycleAfterResponseAndReconcilesEarlyDecodeEvidence()
            throws Exception {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestContext context = directContext(9L);
        when(configService.loadBalanceConfig()).thenReturn(context.data().getConfig());
        requests =
                SchedulerTestSupport.create(
                        configService,
                        mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        try {
            SelectionFixture prefill = selection(RoleType.PREFILL, 9L, "p", 8001, "g1");
            SelectionFixture decode = selection(RoleType.DECODE, 9L, "d", 8002, "g1");
            var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(9L), 2L);
            when(prefillSelector.select(
                            org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                            org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                            org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                            org.mockito.ArgumentMatchers.eq(null),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(
                                    java.util.function.LongSupplier.class)))
                    .thenAnswer(call -> selected(call, prefill));
            when(decodeSelector.select(
                            org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.eq("g1"),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any()))
                    .thenAnswer(call -> selected(call, decode));
            stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
            when(((PrefillEndpoint) prefill.endpoint)
                            .reserveUnqueuedRoute(any(RouteExecution.class), eq(1L)))
                    .thenReturn(
                            CapacityBoundary.Attempt.accepted(
                                    mock(PrefillEndpoint.RequestRecord.class)));
            when(((DecodeEndpoint) decode.endpoint)
                            .tryReserveQueuedRequest(eq("9"), eq(32L), eq(48L), eq(50), isNull()))
                    .thenReturn(
                            new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                    org.flexlb.balance.endpoint.DecodeResources.ReserveStatus
                                            .ACQUIRED,
                                    reservation));
            stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);
            when(((DecodeEndpoint) decode.endpoint).ownership(reservation))
                    .thenReturn(org.flexlb.balance.endpoint.DecodeResources.Ownership.CONFIRMED);
            assertTrue(
                    SchedulerTestServices.entry(scheduler(router(), context))
                            .submit(context)
                            .get(2L, TimeUnit.SECONDS)
                            .isSuccess());
            assertTrue(context.data().getFuture().get(2L, TimeUnit.SECONDS).isSuccess());
            RequestContext requestContext = requests.requests.findActive("9");
            synchronized (requestContext) {
                assertTrue(requestContext.snapshot().engine().decodeAccepted());
                assertTrue(requestContext.snapshot().phase() == ContextResult.Phase.OPEN);
                assertEquals(
                        RequestState.Phase.ACKNOWLEDGED,
                        requests.requests
                                .getRequestState(requestContext.data().getRequestId(), 0L)
                                .state());
            }
            RequestProtocolTestSupport.expireInactiveRequest(
                    requests,
                    requestContext,
                    System.currentTimeMillis()
                            + context.data()
                                    .getConfig()
                                    .getRequestLifecycle()
                                    .getRequest()
                                    .getTimeoutMs());
            assertEquals(
                    RequestState.Phase.TIMED_OUT,
                    SchedulerTestSupport.repository(requests).getRequestState("9", 0L).state());
            assertEquals(0, SchedulerTestSupport.repository(requests).liveRequestCount());
            verify((DecodeEndpoint) decode.endpoint)
                    .release(reservation, DecodeResources.ReleaseReason.EXPIRED);
            verify((PrefillEndpoint) prefill.endpoint).releaseRequest(any(RouteExecution.class));
        } finally {
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(requests);
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(requests)).close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(requests));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directSelectionExceptionPublishesFailureAndClosesLifecycle(boolean fatal)
            throws Exception {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestContext context = directContext(10L);
        when(configService.loadBalanceConfig()).thenReturn(context.data().getConfig());
        requests =
                SchedulerTestSupport.create(
                        configService,
                        mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        try {
            Throwable failure =
                    fatal
                            ? new AssertionError("selection failed")
                            : new IllegalStateException("selection failed");
            when(prefillSelector.select(
                            org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                            org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                            org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                            org.mockito.ArgumentMatchers.eq(null),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(
                                    java.util.function.LongSupplier.class)))
                    .thenThrow(failure);
            var returned =
                    SchedulerTestServices.entry(scheduler(router(), context)).submit(context);
            assertEquals(context.data().getFuture(), returned);
            Response response = context.data().getFuture().get(2L, TimeUnit.SECONDS);
            assertEquals(8510, response.getCode());
            assertEquals("DISPATCH_FAILED", response.getErrorMessage());
            assertEquals(
                    RequestState.Phase.FAILED,
                    SchedulerTestSupport.repository(requests).getRequestState("10", 0L).state());
            SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(requests));
            if (fatal)
                assertSame(
                        failure,
                        SchedulerTestServices.failure(SchedulerTestSupport.runtime(requests)));
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
        var request =
                org.flexlb.dao.loadbalance.Request.builder().seqLen(32L).maxNewTokens(16).build();
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
    @CsvSource({
        "32,16,32,48",
        "500,10000,500,10500",
        "9223372036854775806,100,9223372036854775806,9223372036854775807",
        "-1,100,0,100",
        "500,-1,500,500"
    })
    void reservationAndDispatchUseImmutableDemandAndSharedStartupCapacity(
            long prompt, int output, long hardKv, long expectedKv) {
        var config = SchedulingTestConfig.newConfig();
        var limits = config.getRouter().getRoles().getDecode().getAvailability();
        limits.setMaxEngineRequests(1L);
        limits.setMaxKvUsagePercent(90L);
        var estimator = config.getRouter().getRoles().getDecode().getCostEstimator();
        estimator.setExpression("2 * running_size / max_running_size + 3 * kvcache_used_ratio");
        var capacity = SchedulingTestConfig.capacity(config);
        var selector =
                new org.flexlb.balance.strategy.DecodeSelector(
                        SchedulingTestConfig.configService(config), capacity);
        var formula =
                (org.flexlb.balance.prediction.DecodeCostFormula)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                selector, "costFormula");
        var context = RequestProtocolTestSupport.context(config, 701L);
        context.data()
                .setRequest(
                        context.data().getRequest().toBuilder()
                                .seqLen(prompt)
                                .maxNewTokens(output)
                                .build());
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(73, Long.MAX_VALUE));
        var scheduler = queueScheduler(context);
        org.springframework.test.util.ReflectionTestUtils.setField(
                scheduler.dispatcher, "decodeAdmissionCapacity", capacity);
        var prefill = selection(RoleType.PREFILL, 701L, "10.0.0.1", 8080, "g1");
        var selectedDecode = selection(RoleType.DECODE, 701L, "10.0.0.2", 8080, "g1");
        var decode = (DecodeEndpoint) selectedDecode.endpoint();
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(701L), 1L);
        when(decode.tryReserveQueuedRequest(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        eq(capacity)))
                .thenReturn(
                        new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED,
                                reservation));
        when(decode.acquireDispatchPermit(reservation, capacity))
                .thenReturn(
                        new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                                null));
        limits.setMaxEngineRequests(99L);
        limits.setMaxKvUsagePercent(1L);
        estimator.setExpression("7 * running_size / max_running_size + 11 * kvcache_used_ratio");
        assertNotSame(estimator.compiledFormula(), formula);
        assertEquals(
                "2 * running_size / max_running_size + 3 * kvcache_used_ratio",
                formula.expression());
        when(((PrefillEndpoint) prefill.endpoint()).enqueueRoute(any(), any()))
                .thenAnswer(
                        call ->
                                RequestProtocolTestSupport.recordPrefillReservation(
                                        call.getArgument(0)));
        {
            var route = selectedRoute(context, prefill, selectedDecode);
            try (var plan = RequestProtocolTestSupport.plan(context, route)) {
                var placement = SchedulerTestSupport.enqueueRoute(scheduler, plan);
                assertEquals(PlacementResult.Status.SUCCESS, placement.status());
                var item = placement.value();
                assertSame(reservation, item.decodeReservation());
                {
                    assertSame(context.data().getRequest(), item.ctx().data().getRequest());
                    assertSame(reservation, item.decodeReservation());
                    assertSame(decode, item.decodeEndpoint());
                    assertEquals(hardKv, item.seqLen());
                    var delivery = Delivery.Member.prepare(item, capacity);
                    assertFalse(delivery.accepted());
                    assertTrue(delivery.boundary().unavailable());
                    assertFalse(
                            item.decodeEndpoint()
                                    .shouldRetryDispatch(item.decodeReservation(), capacity));
                    verify(decode).shouldRetryDispatch(reservation, capacity);
                    verify(decode).acquireDispatchPermit(reservation, capacity);
                    verify(decode)
                            .tryReserveQueuedRequest(
                                    eq("701"), eq(hardKv), eq(expectedKv), eq(73), eq(capacity));
                    verify(decode, never())
                            .tryReserveQueuedRequest(
                                    org.mockito.ArgumentMatchers.anyString(),
                                    anyLong(),
                                    anyLong(),
                                    anyInt(),
                                    isNull());
                }
            }
        }
        // Published queue membership now owns the reservation; closing the selection cannot release
        // it.
        verify(decode, never()).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedDecodeAdoptionReleasesOnceAndPreservesOriginalFailure(boolean adoptionThrows) {
        var context = context(702L);
        var prefill = selection(RoleType.PREFILL, 702L, "p", 8001, "g1");
        var selectedDecode = selection(RoleType.DECODE, 702L, "d", 8002, "g1");
        var decode = (DecodeEndpoint) selectedDecode.endpoint();
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(702L), 1L);
        var adoptionFailure = new IllegalStateException("adoption failed");
        var cleanupFailure = new IllegalStateException("cleanup failed");
        if (adoptionThrows) {
            when(decode.markQueued(reservation)).thenThrow(adoptionFailure);
        }
        doThrow(cleanupFailure)
                .when(decode)
                .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);

        {
            var route = selectedRoute(context, prefill, selectedDecode);
            var actual =
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    RequestProtocolTestSupport.adopt(
                                            queueScheduler(context),
                                            RequestProtocolTestSupport.plan(context, route),
                                            new org.flexlb.balance.eviction.DecodePreemption
                                                    .PreemptionResult(reservation, false, "test")));
            assertSame(adoptionThrows ? adoptionFailure : cleanupFailure, actual);
            assertEquals(
                    adoptionThrows ? List.of(cleanupFailure) : List.of(),
                    List.of(actual.getSuppressed()));
        }
        verify(decode).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(prefill.selectedEndpoint(), never()).close();
        verify(selectedDecode.selectedEndpoint(), never()).close();
    }

    @Test
    void closingRouteDoesNotRetireEndpointOrReleaseCommittedCapacity() {
        var context = context(703L);
        var prefill = selection(RoleType.PREFILL, 703L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 703L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(703L), 1L);
        var cleanupFailure =
                new IllegalStateException("selectedEndpoint cleanup failed after publication");
        var route = selectedRoute(context, prefill, decode);
        var scheduler = queueScheduler(context);
        var plan = RequestProtocolTestSupport.plan(context, route);
        when(endpoint.markQueued(reservation))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED,
                                false));
        assertTrue(
                RequestProtocolTestSupport.adopt(
                        scheduler,
                        plan,
                        new org.flexlb.balance.eviction.DecodePreemption.PreemptionResult(
                                reservation, false, "test")));
        when(route.prefillEndpoint().enqueueRoute(any(), any()))
                .thenAnswer(
                        call ->
                                RequestProtocolTestSupport.recordPrefillReservation(
                                        call.getArgument(0)));
        doThrow(cleanupFailure).when(prefill.selectedEndpoint()).close();
        assertEquals(
                PlacementResult.Status.SUCCESS,
                SchedulerTestSupport.enqueueRoute(scheduler, plan).status());
        verify(prefill.selectedEndpoint(), never()).close();
        verify(decode.selectedEndpoint(), never()).close();
        plan.close();
        org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                context.data().getRequestScheduler().tasks::throwIfFailed);
        plan.close();
        verify(prefill.selectedEndpoint(), never()).close();
        verify(decode.selectedEndpoint(), never()).close();
        verify(endpoint, never())
                .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
    }

    @Test
    void attachingCapacityDoesNotAssignRequestFifoIdentity() {
        var context = context(708L);
        var prefill = selection(RoleType.PREFILL, 708L, "p", 8001, "g1");
        {
            var selected = selectedRoute(context, prefill);
            var route = selected.attachReservation(null);
            assertSame(context, route.ctx());
            assertEquals(0L, context.data().getWorkerEnqueueSequence());
            assertEquals(0L, context.data().getFirstWorkerEnqueueTime());
        }
    }

    @Test
    void callerAndDeliveredResponseChangesCannotMutateTheRouteTemplate() {
        var context = context(707L);
        var prefill = selection(RoleType.PREFILL, 707L, "p", 8001, "g1");
        {
            var route = selectedRoute(context, prefill);
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
        verify(prefill.selectedEndpoint(), never()).close();
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void unattachedDecodeReservationIsReleased(boolean direct, boolean cleanupFails) {
        var context = direct ? directContext(703L) : context(703L);
        SchedulerTestSupport.bindOwner(context, requests);
        var prefill = selection(RoleType.PREFILL, 703L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 703L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(703L), 1L);
        when(endpoint.tryReserveQueuedRequest(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        any()))
                .thenReturn(
                        new DecodeResources.Acquisition<>(
                                DecodeResources.ReserveStatus.ACQUIRED, reservation));
        var primary = new IllegalStateException("route rejected resource attachment");
        var cleanup = new IllegalStateException("resource rollback failed");
        if (cleanupFails) {
            doThrow(cleanup)
                    .when(endpoint)
                    .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        } else {
            when(endpoint.release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK))
                    .thenReturn(
                            DecodeResources.ResourceChange.of(
                                    DecodeResources.ReservationReleaseResult.RELEASED, true));
        }
        {
            var route = org.mockito.Mockito.spy(selectedRoute(context, prefill, decode));
            doThrow(primary).when(route).attachReservation(reservation);
            var actual =
                    assertThrows(
                            IllegalStateException.class,
                            () -> {
                                if (direct) requests.commitDirectRoute(route);
                                else
                                    requests.enqueueRoute(
                                            route,
                                            QueueExecutionSettings.capture(
                                                    context.data().getConfig(),
                                                    org.flexlb.balance.planner.BuiltinGroupPlanner
                                                            .from(context.data().getConfig())));
                            });
            assertSame(primary, actual);
            assertEquals(
                    cleanupFails ? List.of(cleanup) : List.of(), List.of(actual.getSuppressed()));
            org.junit.jupiter.api.Assertions.assertNull(route.decodeReservation());
            verify(requests, never()).beginRoute(any());
            verify(endpoint).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
        verify(prefill.selectedEndpoint(), never()).close();
        verify(decode.selectedEndpoint(), never()).close();
    }

    @Test
    void invalidResultConstructionLeavesPinOwnershipWithTheSelector() {
        var context = context(706L);
        var prefill = selection(RoleType.PREFILL, 706L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 706L, "d", 8002, "g1");
        decode.status().setRequestId(String.valueOf(999L));
        assertThrows(IllegalStateException.class, () -> selectedRoute(context, prefill, decode));
        verify(prefill.selectedEndpoint(), never()).close();
        verify(decode.selectedEndpoint(), never()).close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void routeConstructionFailureRollsBackCapacityAndClosesWorkers(boolean direct) {
        var context = org.mockito.Mockito.spy(direct ? directContext(704L) : context(704L));
        var prefill = selection(RoleType.PREFILL, 704L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 704L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(704L), 1L);
        stubDecodePermit(endpoint, reservation);
        stubPrefillCommit((PrefillEndpoint) prefill.endpoint());
        when(endpoint.tryReserveQueuedRequest(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        any()))
                .thenReturn(
                        new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                org.flexlb.balance.endpoint.DecodeResources.ReserveStatus.ACQUIRED,
                                reservation));
        var primary = new IllegalStateException("route construction failed");
        var cleanup = new IllegalStateException("rollback failed");

        doThrow(cleanup)
                .when(endpoint)
                .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        var routing = selectedRoute(context, prefill, decode);
        var actual =
                assertThrows(
                        IllegalStateException.class,
                        () -> {
                            if (direct) {
                                SchedulerTestSupport.bindOwner(context, requests);
                                doThrow(primary).when(requests).beginRoute(any());
                                var admission = RequestProtocolTestSupport.admissionHandle(context);
                                org.mockito.Mockito.doNothing()
                                        .when(
                                                RequestProtocolTestSupport.placementScheduler(
                                                        admission))
                                        .completePlacement(
                                                org.mockito.ArgumentMatchers.same(admission),
                                                org.mockito.ArgumentMatchers.eq(
                                                        ContextResult.PublicationResult.FAILED),
                                                any());
                                requests.publishPlacement(routing, admission, null);
                            } else {
                                var queued = queueScheduler(context);
                                doThrow(primary)
                                        .when(SchedulerTestSupport.lifecycle(queued))
                                        .beginRoute(any());
                                try (var plan = RequestProtocolTestSupport.plan(context, routing)) {
                                    // This selector fixture owns no Lifecycle admission counter.
                                    var admission = context.snapshot().placement().operation();
                                    org.mockito.Mockito.doNothing()
                                            .when(
                                                    RequestProtocolTestSupport.placementScheduler(
                                                            admission))
                                            .completePlacement(
                                                    org.mockito.ArgumentMatchers.same(admission),
                                                    org.mockito.ArgumentMatchers.eq(
                                                            ContextResult.PublicationResult.FAILED),
                                                    any());
                                    SchedulerTestSupport.enqueueRoute(queued, plan);
                                }
                            }
                        });
        assertSame(primary, actual);
        if (direct) {
            assertEquals(List.of(cleanup), List.of(actual.getSuppressed()));
        } else {
            assertSame(
                    cleanup,
                    assertThrows(
                            IllegalStateException.class,
                            context.data().getRequestScheduler().tasks::throwIfFailed));
        }
        verify(endpoint).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(prefill.selectedEndpoint(), never()).close();
        verify(decode.selectedEndpoint(), never()).close();
    }

    @Test
    void conflictingPreemptionResultDoesNotOverwriteOrReleaseExistingCapacityTwice() {
        var context = context(705L);
        var prefill = selection(RoleType.PREFILL, 705L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 705L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var existing = new DecodeResources.RequestReservation(1L, String.valueOf(705L), 1L);
        var incoming = new DecodeResources.RequestReservation(1L, String.valueOf(705L), 2L);
        when(endpoint.markQueued(existing))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED,
                                false));
        var scheduler = queueScheduler(context);
        var plan =
                RequestProtocolTestSupport.plan(context, selectedRoute(context, prefill, decode));
        try (plan) {
            assertTrue(
                    RequestProtocolTestSupport.adopt(
                            scheduler,
                            plan,
                            new org.flexlb.balance.eviction.DecodePreemption.PreemptionResult(
                                    existing, false, "first")));
            assertFalse(
                    RequestProtocolTestSupport.adopt(
                            scheduler,
                            plan,
                            new org.flexlb.balance.eviction.DecodePreemption.PreemptionResult(
                                    incoming, false, "conflict")));
            assertEquals(
                    existing, RequestProtocolTestSupport.planRouting(plan).decodeReservation());
            assertFalse(plan.accept(endpoint, incoming));
            verify(endpoint, never())
                    .release(existing, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
        plan.close();
        verify(endpoint).release(incoming, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(endpoint).release(existing, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        verify(endpoint, never()).markQueued(incoming);
        verify(prefill.selectedEndpoint(), never()).close();
        verify(decode.selectedEndpoint(), never()).close();
    }

    private QueuedRequestScheduler queueScheduler(RequestContext context) {

        var config = SchedulingTestConfig.newConfig();
        var scheduler =
                mock(
                        QueuedRequestScheduler.class,
                        org.mockito.Mockito.withSettings()
                                .useConstructor(
                                        SchedulerTestServices.constructorArguments(
                                                config,
                                                mock(RequestRouteSelector.class),
                                                mock(
                                                        org.flexlb.balance.eviction.DecodePreemption
                                                                .class),
                                                SchedulerTestSupport.runtime(requests)
                                                        .newLifecycle(
                                                                config,
                                                                SchedulingTestConfig.capacity(
                                                                        config),
                                                                org.mockito.Mockito.mock(
                                                                        org.flexlb.service
                                                                                .TheoryCacheHitReporter
                                                                                .class)),
                                                new PlacementAvailability(),
                                                new OrderedRequestQueue(
                                                        config.isPriorityOrdering()),
                                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                        config)))
                                .defaultAnswer(
                                        invocation ->
                                                AbstractRequestScheduler.class.isAssignableFrom(
                                                                invocation
                                                                        .getMethod()
                                                                        .getDeclaringClass())
                                                        ? invocation.callRealMethod()
                                                        : org.mockito.Answers.RETURNS_DEFAULTS
                                                                .answer(invocation)));
        var lifecycle = SchedulerTestSupport.spyLifecycle(scheduler);
        org.springframework.test.util.ReflectionTestUtils.setField(
                context.data(), "requestScheduler", lifecycle);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var route = call.getArgument(0, RouteExecution.class);
                            var handle = RequestProtocolTestSupport.admissionHandle(route.ctx());
                            route.ctx()
                                    .bindPlacement(
                                            handle,
                                            route,
                                            ContextResult.PublicationResult.PREPARED);
                            return handle;
                        })
                .when(lifecycle)
                .beginRoute(any());
        return scheduler;
    }

    private static org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot
            stubPrefillCommit(PrefillEndpoint prefill) {
        return stubRouteCommit(
                prefill,
                new org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot(
                        System.currentTimeMillis(), List.of(), List.of(), 0L));
    }

    private static org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot stubRouteCommit(
            PrefillEndpoint endpoint,
            org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot precedingWork) {
        var capture = mock(org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot.class);
        when(capture.calculate()).thenReturn(precedingWork);
        when(endpoint.commitDirect(any(), any()))
                .thenAnswer(
                        call -> {
                            RouteExecution route = call.getArgument(0);
                            PrefillEndpoint.RequestRecord reservation = call.getArgument(1);
                            when(reservation.belongsTo(route)).thenReturn(true);
                            route.attachPrefillReservation(reservation);
                            return capture;
                        });
        return capture;
    }

    private static DecodeEndpoint.EngineDispatchPermit stubDecodePermit(
            DecodeEndpoint endpoint, DecodeResources.RequestReservation reservation) {
        var permit = mock(DecodeEndpoint.EngineDispatchPermit.class);
        when(endpoint.acquireDispatchPermit(eq(reservation), any()))
                .thenReturn(
                        new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                                permit));
        when(permit.belongsTo(endpoint, reservation)).thenReturn(true);
        when(permit.dispatch())
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                                false));
        return permit;
    }

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "DECODE", "PDFUSION", "VIT"})
    void missingRequiredRoleReturnsItsExactWaitDomain(RoleType role) {
        if (role == RoleType.VIT)
            when(endpoints.endpointAddressSnapshot(role)).thenReturn(List.of());
        when(modelMeta.requiredRoles()).thenReturn(List.of(role));
        RequestRouteSelector router = router();
        RequestContext context = context(11L);
        stubQueueSelection(context, role, null, PlacementResult.blocked(role));

        PlacementResult<RouteExecution, PlacementKey> blocked =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(role, null, null), blocked.blocker());
    }

    @Test
    void invalidVitChoiceCannotAcquireAnEndpoint() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.VIT));
        when(vitSelector.selectIndex(any())).thenReturn(1);
        assertThrows(IllegalArgumentException.class, () -> router().select(context(109L), null));
        verify(endpoints, never()).selectEndpoint(any(), any());
    }

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION", "DECODE", "VIT"})
    void retirementBetweenEndpointLookupAndAddReturnsEndpointChanged(RoleType role) {
        var context = context(113L);
        var prefillRole = role == RoleType.PDFUSION ? role : RoleType.PREFILL;
        when(modelMeta.requiredRoles())
                .thenReturn(role.supportsPrefill() ? List.of(role) : List.of(prefillRole, role));
        var prefill = selection(prefillRole, 113L, "p", 8001, "g1");
        when(prefillSelector.select(
                        any(),
                        any(),
                        eq(prefillRole),
                        any(),
                        org.mockito.ArgumentMatchers.anyList(),
                        any()))
                .thenAnswer(call -> selected(call, prefill));
        var chosen = role.supportsPrefill() ? prefill : selection(role, 113L, "other", 8002, "g1");
        if (role == RoleType.DECODE) {
            var decode = (DecodeEndpoint) chosen.endpoint();
            var view = decode.routingView();
            when(decodeSelector.select(
                            any(), any(), any(), org.mockito.ArgumentMatchers.anyList(), any()))
                    .thenAnswer(call -> PlacementResult.success(decode.routingView()));
            doAnswer(
                            call -> {
                                when(chosen.endpoint().isRetired()).thenReturn(true);
                                return chosen.endpoint();
                            })
                    .when(endpoints)
                    .selectDecodeEndpoint(view);
        } else {
            doAnswer(
                            call -> {
                                when(chosen.endpoint().isRetired()).thenReturn(true);
                                return chosen.endpoint();
                            })
                    .when(endpoints)
                    .selectEndpoint(eq(role), any());
        }
        var result = RouteExecution.fromSelection(context, router().select(context, null));
        assertEquals(PlacementResult.Status.ENDPOINT_CHANGED, result.status());
        assertEquals(role, result.blocker().role());
        verify(chosen.endpoint(), never()).close();
    }

    @Test
    void customDecodeFailureKeepsItsReasonLocationAndDiagnostics() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.DECODE));
        var context = context(110L);
        var failure =
                PlacementResult.<DecodeResources.DecodeRoutingView, PlacementKey>blocked(
                        PlacementKey.exact(RoleType.DECODE, "chosen-group", "worker:8000"),
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
                        java.util.Map.of("custom_reason", "reserved for another batch"));
        decodeSelector =
                (request, metadata, group, candidates, capacity) -> {
                    assertThrows(UnsupportedOperationException.class, candidates::clear);
                    return failure;
                };
        var router = router();
        assertSame(failure, RouteExecution.fromSelection(context, router.select(context, null)));
        verify(endpoints, never()).selectDecodeEndpoint(any());
        verifyNoInteractions(prefillSelector, vitSelector);
    }

    @Test
    void decodeGenerationChangeRetainsSuccessfulSelectionWithoutSelectingAgain() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.DECODE));
        var context = context(112L);
        var fixture = selection(RoleType.DECODE, 112L, "d", 8002, "g1");
        var view = ((DecodeEndpoint) fixture.endpoint()).routingView();
        var selected =
                PlacementResult.<DecodeResources.DecodeRoutingView, PlacementKey>success(view);
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        decodeSelector =
                (request, metadata, group, candidates, capacity) -> {
                    calls.incrementAndGet();
                    return selected;
                };
        when(endpoints.selectDecodeEndpoint(view)).thenReturn(null);
        var result = router().select(context, null);
        assertEquals(PlacementResult.Status.ENDPOINT_CHANGED, result.status());
        assertSame(selected, result.selection());
        assertSame(view, result.selection().value());
        assertEquals(new PlacementKey(RoleType.DECODE, null, null), result.blocker());
        assertEquals(1, calls.get(), "the scheduler owns retry timing");
        verify(fixture.selectedEndpoint(), never()).close();
    }

    @Test
    void projectedDecodeBlockUsesDecodeWaitDomain() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestRouteSelector router = router();
        RequestContext context = context(111L);
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenReturn(PlacementResult.blocked(PlacementKey.anyGroup(RoleType.DECODE)));

        PlacementResult<RouteExecution, PlacementKey> blocked =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.DECODE, null, null), blocked.blocker());
    }

    @Test
    void staticCapacityFailureIsTerminalAndReleasesEarlierSelections() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestRouteSelector router = router();
        RequestContext context = context(12L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 12L, "p", 8001, "g1");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));
        when(decodeSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any()))
                .thenReturn(
                        PlacementResult.rejected(
                                Response.error(StrategyErrorType.RESOURCE_EXHAUSTED)));

        PlacementResult<RouteExecution, PlacementKey> rejected =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.REJECTED, rejected.status());

        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), rejected.failure().getCode());
        verify(prefill.selectedEndpoint, never()).close();
    }

    @Test
    void selectionFailurePreservesCauseAndDoesNotRetireEndpoints() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.VIT));
        RequestContext context = context(120L);
        var prefill = selection(RoleType.PREFILL, 120L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 120L, "d", 8002, "g1");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));
        when(decodeSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any()))
                .thenAnswer(call -> selected(call, decode));
        var primary = new IllegalStateException("selection failed");
        var decodeClose = new IllegalStateException("Decode selectedEndpoint close failed");
        var prefillClose = new IllegalStateException("Prefill selectedEndpoint close failed");
        when(vitSelector.selectIndex(org.mockito.ArgumentMatchers.anyList())).thenThrow(primary);
        doThrow(decodeClose).when(decode.selectedEndpoint).close();
        doThrow(prefillClose).when(prefill.selectedEndpoint).close();

        assertSame(
                primary,
                assertThrows(IllegalStateException.class, () -> router().select(context, null)));

        var order = org.mockito.Mockito.inOrder(decode.selectedEndpoint, prefill.selectedEndpoint);
        order.verify(decode.selectedEndpoint, never()).close();
        order.verify(prefill.selectedEndpoint, never()).close();
        assertEquals(List.of(), List.of(primary.getSuppressed()));
        assertEquals(List.of(), List.of(decodeClose.getSuppressed()));
    }

    @Test
    void queueRouteTransfersTheExactPrefillPinIntoAdmissionOwnership() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestRouteSelector router = router();
        RequestContext context = context(21L);
        SelectionFixture prefill =
                selection(
                        RoleType.PREFILL,
                        Long.parseLong(context.data().getRequestId()),
                        "p",
                        8001,
                        "g1");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));

        PlacementResult<RouteExecution, PlacementKey> admitted =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());

        assertTrue(admitted.value().successResponse(false).isSuccess());
        assertEquals(
                List.of(prefill.status), admitted.value().successResponse(false).getServerStatus());

        verify(prefill.selectedEndpoint, never()).close();
        verify(prefill.selectedEndpoint, never()).close();
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
        when(prefillSelector.select(
                        same(inputs),
                        same(metadata),
                        eq(RoleType.PREFILL),
                        isNull(),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));

        var result = RouteExecution.fromSelection(context, router().select(context, null));
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        {
            var route = result.value();
            assertSame(inputs, route.ctx().data().getRequest());
            assertEquals(32L, route.seqLen());
            assertEquals(50, route.priority());
        }
        verify(prefillSelector)
                .select(
                        same(inputs),
                        same(metadata),
                        eq(RoleType.PREFILL),
                        isNull(),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class));
    }

    @Test
    void firstSelectedGroupChainsToLaterRolesWhenPolicyDidNotForceOne() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestRouteSelector router = router();
        RequestContext context = context(31L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 31L, "p", 8001, "selected-group");
        SelectionFixture vit = selection(RoleType.VIT, 31L, "v", 8002, "selected-group");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));
        when(vitSelector.selectIndex(org.mockito.ArgumentMatchers.anyList())).thenReturn(0);

        PlacementResult<RouteExecution, PlacementKey> admitted =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());

        verify(prefillSelector)
                .select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class));
        verify(vitSelector).selectIndex(org.mockito.ArgumentMatchers.anyList());
        verify(vit.selectedEndpoint, never()).close();
        verify(prefill.selectedEndpoint, never()).close();
    }

    @Test
    void policyGroupRemainsAuthoritativeAcrossEveryRole() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestRouteSelector router = router();
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        TrafficPolicyConfig groupSelector = new TrafficPolicyConfig();
        var target = new TrafficPolicyConfig.Target();
        target.setGroup("forced");
        groupSelector.setDefaultTargets(List.of(target));
        config.getRouter().setGroupSelector(groupSelector);
        RequestContext context = context(41L, config);
        SelectionFixture prefill = selection(RoleType.PREFILL, 41L, "p", 8001, "other");
        SelectionFixture vit = selection(RoleType.VIT, 41L, "v", 8002, "forced");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq("forced"),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));
        when(vitSelector.selectIndex(org.mockito.ArgumentMatchers.anyList())).thenReturn(0);

        PlacementResult<RouteExecution, PlacementKey> admitted =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());

        verify(prefillSelector)
                .select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq("forced"),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class));
        verify(vitSelector).selectIndex(org.mockito.ArgumentMatchers.anyList());
    }

    @Test
    void missingVitAfterPrefillSelectionKeepsThePrefillEndpointAvailable() {
        when(endpoints.endpointAddressSnapshot(RoleType.VIT)).thenReturn(List.of());
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestRouteSelector router = router();
        RequestContext context = context(51L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 51L, "p", 8001, "g1");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));
        when(vitSelector.selectIndex(org.mockito.ArgumentMatchers.anyList())).thenReturn(0);

        PlacementResult<RouteExecution, PlacementKey> blocked =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.VIT, "g1", null), blocked.blocker());
        verify(prefill.selectedEndpoint, never()).close();
    }

    @Test
    void decodeMissAfterPrefillSuccessReleasesThePrefillPinOwner() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestRouteSelector router = router();
        RequestContext context = context(52L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 52L, "p", 8001, "g1");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));
        when(decodeSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any()))
                .thenReturn(PlacementResult.blocked(new PlacementKey(RoleType.DECODE, "g1", null)));

        PlacementResult<RouteExecution, PlacementKey> blocked =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.DECODE, "g1", null), blocked.blocker());
        verify(prefill.selectedEndpoint, never()).close();
    }

    @Test
    void staleSelectedGenerationFailsClosedAndReleasesCapturedWorker() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestRouteSelector router = router();
        RequestContext context = context(61L);
        SelectionFixture foreign = selection(RoleType.PREFILL, 999L, "p", 8001, "g1");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(
                        call ->
                                PlacementResult.success(
                                        new PrefillSelectionStrategy.Selection<>(
                                                "p:8001",
                                                foreign.selectedEndpoint()
                                                                .getStatus()
                                                                .getGenerationId()
                                                        + 1,
                                                0L,
                                                1L,
                                                1L,
                                                0L,
                                                0L,
                                                0L,
                                                null,
                                                1,
                                                new EndpointRegistry.PrefillRoutingEntry(
                                                        "p:8001",
                                                        (PrefillEndpoint) foreign.endpoint()),
                                                null)));

        var handoff =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.ENDPOINT_CHANGED, handoff.status());
        assertEquals(PlacementResult.Status.SUCCESS, handoff.selection().status());
        var selected = (PrefillSelectionStrategy.Selection<?>) handoff.selection().value();
        assertEquals(
                foreign.selectedEndpoint().getStatus().getGenerationId() + 1,
                selected.generationId());
        verify(foreign.selectedEndpoint, never()).close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void blockedPolicyKeepsAffinityObservationWithoutCapturingResources(boolean reporterFails) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        var router = router();
        var context = context(62L);
        selection(RoleType.PREFILL, 62L, "p", 8001, "g1");
        var reporter =
                (org.flexlb.cache.telemetry.CacheMetricsReporter)
                        ReflectionTestUtils.getField(router, "cacheReporter");
        if (reporterFails) {
            doThrow(new IllegalStateException("metrics unavailable"))
                    .when(reporter)
                    .reportCacheAffinityDecision(RoleType.PREFILL, "p", "NO_CACHE_LEAD");
        }
        var key = PlacementKey.exact(RoleType.PREFILL, "g1", "p:8001");
        var failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        var diagnostics =
                java.util.Map.<String, Object>of(
                        "details",
                        java.util.Map.of(
                                "reason",
                                "PREFILL_CAPACITY",
                                "selectedEndpoint",
                                "p:8001",
                                "affinityReason",
                                "NO_CACHE_LEAD"));
        when(prefillSelector.select(any(), any(), any(), any(), any(), any()))
                .thenReturn(PlacementResult.blocked(key, failure, diagnostics));

        var result = RouteExecution.fromSelection(context, router.select(context, null));

        assertEquals(PlacementResult.Status.BLOCKED, result.status());
        assertSame(key, result.blocker());
        assertSame(failure, result.failure());
        assertEquals(diagnostics, result.diagnostics());
        verify(reporter).reportCacheAffinityDecision(RoleType.PREFILL, "p", "NO_CACHE_LEAD");
        verify(endpoints, never()).selectEndpoint(any(), any());
    }

    @Test
    void customPrefillSuccessIsNotReclassifiedByTheCaller() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        var router = router();
        var context = context(70L);
        var selected = selection(RoleType.PREFILL, 70L, "p", 8001, "g1");
        var endpoint = (PrefillEndpoint) selected.selectedEndpoint();
        when(endpoint.requestCapacityAvailable(anyLong(), anyInt())).thenReturn(false);
        when(endpoint.admissionSummary(anyInt(), anyLong()))
                .thenReturn(
                        new org.flexlb.balance.strategy.PrefillSelectionStrategy.AdmissionSummary(
                                1L, 1L, 0L, 0L, 0L, 1L));
        when(prefillSelector.select(any(), any(), any(), any(), any(), any()))
                .thenAnswer(call -> selected(call, selected));

        var result = RouteExecution.fromSelection(context, router.select(context, null));

        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        {
            var route = result.value();
            assertSame(endpoint, route.prefillEndpoint());
            verify(endpoint, never()).requestCapacityAvailable(anyLong(), anyInt());
            verify(endpoint, never()).admissionSummary(anyInt(), anyLong());
        }
    }

    @Test
    void requiredTopologyIsSnapshottedAtConstruction() {
        List<RoleType> mutable = new ArrayList<>();
        mutable.add(RoleType.PREFILL);
        when(modelMeta.requiredRoles()).thenReturn(mutable);
        RequestRouteSelector router = router();
        mutable.clear();
        RequestContext context = context(71L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 71L, "p", 8001, "g1");
        when(prefillSelector.select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class)))
                .thenAnswer(call -> selected(call, prefill));

        PlacementResult<RouteExecution, PlacementKey> admitted =
                RouteExecution.fromSelection(
                        context, router.select(context, router.resolvePolicyGroup(context)));
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());

        verify(prefillSelector)
                .select(
                        org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                        org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.eq(null),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class));
    }

    private AbstractRequestScheduler scheduler(
            RequestRouteSelector router, RequestContext context) {

        when(configService.loadBalanceConfig()).thenReturn(context.data().getConfig());
        return RequestProtocolTestSupport.injectPlacementDependencies(
                requests,
                configService,
                router,
                mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                mock(org.flexlb.balance.eviction.DecodePreemption.class),
                new PlacementAvailability());
    }

    private RequestRouteSelector router() {
        return new RequestRouteSelector(
                endpoints,
                prefillSelector,
                decodeSelector,
                vitSelector,
                org.mockito.Mockito.mock(
                        org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class),
                null,
                mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                mock(org.flexlb.cache.telemetry.CacheMetricsReporter.class),
                configService,
                modelMeta);
    }

    private void stubQueueSelection(
            RequestContext context,
            RoleType role,
            String group,
            PlacementResult<ServerStatus, RoleType> result) {
        switch (role) {
            case PREFILL, PDFUSION ->
                    when(prefillSelector.select(
                                    org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                                    org.mockito.ArgumentMatchers.eq(
                                            context.data().getSchedulingMetadata()),
                                    org.mockito.ArgumentMatchers.eq(role),
                                    org.mockito.ArgumentMatchers.eq(group),
                                    org.mockito.ArgumentMatchers.anyList(),
                                    org.mockito.ArgumentMatchers.any(
                                            java.util.function.LongSupplier.class)))
                            .thenReturn(
                                    new PlacementResult<>(
                                            result.status(),
                                            null,
                                            result.failure(),
                                            result.blocker() == null
                                                    ? null
                                                    : new PlacementKey(
                                                            result.blocker(), group, null),
                                            result.diagnostics(),
                                            null));
            case DECODE ->
                    when(decodeSelector.select(
                                    org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                                    org.mockito.ArgumentMatchers.any(),
                                    org.mockito.ArgumentMatchers.eq(group),
                                    org.mockito.ArgumentMatchers.anyList(),
                                    org.mockito.ArgumentMatchers.any()))
                            .thenReturn(
                                    new PlacementResult<>(
                                            result.status(),
                                            null,
                                            result.failure(),
                                            result.blocker() == null
                                                    ? null
                                                    : new PlacementKey(
                                                            result.blocker(), group, null),
                                            result.diagnostics(),
                                            null));
            case VIT ->
                    when(vitSelector.selectIndex(org.mockito.ArgumentMatchers.anyList()))
                            .thenReturn(0);
            case FRONTEND -> throw new IllegalArgumentException();
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION"})
    void reportedBlockerResolvesExactRoleAndCapturedCapacityVersion(RoleType prefillRole) {
        var context = context(990L);
        var prefill = selection(prefillRole, 990L, "shared", 8080, "g1");
        var decode = selection(RoleType.DECODE, 990L, "shared", 8080, "g1");
        when(prefill.endpoint().ipPort()).thenReturn("shared:8080");
        when(decode.endpoint().ipPort()).thenReturn("shared:8080");
        var prefillEndpoint = (PrefillEndpoint) prefill.endpoint();
        var decodeEndpoint = (DecodeEndpoint) decode.endpoint();
        when(prefillEndpoint.progress()).thenReturn(new PrefillEndpoint.Progress(0L, 0L, 11L, 0));
        when(decodeEndpoint.placementVersion()).thenReturn(23L);
        {
            var route = selectedRoute(context, prefill, decode);
            var prefillKey = route.placementKey(RoleType.PREFILL);
            var decodeKey = route.placementKey(RoleType.DECODE);
            assertSame(prefillEndpoint, route.blockedEndpointIfCurrent(prefillKey));
            assertSame(decodeEndpoint, route.blockedEndpointIfCurrent(decodeKey));
            // Changing one role does not invalidate the other role at the same address.
            when(decodeEndpoint.placementVersion()).thenReturn(24L);
            org.junit.jupiter.api.Assertions.assertNull(route.blockedEndpointIfCurrent(decodeKey));
            assertSame(prefillEndpoint, route.blockedEndpointIfCurrent(prefillKey));
            when(prefillEndpoint.progress())
                    .thenReturn(new PrefillEndpoint.Progress(0L, 0L, 12L, 0));
            org.junit.jupiter.api.Assertions.assertNull(route.blockedEndpointIfCurrent(prefillKey));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            route.blockedEndpointIfCurrent(
                                    PlacementKey.exact(RoleType.DECODE, "g1", "other:8080")));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            route.blockedEndpointIfCurrent(
                                    PlacementKey.exact(
                                            RoleType.DECODE, "other-group", "shared:8080")));
        }
        verify(prefill.selectedEndpoint(), never()).close();
        verify(decode.selectedEndpoint(), never()).close();
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
        Request request =
                Request.builder()
                        .requestId(String.valueOf(requestId))
                        .seqLen(32L)
                        .maxNewTokens(16)
                        .build();
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(
                                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    private SelectionFixture selection(
            RoleType role, long requestId, String ip, int httpPort, String group) {
        WorkerEndpoint endpoint =
                switch (role) {
                    case PREFILL, PDFUSION -> mock(PrefillEndpoint.class);
                    case DECODE -> RequestProtocolTestSupport.decodeEndpoint();
                    default -> mock(WorkerEndpoint.class);
                };
        WorkerEndpoint selectedEndpoint = endpoint;
        org.flexlb.balance.endpoint.EndpointTestSupport.useStateLock(
                endpoint, new java.util.concurrent.locks.ReentrantLock());
        if (endpoint instanceof PrefillEndpoint prefill) {
            var ledger = prefill;
            when(ledger.progress()).thenReturn(new PrefillEndpoint.Progress(0L, 0L, 0L, 0));
            when(ledger.requestCapacityAvailable(anyLong(), anyInt())).thenReturn(true);
            org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(prefill, ledger);
        }
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setRequestId(String.valueOf(requestId));
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(httpPort + 1);
        status.setGroup(group);
        status.setPrefillTime(role.supportsPrefill() ? 1L : 0L);
        if (role.supportsPrefill()) {
            status.setDebugInfo(new org.flexlb.dao.loadbalance.DebugInfo());
        }
        when(endpoint.getIp()).thenReturn(ip);
        when(endpoint.getHttpPort()).thenReturn(httpPort);
        var worker =
                org.flexlb.dao.master.WorkerStatus.createDiscovered(
                        role, group, ip, httpPort, httpPort + 1, null);
        when(endpoint.getStatus()).thenReturn(worker);
        String address = ip + ":" + httpPort;
        when(endpoints.selectEndpoint(role, address)).thenReturn(selectedEndpoint);
        if (role == RoleType.VIT)
            when(endpoints.endpointAddressSnapshot(role)).thenReturn(List.of(address));
        if (endpoint instanceof PrefillEndpoint prefill) {
            when(endpoints.prefillRoutingSnapshot(role))
                    .thenReturn(
                            List.of(
                                    new org.flexlb.balance.endpoint.EndpointRegistry
                                            .PrefillRoutingEntry(address, prefill)));
        } else if (endpoint instanceof DecodeEndpoint decode) {
            var view = mock(DecodeResources.DecodeRoutingView.class);
            when(view.address()).thenReturn(address);
            when(view.generationId()).thenReturn(worker.getGenerationId());
            when(view.topology()).thenReturn(worker.topologySnapshot());
            when(view.workerStatus())
                    .thenReturn(
                            new org.flexlb.dao.master.WorkerStatus.CommittedWorkerStatus(
                                    worker.committedEngineObservation(),
                                    worker.appliedStatusCursor()));
            when(decode.routingView()).thenReturn(view);
            when(endpoints.selectDecodeEndpoint(view)).thenReturn(selectedEndpoint);
            doAnswer(
                            call -> {
                                call.getArgument(1, List.class).add(view);
                                return null;
                            })
                    .when(endpoints)
                    .decodeRoutingSnapshot(any(), any());
        }
        return new SelectionFixture(selectedEndpoint, endpoint, status);
    }

    private static PlacementResult<?, ?> selected(
            org.mockito.invocation.InvocationOnMock call, SelectionFixture selected) {
        if (selected.endpoint() instanceof PrefillEndpoint endpoint) {
            String address =
                    selected.status().getServerIp() + ":" + selected.status().getHttpPort();
            return PlacementResult.success(
                    new PrefillSelectionStrategy.Selection<>(
                            address,
                            selected.selectedEndpoint().getStatus().getGenerationId(),
                            endpoint.progress().ownershipVersion(),
                            selected.status().getPrefillTime(),
                            selected.status().getPrefillTime(),
                            0L,
                            0L,
                            0L,
                            null,
                            1,
                            new EndpointRegistry.PrefillRoutingEntry(address, endpoint),
                            null));
        }
        if (selected.endpoint() instanceof DecodeEndpoint endpoint) {
            return PlacementResult.success(endpoint.routingView());
        }
        RouteResult.Builder route = call.getArgument(call.getArguments().length - 1);
        return PlacementResult.success(
                route.add(
                        selected.selectedEndpoint(),
                        selected.status(),
                        selected.status().getPrefillTime(),
                        selected.endpoint() instanceof PrefillEndpoint ep
                                ? ep.progress().ownershipVersion()
                                : selected.endpoint() instanceof DecodeEndpoint ep
                                        ? ep.placementVersion()
                                        : 0L));
    }

    private static RouteExecution selectedRoute(
            RequestContext context, SelectionFixture... selections) {
        return SchedulingTestConfig.prepareRoute(
                        context,
                        route -> {
                            for (var selected : selections) {
                                route.add(
                                        selected.selectedEndpoint(),
                                        selected.status(),
                                        selected.status().getPrefillTime(),
                                        selected.endpoint() instanceof PrefillEndpoint ep
                                                ? ep.progress().ownershipVersion()
                                                : selected.endpoint() instanceof DecodeEndpoint ep
                                                        ? ep.placementVersion()
                                                        : 0L);
                            }
                            return PlacementResult.success(route);
                        })
                .value();
    }

    private record SelectionFixture(
            WorkerEndpoint selectedEndpoint, WorkerEndpoint endpoint, ServerStatus status) {}
}
