package org.flexlb.balance.schedule;

import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.balance.strategy.VitStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.routing.TrafficPolicyConfig;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
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
@org.junit.jupiter.api.Timeout(15)
class RequestRouteSelectorTest {
    private CostBasedPrefillStrategy prefillStrategy;
    private EndpointRegistry endpoints;
    private DecodeStrategy decodeStrategy;
    private VitStrategy vitStrategy;
    private ConfigService configService;
    private ModelMetaConfig modelMeta;
    private AbstractRequestScheduler requests;

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        endpoints = mock(EndpointRegistry.class);
        when(endpoints.endpoints(
                        org.mockito.ArgumentMatchers.eq(RoleType.VIT),
                        org.mockito.ArgumentMatchers.any()))
                .thenAnswer(call -> List.of(endpoints.getEndpoint(RoleType.VIT, "vit:8080")));
        configService = mock(ConfigService.class);
        modelMeta = mock(ModelMetaConfig.class);
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        when(configService.loadBalanceConfig()).thenReturn(config);
        prefillStrategy =
                org.mockito.Mockito.spy(
                        new CostBasedPrefillStrategy(
                                configService,
                                endpoints,
                                org.mockito.Mockito.mock(
                                        org.flexlb.cache.match.CacheAwareService.class),
                                org.mockito.Mockito.mock(
                                        org.flexlb.cache.metrics.CacheMetricsReporter.class)));
        decodeStrategy =
                org.mockito.Mockito.spy(
                        new DecodeStrategy(
                                configService,
                                SchedulingTestConfig.capacity(configService.loadBalanceConfig()),
                                endpoints));
        vitStrategy =
                org.mockito.Mockito.spy(
                        new VitStrategy(
                                org.mockito.Mockito.mock(org.flexlb.cache.VitCacheDirectory.class),
                                endpoints));
        var runtime =
                SchedulerTestSupport.runtime(
                        SchedulerTestSupport.create(
                                configService,
                                mock(org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                                mock(org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                                mock(org.flexlb.balance.requestlog.TheoryCacheHitReporter.class)));
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
                                                                org.flexlb.balance.requestlog
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
                .tryBeginRouteSelection(any(RequestContext.class));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var context = call.getArgument(0, RequestContext.class);
                            var future =
                                    new ResponseCompletionExecutor.RequestFuture(
                                            (kind, response, failure) -> false);
                            assertEquals(
                                    RequestRepository.RegistrationResult.REGISTERED,
                                    requests.repository.register(
                                            context,
                                            requests,
                                            future,
                                            60_000L,
                                            StrategyErrorType.QUEUE_TIMEOUT));
                            requests.scheduleRequest(context);
                            return future;
                        })
                .when(requests)
                .submit(any(), any(Runnable.class));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var route = call.getArgument(0, RequestResult.class);
                            var handle = RequestProtocolTestSupport.admissionHandle(route.ctx());
                            return handle;
                        })
                .when(requests)
                .currentSelectionAttempt(any());
    }

    @Test
    void invalidRequestFailsBeforePolicyOrEndpointSelection() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestRouteSelector router = router();
        org.mockito.Mockito.clearInvocations(configService);

        RequestResult result =
                router.select(new RequestContext(SchedulingTestConfig.newConfig()), null);

        assertEquals(RoutingResult.Status.REJECTED, result.status());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), result.failure().getCode());
        verify(configService, never()).loadBalanceConfig();
        verifyNoInteractions(prefillStrategy, decodeStrategy, vitStrategy);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directRouteCommitsRolesAndReleasesSelectedWorkers(boolean pinCloseFails) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestRouteSelector router = router();
        RequestContext context = directContext(7L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 7L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 7L, "d", 8002, "g1");
        PrefillEndpoint.RequestRegistration registration =
                mock(PrefillEndpoint.RequestRegistration.class);
        DecodeEndpoint.RequestRegistration reservation =
                new DecodeEndpoint.RequestRegistration(1L, String.valueOf(7L));
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, decode)))
                .when(decodeStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList());
        var capture = stubPrefillCommit((PrefillEndpoint) prefill.endpoint);

        when(((DecodeEndpoint) decode.endpoint)
                        .tryRegisterSelection(
                                org.mockito.ArgumentMatchers.any(),
                                eq("7"),
                                eq(32L),
                                eq(48L),
                                eq(50),
                                any()))
                .thenReturn(
                        new WorkerEndpoint.Registration<>(
                                WorkerEndpoint.RegistrationStatus.REGISTERED, reservation));
        stubDecodeReservation((DecodeEndpoint) decode.endpoint, reservation);
        if (pinCloseFails) {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                    doThrow(
                                    new IllegalStateException(
                                            "selectedEndpoint close failed after direct handoff"))
                            .when(prefill.selectedEndpoint));
        }

        Response response =
                SchedulerTestServices.entry(scheduler(router, context)).submit(context).join();
        assertTrue(response.isSuccess());
        assertEquals(List.of(prefill.status, decode.status), response.getServerStatus());
        verify((PrefillEndpoint) prefill.endpoint, never())
                .releaseRequest(any(RequestResult.class));
        assertTrue(response.isSuccess());
        verify((DecodeEndpoint) decode.endpoint, never()).releaseRequest(reservation);
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint, never()));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(decode.selectedEndpoint, never()));
    }

    @Test
    void directRouteDoesNotSelectDecodeWhenPrefillRegistrationIsFull() throws Exception {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestRouteSelector router = router();
        RequestContext context = directContext(8L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 8L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 8L, "d", 8002, "g1");
        DecodeEndpoint.RequestRegistration reservation =
                new DecodeEndpoint.RequestRegistration(1L, String.valueOf(8L));
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, decode)))
                .when(decodeStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList());
        stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
        when(((DecodeEndpoint) decode.endpoint)
                        .tryRegisterSelection(
                                org.mockito.ArgumentMatchers.any(),
                                eq("8"),
                                eq(32L),
                                eq(48L),
                                eq(50),
                                any()))
                .thenReturn(
                        new WorkerEndpoint.Registration<>(
                                WorkerEndpoint.RegistrationStatus.REGISTERED, reservation));
        stubDecodeReservation((DecodeEndpoint) decode.endpoint, reservation);
        org.mockito.Mockito.doReturn(
                        new WorkerEndpoint.Registration<PrefillEndpoint.RequestRegistration>(
                                WorkerEndpoint.RegistrationStatus.CAPACITY_FULL, null))
                .when((PrefillEndpoint) prefill.endpoint)
                .tryRegisterSelection(any(), any(), anyLong());

        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                SchedulerTestServices.entry(scheduler(router, context))
                        .submit(context)
                        .get(5, TimeUnit.SECONDS)
                        .getCode());
        verify(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        verify((DecodeEndpoint) decode.endpoint, never()).releaseRequest(reservation);
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint, never()));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(decode.selectedEndpoint, never()));
    }

    @Test
    void directRequestRemainsInCanonicalLifecycleAfterResponseAndReconcilesEarlyDecodeEvidence()
            throws Exception {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestContext context = directContext(9L);
        when(configService.loadBalanceConfig()).thenReturn(context.getConfig());
        requests =
                SchedulerTestSupport.create(
                        configService,
                        mock(org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                        mock(org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        try {
            SelectionFixture prefill = selection(RoleType.PREFILL, 9L, "p", 8001, "g1");
            SelectionFixture decode = selection(RoleType.DECODE, 9L, "d", 8002, "g1");
            var reservation = new DecodeEndpoint.RequestRegistration(1L, String.valueOf(9L));
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.selectionResult(
                                            call, selected(call, prefill)))
                    .when(prefillStrategy)
                    .chooseBest(
                            org.mockito.ArgumentMatchers.eq(context.getRequest()),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any());
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.selectionResult(
                                            call, selected(call, decode)))
                    .when(decodeStrategy)
                    .chooseBest(
                            org.mockito.ArgumentMatchers.eq("g1"),
                            org.mockito.ArgumentMatchers.anyList());
            stubPrefillCommit((PrefillEndpoint) prefill.endpoint);

            when(((DecodeEndpoint) decode.endpoint)
                            .tryRegisterSelection(
                                    org.mockito.ArgumentMatchers.any(),
                                    eq("9"),
                                    eq(32L),
                                    eq(48L),
                                    eq(50),
                                    any()))
                    .thenReturn(
                            new WorkerEndpoint.Registration<>(
                                    WorkerEndpoint.RegistrationStatus.REGISTERED, reservation));
            stubDecodeReservation((DecodeEndpoint) decode.endpoint, reservation);
            when(((DecodeEndpoint) decode.endpoint).registrationState(reservation))
                    .thenReturn(
                            org.flexlb.worker.endpoint.DecodeResources.RegistrationState.CONFIRMED);
            assertTrue(
                    SchedulerTestServices.entry(scheduler(router(), context))
                            .submit(context)
                            .get(2L, TimeUnit.SECONDS)
                            .isSuccess());
            assertTrue(context.getFuture().get(2L, TimeUnit.SECONDS).isSuccess());
            RequestContext requestContext = requests.repository.findActive("9");
            synchronized (requestContext) {
                assertEquals(
                        DecodeResources.RegistrationState.CONFIRMED,
                        ((DecodeEndpoint) decode.endpoint).registrationState(reservation));
                assertTrue(requestContext.snapshot().phase() == RequestContext.Phase.OPEN);
                assertEquals(
                        RequestResult.State.ACKNOWLEDGED,
                        requests.repository.getRequestState(requestContext.getRequestId()).state());
            }
            RequestProtocolTestSupport.expireInactiveRequest(
                    requests,
                    requestContext,
                    System.currentTimeMillis()
                            + context.getConfig()
                                    .getRequestLifecycle()
                                    .getRequest()
                                    .getTimeoutMs());
            assertEquals(
                    RequestResult.State.TIMED_OUT,
                    SchedulerTestSupport.repository(requests).getRequestState("9").state());
            assertEquals(0, SchedulerTestSupport.repository(requests).liveRequestCount());
            verify((DecodeEndpoint) decode.endpoint).releaseRequest(reservation);
            verify((PrefillEndpoint) prefill.endpoint).releaseRequest(any(RequestResult.class));
        } finally {
            RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(requests);
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
        when(configService.loadBalanceConfig()).thenReturn(context.getConfig());
        requests =
                SchedulerTestSupport.create(
                        configService,
                        mock(org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                        mock(org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        selection(RoleType.PREFILL, 10L, "p", 8001, "g1");
        try {
            Throwable failure =
                    fatal
                            ? new AssertionError("selection failed")
                            : new IllegalStateException("selection failed");
            when(prefillStrategy.chooseBest(
                            org.mockito.ArgumentMatchers.eq(context.getRequest()),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any()))
                    .thenThrow(failure);
            var returned =
                    SchedulerTestServices.entry(scheduler(router(), context)).submit(context);
            assertEquals(context.getFuture(), returned);
            Response response = context.getFuture().get(2L, TimeUnit.SECONDS);
            assertEquals(8510, response.getCode());
            assertEquals(
                    fatal
                            ? "DISPATCH_FAILED"
                            : "Selection or publication failed: IllegalStateException: selection failed",
                    response.getErrorMessage());
            assertEquals(
                    RequestResult.State.FAILED,
                    SchedulerTestSupport.repository(requests).getRequestState("10").state());
            SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(requests));
            if (fatal)
                assertSame(
                        failure,
                        SchedulerTestServices.failure(SchedulerTestSupport.runtime(requests)));
            verifyNoInteractions(decodeStrategy, vitStrategy);
        } finally {
            RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(requests);
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(requests)).close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(requests));
        }
    }

    @Test
    void requestDemandUsesImmutablePromptAndOutputInputs() {
        var request =
                org.flexlb.worker.endpoint.Request.builder().seqLen(32L).maxNewTokens(16).build();
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
    @ValueSource(booleans = {false, true})
    void failedReclaimedSelectionReleasesOnceAndPreservesOriginalFailure(boolean adoptionThrows) {
        var context = context(702L);
        var prefill = selection(RoleType.PREFILL, 702L, "p", 8001, "g1");
        var selectedDecode = selection(RoleType.DECODE, 702L, "d", 8002, "g1");
        var decode = (DecodeEndpoint) selectedDecode.endpoint();
        var reservation = new DecodeEndpoint.RequestRegistration(1L, String.valueOf(702L));
        var adoptionFailure = new IllegalStateException("adoption failed");
        var cleanupFailure = new IllegalStateException("cleanup failed");
        if (adoptionThrows) {
            when(decode.registrationState(reservation)).thenThrow(adoptionFailure);
        }
        doThrow(cleanupFailure).when(decode).rollbackSelection(reservation);

        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        var actual =
                assertThrows(
                        IllegalStateException.class,
                        () -> router().select(context, null, decode, reservation));
        assertSame(adoptionThrows ? adoptionFailure : cleanupFailure, actual);
        assertEquals(
                adoptionThrows ? List.of(cleanupFailure) : List.of(),
                List.of(actual.getSuppressed()));
        verify(decode).rollbackSelection(reservation);
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint(), never()));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(selectedDecode.selectedEndpoint(), never()));
    }

    @Test
    void callerAndDeliveredResponseChangesCannotMutateTheRouteTemplate() {
        var context = context(707L);
        var prefill = selection(RoleType.PREFILL, 707L, "p", 8001, "g1");
        {
            var route = selectedRoute(context, prefill);
            prefill.status().setGroup("caller changed");
            var exported = route.successResponse();
            exported.setCode(503);
            exported.getServerStatus().getFirst().setGroup("frontend changed");
            var subsequent = route.successResponse();
            assertTrue(subsequent.isSuccess());
            assertEquals(200, subsequent.getCode());
            assertEquals("g1", subsequent.getServerStatus().getFirst().getGroup());
            assertFalse(subsequent.isEnqueuedByMaster());
        }
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint(), never()));
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void unattachedDecodeReservationIsReleased(boolean direct, boolean cleanupFails) {
        var context = direct ? directContext(703L) : context(703L);
        SchedulerTestSupport.bindOwner(context, requests);
        var prefill = selection(RoleType.PREFILL, 703L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 703L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var reservation = new DecodeEndpoint.RequestRegistration(1L, String.valueOf(703L));
        when(endpoint.tryRegisterSelection(
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        anyLong(),
                        anyInt(),
                        any()))
                .thenReturn(
                        new WorkerEndpoint.Registration<>(
                                WorkerEndpoint.RegistrationStatus.REGISTERED, reservation));
        var primary = new IllegalStateException("route rejected resource attachment");
        var cleanup = new IllegalStateException("resource rollback failed");
        if (cleanupFails) {
            doThrow(cleanup).when(endpoint).rollbackSelection(reservation);
        } else {
            when(endpoint.rollbackSelection(reservation)).thenReturn(true);
        }
        {
            when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.selectionResult(
                                            call, selected(call, prefill)))
                    .when(prefillStrategy)
                    .chooseBest(any(), any(), org.mockito.ArgumentMatchers.any());
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.selectionResult(
                                            call, selected(call, decode)))
                    .when(decodeStrategy)
                    .chooseBest(any(), any());
            var foreign = new DecodeEndpoint.RequestRegistration(1L, "foreign-request");
            when(endpoint.tryRegisterSelection(any(), any(), anyLong(), anyLong(), anyInt(), any()))
                    .thenReturn(
                            new WorkerEndpoint.Registration<>(
                                    WorkerEndpoint.RegistrationStatus.REGISTERED, foreign));
            if (cleanupFails) doThrow(cleanup).when(endpoint).rollbackSelection(eq(foreign));
            var actual =
                    assertThrows(
                            IllegalArgumentException.class, () -> router().select(context, null));
            assertEquals("Decode reservation belongs to another request", actual.getMessage());
            assertEquals(
                    cleanupFails ? List.of(cleanup) : List.of(), List.of(actual.getSuppressed()));
            verify(requests, never()).currentSelectionAttempt(any());
            verify(endpoint).rollbackSelection(foreign);
            verify((PrefillEndpoint) prefill.endpoint())
                    .rollbackSelection(any(PrefillEndpoint.RequestRegistration.class));
        }
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint(), never()));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(decode.selectedEndpoint(), never()));
    }

    @Test
    void invalidResultConstructionLeavesPinOwnershipWithTheSelector() {
        var context = context(706L);
        var prefill = selection(RoleType.PREFILL, 706L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 706L, "d", 8002, "g1");
        decode.status().setRequestId(String.valueOf(999L));
        assertThrows(IllegalStateException.class, () -> selectedRoute(context, prefill, decode));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint(), never()));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(decode.selectedEndpoint(), never()));
    }

    @Test
    void reclaimedDecodeIsReleasedOnceWhenPrefillCannotBeSelected() {
        var context = context(705L);
        var decode = selection(RoleType.DECODE, 705L, "d", 8002, "g1");
        var endpoint = (DecodeEndpoint) decode.endpoint();
        var reservation = new DecodeEndpoint.RequestRegistration(1L, "705");
        when(endpoint.registrationState(reservation))
                .thenReturn(DecodeResources.RegistrationState.UNCONFIRMED);
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        stubQueueSelection(
                context, RoleType.PREFILL, "g1", RoutingResult.blocked(RoleType.PREFILL));

        var result = router().select(context, null, endpoint, reservation);

        assertEquals(RoutingResult.Status.BLOCKED, result.status());
        verify(endpoint).rollbackSelection(reservation);
        verifyNoInteractions(decodeStrategy);
    }

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "DECODE", "PDFUSION", "VIT"})
    void missingRequiredRoleReturnsItsExactWaitDomain(RoleType role) {
        if (role == RoleType.VIT)
            when(endpoints.endpoints(
                            org.mockito.ArgumentMatchers.eq(role),
                            org.mockito.ArgumentMatchers.any()))
                    .thenReturn(List.of());
        when(modelMeta.requiredRoles()).thenReturn(List.of(role));
        RequestRouteSelector router = router();
        RequestContext context = context(11L);
        stubQueueSelection(context, role, null, RoutingResult.blocked(role));

        RequestResult blocked = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.BLOCKED, blocked.status());

        assertEquals(new CapacityKey(role, null, null), blocked.blocker());
    }

    @Test
    void invalidVitChoiceCannotPublishARoute() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.VIT));
        selection(RoleType.VIT, 109L, "v", 8003, "g1");
        when(vitStrategy.select(org.mockito.ArgumentMatchers.anyList()))
                .thenAnswer(
                        call -> {
                            var original =
                                    (VitStrategy.Candidate)
                                            call.getArgument(0, java.util.List.class).getFirst();
                            return java.util.Optional.of(
                                    new VitStrategy.Candidate(
                                            original.endpoint(), 0, 0, 0, 0, null));
                        });
        var context = context(109L);
        assertThrows(IllegalArgumentException.class, () -> router().select(context, null));
        assertNull(context.snapshot().route());
    }

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION", "DECODE", "VIT"})
    void retirementBetweenEndpointLookupAndAddRetriesSelection(RoleType role) {
        var context = context(113L);
        var prefillRole = role == RoleType.PDFUSION ? role : RoleType.PREFILL;
        when(modelMeta.requiredRoles())
                .thenReturn(role.supportsPrefill() ? List.of(role) : List.of(prefillRole, role));
        var prefill = selection(prefillRole, 113L, "p", 8001, "g1");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        any(),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        var chosen = role.supportsPrefill() ? prefill : selection(role, 113L, "other", 8002, "g1");
        if (role == RoleType.DECODE) {
            var decode = (DecodeEndpoint) chosen.endpoint();
            var view = decode.selectionSnapshot();
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.selectionResult(
                                            call,
                                            RoutingResult.success(decode.selectionSnapshot())))
                    .when(decodeStrategy)
                    .chooseBest(any(), org.mockito.ArgumentMatchers.anyList());
            doAnswer(
                            call -> {
                                when(chosen.endpoint().hasStoppedAcceptingRequests())
                                        .thenReturn(true);
                                return chosen.endpoint();
                            })
                    .when(endpoints)
                    .getEndpoint(org.flexlb.config.service.RoleType.DECODE, view.address());
        } else {
            doAnswer(
                            call -> {
                                when(chosen.endpoint().hasStoppedAcceptingRequests())
                                        .thenReturn(true);
                                return chosen.endpoint();
                            })
                    .when(endpoints)
                    .getEndpoint(eq(role), any());
        }
        RequestProtocolTestSupport.submitForManualPlacement(requests, context);
        var attempt = RequestProtocolTestSupport.beginAdmission(requests, context);
        var router = router();
        var result =
                requests.executeRouteSelection(attempt, () -> router.select(context, null), false);
        assertEquals(RoutingResult.Status.RETRY, result.status());
        if (!role.supportsPrefill()) assertNull(result.blocker());
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(chosen.endpoint(), never()));
    }

    @Test
    void customDecodeStrategyReceivesReadOnlyEvaluatedCandidates() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.DECODE));
        var context = context(110L);
        selection(RoleType.DECODE, 110L, "d", 8002, "g1");
        decodeStrategy =
                org.mockito.Mockito.spy(
                        new DecodeStrategy(
                                configService,
                                SchedulingTestConfig.capacity(configService.loadBalanceConfig()),
                                endpoints));
        doAnswer(
                        call -> {
                            java.util.List<DecodeEndpoint.SelectionSnapshot> candidates =
                                    call.getArgument(1);
                            assertEquals(1, candidates.size());
                            assertThrows(UnsupportedOperationException.class, candidates::clear);
                            return java.util.Optional.empty();
                        })
                .when(decodeStrategy)
                .chooseBest(any(), any());
        var result = router().select(context, null);
        assertEquals(RoutingResult.Status.BLOCKED, result.status());
        assertEquals(new CapacityKey(RoleType.DECODE, null, null), result.blocker());
        verify(endpoints, never())
                .getEndpoint(
                        org.mockito.ArgumentMatchers.eq(org.flexlb.config.service.RoleType.DECODE),
                        org.mockito.ArgumentMatchers.anyString());
        verifyNoInteractions(prefillStrategy, vitStrategy);
    }

    @Test
    void customDecodeStrategyReportsRetryWhenSelectedGenerationDisappears() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.DECODE));
        var context = context(112L);
        var fixture = selection(RoleType.DECODE, 112L, "d", 8002, "g1");
        var view = ((DecodeEndpoint) fixture.endpoint()).selectionSnapshot();
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        decodeStrategy =
                org.mockito.Mockito.spy(
                        new DecodeStrategy(
                                configService,
                                SchedulingTestConfig.capacity(configService.loadBalanceConfig()),
                                endpoints));
        doAnswer(
                        call -> {
                            calls.incrementAndGet();
                            return java.util.Optional.of(view);
                        })
                .when(decodeStrategy)
                .chooseBest(any(), any());
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var capturedViews =
                                    new java.util.ArrayList<
                                            org.flexlb.worker.endpoint.DecodeEndpoint
                                                    .SelectionSnapshot>();

                            ((java.util.List<DecodeEndpoint.SelectionSnapshot>) capturedViews)
                                    .add(view);
                            return org.flexlb.worker.endpoint.EndpointTestSupport.decodeCandidates(
                                    capturedViews);
                        })
                .when(endpoints)
                .endpoints(
                        org.mockito.ArgumentMatchers.eq(org.flexlb.config.service.RoleType.DECODE),
                        org.mockito.ArgumentMatchers.any());
        when(endpoints.getEndpoint(org.flexlb.config.service.RoleType.DECODE, view.address()))
                .thenReturn(null);
        var result = router().select(context, null);
        assertEquals(RoutingResult.Status.RETRY, result.status());
        org.junit.jupiter.api.Assertions.assertNull(result.blocker());
        assertEquals(
                1,
                calls.get(),
                "losing the last candidate returns to the scheduler for a fresh directory");
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(fixture.selectedEndpoint(), never()));
    }

    @Test
    void prefillStrategyCannotRedirectCapacityWaitToAnotherRole() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestRouteSelector router = router();
        RequestContext context = context(111L);
        org.mockito.Mockito.doAnswer(
                        call ->
                                SelectionTestSupport.selectionResult(
                                        call,
                                        RoutingResult.blocked(
                                                CapacityKey.anyGroup(RoleType.DECODE))))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());

        RequestResult blocked = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.BLOCKED, blocked.status());

        assertEquals(new CapacityKey(RoleType.PREFILL, null, null), blocked.blocker());
    }

    @Test
    void staticCapacityFailureIsTerminalAndReleasesEarlierSelections() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestRouteSelector router = router();
        RequestContext context = context(12L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 12L, "p", 8001, "g1");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        var decode = selection(RoleType.DECODE, 12L, "d", 8002, "g1");
        var before = ((DecodeEndpoint) decode.endpoint()).selectionSnapshot();
        var tooSmall =
                new DecodeEndpoint.SelectionSnapshot(
                        before.address(),
                        before.generationId(),
                        before.topology(),
                        before.workerStatus(),
                        before.admissionVersion(),
                        new DecodeResources.CapacityUsage(0, 1, 1, 0, 0));
        doAnswer(
                        call -> {
                            var capturedViews =
                                    new java.util.ArrayList<
                                            org.flexlb.worker.endpoint.DecodeEndpoint
                                                    .SelectionSnapshot>();

                            capturedViews.add(tooSmall);
                            return org.flexlb.worker.endpoint.EndpointTestSupport.decodeCandidates(
                                    capturedViews);
                        })
                .when(endpoints)
                .endpoints(
                        org.mockito.ArgumentMatchers.eq(org.flexlb.config.service.RoleType.DECODE),
                        org.mockito.ArgumentMatchers.any());

        RequestResult rejected = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.REJECTED, rejected.status());

        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), rejected.failure().getCode());
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint, never()));
    }

    @Test
    void selectionFailurePreservesCauseAndDoesNotRetireEndpoints() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.VIT));
        RequestContext context = context(120L);
        var prefill = selection(RoleType.PREFILL, 120L, "p", 8001, "g1");
        var decode = selection(RoleType.DECODE, 120L, "d", 8002, "g1");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, decode)))
                .when(decodeStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList());
        var primary = new IllegalStateException("selection failed");
        var decodeClose = new IllegalStateException("Decode selectedEndpoint close failed");
        var prefillClose = new IllegalStateException("Prefill selectedEndpoint close failed");
        selection(RoleType.VIT, 120L, "v", 8003, "g1");
        when(vitStrategy.select(org.mockito.ArgumentMatchers.anyList())).thenThrow(primary);
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                doThrow(decodeClose).when(decode.selectedEndpoint));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                doThrow(prefillClose).when(prefill.selectedEndpoint));

        assertSame(
                primary,
                assertThrows(IllegalStateException.class, () -> router().select(context, null)));

        var order = org.mockito.Mockito.inOrder(decode.selectedEndpoint, prefill.selectedEndpoint);
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                order.verify(decode.selectedEndpoint, never()));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                order.verify(prefill.selectedEndpoint, never()));
        assertEquals(List.of(), List.of(primary.getSuppressed()));
        assertEquals(List.of(), List.of(decodeClose.getSuppressed()));
    }

    @Test
    void queueRouteTransfersTheExactPrefillPinIntoAdmissionOwnership() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestRouteSelector router = router();
        RequestContext context = context(21L);
        RequestProtocolTestSupport.register(
                context,
                context.getRequestId(),
                System.currentTimeMillis(),
                context.getRequestExpiresAtMs(),
                60_000L,
                StrategyErrorType.QUEUE_TIMEOUT);
        var attempt = context.beginRouteSelection();
        SelectionFixture prefill =
                selection(
                        RoleType.PREFILL, Long.parseLong(context.getRequestId()), "p", 8001, "g1");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());

        RequestResult admitted = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.SUCCESS, admitted.status());

        assertTrue(admitted.successResponse().isSuccess());
        assertEquals(List.of(prefill.status), admitted.successResponse().getServerStatus());
        assertNull(context.snapshot().route());
        assertEquals(attempt, context.currentSelectionAttempt());

        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint, never()));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint, never()));
    }

    @Test
    void prefillSelectionUsesRegisteredInputsAndOriginalConfiguration() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        var context = context(30L);
        var inputs = context.getRequest();
        var metadata = context.getSchedulingMetadata();
        var changed = inputs.toBuilder().seqLen(4_096L).build();
        assertEquals(4_096L, changed.getSeqLen());
        var prefill = selection(RoleType.PREFILL, 30L, "p", 8001, "g1");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        same(inputs),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());

        var result = router().select(context, null);
        assertEquals(RoutingResult.Status.SUCCESS, result.status());
        {
            var route = result;
            assertSame(inputs, route.ctx().getRequest());
            assertEquals(32L, route.ctx().getRequest().getSeqLen());
            assertEquals(50, route.priority());
        }
        verify(prefillStrategy)
                .chooseBest(
                        same(inputs),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
    }

    @Test
    void firstSelectedGroupChainsToLaterRolesWhenPolicyDidNotForceOne() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestRouteSelector router = router();
        RequestContext context = context(31L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 31L, "p", 8001, "selected-group");
        SelectionFixture vit = selection(RoleType.VIT, 31L, "v", 8002, "selected-group");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        when(vitStrategy.select(org.mockito.ArgumentMatchers.anyList()))
                .thenAnswer(
                        call ->
                                java.util.Optional.of(
                                        call.getArgument(0, java.util.List.class).get(0)));

        RequestResult admitted = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.SUCCESS, admitted.status());

        verify(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        verify(vitStrategy).select(org.mockito.ArgumentMatchers.anyList());
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(vit.selectedEndpoint, never()));
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint, never()));
    }

    @Test
    void policyGroupRemainsAuthoritativeAcrossEveryRole() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestRouteSelector router = router();
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        TrafficPolicyConfig groupSelector = new TrafficPolicyConfig();
        var target = new TrafficPolicyConfig.Target();
        target.setGroup("forced");
        groupSelector.setDefaultTargets(List.of(target));
        config.getRouter().setGroupSelector(groupSelector);
        RequestContext context = context(41L, config);
        SelectionFixture prefill = selection(RoleType.PREFILL, 41L, "p", 8001, "other");
        SelectionFixture vit = selection(RoleType.VIT, 41L, "v", 8002, "forced");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        when(vitStrategy.select(org.mockito.ArgumentMatchers.anyList()))
                .thenAnswer(
                        call ->
                                java.util.Optional.of(
                                        call.getArgument(0, java.util.List.class).get(0)));

        RequestResult admitted = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.SUCCESS, admitted.status());

        verify(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        verify(vitStrategy).select(org.mockito.ArgumentMatchers.anyList());
    }

    @Test
    void missingVitAfterPrefillSelectionKeepsThePrefillEndpointAvailable() {
        when(endpoints.endpoints(
                        org.mockito.ArgumentMatchers.eq(RoleType.VIT),
                        org.mockito.ArgumentMatchers.any()))
                .thenReturn(List.of());
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        RequestRouteSelector router = router();
        RequestContext context = context(51L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 51L, "p", 8001, "g1");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        when(vitStrategy.select(org.mockito.ArgumentMatchers.anyList()))
                .thenAnswer(
                        call ->
                                java.util.Optional.of(
                                        call.getArgument(0, java.util.List.class).get(0)));

        RequestResult blocked = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.BLOCKED, blocked.status());

        assertEquals(new CapacityKey(RoleType.VIT, "g1", null), blocked.blocker());
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint, never()));
    }

    @Test
    void decodeMissAfterPrefillSuccessReleasesThePrefillPinOwner() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        RequestRouteSelector router = router();
        RequestContext context = context(52L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 52L, "p", 8001, "g1");
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
        org.mockito.Mockito.doAnswer(
                        call ->
                                SelectionTestSupport.selectionResult(
                                        call,
                                        RoutingResult.blocked(
                                                new CapacityKey(RoleType.DECODE, "g1", null))))
                .when(decodeStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq("g1"),
                        org.mockito.ArgumentMatchers.anyList());

        RequestResult blocked = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.BLOCKED, blocked.status());

        assertEquals(new CapacityKey(RoleType.DECODE, "g1", null), blocked.blocker());
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(prefill.selectedEndpoint, never()));
    }

    @Test
    void staleSelectedGenerationFailsClosedAndReleasesCapturedWorker() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        RequestRouteSelector router = router();
        RequestContext context = context(61L);
        SelectionFixture foreign = selection(RoleType.PREFILL, 999L, "p", 8001, "g1");
        org.mockito.Mockito.doAnswer(
                        call ->
                                SelectionTestSupport.selectionResult(
                                        call,
                                        RoutingResult.success(
                                                new CostBasedPrefillStrategy.Selection(
                                                        "p:8001@0",
                                                        foreign.selectedEndpoint()
                                                                        .getStatus()
                                                                        .getGenerationId()
                                                                + 1,
                                                        1L,
                                                        0L,
                                                        0L,
                                                        0L,
                                                        null,
                                                        (PrefillEndpoint) foreign.endpoint(),
                                                        null,
                                                        ((PrefillEndpoint) foreign.endpoint())
                                                                .selectionSnapshot()))))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());

        var handoff = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.RETRY, handoff.status());
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                verify(foreign.selectedEndpoint, never()));
    }

    @Test
    void noSelectionDoesNotRegisterResourcesOrReportASelectedWorker() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        var context = context(62L);
        var prefill = selection(RoleType.PREFILL, 62L, "p", 8001, "g1");
        org.mockito.Mockito.doReturn(java.util.Optional.empty())
                .when(prefillStrategy)
                .chooseBest(any(), any(), any());
        var router = router();
        var result = router.select(context, null);
        assertEquals(RoutingResult.Status.BLOCKED, result.status());
        assertEquals(new CapacityKey(RoleType.PREFILL, null, null), result.blocker());
        verify((PrefillEndpoint) prefill.endpoint(), never())
                .tryRegisterSelection(any(), any(), anyLong());
        verifyNoInteractions(ReflectionTestUtils.getField(prefillStrategy, "cacheReporter"));
    }

    @Test
    void affinityReporterFailureDoesNotDiscardRegisteredSelection() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        var context = context(63L);
        var prefill = selection(RoleType.PREFILL, 63L, "p", 8001, "g1");
        var snapshot =
                new PrefillEndpoint.SelectionSnapshot(
                        prefill.endpoint().getStatus().getGenerationId(), 0, 0, 1000);
        when(((PrefillEndpoint) prefill.endpoint()).selectionSnapshot()).thenReturn(snapshot);
        prefillStrategy =
                new CostBasedPrefillStrategy(
                        configService,
                        endpoints,
                        org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.cache.metrics.CacheMetricsReporter.class));
        var router = router();
        var reporter =
                (org.flexlb.cache.metrics.CacheMetricsReporter)
                        ReflectionTestUtils.getField(prefillStrategy, "cacheReporter");
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(reporter)
                .reportCacheAffinityDecision(RoleType.PREFILL, "p", "MIN_PREFILL_COST");
        assertEquals(RoutingResult.Status.SUCCESS, router.select(context, null).status());
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
        org.mockito.Mockito.doAnswer(
                        call -> SelectionTestSupport.selectionResult(call, selected(call, prefill)))
                .when(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());

        RequestResult admitted = router.select(context, router.resolvePolicyGroup(context));
        assertEquals(RoutingResult.Status.SUCCESS, admitted.status());

        verify(prefillStrategy)
                .chooseBest(
                        org.mockito.ArgumentMatchers.eq(context.getRequest()),
                        org.mockito.ArgumentMatchers.anyList(),
                        org.mockito.ArgumentMatchers.any());
    }

    private QueuedRequestScheduler queueScheduler(RequestContext context) {

        var config = SchedulingTestConfig.newConfig();
        var scheduler = mock(QueuedRequestScheduler.class);
        var lifecycle = SchedulerTestSupport.spyLifecycle(scheduler);
        org.springframework.test.util.ReflectionTestUtils.setField(
                context, "requestScheduler", lifecycle);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var route = call.getArgument(0, RequestResult.class);
                            var handle = RequestProtocolTestSupport.admissionHandle(route.ctx());
                            return handle;
                        })
                .when(lifecycle)
                .currentSelectionAttempt(any());
        return scheduler;
    }

    private static void stubDecodeReservation(
            DecodeEndpoint endpoint, DecodeEndpoint.RequestRegistration reservation) {
        when(endpoint.registrationState(reservation))
                .thenReturn(DecodeResources.RegistrationState.UNCONFIRMED);
    }

    private AbstractRequestScheduler scheduler(
            RequestRouteSelector router, RequestContext context) {

        when(configService.loadBalanceConfig()).thenReturn(context.getConfig());
        return RequestProtocolTestSupport.injectPlacementDependencies(
                requests,
                configService,
                router,
                mock(org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                mock(org.flexlb.balance.schedule.DecodePreemption.class),
                new CapacityNotifier());
    }

    private RequestRouteSelector router() {
        return new RequestRouteSelector(
                endpoints,
                prefillStrategy,
                decodeStrategy,
                vitStrategy,
                org.mockito.Mockito.mock(EncoderStrategy.class),
                modelMeta);
    }

    private void stubQueueSelection(
            RequestContext context,
            RoleType role,
            String group,
            RoutingResult<ServerStatus, RoleType> result) {
        switch (role) {
            case PREFILL, PDFUSION ->
                    org.mockito.Mockito.doAnswer(
                                    call ->
                                            SelectionTestSupport.selectionResult(
                                                    call,
                                                    new RoutingResult<>(
                                                            result.status(),
                                                            null,
                                                            result.failure(),
                                                            result.blocker() == null
                                                                    ? null
                                                                    : new CapacityKey(
                                                                            result.blocker(),
                                                                            group,
                                                                            null),
                                                            result.diagnostics())))
                            .when(prefillStrategy)
                            .chooseBest(
                                    org.mockito.ArgumentMatchers.eq(context.getRequest()),
                                    org.mockito.ArgumentMatchers.anyList(),
                                    org.mockito.ArgumentMatchers.any());
            case DECODE ->
                    org.mockito.Mockito.doAnswer(
                                    call ->
                                            SelectionTestSupport.selectionResult(
                                                    call,
                                                    new RoutingResult<>(
                                                            result.status(),
                                                            null,
                                                            result.failure(),
                                                            result.blocker() == null
                                                                    ? null
                                                                    : new CapacityKey(
                                                                            result.blocker(),
                                                                            group,
                                                                            null),
                                                            result.diagnostics())))
                            .when(decodeStrategy)
                            .chooseBest(
                                    org.mockito.ArgumentMatchers.eq(group),
                                    org.mockito.ArgumentMatchers.anyList());
            case VIT ->
                    when(vitStrategy.select(org.mockito.ArgumentMatchers.anyList()))
                            .thenAnswer(
                                    call ->
                                            java.util.Optional.of(
                                                    call.getArgument(0, java.util.List.class)
                                                            .getFirst()));
            case FRONTEND -> throw new IllegalArgumentException();
        }
    }

    private static RequestContext directContext(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        return context(requestId, config);
    }

    private static RequestContext context(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
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
        context.setRequest(request);
        context.setSchedulingMetadata(
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
        org.flexlb.worker.endpoint.EndpointTestSupport.useStateLock(
                endpoint, new java.util.concurrent.locks.ReentrantLock());
        if (endpoint instanceof PrefillEndpoint prefill) {
            var ledger = prefill;

            org.flexlb.worker.endpoint.EndpointTestSupport.stubResources(prefill, ledger);
            when(prefill.tryRegisterSelection(any(), any(), org.mockito.ArgumentMatchers.anyLong()))
                    .thenAnswer(
                            call ->
                                    new WorkerEndpoint.Registration<
                                            PrefillEndpoint.RequestRegistration>(
                                            prefill.acceptsNewRequests()
                                                    ? WorkerEndpoint.RegistrationStatus.REGISTERED
                                                    : WorkerEndpoint.RegistrationStatus.UNAVAILABLE,
                                            prefill.acceptsNewRequests()
                                                    ? mock(
                                                            PrefillEndpoint.RequestRegistration
                                                                    .class)
                                                    : null));
        }
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setRequestId(String.valueOf(requestId));
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(httpPort + 1);
        status.setGroup(group);
        status.setPrefillTime(0L);
        if (role.supportsPrefill()) {
            status.setDebugInfo(new org.flexlb.worker.status.DebugInfo());
        }
        when(endpoint.getIp()).thenReturn(ip);
        when(endpoint.getHttpPort()).thenReturn(httpPort);
        var worker =
                org.flexlb.worker.status.WorkerStatus.createDiscovered(
                        role, group, ip, httpPort, httpPort + 1, null);
        when(endpoint.getStatus()).thenReturn(worker);
        String address = worker.getLogicalIpPort();
        when(endpoint.ipPort()).thenReturn(address);
        when(endpoints.getEndpoint(role, address)).thenReturn(selectedEndpoint);
        if (role == RoleType.VIT) {
            when(endpoints.endpoints(
                            org.mockito.ArgumentMatchers.eq(role),
                            org.mockito.ArgumentMatchers.any()))
                    .thenReturn(List.of(endpoint));
            when(vitStrategy.select(org.mockito.ArgumentMatchers.anyList()))
                    .thenAnswer(
                            call -> {
                                java.util.List<VitStrategy.Candidate> candidates =
                                        call.getArgument(0);
                                return candidates.isEmpty()
                                        ? java.util.Optional.empty()
                                        : java.util.Optional.of(candidates.getFirst());
                            });
        }
        if (endpoint instanceof PrefillEndpoint prefill) {
            when(endpoints.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                            org.mockito.ArgumentMatchers.eq(role),
                            org.mockito.ArgumentMatchers.any()))
                    .thenReturn(List.of(prefill));
        } else if (endpoint instanceof DecodeEndpoint decode) {
            var view =
                    new DecodeEndpoint.SelectionSnapshot(
                            address,
                            worker.getGenerationId(),
                            worker.topologySnapshot(),
                            new org.flexlb.worker.status.WorkerStatus.CommittedWorkerStatus(
                                    worker.committedEngineObservation(),
                                    worker.appliedStatusCursor()),
                            0L,
                            new DecodeResources.CapacityUsage(0L, 1000000L, 1000000L, 0L, 0L));
            when(decode.selectionSnapshot()).thenReturn(view);
            when(endpoints.getEndpoint(org.flexlb.config.service.RoleType.DECODE, view.address()))
                    .thenReturn(selectedEndpoint);
            doAnswer(
                            call -> {
                                var capturedViews =
                                        new java.util.ArrayList<
                                                org.flexlb.worker.endpoint.DecodeEndpoint
                                                        .SelectionSnapshot>();

                                capturedViews.add(view);
                                return org.flexlb.worker.endpoint.EndpointTestSupport
                                        .decodeCandidates(capturedViews);
                            })
                    .when(endpoints)
                    .endpoints(
                            org.mockito.ArgumentMatchers.eq(
                                    org.flexlb.config.service.RoleType.DECODE),
                            org.mockito.ArgumentMatchers.any());
        }
        return new SelectionFixture(selectedEndpoint, endpoint, status);
    }

    private static RoutingResult<?, CapacityKey> selected(
            org.mockito.invocation.InvocationOnMock call, SelectionFixture selected) {
        if (selected.endpoint() instanceof PrefillEndpoint endpoint) {
            String address = selected.endpoint().ipPort();
            return RoutingResult.success(
                    new CostBasedPrefillStrategy.Selection(
                            address,
                            selected.selectedEndpoint().getStatus().getGenerationId(),
                            selected.status().getPrefillTime(),
                            0L,
                            0L,
                            0L,
                            null,
                            ((java.util.List<PrefillEndpoint>) call.getArgument(1))
                                    .stream()
                                            .filter(input -> input.ipPort().equals(address))
                                            .findFirst()
                                            .orElseThrow(),
                            null,
                            endpoint.selectionSnapshot()));
        }
        if (selected.endpoint() instanceof DecodeEndpoint endpoint) {
            return RoutingResult.success(endpoint.selectionSnapshot());
        }
        throw new AssertionError("P/D strategy fixture expected");
    }

    private static RequestResult selectedRoute(
            RequestContext context, SelectionFixture... selections) {
        return selectedRoute(context, null, selections);
    }

    private static RequestResult selectedRoute(
            RequestContext context,
            DecodeEndpoint.RequestRegistration decodeRegistration,
            SelectionFixture... selections) {
        var roles = new java.util.ArrayList<RoleSelection>();
        for (var selected : selections) {
            if (selected.endpoint() instanceof PrefillEndpoint prefill) {
                roles.add(
                        RoleSelection.prefill(
                                prefill,
                                selected.status(),
                                selected.status().getPrefillTime(),
                                mock(PrefillEndpoint.RequestRegistration.class)));
            } else if (selected.endpoint() instanceof DecodeEndpoint decode) {
                roles.add(RoleSelection.decode(decode, selected.status(), decodeRegistration));
            } else {
                roles.add(RoleSelection.auxiliary(selected.endpoint(), selected.status()));
            }
        }
        var route = new RequestResult(context, roles);
        return route;
    }

    private static DecodeEndpoint.RequestRegistration registerDecode(
            RequestContext context,
            DecodeEndpoint endpoint,
            DecodeResources.AdmissionCapacity capacity) {
        var request = context.getRequest();
        var acquired =
                endpoint.tryRegisterSelection(
                        endpoint.selectionSnapshot(),
                        request.getRequestId(),
                        request.getHardKvTokens(),
                        request.getExpectedKvTokens(),
                        context.getPriority(),
                        capacity);
        assertEquals(WorkerEndpoint.RegistrationStatus.REGISTERED, acquired.status());
        return acquired.value();
    }

    private static boolean stubPrefillCommit(PrefillEndpoint endpoint) {
        when(endpoint.commitSelection(any())).thenReturn(true);
        return true;
    }

    private record SelectionFixture(
            WorkerEndpoint selectedEndpoint, WorkerEndpoint endpoint, ServerStatus status) {}
}
