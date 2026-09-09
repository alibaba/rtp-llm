package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeMode;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** Final selector/pin ownership contracts for {@link DefaultRouter}. */
class DefaultRouterTest {

    private CostBasedPrefillStrategy prefillSelector;
    private DecodeSelector decodeSelector;
    private RandomStrategy vitSelector;
    private ConfigService configService;
    private ModelMetaConfig modelMeta;
    private RequestRegistry requests;

    @BeforeEach
    void setUp() {
        prefillSelector = mock(CostBasedPrefillStrategy.class);
        decodeSelector = mock(DecodeSelector.class);
        vitSelector = mock(RandomStrategy.class);
        configService = mock(ConfigService.class);
        modelMeta = mock(ModelMetaConfig.class);
        requests = mock(RequestRegistry.class);
        when(requests.claimAdmissionMutation(anyLong(), any())).thenReturn(mock(AdmissionMutation.class));
        when(requests.register(any())).thenReturn(new CompletableFuture<>());
        when(requests.commitRoute(any(), any())).thenAnswer(call ->
                ((java.util.function.BooleanSupplier) call.getArgument(1)).getAsBoolean()
                        ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED);
        when(requests.tryClaimRouteDelivery(any(), any())).thenAnswer(call -> {
            ((java.util.function.BooleanSupplier) call.getArgument(1)).getAsBoolean();
            RequestRegistry.DeliveryClaim claim = mock(RequestRegistry.DeliveryClaim.class);
            when(claim.item()).thenReturn(call.getArgument(0));
            return claim;
        });
        doAnswer(call -> {
            ScheduledRequest item = call.getArgument(0, RequestRegistry.DeliveryClaim.class).item();
            item.future().complete(item.routeResponse());
            return null;
        }).when(requests).beginRouteDelivery(any(), any(), anyLong());
        when(requests.publishDecisionResponseAsync(anyLong(), any(), any())).thenAnswer(call ->
                call.getArgument(1, CompletableFuture.class).complete(call.getArgument(2)));
    }

    @Test
    void invalidRequestFailsBeforePolicyOrEndpointSelection() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();

        PlacementResult<RouteAdmission, PlacementKey> result = router.select(new BalanceContext());

        assertEquals(PlacementResult.Status.REJECTED, result.status());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                result.rejection().getCode());
        verify(configService, never()).loadBalanceConfig();
        verifyNoInteractions(prefillSelector, decodeSelector, vitSelector);
    }

    @Test
    void directRouteCommitsRolesAndReleasesGenerationPins() {
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.PREFILL, RoleType.DECODE));
        DefaultRouter router = router();
        BalanceContext context = context(7L);
        context.getConfig().setScheduler(org.flexlb.config.SchedulerConfig.direct());
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        SelectionFixture prefill = selection(RoleType.PREFILL, 7L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 7L, "d", 8002, "g1");
        PrefillState.RouteReservation registration = mock(PrefillState.RouteReservation.class);
        DecodeEndpoint.ReservationHandle reservation = new DecodeEndpoint.ReservationHandle(1L, 7L, 2L);
        when(prefillSelector.select(context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(DecodeBinding.capture(context), "g1"))
                .thenReturn(PlacementResult.success(decode.selection));
        stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
        when(((PrefillEndpoint) prefill.endpoint).reserveUnqueuedRoute(eq(prefill.pin), any(ScheduledRequest.class), eq(1L)))
                .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, registration));
        when(((DecodeEndpoint) decode.endpoint).tryReservePlacementPinned(
                eq(decode.pin), eq(7L), eq(32L), eq(48L), eq(50)))
                .thenReturn(reservation);
        var permit = stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);

        Response response = scheduler(router, context).submit(context).join();

        assertTrue(response.isSuccess());
        assertEquals(List.of(prefill.status, decode.status),
                response.getServerStatus());
        verify(registration).close();
        verify(permit).transferToEngineLifecycle();
        verify((DecodeEndpoint) decode.endpoint, never())
                .releaseReservationExact(reservation);
        verify(prefill.pin).close();
        verify(decode.pin).close();
    }

    @Test
    void directRouteRollsBackDecodeWhenPrefillAdmissionIsFull() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        DefaultRouter router = router();
        BalanceContext context = context(8L);
        context.getConfig().setScheduler(org.flexlb.config.SchedulerConfig.direct());
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        SelectionFixture prefill = selection(RoleType.PREFILL, 8L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 8L, "d", 8002, "g1");
        DecodeEndpoint.ReservationHandle reservation = new DecodeEndpoint.ReservationHandle(1L, 8L, 2L);
        when(prefillSelector.select(context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(DecodeBinding.capture(context), "g1"))
                .thenReturn(PlacementResult.success(decode.selection));
        stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
        when(((DecodeEndpoint) decode.endpoint).tryReservePlacementPinned(
                eq(decode.pin), eq(8L), eq(32L), eq(48L), eq(50)))
                .thenReturn(reservation);
        stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);
        when(((PrefillEndpoint) prefill.endpoint).reserveUnqueuedRoute(eq(prefill.pin), any(ScheduledRequest.class), eq(1L)))
                .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.CAPACITY_FULL, null));

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                scheduler(router, context).submit(context).join().getCode());
        verify(prefillSelector).select(context, RoleType.PREFILL, null);
        verify((DecodeEndpoint) decode.endpoint).releaseReservationExact(reservation);
        verify(prefill.pin).close();
        verify(decode.pin).close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directAdmissionPassesZeroAndUnknownTimeToLifecycle(boolean unknown) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        BalanceContext context = context(81L);
        context.getConfig().setScheduler(org.flexlb.config.SchedulerConfig.direct());
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        SelectionFixture selected = selection(RoleType.PREFILL, 81L, "p", 8001, "g1");
        when(selected.selection.prefillWorkMs()).thenReturn(0L);
        when(prefillSelector.select(context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(selected.selection));
        PrefillEndpoint endpoint = (PrefillEndpoint) selected.endpoint;
        long committedAtMs = System.currentTimeMillis();
        stubRouteCommit(endpoint, new org.flexlb.balance.projection.WorkSnapshot(
                committedAtMs, List.of(), List.of(), unknown ? 1L : 0L));
        var registration = mock(PrefillState.RouteReservation.class);
        when(endpoint.reserveUnqueuedRoute(eq(selected.pin), any(), eq(0L)))
                .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, registration));

        assertTrue(scheduler(router(), context).submit(context).join().isSuccess());

        var precedingWork = org.mockito.ArgumentCaptor.forClass(org.flexlb.balance.projection.WorkSnapshot.class);
        verify(requests).beginRouteDelivery(any(), precedingWork.capture(), eq(0L));
        assertEquals(unknown ? java.util.OptionalLong.empty() : java.util.OptionalLong.of(0L),
                precedingWork.getValue().totalRemainingWorkMs());
        verify(registration).close();
    }

    @Test
    void directRequestRemainsInCanonicalLifecycleAfterResponseAndReconcilesEarlyDecodeEvidence() throws Exception {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        BalanceContext context = context(9L);
        context.getConfig().setScheduler(org.flexlb.config.SchedulerConfig.direct());
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        when(configService.loadBalanceConfig()).thenReturn(context.getConfig());
        requests = new RequestRegistry(configService,
                mock(org.flexlb.service.monitor.BatchSchedulerReporter.class),
                mock(org.flexlb.service.monitor.RequestSchedulerReporter.class));
        try {
            SelectionFixture prefill = selection(RoleType.PREFILL, 9L, "p", 8001, "g1");
            SelectionFixture decode = selection(RoleType.DECODE, 9L, "d", 8002, "g1");
            var reservation = new DecodeEndpoint.ReservationHandle(1L, 9L, 2L);
            when(prefillSelector.select(context, RoleType.PREFILL, null))
                    .thenReturn(PlacementResult.success(prefill.selection));
            when(decodeSelector.select(DecodeBinding.capture(context), "g1"))
                    .thenReturn(PlacementResult.success(decode.selection));
            stubPrefillCommit((PrefillEndpoint) prefill.endpoint);
            when(((PrefillEndpoint) prefill.endpoint).reserveUnqueuedRoute(eq(prefill.pin), any(ScheduledRequest.class), eq(1L)))
                    .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED,
                            mock(PrefillState.RouteReservation.class)));
            when(((DecodeEndpoint) decode.endpoint).tryReservePlacementPinned(
                    eq(decode.pin), eq(9L), eq(32L), eq(48L), eq(50))).thenReturn(reservation);
            stubDecodePermit((DecodeEndpoint) decode.endpoint, reservation);
            when(((DecodeEndpoint) decode.endpoint).isReservationAccepted(reservation)).thenReturn(true);

            assertTrue(scheduler(router(), context).submit(context).get(2L, TimeUnit.SECONDS).isSuccess());
            assertTrue(context.getFuture().get(2L, TimeUnit.SECONDS).isSuccess());
            RequestSlot slot = requests.requestSlot(9L);
            synchronized (slot) {
                assertTrue(slot.decodeOwnsRequest());
                assertTrue(slot.isLiveGeneration());
                assertEquals(RequestState.Phase.ACKNOWLEDGED, slot.snapshot().state());
            }
            requests.expireInactiveRequest(slot, System.currentTimeMillis()
                    + context.getConfig().getRequestLifecycle().getRequest().getTimeoutMs());
            assertEquals(RequestState.Phase.TIMED_OUT, requests.getRequestState(9L, 0L).state());
            assertEquals(0, requests.liveRequestCount());
            verify((DecodeEndpoint) decode.endpoint).expireReservationExact(reservation);
            verify((PrefillEndpoint) prefill.endpoint).expireCommittedItem(any(ScheduledRequest.class));
        } finally {
            requests.closeAdmissionAndAwaitMutations();
            requests.closeOutstandingAndTerminalize();
            requests.closeExpiration();
            requests.closePublisher();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directSelectionExceptionPublishesFailureAndClosesLifecycle(boolean fatal) throws Exception {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        BalanceContext context = context(10L);
        context.getConfig().setScheduler(org.flexlb.config.SchedulerConfig.direct());
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        when(configService.loadBalanceConfig()).thenReturn(context.getConfig());
        requests = new RequestRegistry(configService,
                mock(org.flexlb.service.monitor.BatchSchedulerReporter.class),
                mock(org.flexlb.service.monitor.RequestSchedulerReporter.class));
        try {
            Throwable failure = fatal ? new AssertionError("selection failed")
                    : new IllegalStateException("selection failed");
            when(prefillSelector.select(context, RoleType.PREFILL, null)).thenThrow(failure);
            if (fatal) {
                assertEquals(failure, assertThrows(AssertionError.class, () -> scheduler(router(), context).submit(context)));
            } else {
                var returned = scheduler(router(), context).submit(context);
                assertEquals(context.getFuture(), returned);
            }
            Response response = context.getFuture().get(2L, TimeUnit.SECONDS);
            assertEquals(8510, response.getCode());
            assertEquals("DISPATCH_FAILED", response.getErrorMessage());
            assertEquals(RequestState.Phase.FAILED, requests.getRequestState(10L, 0L).state());
            verifyNoInteractions(decodeSelector, vitSelector);
        } finally {
            requests.closeAdmissionAndAwaitMutations();
            requests.closeOutstandingAndTerminalize();
            requests.closeExpiration();
            requests.closePublisher();
        }
    }

    @ParameterizedTest
    @CsvSource({"32,16,32,48", "500,10000,500,10500",
            "9223372036854775806,100,9223372036854775806,9223372036854775807",
            "-1,100,0,100", "500,-1,500,500"})
    void queuedRouteRetainsSelectedDemandAndLimitsAcrossLaterContextChanges(
            long prompt, int output, long hardKv, long expectedKv) {
        var config = SchedulingTestConfig.newConfig();
        var context = RequestLifecycleTestSupport.context(config, 701L);
        context.getRequest().setSeqLen(prompt);
        context.getRequest().setMaxNewTokens(output);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(73, Long.MAX_VALUE));
        var limits = config.getRouter().getRoles().getDecode().getAvailability();
        limits.setMaxEngineRequests(1L);
        limits.setMaxKvUsagePercent(90L);
        var frozen = DecodeBinding.capture(context);
        var prefill = selection(RoleType.PREFILL, 701L, "10.0.0.1", 8080, "g1");
        var selectedDecode = selection(RoleType.DECODE, 701L, "10.0.0.2", 8080, "g1");
        var decode = (DecodeEndpoint) selectedDecode.endpoint();
        var reservation = new DecodeEndpoint.ReservationHandle(1L, 701L, 1L);
        when(decode.tryReservePlacementPinned(any(), anyLong(), anyLong(), anyLong(), anyInt()))
                .thenReturn(reservation);
        when(decode.acquireEngineDispatchPermit(reservation, frozen.capacity())).thenReturn(
                new DecodeEndpoint.EngineDispatchPermitAcquisition(
                        DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL, null));

        // Changes after selection must not change this request's publication mode or demand.
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_RESERVED);
        limits.setMaxEngineRequests(99L);
        limits.setMaxKvUsagePercent(1L);
        context.getRequest().setSeqLen(4_096L);
        context.getRequest().setMaxNewTokens(1_024);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(4, Long.MAX_VALUE));

        try (var route = RouteAdmission.prepare(context,
                List.of(prefill.selection(), selectedDecode.selection()), new Response(), frozen)) {
            assertTrue(route.reserveDecode());
            var item = route.createScheduledRequest(context, new CompletableFuture<>(), System.currentTimeMillis());
            assertSame(frozen.capacity(), route.decodeBinding().capacity());
            assertSame(frozen.capacity(), item.decodeBinding().capacity());
            assertSame(reservation, item.decodeBinding().reservation());
            assertSame(decode, item.decodeBinding().endpoint());
            assertEquals(hardKv, item.seqLen());
            assertEquals(DecodeMode.WAIT_AT_DISPATCH, frozen.mode());

            var delivery = PrefillAdmissionResources.prepareMember(item);
            assertFalse(delivery.accepted());
            assertTrue(delivery.boundary().unavailable());
            assertFalse(delivery.boundary().availability().isAvailable());
            verify(decode).isEngineDispatchPermitAvailable(701L, frozen.capacity());
            verify(decode).acquireEngineDispatchPermit(reservation, frozen.capacity());
            verify(decode).tryReservePlacementPinned(any(), eq(701L),
                    eq(hardKv), eq(expectedKv),
                    eq(73));
            verify(decode, never()).tryReservePlacementPinned(
                    any(), anyLong(), anyLong(), anyLong(), anyInt(), any());
        }
        verify(decode).releaseReservationExact(reservation);
    }

    private static void stubPrefillCommit(PrefillEndpoint prefill) {
        stubRouteCommit(prefill, new org.flexlb.balance.projection.WorkSnapshot(System.currentTimeMillis(), List.of(), List.of(), 0L));
    }

    private static void stubRouteCommit(PrefillEndpoint endpoint,
            org.flexlb.balance.projection.WorkSnapshot precedingWork) {
        var commit = mock(PrefillEndpoint.RouteCommitAdmission.class);
        var handoff = mock(PrefillState.CommittedHandoff.class);
        when(handoff.precedingWork()).thenReturn(precedingWork);
        when(endpoint.tryBeginRouteCommitAdmission()).thenReturn(commit);
        when(commit.commit(any(), any())).thenReturn(handoff);
    }

    private static DecodeEndpoint.EngineDispatchPermit stubDecodePermit(
            DecodeEndpoint endpoint, DecodeEndpoint.ReservationHandle reservation) {
        var permit = mock(DecodeEndpoint.EngineDispatchPermit.class);
        when(endpoint.acquireEngineDispatchPermit(eq(reservation), any()))
                .thenReturn(new DecodeEndpoint.EngineDispatchPermitAcquisition(
                        DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, permit));
        when(permit.transferToEngineLifecycle())
                .thenReturn(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED);
        return permit;
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {
            "PREFILL", "DECODE", "PDFUSION", "VIT"})
    void missingRequiredRoleReturnsItsExactWaitDomain(RoleType role) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(role));
        DefaultRouter router = router();
        BalanceContext context = context(11L);
        stubQueueSelection(
                context, role, null, PlacementResult.blocked(role));

        PlacementResult<RouteAdmission, PlacementKey> blocked = router.select(context);
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(role, null), blocked.blocker());
    }

    @Test
    void projectedDecodeBlockUsesDecodeWaitDomain() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();
        BalanceContext context = context(111L);
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.blocked(RoleType.DECODE));

        PlacementResult<RouteAdmission, PlacementKey> blocked = router.select(context);
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.DECODE, null),
                blocked.blocker());
    }

    @Test
    void staticCapacityFailureIsTerminalAndReleasesEarlierSelections() {
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.PREFILL, RoleType.DECODE));
        DefaultRouter router = router();
        BalanceContext context = context(12L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 12L, "p", 8001, "g1");
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(
                DecodeBinding.capture(context), "g1"))
                .thenReturn(PlacementResult.rejected(
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED)));

        PlacementResult<RouteAdmission, PlacementKey> rejected = router.select(context);
        assertEquals(PlacementResult.Status.REJECTED, rejected.status());

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                rejected.rejection().getCode());
        verify(prefill.selection).close();
    }

    @Test
    void queueRouteTransfersTheExactPrefillPinIntoAdmissionOwnership() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();
        BalanceContext context = context(21L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, context.getRequestId(), "p", 8001, "g1");
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));

        PlacementResult<RouteAdmission, PlacementKey> admitted = router.select(context);
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());

        assertTrue(admitted.value().response().isSuccess());
        assertEquals(List.of(prefill.status),
                admitted.value().response().getServerStatus());
        verify(prefill.selection).takeGenerationPin();
        verify(prefill.pin, never()).close();

        admitted.value().close();
        verify(prefill.pin).close();
    }

    @Test
    void firstSelectedGroupChainsToLaterRolesWhenPolicyDidNotForceOne() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        DefaultRouter router = router();
        BalanceContext context = context(31L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 31L, "p", 8001, "selected-group");
        SelectionFixture vit = selection(
                RoleType.VIT, 31L, "v", 8002, "selected-group");
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context, RoleType.VIT, "selected-group"))
                .thenReturn(vit.selection);

        PlacementResult<RouteAdmission, PlacementKey> admitted = router.select(context);
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).select(
                context, RoleType.PREFILL, null);
        verify(vitSelector).select(
                context, RoleType.VIT, "selected-group");
        verify(vit.pin).close();
        verify(prefill.pin).close();
    }

    @Test
    void policyGroupRemainsAuthoritativeAcrossEveryRole() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        DefaultRouter router = router();
        BalanceContext context = context(41L);
        TrafficPolicyConfig groupSelector = mock(TrafficPolicyConfig.class);
        context.getConfig().getRouter().setGroupSelector(groupSelector);
        when(groupSelector.resolveTargetGroup(context.getRequest()))
                .thenReturn(Optional.of("forced"));
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 41L, "p", 8001, "other");
        SelectionFixture vit = selection(
                RoleType.VIT, 41L, "v", 8002, "other");
        when(prefillSelector.select(
                context, RoleType.PREFILL, "forced"))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context, RoleType.VIT, "forced"))
                .thenReturn(vit.selection);

        PlacementResult<RouteAdmission, PlacementKey> admitted = router.select(context);
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).select(
                context, RoleType.PREFILL, "forced");
        verify(vitSelector).select(context, RoleType.VIT, "forced");
    }

    @Test
    void laterSelectionFailureClosesEveryEarlierExactPinOwner() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        DefaultRouter router = router();
        BalanceContext context = context(51L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 51L, "p", 8001, "g1");
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context, RoleType.VIT, "g1"))
                .thenReturn(null);

        PlacementResult<RouteAdmission, PlacementKey> blocked = router.select(context);
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.VIT, "g1"),
                blocked.blocker());
        verify(prefill.selection).close();
    }

    @Test
    void decodeMissAfterPrefillSuccessReleasesThePrefillPinOwner() {
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.PREFILL, RoleType.DECODE));
        DefaultRouter router = router();
        BalanceContext context = context(52L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 52L, "p", 8001, "g1");
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(
                DecodeBinding.capture(context), "g1"))
                .thenReturn(PlacementResult.blocked(RoleType.DECODE));

        PlacementResult<RouteAdmission, PlacementKey> blocked = router.select(context);
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(RoleType.DECODE, "g1"),
                blocked.blocker());
        verify(prefill.selection).close();
        verify(prefill.selection, never()).takeGenerationPin();
    }

    @Test
    void mismatchedSelectedRequestFailsClosedAndReleasesPins() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();
        BalanceContext context = context(61L);
        SelectionFixture foreign = selection(
                RoleType.PREFILL, 999L, "p", 8001, "g1");
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(foreign.selection));

        assertThrows(IllegalStateException.class,
                () -> router.select(context));

        verify(foreign.selection).close();
        verify(foreign.selection, never()).takeGenerationPin();
    }

    @Test
    void requiredTopologyIsSnapshottedAtConstruction() {
        List<RoleType> mutable = new ArrayList<>();
        mutable.add(RoleType.PREFILL);
        when(modelMeta.requiredRoles()).thenReturn(mutable);
        DefaultRouter router = router();
        mutable.clear();
        BalanceContext context = context(71L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 71L, "p", 8001, "g1");
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));

        PlacementResult<RouteAdmission, PlacementKey> admitted = router.select(context);
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).select(
                context, RoleType.PREFILL, null);
    }

    private RequestScheduler scheduler(DefaultRouter router, BalanceContext context) {
        when(configService.loadBalanceConfig()).thenReturn(context.getConfig());
        return new RequestScheduler(configService, router,
                mock(org.flexlb.balance.endpoint.EndpointRegistry.class),
                mock(org.flexlb.service.monitor.BatchSchedulerReporter.class),
                mock(org.flexlb.balance.eviction.EvictionManager.class), requests,
                new PlacementAvailability());
    }

    private DefaultRouter router() {
        return new DefaultRouter(
                prefillSelector,
                decodeSelector,
                vitSelector,
                configService,
                modelMeta);
    }

    private void stubQueueSelection(
            BalanceContext context,
            RoleType role,
            String group,
            PlacementResult<SelectedRole, RoleType> result) {
        switch (role) {
            case PREFILL, PDFUSION -> when(prefillSelector.select(
                    context, role, group)).thenReturn(result);
            case DECODE -> when(decodeSelector.select(
                    DecodeBinding.capture(context), group)).thenReturn(result);
            case VIT -> when(vitSelector.select(context, role, group))
                    .thenReturn(result.value());
            case FRONTEND -> throw new IllegalArgumentException();
        }
    }

    private static BalanceContext context(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(32L);
        request.setMaxNewTokens(16);
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    private static SelectionFixture selection(
            RoleType role,
            long requestId,
            String ip,
            int httpPort,
            String group) {
        SelectedRole selection = mock(SelectedRole.class);
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        WorkerEndpoint endpoint = switch (role) {
            case PREFILL, PDFUSION -> mock(PrefillEndpoint.class);
            case DECODE -> mock(DecodeEndpoint.class);
            default -> mock(WorkerEndpoint.class);
        };
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setRequestId(requestId);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGroup(group);
        when(selection.serverStatus()).thenReturn(status);
        when(selection.prefillWorkMs()).thenReturn(1L);
        when(selection.takeGenerationPin()).thenReturn(pin);
        when(pin.endpoint()).thenReturn(endpoint);
        return new SelectionFixture(selection, pin, endpoint, status);
    }

    private record SelectionFixture(
            SelectedRole selection,
            WorkerEndpoint.GenerationPin pin,
            WorkerEndpoint endpoint,
            ServerStatus status) {
    }

}
