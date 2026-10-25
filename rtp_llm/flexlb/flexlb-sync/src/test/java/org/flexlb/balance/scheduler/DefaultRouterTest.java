package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.endpoint.prefill.CommittedHandoff;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
import org.flexlb.balance.scheduler.decode.DecodeDecideInfo;
import org.flexlb.balance.scheduler.decode.DecodeMode;
import org.flexlb.balance.scheduler.decode.DecodeRequirement;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.EncoderStrategy;
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
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** Final selector/pin ownership contracts for {@link DefaultRouter}. */
class DefaultRouterTest {
    private DeliveryClaim lastDelivery;

    private CostBasedPrefillStrategy prefillSelector;
    private DecodeSelector decodeSelector;
    private RandomStrategy vitSelector;
    private EncoderStrategy encoderSelector;
    private ConfigService configService;
    private ModelMetaConfig modelMeta;
    private RequestRegistry requests;

    @BeforeEach
    void setUp() {
        prefillSelector = mock(CostBasedPrefillStrategy.class);
        decodeSelector = mock(DecodeSelector.class);
        vitSelector = mock(RandomStrategy.class);
        encoderSelector = mock(EncoderStrategy.class);
        configService = mock(ConfigService.class);
        modelMeta = mock(ModelMetaConfig.class);
        requests = mock(RequestRegistry.class);
        when(requests.claimAdmissionHandle(anyString(), any())).thenReturn(mock(AdmissionHandle.class));
        when(requests.register(any())).thenReturn(new CompletableFuture<>());
        when(requests.commitRoute(any(), any())).thenAnswer(call ->
                ((java.util.function.BooleanSupplier) call.getArgument(1)).getAsBoolean()
                        ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED);
        when(requests.claimRouteDelivery(any(), any())).thenAnswer(call -> {
            call.getArgument(1, PrefillAdmissionResources.CommittedAdmissionOwner.class).transferToEndpoint(call.getArgument(0));
            DeliveryClaim claim = mock(DeliveryClaim.class);
            ScheduledRequest item = call.getArgument(0);
            lastDelivery = claim;
            doAnswer(inv -> {
                item.future().complete(item.routeResponse());
                return null;
            }).when(requests).publishRoute(eq(claim), any(), anyLong());
            return claim;
        });
        when(requests.publishDecisionResponseAsync(anyString(), any(), any())).thenAnswer(call ->
                call.getArgument(1, CompletableFuture.class).complete(call.getArgument(2)));
    }

    @Test
    void invalidRequestFailsBeforePolicyOrEndpointSelection() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();

        PlacementResult<RouteAdmission, PlacementKey> result =
                select(router, new BalanceContext(SchedulingTestConfig.newConfig()));

        assertEquals(PlacementResult.Status.REJECTED, result.status());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                result.failure().getCode());
        verify(configService, never()).loadBalanceConfig();
        verifyNoInteractions(prefillSelector, decodeSelector, vitSelector);
    }

    @Test
    void explicitRolesSelectOnlyConfiguredRequestedRoles() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        BalanceContext context = context(701L);
        context.setRequestedRoles(Set.of(RoleType.PREFILL));
        SelectionFixture prefill = selection(RoleType.PREFILL, 701L, "p", 8001, "g1");
        when(prefillSelector.select(context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));

        try (var result = select(router(), context).value()) {
            assertEquals(RoleType.PREFILL, result.response().getServerStatus().getFirst().getRole());
        }
        verifyNoInteractions(decodeSelector, encoderSelector);
    }

    @Test
    void explicitUnconfiguredRoleIsInvalidRequest() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        BalanceContext context = context(702L);
        context.setRequestedRoles(Set.of(RoleType.ENCODER));

        var result = select(router(), context);

        assertEquals(PlacementResult.Status.REJECTED, result.status());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), result.failure().getCode());
        verifyNoInteractions(prefillSelector, encoderSelector);
    }

    @Test
    void encoderOnlyUsesEncoderQueueWhenQueueingIsEnabled() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.ENCODER));
        BalanceContext context = context(703L);
        context.setRequestedRoles(Set.of(RoleType.ENCODER));
        SelectionFixture encoder = selection(RoleType.ENCODER, 703L, "encoder", 8003, "g1");
        when(encoderSelector.select(context, null)).thenReturn(encoder.selection);
        when(requests.claimEncoderRoute(eq("703"), any(), eq(encoder.pin)))
                .thenReturn(true);
        when(requests.publishEncoderRoute(eq("703"), any(), any()))
                .thenAnswer(call -> {
                    call.getArgument(1, CompletableFuture.class)
                            .complete(call.getArgument(2, Response.class));
                    return true;
                });

        RequestScheduler scheduler = scheduler(router(), context);
        Response response;
        try {
            response = scheduler.submit(context).join();
        } finally {
            scheduler.closePlacement();
        }

        assertTrue(response.isSuccess());
        assertEquals(RequestPhase.ENCODER, context.getRequestPhase());
        assertEquals(List.of(encoder.status), response.getServerStatus());
        verify(encoder.pin).close();
        verifyNoInteractions(prefillSelector, decodeSelector, vitSelector);
    }

    @Test
    void unavailableEncoderRetainsItsFailureReasonAndPolicyGroup() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.ENCODER));
        BalanceContext context = context(704L);
        context.setRequestedRoles(Set.of(RoleType.ENCODER));
        PlacementResult<SelectedRole, PlacementKey> result = router().selectEncoder(context);

        assertEquals(PlacementResult.Status.BLOCKED, result.status());
        assertEquals(StrategyErrorType.NO_ENCODER_WORKER.getErrorCode(), result.failure().getCode());
        assertEquals(new PlacementKey(RoleType.ENCODER, null), result.blocker());
        verify(encoderSelector).select(context, null);
        verifyNoInteractions(configService);
    }

    @ParameterizedTest
    @CsvSource({"32,16,32,48", "500,10000,500,10500", "9223372036854775806,100,9223372036854775806,9223372036854775807",
            "-1,100,0,100", "500,-1,500,500"})
    void queuedRouteKeepsFrozenDecodeRequirement(long prompt, int output, long requiredKv, long kvBudget) {
        var config = SchedulingTestConfig.newConfig();
        var context = RequestLifecycleTestSupport.context(config, 701L);
        context.getRequest().setSeqLen(prompt);
        context.getRequest().setMaxNewTokens(output);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(73, Long.MAX_VALUE));
        var limits = config.getRouter().getRoles().getDecode().getAvailability();
        limits.setMaxEngineRequests(1L);
        limits.setMaxKvUsagePercent(90L);
        var estimator = config.getRouter().getRoles().getDecode().getCostEstimator();
        estimator.setExpression("2 * running_size / max_running_size + 3 * kvcache_used_ratio");
        var frozen = DecodeDecideInfo.capture(context);
        var prefill = selection(RoleType.PREFILL, 701L, "10.0.0.1", 8080, "g1");
        var selectedDecode = selection(RoleType.DECODE, 701L, "10.0.0.2", 8080, "g1");
        var decode = (DecodeEndpoint) selectedDecode.endpoint();
        var reservation = new DecodeEndpoint.ReservationHandle(1L, "701", 1L);
        when(decode.reserve(any(), anyString(), anyLong(), anyLong(), anyInt(), isNull()))
                .thenReturn(reservation);
        when(decode.acquireDispatchPermit(reservation, frozen.requirement().capacity()))
                .thenReturn(new DecodeEndpoint.EngineDispatchPermitAcquisition(
                        DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL, null));

        // Changes after selection must not change this request's publication mode, demand or cost.
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_RESERVED);
        limits.setMaxEngineRequests(99L);
        limits.setMaxKvUsagePercent(1L);
        estimator.setExpression("7 * running_size / max_running_size + 11 * kvcache_used_ratio");
        assertNotSame(estimator.compiledFormula(), frozen.requirement().costFormula());
        assertEquals("2 * running_size / max_running_size + 3 * kvcache_used_ratio",
                frozen.requirement().costFormula().expression());
        context.getRequest().setSeqLen(4_096L);
        context.getRequest().setMaxNewTokens(1_024);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(4, Long.MAX_VALUE));

        try (var route = RouteAdmission.prepare(context,
                List.of(prefill.selection(), selectedDecode.selection()), new Response(), frozen)) {
            assertTrue(route.reserveDecode());
            var item = route.createScheduledRequest(context, new CompletableFuture<>(), System.currentTimeMillis());
            assertSame(frozen.requirement().capacity(), route.decodeDecideInfo().requirement().capacity());
            assertSame(frozen.requirement().capacity(), item.decodeDecideInfo().requirement().capacity());
            assertSame(frozen.requirement().costFormula(), route.decodeDecideInfo().requirement().costFormula());
            assertSame(frozen.requirement().costFormula(), item.decodeDecideInfo().requirement().costFormula());
            assertSame(reservation, item.decodeDecideInfo().reservation());
            assertSame(decode, item.decodeDecideInfo().endpoint());
            assertEquals(requiredKv, item.seqLen());
            assertEquals(DecodeMode.WAIT_AT_PLACEMENT, frozen.requirement().mode());

            var delivery = PrefillAdmissionResources.prepareMember(item);
            assertFalse(delivery.accepted());
            assertTrue(delivery.blockReason().unavailable());
            assertFalse(delivery.blockReason().availability().isAvailable());
            verify(decode).shouldRetryDispatch("701", frozen.requirement().capacity());
            verify(decode).acquireDispatchPermit(reservation, frozen.requirement().capacity());
            verify(decode).reserve(any(), eq("701"), eq(requiredKv), eq(kvBudget), eq(73), isNull());
            verify(decode, never()).reserve(any(), anyString(), anyLong(), anyLong(), anyInt());
        }
        verify(decode).release(reservation, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
    }

    @Test
    void configReplacementAppliesDecodeRequestLimitOnlyToNewRequests() {
        FlexlbConfig initial = SchedulingTestConfig.newConfig();
        initial.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        AtomicReference<FlexlbConfig> current = new AtomicReference<>(initial);
        BalanceContext waiting = RequestLifecycleTestSupport.context(current.get(), 701L);
        var usage = new DecodeEndpoint.CapacityUsage(2L, 1_000L, 1_000L, 0L, 0L);
        DecodeRequirement firstAttempt = DecodeRequirement.capture(waiting);
        assertFalse(firstAttempt.capacity().evaluate(usage, 10L, 10L).fits());

        FlexlbConfig increased = SchedulingTestConfig.newConfig();
        increased.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(4L);
        current.set(increased);
        DecodeRequirement retried = DecodeRequirement.capture(waiting);
        DecodeRequirement newRequest =
                DecodeRequirement.capture(RequestLifecycleTestSupport.context(current.get(), 702L));
        assertEquals(1L, firstAttempt.capacity().maxEngineRequests());
        assertEquals(1L, retried.capacity().maxEngineRequests());
        assertFalse(retried.capacity().evaluate(usage, 10L, 10L).fits());
        assertEquals(4L, newRequest.capacity().maxEngineRequests());
        assertTrue(newRequest.capacity().evaluate(usage, 10L, 10L).fits());

        FlexlbConfig reduced = SchedulingTestConfig.newConfig();
        reduced.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
        current.set(reduced);
        assertEquals(1L, DecodeRequirement.capture(waiting).capacity().maxEngineRequests());
        assertEquals(4L, newRequest.capacity().maxEngineRequests());
        DecodeRequirement nextAttempt =
                DecodeRequirement.capture(RequestLifecycleTestSupport.context(current.get(), 703L));
        assertEquals(2L, nextAttempt.capacity().maxEngineRequests());
        assertFalse(nextAttempt.capacity().evaluate(usage, 10L, 10L).fits());
    }

    @Test
    void configReplacementAppliesDecodeKvPercentageOnlyToNewRequests() {
        FlexlbConfig initial = SchedulingTestConfig.newConfig();
        initial.getRouter().getRoles().getDecode().getAvailability().setMaxKvUsagePercent(90L);
        AtomicReference<FlexlbConfig> current = new AtomicReference<>(initial);
        BalanceContext waiting = RequestLifecycleTestSupport.context(current.get(), 704L);
        waiting.getRequest().setSeqLen(10L);
        waiting.getRequest().setMaxNewTokens(20);
        var usage = new DecodeEndpoint.CapacityUsage(0L, 1_000L, 120L, 0L, 880L);
        DecodeRequirement firstAttempt = DecodeRequirement.capture(waiting);
        assertFalse(firstAttempt.capacity().evaluate(usage, 10L, 30L).fits());

        FlexlbConfig increased = SchedulingTestConfig.newConfig();
        increased.getRouter().getRoles().getDecode().getAvailability().setMaxKvUsagePercent(95L);
        current.set(increased);
        DecodeRequirement retried = DecodeRequirement.capture(waiting);
        DecodeRequirement newRequest =
                DecodeRequirement.capture(RequestLifecycleTestSupport.context(current.get(), 705L));
        assertEquals(90L, retried.capacity().maxKvUsagePercent());
        assertFalse(retried.capacity().evaluate(usage, 10L, 30L).fits());
        assertEquals(95L, newRequest.capacity().maxKvUsagePercent());
        assertTrue(newRequest.capacity().evaluate(usage, 10L, 30L).fits());

        FlexlbConfig reduced = SchedulingTestConfig.newConfig();
        reduced.getRouter().getRoles().getDecode().getAvailability().setMaxKvUsagePercent(80L);
        current.set(reduced);
        assertEquals(90L, DecodeRequirement.capture(waiting).capacity().maxKvUsagePercent());
        assertEquals(95L, newRequest.capacity().maxKvUsagePercent());
        assertEquals(80L,
                DecodeRequirement.capture(RequestLifecycleTestSupport.context(current.get(), 706L))
                        .capacity()
                        .maxKvUsagePercent());
    }

    private static void stubPrefillCommit(PrefillEndpoint prefill) {
        stubRouteCommit(prefill, new org.flexlb.balance.estimate.WorkView(System.currentTimeMillis(), List.of(), List.of(), 0L));
    }

    private static void stubRouteCommit(PrefillEndpoint endpoint, org.flexlb.balance.estimate.WorkView precedingWork) {
        var commit = mock(PrefillEndpoint.RouteCommitAdmission.class);
        var handoff = mock(CommittedHandoff.class);
        when(handoff.precedingWork()).thenReturn(precedingWork);
        when(endpoint.tryBeginRouteCommitAdmission()).thenReturn(commit);
        when(commit.commit(any(), any())).thenReturn(handoff);
    }

    private static DecodeEndpoint.EngineDispatchPermit stubDecodePermit(
            DecodeEndpoint endpoint, DecodeEndpoint.ReservationHandle reservation) {
        var permit = mock(DecodeEndpoint.EngineDispatchPermit.class);
        when(endpoint.acquireDispatchPermit(eq(reservation), any()))
                .thenReturn(new DecodeEndpoint.EngineDispatchPermitAcquisition(
                        DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, permit));
        when(permit.dispatch())
                .thenReturn(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED);
        return permit;
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE", "PDFUSION", "VIT"})
    void missingRequiredRoleReturnsItsExactWaitDomain(RoleType role) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(role));
        DefaultRouter router = router();
        BalanceContext context = context(11L);
        stubQueueSelection(
                context, role, null, PlacementResult.blocked(role));

        PlacementResult<RouteAdmission, PlacementKey> blocked = select(router, context);
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(role, null), blocked.blocker());
    }

    @Test
    void estimatedDecodeBlockUsesDecodeWaitDomain() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();
        BalanceContext context = context(111L);
        when(prefillSelector.select(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.blocked(RoleType.DECODE));

        PlacementResult<RouteAdmission, PlacementKey> blocked = select(router, context);
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
        when(decodeSelector.select(context, DecodeRequirement.capture(context), "g1"))
                .thenReturn(PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED)));

        PlacementResult<RouteAdmission, PlacementKey> rejected = select(router, context);
        assertEquals(PlacementResult.Status.REJECTED, rejected.status());

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                rejected.failure().getCode());
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

        PlacementResult<RouteAdmission, PlacementKey> admitted = select(router, context);
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

        PlacementResult<RouteAdmission, PlacementKey> admitted = select(router, context);
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

        PlacementResult<RouteAdmission, PlacementKey> admitted = select(router, context);
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

        PlacementResult<RouteAdmission, PlacementKey> blocked = select(router, context);
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
        when(decodeSelector.select(context, DecodeRequirement.capture(context), "g1"))
                .thenReturn(PlacementResult.blocked(RoleType.DECODE));

        PlacementResult<RouteAdmission, PlacementKey> blocked = select(router, context);
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

        assertThrows(IllegalStateException.class, () -> select(router, context));

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

        PlacementResult<RouteAdmission, PlacementKey> admitted = select(router, context);
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
                new PlacementAvailability(), new QueueCapacity());
    }

    private DefaultRouter router() {
        return new DefaultRouter(
                prefillSelector,
                decodeSelector,
                vitSelector,
                encoderSelector,
                configService,
                modelMeta);
    }

    private void stubQueueSelection(BalanceContext context,
                                    RoleType role,
                                    String group,
                                    PlacementResult<SelectedRole, RoleType> result) {
        switch (role) {
            case PREFILL, PDFUSION -> when(prefillSelector.select(
                    context, role, group)).thenReturn(result);
            case DECODE ->
                when(decodeSelector.select(context, DecodeRequirement.capture(context), group)).thenReturn(result);
            case VIT -> when(vitSelector.select(context, role, group))
                    .thenReturn(result.value());
            case FRONTEND -> throw new IllegalArgumentException();
        }
    }

    private static BalanceContext context(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        Request request = new Request();
        request.setRequestId(Long.toString(requestId));
        request.setSeqLen(32L);
        request.setMaxNewTokens(16);
        BalanceContext context = new BalanceContext(config);
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
        return selection(role, Long.toString(requestId), ip, httpPort, group);
    }

    private static SelectionFixture selection(
            RoleType role,
            String requestId,
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
        when(endpoint.ipPort()).thenReturn(status.getLogicalIpPort());
        return new SelectionFixture(selection, pin, endpoint, status);
    }

    private record SelectionFixture(
            SelectedRole selection,
            WorkerEndpoint.GenerationPin pin,
            WorkerEndpoint endpoint,
            ServerStatus status) {
    }

    private static PlacementResult<RouteAdmission, PlacementKey> select(DefaultRouter router, BalanceContext context) {
        return router.select(context, router.resolvePolicyGroup(context));
    }
}
