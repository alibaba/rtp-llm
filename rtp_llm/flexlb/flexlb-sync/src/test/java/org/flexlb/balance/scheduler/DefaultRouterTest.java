package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
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
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Final selector/pin ownership contracts for {@link DefaultRouter}. */
class DefaultRouterTest {

    private CostBasedPrefillStrategy prefillSelector;
    private CostBasedDecodeStrategy decodeSelector;
    private RandomStrategy vitSelector;
    private ConfigService configService;
    private ModelMetaConfig modelMeta;

    @BeforeEach
    void setUp() {
        prefillSelector = mock(CostBasedPrefillStrategy.class);
        decodeSelector = mock(CostBasedDecodeStrategy.class);
        vitSelector = mock(RandomStrategy.class);
        configService = mock(ConfigService.class);
        modelMeta = mock(ModelMetaConfig.class);
    }

    @Test
    void invalidRequestFailsBeforePolicyOrEndpointSelection() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();

        PlacementResult<QueueRouteAdmission, PlacementKey> result = router.routeForQueue(new BalanceContext());

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
        SelectionFixture prefill = selection(RoleType.PREFILL, 7L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 7L, "d", 8002, "g1");
        PrefillState.DirectRegistration registration = mock(PrefillState.DirectRegistration.class);
        DecodeEndpoint.ReservationHandle reservation = new DecodeEndpoint.ReservationHandle(1L, 7L, 2L);
        when(prefillSelector.select(context, RoleType.PREFILL, null)).thenReturn(prefill.selection);
        when(decodeSelector.select(context, RoleType.DECODE, "g1"))
                .thenReturn(PlacementResult.success(decode.selection));
        when(((PrefillEndpoint) prefill.endpoint).registerDirectRequest(
                prefill.pin, 7L, 0L)).thenReturn(registration);
        when(((DecodeEndpoint) decode.endpoint).reservePinned(
                decode.pin, 7L, 32L, 48L, 50))
                .thenReturn(reservation);

        Response response = router.routeDirect(context);

        assertTrue(response.isSuccess());
        assertEquals(List.of(prefill.status, decode.status),
                response.getServerStatus());
        verify(registration).commit();
        verify(registration, never()).close();
        verify((DecodeEndpoint) decode.endpoint, never())
                .releaseReservationExact(reservation);
        verify(prefill.pin).close();
        verify(decode.pin).close();
    }

    @Test
    void directRouteRollsBackEarlierRolesWhenALaterReservationFails() {
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.PREFILL, RoleType.DECODE));
        DefaultRouter router = router();
        BalanceContext context = context(8L);
        SelectionFixture prefill = selection(RoleType.PREFILL, 8L, "p", 8001, "g1");
        SelectionFixture decode = selection(RoleType.DECODE, 8L, "d", 8002, "g1");
        PrefillState.DirectRegistration registration = mock(PrefillState.DirectRegistration.class);
        when(prefillSelector.select(context, RoleType.PREFILL, null)).thenReturn(prefill.selection);
        when(decodeSelector.select(context, RoleType.DECODE, "g1"))
                .thenReturn(PlacementResult.success(decode.selection));
        when(((PrefillEndpoint) prefill.endpoint).registerDirectRequest(
                prefill.pin, 8L, 0L)).thenReturn(registration);
        when(((DecodeEndpoint) decode.endpoint).reservePinned(
                decode.pin, 8L, 32L, 48L, 50))
                .thenThrow(new IllegalStateException("decode full"));

        assertThrows(IllegalStateException.class,
                () -> router.routeDirect(context));

        verify(registration, never()).commit();
        verify(registration).close();
        verify(prefill.pin).close();
        verify(decode.pin).close();
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

        PlacementResult<QueueRouteAdmission, PlacementKey> blocked = router.routeForQueue(context);
        assertEquals(PlacementResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(role, null), blocked.blocker());
    }

    @Test
    void projectedDecodeBlockUsesDecodeWaitDomain() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();
        BalanceContext context = context(111L);
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.blocked(RoleType.DECODE));

        PlacementResult<QueueRouteAdmission, PlacementKey> blocked = router.routeForQueue(context);
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
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(
                context, RoleType.DECODE, "g1"))
                .thenReturn(PlacementResult.rejected(
                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED)));

        PlacementResult<QueueRouteAdmission, PlacementKey> rejected = router.routeForQueue(context);
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
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));

        PlacementResult<QueueRouteAdmission, PlacementKey> admitted = router.routeForQueue(context);
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
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context, RoleType.VIT, "selected-group"))
                .thenReturn(vit.selection);

        PlacementResult<QueueRouteAdmission, PlacementKey> admitted = router.routeForQueue(context);
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).selectForQueue(
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
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, "forced"))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context, RoleType.VIT, "forced"))
                .thenReturn(vit.selection);

        PlacementResult<QueueRouteAdmission, PlacementKey> admitted = router.routeForQueue(context);
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).selectForQueue(
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
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(vitSelector.select(context, RoleType.VIT, "g1"))
                .thenReturn(null);

        PlacementResult<QueueRouteAdmission, PlacementKey> blocked = router.routeForQueue(context);
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
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));
        when(decodeSelector.select(
                context, RoleType.DECODE, "g1"))
                .thenReturn(PlacementResult.blocked(RoleType.DECODE));

        PlacementResult<QueueRouteAdmission, PlacementKey> blocked = router.routeForQueue(context);
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
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(foreign.selection));

        assertThrows(IllegalStateException.class,
                () -> router.routeForQueue(context));

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
        when(prefillSelector.selectForQueue(
                context, RoleType.PREFILL, null))
                .thenReturn(PlacementResult.success(prefill.selection));

        PlacementResult<QueueRouteAdmission, PlacementKey> admitted = router.routeForQueue(context);
        assertEquals(PlacementResult.Status.SUCCESS, admitted.status());
        admitted.value().close();

        verify(prefillSelector).selectForQueue(
                context, RoleType.PREFILL, null);
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
            case PREFILL, PDFUSION -> when(prefillSelector.selectForQueue(
                    context, role, group)).thenReturn(result);
            case DECODE -> when(decodeSelector.select(
                    context, role, group)).thenReturn(result);
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
