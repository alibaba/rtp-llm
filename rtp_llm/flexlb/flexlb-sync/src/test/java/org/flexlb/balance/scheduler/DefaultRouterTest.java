package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.policy.GroupRoutingPolicy;
import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
import org.flexlb.balance.strategy.EndpointSelection;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.balance.strategy.StaticCapacityExceededException;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Final selector/pin ownership contracts for {@link DefaultRouter}. */
class DefaultRouterTest {

    private ConfiguredLoadBalanceSelector selector;
    private GroupRoutingPolicy groupPolicy;
    private ModelMetaConfig modelMeta;

    @BeforeEach
    void setUp() {
        selector = mock(ConfiguredLoadBalanceSelector.class);
        groupPolicy = mock(GroupRoutingPolicy.class);
        modelMeta = mock(ModelMetaConfig.class);
        when(groupPolicy.route(org.mockito.ArgumentMatchers.any()))
                .thenReturn(GroupRoutingDecision.none());
    }

    @Test
    void invalidRequestFailsBeforePolicyOrEndpointSelection() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();

        QueueRoutingResult result = router.routeForQueue(new BalanceContext());

        assertEquals(QueueRoutingResult.Status.REJECTED, result.status());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                result.response().getCode());
        verify(groupPolicy, never()).route(org.mockito.ArgumentMatchers.any());
        verify(selector, never()).selectForQueue(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any());
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {
            "PREFILL", "DECODE", "PDFUSION", "VIT"})
    void missingRequiredRoleReturnsItsExactWaitDomain(RoleType role) {
        when(modelMeta.requiredRoles()).thenReturn(List.of(role));
        DefaultRouter router = router();
        BalanceContext context = context(11L);
        when(selector.selectForQueue(context, role, null))
                .thenReturn(EndpointSelection.unavailable(role));

        QueueRoutingResult blocked = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.BLOCKED, blocked.status());

        assertEquals(new PlacementKey(role, null), blocked.blocker());
    }

    @Test
    void projectedDecodeBlockUsesDecodeWaitDomain() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();
        BalanceContext context = context(111L);
        when(selector.selectForQueue(context, RoleType.PREFILL, null))
                .thenReturn(EndpointSelection.unavailable(RoleType.DECODE));

        QueueRoutingResult blocked = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.BLOCKED, blocked.status());

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
        when(selector.selectForQueue(context, RoleType.PREFILL, null))
                .thenReturn(EndpointSelection.selected(prefill.selection));
        when(selector.selectForQueue(context, RoleType.DECODE, "g1"))
                .thenThrow(new StaticCapacityExceededException(
                        "seq_len exceeds every Decode worker"));

        QueueRoutingResult rejected = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.REJECTED, rejected.status());

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                rejected.response().getCode());
        verify(prefill.selection).close();
    }

    @Test
    void queueRouteTransfersTheExactPrefillPinIntoAdmissionOwnership() {
        when(modelMeta.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        DefaultRouter router = router();
        BalanceContext context = context(21L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, context.getRequestId(), "p", 8001, "g1");
        when(selector.selectForQueue(context, RoleType.PREFILL, null))
                .thenReturn(EndpointSelection.selected(prefill.selection));

        QueueRoutingResult admitted = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.ADMITTED, admitted.status());

        assertTrue(admitted.admission().response().isSuccess());
        assertEquals(List.of(prefill.status),
                admitted.admission().response().getServerStatus());
        verify(prefill.selection).takeGenerationPin();
        verify(prefill.pin, never()).close();

        admitted.admission().close();
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
        when(selector.selectForQueue(context, RoleType.PREFILL, null))
                .thenReturn(EndpointSelection.selected(prefill.selection));
        when(selector.selectForQueue(
                context, RoleType.VIT, "selected-group"))
                .thenReturn(EndpointSelection.selected(vit.selection));

        QueueRoutingResult admitted = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.ADMITTED, admitted.status());
        admitted.admission().close();

        verify(selector).selectForQueue(context, RoleType.PREFILL, null);
        verify(selector).selectForQueue(
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
        when(groupPolicy.route(context))
                .thenReturn(GroupRoutingDecision.of("forced", "test-policy"));
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 41L, "p", 8001, "other");
        SelectionFixture vit = selection(
                RoleType.VIT, 41L, "v", 8002, "other");
        when(selector.selectForQueue(context, RoleType.PREFILL, "forced"))
                .thenReturn(EndpointSelection.selected(prefill.selection));
        when(selector.selectForQueue(context, RoleType.VIT, "forced"))
                .thenReturn(EndpointSelection.selected(vit.selection));

        QueueRoutingResult admitted = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.ADMITTED, admitted.status());
        admitted.admission().close();

        verify(selector).selectForQueue(
                context, RoleType.PREFILL, "forced");
        verify(selector).selectForQueue(context, RoleType.VIT, "forced");
    }

    @Test
    void laterSelectionFailureClosesEveryEarlierExactPinOwner() {
        when(modelMeta.requiredRoles())
                .thenReturn(List.of(RoleType.PREFILL, RoleType.VIT));
        DefaultRouter router = router();
        BalanceContext context = context(51L);
        SelectionFixture prefill = selection(
                RoleType.PREFILL, 51L, "p", 8001, "g1");
        when(selector.selectForQueue(context, RoleType.PREFILL, null))
                .thenReturn(EndpointSelection.selected(prefill.selection));
        when(selector.selectForQueue(context, RoleType.VIT, "g1"))
                .thenReturn(EndpointSelection.unavailable(RoleType.VIT));

        QueueRoutingResult blocked = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.BLOCKED, blocked.status());

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
        when(selector.selectForQueue(context, RoleType.PREFILL, null))
                .thenReturn(EndpointSelection.selected(prefill.selection));
        when(selector.selectForQueue(context, RoleType.DECODE, "g1"))
                .thenReturn(EndpointSelection.unavailable(RoleType.DECODE));

        QueueRoutingResult blocked = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.BLOCKED, blocked.status());

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
        when(selector.selectForQueue(context, RoleType.PREFILL, null))
                .thenReturn(EndpointSelection.selected(foreign.selection));

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
        when(selector.selectForQueue(context, RoleType.PREFILL, null))
                .thenReturn(EndpointSelection.selected(prefill.selection));

        QueueRoutingResult admitted = router.routeForQueue(context);
        assertEquals(QueueRoutingResult.Status.ADMITTED, admitted.status());
        admitted.admission().close();

        verify(selector).selectForQueue(context, RoleType.PREFILL, null);
    }

    private DefaultRouter router() {
        return new DefaultRouter(selector, groupPolicy, modelMeta);
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
        WorkerEndpoint endpoint = role == RoleType.PREFILL
                || role == RoleType.PDFUSION
                ? mock(PrefillEndpoint.class)
                : mock(WorkerEndpoint.class);
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
        return new SelectionFixture(selection, pin, status);
    }

    private record SelectionFixture(
            SelectedRole selection,
            WorkerEndpoint.GenerationPin pin,
            ServerStatus status) {
    }
}
