package org.flexlb.balance.schedulev2.main;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.Request;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.RequestTable;
import org.flexlb.balance.schedulev2.Stage;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class MainQueueLoopTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final QueueCapacity queueCapacity = new QueueCapacity();
    private final LoopSignal loopSignal = new LoopSignal();
    private final RequestTable requestTable = new RequestTable();
    private final MainQueue mainQueue = new MainQueue(queueCapacity, loopSignal, false);
    private final RequestCleanup cleanup = new RequestCleanup(requestTable, mainQueue, queueCapacity, loopSignal);
    private final EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
    private final PrefillPicker picker = mock(PrefillPicker.class);
    private final List<Request> sent = new ArrayList<>();
    private final List<SelectedRole> sentSelections = new ArrayList<>();
    private final MainQueueLoop loop;

    MainQueueLoopTest() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(endpointRegistry.getEndpointCount(RoleType.PREFILL)).thenReturn(1);
        loop = new MainQueueLoop(mainQueue, loopSignal, cleanup, queueCapacity, endpointRegistry,
                new PlacementAvailability(), configService, picker, RoleType.PREFILL, (request, selected) -> {
                    sent.add(request);
                    sentSelections.add(selected);
                });
    }

    @Test
    void pickedRequestHoldsQueueSlotAndGoesToPrefillQueue() {
        Request request = add("1", "g");
        SelectedRole selected = mock(SelectedRole.class);
        when(picker.pick(request.ctx(), RoleType.PREFILL, "g")).thenReturn(PlacementResult.success(selected));

        assertTrue(loop.runOnce());

        assertEquals(List.of(request), sent);
        assertSame(selected, sentSelections.get(0));
        assertEquals(Stage.PREFILL_WAITING, request.stage());
        assertEquals(0, mainQueue.size());
        assertEquals(1, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    @Test
    void requestWaitsWhenQueueSlotsAreFull() {
        when(endpointRegistry.getEndpointCount(RoleType.PREFILL)).thenReturn(0);
        Request request = add("1", "g");

        assertFalse(loop.runOnce());

        assertEquals(Stage.MAIN_WAITING, request.stage());
        assertEquals(1, mainQueue.size());
    }

    @Test
    void blockedGroupDoesNotStopOtherGroups() {
        Request blocked = add("1", "a");
        Request other = add("2", "b");
        Request sameGroup = add("3", "a");
        when(picker.pick(blocked.ctx(), RoleType.PREFILL, "a")).thenReturn(PlacementResult.blocked(RoleType.PREFILL));
        when(picker.pick(other.ctx(), RoleType.PREFILL, "b"))
                .thenReturn(PlacementResult.success(mock(SelectedRole.class)));

        assertTrue(loop.runOnce());

        assertEquals(List.of(other), sent);
        assertEquals(Stage.MAIN_WAITING, blocked.stage());
        assertEquals(Stage.MAIN_WAITING, sameGroup.stage());
        assertNull(blocked.takeQueueSlot());
        assertEquals(1, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
    }

    @Test
    void rejectedSelectionFailsRequestAndReleasesEverything() {
        Request request = add("1", "g");
        Response failure = Response.buildErrorResponse(StrategyErrorType.INVALID_REQUEST, "bad request");
        when(picker.pick(request.ctx(), RoleType.PREFILL, "g")).thenReturn(PlacementResult.rejected(failure));

        assertTrue(loop.runOnce());

        assertSame(failure, request.future().join());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(0, mainQueue.size());
    }

    @Test
    void requestCancelledDuringPickClosesSelectionAndReleasesSlot() {
        Request request = add("1", "g");
        SelectedRole selected = mock(SelectedRole.class);
        when(picker.pick(eq(request.ctx()), eq(RoleType.PREFILL), any())).thenAnswer(call -> {
            request.cancel(CancelReason.CLIENT_CANCELLED);
            return PlacementResult.success(selected);
        });

        loop.runOnce();

        verify(selected).close();
        assertTrue(sent.isEmpty());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    private Request add(String requestId, String group) {
        org.flexlb.dao.loadbalance.Request body = new org.flexlb.dao.loadbalance.Request();
        body.setRequestId(requestId);
        body.setSeqLen(16L);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(body);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 60_000L));
        Request request = new Request(context, new CompletableFuture<>(), System.currentTimeMillis());
        request.future().whenComplete((response, failure) -> cleanup.releaseEnded(request));
        requestTable.add(request);
        when(picker.groupOf(context)).thenReturn(group);
        assertNull(mainQueue.add(request, config));
        return request;
    }
}
