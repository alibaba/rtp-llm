package org.flexlb.balance.schedulev2.main;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.decode.DecodeQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestTable;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class MainQueueLoopTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final QueueCapacity queueCapacity = new QueueCapacity();
    private final LoopSignal loopSignal = new LoopSignal();
    private final RequestTable requestTable = new RequestTable();
    private final MainQueue mainQueue = new MainQueue(queueCapacity, loopSignal, false);
    private final RequestCleanup cleanup = new RequestCleanup(requestTable, mainQueue, queueCapacity, loopSignal,
            new PrefillQueueRegistry(mock(ConfigService.class)), new LoopSignal(), new DecodeQueue(false),
            new LoopSignal());
    private final PrefillEndpoint endpoint = V2TestSupport.prefillEndpoint("p1:8000");
    private final EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
    private final PrefillPicker picker = mock(PrefillPicker.class);
    private final List<Request> sent = new ArrayList<>();
    private final MainQueueLoop loop;

    MainQueueLoopTest() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(endpointRegistry.getEndpointCount(RoleType.PREFILL)).thenReturn(1);
        loop = new MainQueueLoop(mainQueue, loopSignal, cleanup, queueCapacity, endpointRegistry,
                new PlacementAvailability(), configService, picker, RoleType.PREFILL, sent::add);
    }

    @Test
    void pickedRequestHoldsQueueSlotAndGoesToPrefillQueue() {
        Request request = add("1", "g");
        SelectedRole selected = V2TestSupport.selected(endpoint, 4L, 30L);
        WorkerEndpoint.EndpointHold hold = selected.takeEndpointHold();
        when(picker.pick(request.ctx(), RoleType.PREFILL, "g")).thenReturn(PlacementResult.success(selected));

        assertTrue(loop.runOnce());

        assertEquals(List.of(request), sent);
        assertSame(endpoint, request.prefillChoice().endpoint());
        assertEquals(4L, request.prefillChoice().hitCacheTokens());
        assertEquals(30L, request.prefillChoice().predictedMs());
        verify(hold).close();
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
        SelectedRole selected = V2TestSupport.selected(endpoint, 0L, 10L);
        when(picker.pick(other.ctx(), RoleType.PREFILL, "b")).thenReturn(PlacementResult.success(selected));

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
    void requestCancelledDuringPickClosesEndpointHoldAndReleasesSlot() {
        Request request = add("1", "g");
        SelectedRole selected = V2TestSupport.selected(endpoint, 0L, 10L);
        WorkerEndpoint.EndpointHold hold = selected.takeEndpointHold();
        when(picker.pick(eq(request.ctx()), eq(RoleType.PREFILL), any())).thenAnswer(call -> {
            request.cancel(CancelReason.CLIENT_CANCELLED);
            return PlacementResult.success(selected);
        });

        loop.runOnce();

        verify(hold).close();
        assertTrue(sent.isEmpty());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    private Request add(String requestId, String group) {
        Request request = V2TestSupport.request(config, requestId, 50, 16L);
        request.future().whenComplete((response, failure) -> cleanup.releaseEnded(request));
        requestTable.add(request);
        when(picker.groupOf(request.ctx())).thenReturn(group);
        assertNull(mainQueue.add(request, config));
        return request;
    }

    @Test
    void skippedRequestsDoNotHideAnotherGroupBeyondScanLimit() {
        config.getScheduler().setMaxQueuedRequests(1_000);
        Request first = add("1", "a");
        when(picker.pick(first.ctx(), RoleType.PREFILL, "a")).thenReturn(PlacementResult.blocked(RoleType.PREFILL));
        for (int i = 2; i <= 300; i++) {
            add(Integer.toString(i), "a");
        }
        Request other = add("301", "b");
        doAnswer(call -> PlacementResult.success(V2TestSupport.selected(endpoint, 0L, 10L))).when(picker)
                .pick(other.ctx(), RoleType.PREFILL, "b");
        assertTrue(loop.runOnce());
        assertEquals(List.of(other), sent);
    }

    @Test
    void loopContinuesAfterError() throws InterruptedException {
        add("1", "g");
        CountDownLatch nextRound = new CountDownLatch(1);
        AtomicInteger calls = new AtomicInteger();
        doAnswer(call -> {
            if (calls.getAndIncrement() == 0) {
                throw new AssertionError("test round failure");
            }
            nextRound.countDown();
            return 0;
        }).when(endpointRegistry).getEndpointCount(RoleType.PREFILL);
        loop.start();
        try {
            assertTrue(nextRound.await(5, TimeUnit.SECONDS));
        } finally {
            loop.close();
        }
    }
}
