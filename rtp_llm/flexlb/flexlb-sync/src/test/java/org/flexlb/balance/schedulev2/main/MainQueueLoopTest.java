package org.flexlb.balance.schedulev2.main;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.schedulev2.DirectExecutorService;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.ProcessWorkers;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.decode.DecodeQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestTable;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.balance.strategy.PrefillInputs;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.V2QueueReporter;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
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
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class MainQueueLoopTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final ConfigService configService = mock(ConfigService.class);
    private final QueueCapacity queueCapacity = new QueueCapacity();
    private final LoopSignal loopSignal = new LoopSignal();
    private final RequestTable requestTable = new RequestTable();
    private final V2QueueReporter reporter = mock(V2QueueReporter.class);
    private final MainQueue mainQueue = new MainQueue(queueCapacity, loopSignal, true, reporter);
    private final PrefillQueueRegistry prefillQueues;
    private final RequestCleanup cleanup;
    private final PrefillEndpoint endpoint = V2TestSupport.prefillEndpoint("p1:8000");
    private final EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
    private final PrefillPicker picker = mock(PrefillPicker.class);
    private final PrefillInputs inputs =
            new PrefillInputs(CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC), List.of());
    private final List<Request> sent = new ArrayList<>();
    private final MainQueueLoop loop;

    MainQueueLoopTest() {
        config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH_V2);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(picker.prepare(any(), any(), any())).thenReturn(inputs);
        when(picker.inputsAreCurrent(any(), any(), any())).thenReturn(true);
        prefillQueues = new PrefillQueueRegistry();
        V2TestSupport.register(prefillQueues, endpoint);
        cleanup = new RequestCleanup(requestTable, mainQueue, queueCapacity, loopSignal, prefillQueues,
                new DecodeQueue(false), new LoopSignal());
        when(endpointRegistry.getEndpointCount(RoleType.PREFILL)).thenReturn(1);
        loop = new MainQueueLoop(mainQueue, loopSignal, cleanup, queueCapacity, endpointRegistry,
                new PlacementAvailability(), configService, picker, prefillQueues, RoleType.PREFILL, request -> {
                    prefillQueues.addWaiting(request);
                    sent.add(request);
                }, new ProcessWorkers<>(new DirectExecutorService(), 2, loopSignal, true), reporter);
    }

    @Test
    void pickedRequestHoldsQueueSlotAndGoesToPrefillQueue() {
        Request request = add("1", "g");
        SelectedRole selected = V2TestSupport.selected(endpoint, 4L, 30L);
        WorkerEndpoint.EndpointHold hold = selected.takeEndpointHold();
        when(picker.pick(request, RoleType.PREFILL, "g", inputs)).thenReturn(PlacementResult.success(selected));

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
        when(picker.pick(blocked, RoleType.PREFILL, "a", inputs)).thenReturn(PlacementResult.blocked(RoleType.PREFILL));
        SelectedRole selected = V2TestSupport.selected(endpoint, 0L, 10L);
        when(picker.pick(other, RoleType.PREFILL, "b", inputs)).thenReturn(PlacementResult.success(selected));

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
        when(picker.pick(request, RoleType.PREFILL, "g", inputs)).thenReturn(PlacementResult.rejected(failure));

        assertTrue(loop.runOnce());

        assertSame(failure, request.future().join());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(0, mainQueue.size());
    }

    @Test
    void requestCancelledDuringPickClosesSelectionAndReleasesSlot() {
        Request request = add("1", "g");
        SelectedRole selected = V2TestSupport.selected(endpoint, 0L, 10L);
        when(picker.pick(eq(request), eq(RoleType.PREFILL), any(), any())).thenAnswer(call -> {
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
        return add(requestId, group, 50);
    }

    private Request add(String requestId, String group, int priority) {
        Request request = V2TestSupport.request(config, requestId, priority, 16L);
        request.future().whenComplete((response, failure) -> {
            if (request.stage().isEnded()) {
                cleanup.releaseEnded(request);
            }
        });
        requestTable.add(request);
        when(picker.groupOf(request.ctx())).thenReturn(group);
        assertNull(mainQueue.add(request, config));
        return request;
    }

    @Test
    void skippedRequestsDoNotHideAnotherGroupBeyondScanLimit() {
        config.getScheduler().setMaxQueuedRequests(1_000);
        Request first = add("1", "a");
        when(picker.pick(first, RoleType.PREFILL, "a", inputs)).thenReturn(PlacementResult.blocked(RoleType.PREFILL));
        for (int i = 2; i <= 300; i++) {
            add(Integer.toString(i), "a");
        }
        Request other = add("301", "b");
        doAnswer(call -> PlacementResult.success(V2TestSupport.selected(endpoint, 0L, 10L))).when(picker)
                .pick(other, RoleType.PREFILL, "b", inputs);

        assertTrue(loop.runOnce());

        assertEquals(List.of(other), sent);
    }

    @Test
    void fullMainQueueRejectsVictimAndReportsBothPriorities() {
        SchedulingTestConfig.usePriorityQueue(config);
        config.priorityOrdering().setHighPriorityThreshold(50);
        config.getScheduler().setMaxQueuedRequests(1);
        Request low = add("low", "g", 40);
        Request high = add("high", "g", 50);

        assertEquals(Stage.FAILED, low.stage());
        assertEquals(8433, low.future().join().getCode());
        assertEquals(Stage.MAIN_WAITING, high.stage());
        assertEquals(1, mainQueue.size());
        assertSame(high, mainQueue.iterator().next());
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
        verify(reporter).mainPreempted(50, 40);
        org.mockito.Mockito.verifyNoMoreInteractions(reporter);
    }

    @Test
    void fullSubqueuesOnlyMarkNewestLowestPriorityAndWaitForReleasedSlot() {
        SchedulingTestConfig.usePriorityQueue(config);
        config.priorityOrdering().setHighPriorityThreshold(60);
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        config.getDispatcher().setPrefillSubqueueLimitMultiplier(2);
        doAnswer(call -> PlacementResult.success(V2TestSupport.selected(endpoint, 0L, 10L))).when(picker)
                .pick(any(), any(), any(), any());
        Request older = add("1", "g", 10);
        Request newer = add("2", "g", 10);
        loop.runOnce();
        Request high = add("3", "g", 80);

        assertFalse(loop.runOnce());
        assertFalse(loop.runOnce());

        assertNull(older.queueProgress().prefillReturnPriority());
        assertEquals(80, newer.queueProgress().prefillReturnPriority());
        assertEquals(Stage.PREFILL_WAITING, newer.stage());
        assertFalse(newer.future().isDone());
        assertEquals(Stage.MAIN_WAITING, high.stage());
        assertEquals(2, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(3L, queueCapacity.globalQueueAndEngineSubQueueCount());
        org.mockito.Mockito.verify(reporter, org.mockito.Mockito.never())
                .preemptedReturn(org.mockito.ArgumentMatchers.anyInt(), org.mockito.ArgumentMatchers.anyInt(), any());
    }

    @Test
    void fullSubqueuesDoNotPreemptAnotherPrefillOrDecodeWaiting() {
        SchedulingTestConfig.usePriorityQueue(config);
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        config.getDispatcher().setPrefillSubqueueLimitMultiplier(1);
        Request low = add("1", "g", 10);
        doAnswer(call -> PlacementResult.success(V2TestSupport.selected(endpoint, 0L, 10L))).when(picker)
                .pick(low, RoleType.PREFILL, "g", inputs);
        loop.runOnce();
        Request high = add("2", "g", 80);
        PrefillEndpoint other = V2TestSupport.prefillEndpoint("p2:8000");
        doAnswer(call -> PlacementResult.success(V2TestSupport.selected(other, 0L, 10L))).when(picker)
                .pick(high, RoleType.PREFILL, "g", inputs);

        assertFalse(loop.runOnce());
        assertEquals(Stage.PREFILL_WAITING, low.stage());

        low.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING);
        doAnswer(call -> PlacementResult.success(V2TestSupport.selected(endpoint, 0L, 10L))).when(picker)
                .pick(high, RoleType.PREFILL, "g", inputs);
        assertFalse(loop.runOnce());
        assertEquals(Stage.DECODE_WAITING, low.stage());
        assertEquals(Stage.MAIN_WAITING, high.stage());
    }

    @Test
    void loopContinuesAfterError() throws InterruptedException {
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

    @Test
    void preparationRunsTogetherAndDecisionsUseAlreadyUpdatedQueues() throws Exception {
        Request first = add("1", "g");
        Request second = add("2", "g");
        CountDownLatch bothStarted = new CountDownLatch(2);
        CountDownLatch finishFirst = new CountDownLatch(1);
        CompletableFuture<Request> firstSent = new CompletableFuture<>();
        CompletableFuture<Request> secondSent = new CompletableFuture<>();
        doAnswer(call -> {
            bothStarted.countDown();
            if (call.getArgument(0) == first.ctx()) {
                assertTrue(finishFirst.await(5, TimeUnit.SECONDS));
            }
            return inputs;
        }).when(picker).prepare(any(), any(), any());
        AtomicInteger decisions = new AtomicInteger();
        doAnswer(call -> {
            assertEquals("flexlb-v2-main-queue", Thread.currentThread().getName());
            int waiting = prefillQueues.find(endpoint) == null ? 0 : prefillQueues.find(endpoint).waitingCount();
            assertEquals(decisions.getAndIncrement(), waiting);
            return PlacementResult.success(V2TestSupport.selected(endpoint, 0L, 10L));
        }).when(picker).pick(any(), any(), any(), any());
        ExecutorService executor = Executors.newFixedThreadPool(2);
        MainQueueLoop parallelLoop = new MainQueueLoop(mainQueue, loopSignal, cleanup, queueCapacity, endpointRegistry,
                new PlacementAvailability(), configService, picker, prefillQueues, RoleType.PREFILL, request -> {
                    prefillQueues.addWaiting(request);
                    (request == first ? firstSent : secondSent).complete(request);
                }, new ProcessWorkers<>(executor, 2, loopSignal, true), mock(V2QueueReporter.class));
        parallelLoop.start();
        try {
            assertTrue(bothStarted.await(5, TimeUnit.SECONDS));
            assertSame(second, secondSent.get(5, TimeUnit.SECONDS));
            assertEquals(Stage.MAIN_WAITING, first.stage());
            finishFirst.countDown();
            assertSame(first, firstSent.get(5, TimeUnit.SECONDS));
        } finally {
            finishFirst.countDown();
            parallelLoop.close();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void fullQuotaReusesMaterialsUntilCapacityReturns() {
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        config.getDispatcher().setPrefillSubqueueLimitMultiplier(1);
        assertTrue(queueCapacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        Request request = add("1", "g");
        doAnswer(call -> PlacementResult.success(V2TestSupport.selected(endpoint, 0L, 10L))).when(picker)
                .pick(any(), any(), any(), any());
        for (int i = 0; i < 10; i++) {
            assertFalse(loop.runOnce());
        }
        verify(picker).prepare(request.ctx(), RoleType.PREFILL, "g");
        assertEquals(Stage.MAIN_WAITING, request.stage());

        queueCapacity.releaseEngineSubQueueSlot(RoleType.PREFILL);
        assertTrue(loop.runOnce());
        assertEquals(Stage.PREFILL_WAITING, request.stage());
        verify(picker).prepare(request.ctx(), RoleType.PREFILL, "g");
    }

    @Test
    void changedTtlExpiresExistingMaterialsAndCandidateChangesInvalidateThem() throws Exception {
        Request request = add("1", "g");
        when(picker.pick(request, RoleType.PREFILL, "g", inputs)).thenReturn(PlacementResult.blocked(RoleType.PREFILL));
        loop.runOnce();
        loop.runOnce();
        verify(picker).prepare(request.ctx(), RoleType.PREFILL, "g");

        Thread.sleep(25L);
        config.getDispatcher().setCacheMatchResultTtlMs(10L);
        loop.runOnce();
        verify(picker, times(2)).prepare(request.ctx(), RoleType.PREFILL, "g");

        config.getDispatcher().setCacheMatchResultTtlMs(30_000L);
        when(picker.inputsAreCurrent(any(), any(), any())).thenReturn(false);
        loop.runOnce();
        when(picker.inputsAreCurrent(any(), any(), any())).thenReturn(true);
        loop.runOnce();
        verify(picker, times(3)).prepare(request.ctx(), RoleType.PREFILL, "g");
    }
}
