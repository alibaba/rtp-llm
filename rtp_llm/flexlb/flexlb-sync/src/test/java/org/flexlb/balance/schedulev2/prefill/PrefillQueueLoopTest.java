package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.decode.DecodeQueue;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestTable;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrefillQueueLoopTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final ConfigService configService = mock(ConfigService.class);
    private final QueueCapacity queueCapacity = new QueueCapacity();
    private final LoopSignal mainLoopSignal = new LoopSignal();
    private final LoopSignal prefillLoopSignal = new LoopSignal();
    private final RequestTable requestTable = new RequestTable();
    private final MainQueue mainQueue = new MainQueue(queueCapacity, mainLoopSignal, false);
    private final PrefillQueueRegistry queues;
    private final RequestCleanup cleanup;
    private final EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
    private final PrefillEndpoint endpoint = V2TestSupport.prefillEndpoint("p1:8000");
    private final List<Request> sentToDecode = new ArrayList<>();
    private final RequestSchedulerReporter reporter = mock(RequestSchedulerReporter.class);
    private final PrefillQueueLoop loop;

    PrefillQueueLoopTest() {
        when(configService.loadBalanceConfig()).thenReturn(config);
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        when(endpointRegistry.getEndpointCount(RoleType.PREFILL)).thenReturn(1);
        when(endpointRegistry.get(RoleType.PREFILL, "p1:8000")).thenReturn(endpoint);
        queues = new PrefillQueueRegistry(configService);
        cleanup = new RequestCleanup(requestTable, mainQueue, queueCapacity, mainLoopSignal, queues, prefillLoopSignal,
                new DecodeQueue(false), new LoopSignal());
        loop = new PrefillQueueLoop(queues, prefillLoopSignal, cleanup, mainQueue, endpointRegistry, configService,
                reporter, sentToDecode::add);
    }

    @Test
    void publishedRequestNotReportedByEngineIsRemovedAfterConfirmTimeout() {
        config.getWorkerRegistry().getHealth().setStatusPollIntervalMs(0L);
        Request request = addWaiting("1");
        loop.runOnce();
        assertEquals(1, queues.find(endpoint).dispatchedCount());

        // 发布前不算超时
        loop.runOnce();
        assertEquals(1, queues.find(endpoint).dispatchedCount());

        request.publish(Stage.DECODE_WAITING, List.of());
        loop.runOnce();

        assertEquals(0, queues.find(endpoint).dispatchedCount());
        verify(reporter).reportDispatchConfirmTimeout();
    }

    @Test
    void dispatchKeepsQueueSlotAndCountUntilPublish() {
        Request request = addWaiting("1");

        assertFalse(loop.runOnce());

        assertEquals(List.of(request), sentToDecode);
        assertEquals(Stage.DECODE_WAITING, request.stage());
        assertTrue(request.prefillDecidedAtMs() > 0L);
        assertEquals(1, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(1, queues.find(endpoint).dispatchedCount());
        Request other = addWaiting("2");
        assertTrue(request.publish(Stage.DECODE_WAITING, List.of()));
        cleanup.releasePublished(request);
        cleanup.releasePublished(request);
        assertEquals(1, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(Stage.PREFILL_WAITING, other.stage());
    }

    @Test
    void requestWaitsWhenInflightLimitIsReached() {
        Request first = addWaiting("1");
        Request second = addWaiting("2");

        assertTrue(loop.runOnce());

        assertEquals(List.of(first), sentToDecode);
        assertEquals(Stage.PREFILL_WAITING, second.stage());
        assertEquals(2, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));

        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("1", TaskPhase.RUNNING));
        assertTrue(loop.runOnce());
        V2TestSupport.report(endpoint, 2L, 1_000L, 1_000L, Map.of());
        assertFalse(loop.runOnce());

        assertEquals(List.of(first, second), sentToDecode);
    }

    @Test
    void waitingRequestsGoBackToMainQueueWhenEndpointIsGone() {
        Request request = addWaiting("1");
        when(endpointRegistry.get(RoleType.PREFILL, "p1:8000")).thenReturn(null);

        assertFalse(loop.runOnce());

        assertEquals(Stage.MAIN_WAITING, request.stage());
        assertNull(request.prefillChoice());
        assertEquals(1, mainQueue.size());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertNull(queues.find(endpoint));
        assertTrue(sentToDecode.isEmpty());
    }

    @Test
    void waitingRequestsGoBackToMainQueueWhenSameAddressComesBackOnline() {
        Request request = addWaiting("1");
        PrefillEndpoint restarted = V2TestSupport.prefillEndpoint("p1:8000");
        when(endpointRegistry.get(RoleType.PREFILL, "p1:8000")).thenReturn(restarted);

        assertFalse(loop.runOnce());

        assertEquals(Stage.MAIN_WAITING, request.stage());
        assertEquals(1, mainQueue.size());
        assertNull(queues.find(endpoint));
        assertTrue(sentToDecode.isEmpty());
    }

    @Test
    void cancelledAfterDispatchRemovesDispatchedRecord() {
        Request request = addWaiting("1");
        loop.runOnce();

        request.cancel(CancelReason.CLIENT_CANCELLED);

        assertEquals(0, queues.find(endpoint).dispatchedCount());
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    @Test
    void cancelledWhileWaitingIsNotDispatched() {
        Request request = addWaiting("1");

        request.cancel(CancelReason.CLIENT_CANCELLED);
        assertFalse(loop.runOnce());

        assertTrue(sentToDecode.isEmpty());
        assertEquals(0, queues.find(endpoint).waitingCount());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    /**
     * 模拟主队列已经选好 Prefill：请求在排队总数里、占着一个主队列名额，处于 PREFILL_WAITING。
     */
    private Request addWaiting(String requestId) {
        return addWaiting(requestId, 16L);
    }

    private Request addWaiting(String requestId, long seqLen) {
        Request request = V2TestSupport.request(config, requestId, 50, seqLen);
        request.future().whenComplete((response, failure) -> {
            if (request.stage().isEnded()) {
                cleanup.releaseEnded(request);
            }
        });
        requestTable.add(request);
        assertNull(mainQueue.add(request, config));
        assertTrue(queueCapacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 4));
        request.holdQueueSlot(RoleType.PREFILL);
        V2TestSupport.choose(request, endpoint, 0L, 10L);
        assertTrue(request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING));
        mainQueue.remove(request);
        queues.addWaiting(request);
        return request;
    }

    @Test
    void smallerRequestGoesFirstOnlyWhileKvBlockedHeadHasNotStarved() {
        config.getDispatcher().setMaxInflightPerPrefillWorker(4);
        config.getDispatcher().setDispatchStarvationThresholdMs(60_000L);
        V2TestSupport.report(endpoint, 1L, 50L, 1_000L, Map.of("engine-request", TaskPhase.RUNNING));
        Request head = addWaiting("1", 100L);
        Request small = addWaiting("2", 16L);
        assertTrue(loop.runOnce());
        assertEquals(Stage.PREFILL_WAITING, head.stage());
        assertEquals(List.of(small), sentToDecode);
        head.setPrefillQueuedAtMs(System.currentTimeMillis() - 120_000L);
        Request later = addWaiting("3", 16L);
        loop.runOnce();
        assertEquals(Stage.PREFILL_WAITING, later.stage());
        assertEquals(List.of(small), sentToDecode);
    }

    @Test
    void hotUpdatedThresholdLetsSmallerRequestPassKvBlockedHead() {
        config.getDispatcher().setMaxInflightPerPrefillWorker(4);
        config.getDispatcher().setDispatchStarvationThresholdMs(0L);
        V2TestSupport.report(endpoint, 1L, 50L, 1_000L, Map.of("engine-request", TaskPhase.RUNNING));
        Request head = addWaiting("1", 100L);
        Request small = addWaiting("2", 16L);
        loop.runOnce();
        assertTrue(sentToDecode.isEmpty());
        config.getDispatcher().setDispatchStarvationThresholdMs(60_000L);
        loop.runOnce();
        assertEquals(Stage.PREFILL_WAITING, head.stage());
        assertEquals(List.of(small), sentToDecode);
    }

    @Test
    void inflightFullStopsSmallRequestsEvenBeforeStarvationThreshold() {
        config.getDispatcher().setDispatchStarvationThresholdMs(60_000L);
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("engine-request", TaskPhase.RUNNING));
        Request head = addWaiting("1", 100L);
        Request small = addWaiting("2", 1L);
        loop.runOnce();
        assertEquals(Stage.PREFILL_WAITING, head.stage());
        assertEquals(Stage.PREFILL_WAITING, small.stage());
        assertTrue(sentToDecode.isEmpty());
    }

    @Test
    void decodeWaitingStillCountsTowardTotalQueueLimit() {
        config.getScheduler().setMaxQueuedRequests(1);
        Request request = addWaiting("1");
        loop.runOnce();
        Request next = V2TestSupport.request(config, "2", 50, 16L);
        assertEquals(StrategyErrorType.QUEUE_FULL, mainQueue.add(next, config));
        assertEquals(Stage.DECODE_WAITING, request.stage());
        request.publish(Stage.DECODE_WAITING, List.of());
        cleanup.releasePublished(request);
        assertNull(mainQueue.add(next, config));
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
            return config;
        }).when(configService).loadBalanceConfig();
        loop.start();
        try {
            assertTrue(nextRound.await(5, TimeUnit.SECONDS));
        } finally {
            loop.close();
        }
    }
}
