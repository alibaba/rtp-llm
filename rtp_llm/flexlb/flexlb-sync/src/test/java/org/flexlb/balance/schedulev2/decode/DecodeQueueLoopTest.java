package org.flexlb.balance.schedulev2.decode;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemptionCoordinator;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.scheduler.decode.DecodeRequirement;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.DecodeChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestTable;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DecodeQueueLoopTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final ConfigService configService = mock(ConfigService.class);
    private final QueueCapacity queueCapacity = new QueueCapacity();
    private final RequestTable requestTable = new RequestTable();
    private final LoopSignal decodeLoopSignal = new LoopSignal();
    private final DecodeQueue decodeQueue = new DecodeQueue(false);
    private final RequestCleanup cleanup;
    private final DecodeSelector selector = mock(DecodeSelector.class);
    private final DefaultRouter router = mock(DefaultRouter.class);
    private final DecodeEndpoint decode =
            new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.2", 8080, 8081, "test"),
                    mock(EndpointEventHandler.class));
    private final ServerStatus decodeStatus = new ServerStatus();
    private final RequestSchedulerReporter reporter = mock(RequestSchedulerReporter.class);
    private final EmptyDecodeFinder emptyDecodeFinder = mock(EmptyDecodeFinder.class);
    private final DecodeEndpoint emptyDecode =
            new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.3", 8080, 8081, "test"),
                    mock(EndpointEventHandler.class));
    private final DecodeQueueLoop loop;

    DecodeQueueLoopTest() {
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setSuccess(true);
        decodeStatus.setServerIp("127.0.0.2");
        decodeStatus.setHttpPort(8080);
        LoopSignal mainLoopSignal = new LoopSignal();
        cleanup = new RequestCleanup(requestTable, new MainQueue(queueCapacity, mainLoopSignal, false), queueCapacity,
                mainLoopSignal, new PrefillQueueRegistry(mock(ConfigService.class)), new LoopSignal(), decodeQueue,
                decodeLoopSignal);
        when(selector.select(any(), any(), any())).thenAnswer(invocation -> success());
        loop = new DecodeQueueLoop(decodeQueue, decodeLoopSignal, cleanup, new PlacementAvailability(), selector,
                router, new DecodePreemptionCoordinator(mock(EngineCancelChannel.class), mock(RequestRegistry.class),
                        mock(RequestSchedulerReporter.class)),
                emptyDecodeFinder, configService(), reporter);
    }

    @AfterEach
    void closeEndpoint() {
        decode.close();
        emptyDecode.close();
    }

    @Test
    void requestIsReservedThenPublishedWithPrefillAndDecode() {
        Request request = addWaiting("1");
        queueCapacity.increaseTotalQueueCount(request.future());
        assertTrue(queueCapacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        request.holdQueueSlot(RoleType.PREFILL);

        assertTrue(loop.runOnce());
        assertEquals(Stage.DECODE_RESERVED, request.stage());
        assertSame(decode, request.decodeChoice().endpoint());
        assertEquals(1, decodeQueue.reservedCount());
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(1, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));

        assertTrue(loop.runOnce());
        assertEquals(Stage.PUBLISHED, request.stage());
        Response response = request.future().join();
        assertTrue(response.isSuccess());
        assertEquals(List.of(request.prefillChoice().serverStatus(), decodeStatus), response.getServerStatus());
        assertEquals(0, decodeQueue.reservedCount());
        assertEquals(0, requestTable.size());
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
    }

    @Test
    void requestStaysWaitingWhenDecodeIsFullAndIsReservedAfterRelease() {
        Request first = addWaiting("1");
        Request second = addWaiting("2");

        assertTrue(loop.runOnce());
        assertEquals(Stage.DECODE_RESERVED, first.stage());
        assertEquals(Stage.DECODE_WAITING, second.stage());
        assertEquals(1L, decode.routingView().placementUsage().occupiedRequests());

        first.cancel(CancelReason.CLIENT_CANCELLED);
        cleanup.releaseEnded(first);
        assertTrue(loop.runOnce());

        assertEquals(Stage.DECODE_RESERVED, second.stage());
        assertEquals(1L, decode.routingView().placementUsage().occupiedRequests());
    }

    @Test
    void publishedRequestKeepsLaterRequestWaiting() {
        Request first = addWaiting("1");
        Request second = addWaiting("2");
        loop.runOnce();
        loop.runOnce();

        assertEquals(Stage.PUBLISHED, first.stage());
        assertFalse(loop.runOnce());
        assertEquals(Stage.DECODE_WAITING, second.stage());
        assertEquals(0, decodeQueue.reservedCount());
    }

    @Test
    void cancelledRequestReturnsDecodeReservationInNextRound() {
        Request request = addWaiting("1");
        loop.runOnce();

        assertTrue(request.cancel(CancelReason.CLIENT_CANCELLED));
        cleanup.releaseEnded(request);

        // 清理不动 reserve，请求留在 reserved，Decode 队列下一轮扫到后归还
        assertEquals(1, decodeQueue.reservedCount());
        assertEquals(1L, decode.routingView().placementUsage().occupiedRequests());
        loop.runOnce();
        assertNull(request.decodeChoice());
        assertEquals(0L, decode.routingView().placementUsage().occupiedRequests());
        assertEquals(0, decodeQueue.reservedCount());
        assertEquals(0, decodeQueue.waitingCount());
        assertEquals(0, requestTable.size());
    }

    @Test
    void requestGoesBackToWaitingWhenReservedDecodeIsRetired() {
        Request request = addWaiting("1");
        loop.runOnce();
        doReturn(PlacementResult.blocked(RoleType.DECODE)).when(selector).select(any(), any(), any());

        decode.close();
        assertTrue(loop.runOnce());

        assertEquals(Stage.DECODE_WAITING, request.stage());
        assertNull(request.decodeChoice());
        assertEquals(1, decodeQueue.waitingCount());
        assertEquals(0, decodeQueue.reservedCount());
    }

    @Test
    void requestStaysWaitingWhenNoDecodeIsAvailable() {
        doReturn(PlacementResult.blocked(RoleType.DECODE)).when(selector).select(any(), any(), any());
        Request request = addWaiting("1");

        assertFalse(loop.runOnce());

        assertEquals(Stage.DECODE_WAITING, request.stage());
        assertEquals(1, decodeQueue.waitingCount());
    }

    @Test
    void higherPriorityRequestPreemptsPublishedRequestWhenDecodeIsFull() {
        config.queueScheduler().getOrdering().setType(QueueOrderingConfig.Type.PRIORITY);
        Request low = addWaiting("1", 30);
        loop.runOnce();
        loop.runOnce();
        assertEquals(Stage.PUBLISHED, low.stage());

        ServerStatus highStatus = decodeStatus();
        doAnswer(invocation -> PlacementResult.success(SelectedRole.decode(decode.tryHold(), highStatus)))
                .when(selector)
                .select(any(), any(), any());
        Request high = addWaiting("2", 80);
        assertTrue(loop.runOnce());
        assertEquals(Stage.DECODE_RESERVED, high.stage());
        assertTrue(loop.runOnce());

        assertEquals(Stage.PUBLISHED, high.stage());
        assertEquals(List.of("1"), high.future().join().getServerStatus().get(1).getPreemptRequestIds());
    }

    @Test
    void samePriorityRequestDoesNotPreempt() {
        config.queueScheduler().getOrdering().setType(QueueOrderingConfig.Type.PRIORITY);
        addWaiting("1", 30);
        loop.runOnce();
        loop.runOnce();

        Request second = addWaiting("2", 30);
        assertFalse(loop.runOnce());

        assertEquals(Stage.DECODE_WAITING, second.stage());
    }

    @Test
    void cancelledPreemptingRequestRollsBackPreemption() {
        config.queueScheduler().getOrdering().setType(QueueOrderingConfig.Type.PRIORITY);
        addWaiting("1", 30);
        loop.runOnce();
        loop.runOnce();
        Request high = addWaiting("2", 80);
        loop.runOnce();
        assertEquals(Stage.DECODE_RESERVED, high.stage());

        high.cancel(CancelReason.CLIENT_CANCELLED);
        cleanup.releaseEnded(high);
        loop.runOnce();
        assertNull(decode.reservationHandle("2"));

        // 抢占撤销后，被抢占的请求可以再被别的请求抢占
        Request next = addWaiting("3", 80);
        assertTrue(loop.runOnce());
        assertEquals(Stage.DECODE_RESERVED, next.stage());
    }

    @Test
    void endedRequestIsReleasedEvenWhenItsDecodeHasNoPermit() {
        Request running = addWaiting("1");
        loop.runOnce();
        loop.runOnce();
        assertEquals(Stage.PUBLISHED, running.stage());

        // Decode 上限 1 已经被发布的请求占了，两个请求都在 reserved 里拿不到 permit
        Request first = addReserved("2");
        Request second = addReserved("3");
        second.cancel(CancelReason.CLIENT_CANCELLED);
        cleanup.releaseEnded(second);
        assertNotNull(decode.reservationHandle("3"));

        loop.runOnce();

        assertEquals(Stage.DECODE_RESERVED, first.stage());
        assertNotNull(decode.reservationHandle("2"));
        assertNull(decode.reservationHandle("3"));
        assertEquals(1, decodeQueue.reservedCount());
    }

    @Test
    void publishedReservationNotReportedByEngineIsExpiredAfterConfirmTimeout() {
        config.getWorkerRegistry().getHealth().setStatusPollIntervalMs(0L);
        Request request = addWaiting("1");
        loop.runOnce();
        loop.runOnce();
        assertEquals(Stage.PUBLISHED, request.stage());
        assertEquals(1, decodeQueue.publishedCount());
        assertNotNull(decode.reservationHandle("1"));

        loop.runOnce();

        assertEquals(0, decodeQueue.publishedCount());
        assertNull(decode.reservationHandle("1"));
        verify(reporter).reportDispatchConfirmTimeout();
    }

    @Test
    void publishedRequestIsNoLongerTrackedAfterItsReservationIsGone() {
        Request request = addWaiting("1");
        loop.runOnce();
        loop.runOnce();
        assertEquals(1, decodeQueue.publishedCount());

        decode.release(request.decodeChoice().reservation(), DecodeEndpoint.ReleaseReason.EXPIRED);
        loop.runOnce();

        assertEquals(0, decodeQueue.publishedCount());
        verify(reporter, never()).reportDispatchConfirmTimeout();
    }

    @Test
    void requestGoesToEmptyDecodeWhenSelectedDecodeIsFull() {
        Request first = addWaiting("1");
        loop.runOnce();
        Request second = addWaiting("2");
        ServerStatus emptyStatus = emptyDecodeStatus();
        doAnswer(invocation -> SelectedRole.decode(emptyDecode.tryHold(), emptyStatus)).when(emptyDecodeFinder)
                .find(any(), any());

        // 这一轮 first 发布占满 decode，second 在 decode 上放不下，放到空的 Decode 上直接发布
        assertTrue(loop.runOnce());

        assertEquals(Stage.PUBLISHED, first.stage());
        assertEquals(Stage.PUBLISHED, second.stage());
        assertSame(emptyDecode, second.decodeChoice().endpoint());
        assertEquals(List.of(second.prefillChoice().serverStatus(), emptyStatus),
                second.future().join().getServerStatus());
        assertNotNull(emptyDecode.reservationHandle("2"));
        assertEquals(1L, emptyDecode.routingView().totalLoad());
        assertEquals(2, decodeQueue.publishedCount());
    }

    @Test
    void requestOverEveryDecodeBudgetWaitsForEmptyDecode() {
        doReturn(PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED))).when(selector)
                .select(any(), any(), any());
        Request request = addWaiting("1");

        assertFalse(loop.runOnce());
        assertEquals(Stage.DECODE_WAITING, request.stage());

        ServerStatus emptyStatus = emptyDecodeStatus();
        doAnswer(invocation -> SelectedRole.decode(emptyDecode.tryHold(), emptyStatus)).when(emptyDecodeFinder)
                .find(any(), any());
        assertTrue(loop.runOnce());

        assertEquals(Stage.PUBLISHED, request.stage());
        assertNotNull(emptyDecode.reservationHandle("1"));
    }

    @Test
    void laterRequestWaitsBehindHeadThatDoesNotFitWhenThresholdIsZero() {
        config.getDispatcher().setDispatchStarvationThresholdMs(0L);
        Request head = addWaiting("1");
        head.setPrefillDecidedAtMs(System.currentTimeMillis());
        Request later = addWaiting("2");
        blockOnly(head);

        loop.runOnce();

        assertEquals(Stage.DECODE_WAITING, head.stage());
        assertEquals(Stage.DECODE_WAITING, later.stage());
    }

    @Test
    void laterRequestGoesFirstWhileHeadHasNotWaitedLong() {
        config.getDispatcher().setDispatchStarvationThresholdMs(60_000L);
        Request head = addWaiting("1");
        head.setPrefillDecidedAtMs(System.currentTimeMillis());
        Request later = addWaiting("2");
        blockOnly(head);

        loop.runOnce();

        assertEquals(Stage.DECODE_WAITING, head.stage());
        assertEquals(Stage.DECODE_RESERVED, later.stage());
    }

    @Test
    void laterRequestWaitsBehindHeadThatHasWaitedTooLong() {
        config.getDispatcher().setDispatchStarvationThresholdMs(60_000L);
        Request head = addWaiting("1");
        head.setPrefillDecidedAtMs(System.currentTimeMillis() - 120_000L);
        Request later = addWaiting("2");
        blockOnly(head);

        loop.runOnce();

        assertEquals(Stage.DECODE_WAITING, head.stage());
        assertEquals(Stage.DECODE_WAITING, later.stage());
    }

    @Test
    void changedStarvationThresholdIsUsedInNextRound() {
        config.getDispatcher().setDispatchStarvationThresholdMs(0L);
        Request head = addWaiting("1");
        head.setPrefillDecidedAtMs(System.currentTimeMillis());
        Request later = addWaiting("2");
        blockOnly(head);
        loop.runOnce();
        assertEquals(Stage.DECODE_WAITING, later.stage());

        config.getDispatcher().setDispatchStarvationThresholdMs(60_000L);
        loop.runOnce();

        assertEquals(Stage.DECODE_WAITING, head.stage());
        assertEquals(Stage.DECODE_RESERVED, later.stage());
    }

    /**
     * 只有 blocked 这个请求选不到 Decode，其他请求正常选中。
     */
    private void blockOnly(Request blocked) {
        doAnswer(invocation -> invocation.getArgument(0) == blocked.ctx()
                ? PlacementResult.blocked(RoleType.DECODE)
                : success()).when(selector).select(any(), any(), any());
    }

    private ServerStatus emptyDecodeStatus() {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.DECODE);
        status.setSuccess(true);
        status.setServerIp("127.0.0.3");
        status.setHttpPort(8080);
        return status;
    }

    private ConfigService configService() {
        when(configService.loadBalanceConfig()).thenReturn(config);
        return configService;
    }

    @Test
    void skippedRequestsDoNotHideAnotherGroupBeyondScanLimit() {
        config.getDispatcher().setDispatchStarvationThresholdMs(0L);
        for (int i = 1; i <= 300; i++) {
            Request request = addWaiting(Integer.toString(i));
            when(router.resolvePolicyGroup(request.ctx())).thenReturn("a");
        }
        Request other = addWaiting("301");
        when(router.resolvePolicyGroup(other.ctx())).thenReturn("b");
        doAnswer(call -> "a".equals(call.getArgument(2)) ? PlacementResult.blocked(RoleType.DECODE) : success())
                .when(selector)
                .select(any(), any(), any());
        assertTrue(loop.runOnce());
        assertEquals(Stage.DECODE_RESERVED, other.stage());
        assertEquals(300, decodeQueue.waitingCount());
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

    private Request addWaiting(String requestId) {
        return addWaiting(requestId, 30);
    }

    /**
     * 直接放进 reserved 的请求。reserve 不带容量上限，用来造出 reserve 成功但拿不到 permit 的情况。
     */
    private Request addReserved(String requestId) {
        Request request = addWaiting(requestId);
        decodeQueue.removeWaiting(request);
        DecodeEndpoint.ReservationHandle reservation;
        try (WorkerEndpoint.EndpointHold hold = decode.tryHold()) {
            reservation = decode.reserve(hold, requestId, 16L, 16L, 30, null);
        }
        request.setDecodeChoice(
                new DecodeChoice(decode, reservation, decodeStatus(), DecodeRequirement.capture(request.ctx())));
        request.moveTo(Stage.DECODE_WAITING, Stage.DECODE_RESERVED);
        decodeQueue.addReserved(request);
        return request;
    }

    private Request addWaiting(String requestId, int priority) {
        Request request = V2TestSupport.request(config, requestId, priority, 16L);
        request.setQueueSeq(Long.parseLong(requestId));
        V2TestSupport.choose(request, V2TestSupport.prefillEndpoint("p1:8000"), 0L, 0L);
        request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
        request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING);
        requestTable.add(request);
        decodeQueue.addWaiting(request);
        return request;
    }

    private ServerStatus decodeStatus() {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.DECODE);
        status.setSuccess(true);
        status.setServerIp("127.0.0.2");
        status.setHttpPort(8080);
        return status;
    }

    private PlacementResult<SelectedRole, RoleType> success() {
        return PlacementResult.success(SelectedRole.decode(decode.tryHold(), decodeStatus));
    }
}
