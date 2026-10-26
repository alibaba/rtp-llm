package org.flexlb.balance.schedulev2.decode;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
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
import org.flexlb.balance.schedulev2.PrefillPublishCheck;
import org.flexlb.balance.schedulev2.QueueBlockReason;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.prefill.DispatchedRequest;
import org.flexlb.balance.schedulev2.prefill.PrefillQueue;
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
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.V2QueueReporter;
import org.flexlb.sync.status.WorkerDirectory;
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
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DecodeQueueLoopTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final QueueCapacity queueCapacity = new QueueCapacity();
    private final RequestTable requestTable = new RequestTable();
    private final PrefillPublishCheck prefillCheck = mock(PrefillPublishCheck.class);
    private final DecodeReservations reservations = new DecodeReservations();
    private final V2QueueReporter queueReporter = mock(V2QueueReporter.class);
    private final LoopSignal decodeLoopSignal = new LoopSignal();
    private final DecodeQueue decodeQueue = new DecodeQueue(false);
    private final RequestCleanup cleanup;
    private final MainQueue mainQueue;
    private final DecodeSelector selector = mock(DecodeSelector.class);
    private final DefaultRouter router = mock(DefaultRouter.class);
    private final ConfigService configService = mock(ConfigService.class);
    private final DecodeEndpoint decode =
            new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.2", 8080, 8081, "test"),
                    mock(EndpointEventHandler.class));
    private final ServerStatus decodeStatus = new ServerStatus();
    private final RequestSchedulerReporter reporter = mock(RequestSchedulerReporter.class);
    private final EmptyDecodeFinder emptyDecodeFinder = mock(EmptyDecodeFinder.class);
    private final DecodeEndpoint emptyDecode =
            new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.3", 8080, 8081, "test"),
                    mock(EndpointEventHandler.class));
    private final EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
    private final DecodeQueueLoop loop;

    DecodeQueueLoopTest() {
        when(endpointRegistry.getEndpointCount(RoleType.DECODE)).thenReturn(16);
        when(prefillCheck.isAvailable(any())).thenReturn(true);
        when(prefillCheck.hold(any())).thenAnswer(call -> mock(WorkerEndpoint.EndpointHold.class));
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setSuccess(true);
        decodeStatus.setServerIp("127.0.0.2");
        decodeStatus.setHttpPort(8080);
        LoopSignal mainLoopSignal = new LoopSignal();
        mainQueue = new MainQueue(queueCapacity, mainLoopSignal, false, mock(V2QueueReporter.class));
        cleanup = new RequestCleanup(requestTable, mainQueue, queueCapacity, mainLoopSignal,
                new PrefillQueueRegistry(), decodeQueue, decodeLoopSignal);
        when(selector.select(any(), any(), any())).thenAnswer(invocation -> success());
        loop = new DecodeQueueLoop(decodeQueue, decodeLoopSignal, cleanup, new PlacementAvailability(), selector,
                router,
                new DecodePreemptionCoordinator(mock(EngineCancelChannel.class), mock(RequestRegistry.class),
                        mock(RequestSchedulerReporter.class)),
                emptyDecodeFinder, configService(), reporter, prefillCheck, reservations, queueReporter, endpointRegistry);
    }

    @AfterEach
    void closeEndpoint() {
        decode.close();
        emptyDecode.close();
    }

    @Test
    void prefillBlockedRequestsDoNotHideReadyRequestAfterTwoThousandEntries() {
        when(endpointRegistry.getEndpointCount(RoleType.DECODE)).thenReturn(1);
        doReturn(QueueBlockReason.PREFILL_INFLIGHT_FULL).when(prefillCheck)
                .tryAcquire(any(), org.mockito.ArgumentMatchers.anyInt());
        for (int i = 0; i < 2000; i++) {
            addWaiting(Integer.toString(i + 1));
        }
        Request ready = addWaiting("2001");
        doReturn(null).when(prefillCheck)
                .tryAcquire(org.mockito.ArgumentMatchers.eq(ready), org.mockito.ArgumentMatchers.anyInt());
        assertTrue(loop.runOnce());
        assertEquals(Stage.PUBLISHED, ready.stage());
        assertEquals(2000, decodeQueue.waitingCount());
        verify(selector).select(any(), any(), any());
    }

    @Test
    void decodeBlockedRequestsDoNotConsumePublicationBudget() {
        when(endpointRegistry.getEndpointCount(RoleType.DECODE)).thenReturn(1);
        config.getDispatcher().setDispatchStarvationThresholdMs(Long.MAX_VALUE);
        doReturn(PlacementResult.blocked(RoleType.DECODE)).when(selector).select(any(), any(), any());
        for (int i = 0; i < 2000; i++) {
            addWaiting(Integer.toString(i + 1));
        }
        Request ready = addWaiting("2001");
        doAnswer(call -> success()).when(selector)
                .select(org.mockito.ArgumentMatchers.eq(ready.ctx()), any(), any());
        assertTrue(loop.runOnce());
        assertEquals(Stage.PUBLISHED, ready.stage());
        assertEquals(2000, decodeQueue.waitingCount());
    }

    @Test
    void publicationBudgetTracksDecodeCountEachRound() {
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(20L);
        for (int i = 0; i < 6; i++) {
            addWaiting(Integer.toString(i + 1));
        }
        when(endpointRegistry.getEndpointCount(RoleType.DECODE)).thenReturn(3);
        assertTrue(loop.runOnce());
        assertEquals(3, decodeQueue.waitingCount());
        when(endpointRegistry.getEndpointCount(RoleType.DECODE)).thenReturn(2);
        assertTrue(loop.runOnce());
        assertEquals(1, decodeQueue.waitingCount());
        when(endpointRegistry.getEndpointCount(RoleType.DECODE)).thenReturn(0);
        assertFalse(loop.runOnce());
        assertEquals(1, decodeQueue.waitingCount());
    }

    @Test
    void reservedAndWaitingSharePublicationBudget() {
        when(endpointRegistry.getEndpointCount(RoleType.DECODE)).thenReturn(1);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(20L);
        Request reserved = addReserved("1");
        Request waiting = addWaiting("2");
        assertTrue(loop.runOnce());
        assertEquals(Stage.PUBLISHED, reserved.stage());
        assertEquals(Stage.DECODE_WAITING, waiting.stage());
        assertTrue(loop.runOnce());
        assertEquals(Stage.PUBLISHED, waiting.stage());
    }

    @Test
    void emptyDecodePublicationConsumesBudget() {
        when(endpointRegistry.getEndpointCount(RoleType.DECODE)).thenReturn(1);
        doReturn(PlacementResult.blocked(RoleType.DECODE)).when(selector).select(any(), any(), any());
        doAnswer(call -> SelectedRole.decode(emptyDecode.tryHold(), emptyDecodeStatus())).when(emptyDecodeFinder)
                .find(any(), any());
        Request first = addWaiting("1");
        Request second = addWaiting("2");
        assertTrue(loop.runOnce());
        assertEquals(Stage.PUBLISHED, first.stage());
        assertEquals(Stage.DECODE_WAITING, second.stage());
    }

    @Test
    void publicationResumesWhenActiveRequestsDisappearAndLateFinishedIsHarmless() {
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(20L);
        config.getDispatcher().setMaxInflightPerPrefillWorker(4);
        var prefill = V2TestSupport.prefillEndpoint("p1:8000");
        PrefillQueue queue = new PrefillQueue(prefill, true);
        when(prefill.prefillQueue()).thenReturn(queue);
        var requests = new java.util.ArrayList<Request>();
        for (int i = 1; i <= 6; i++) {
            Request request = addWaiting(Integer.toString(i));
            V2TestSupport.choose(request, prefill, 0L, 0L);
            queue.addDispatched(new DispatchedRequest(request, 16L, 0L));
            requests.add(request);
        }
        doAnswer(call -> queue.tryAcquirePublishSlot(call.getArgument(0), call.getArgument(1))).when(prefillCheck)
                .tryAcquire(any(), org.mockito.ArgumentMatchers.anyInt());
        doAnswer(call -> {
            queue.releasePublishSlot(call.getArgument(0));
            return null;
        }).when(prefillCheck).release(any());

        loop.runOnce();
        assertEquals(4L, requests.stream().filter(r -> r.stage() == Stage.PUBLISHED).count());
        assertEquals(2, decodeQueue.waitingCount());
        assertFalse(loop.runOnce());
        V2TestSupport.report(prefill, 1L, 1000L, 1000L,
                java.util.Map.of("1", org.flexlb.enums.TaskPhase.RUNNING, "2", org.flexlb.enums.TaskPhase.PENDING, "3",
                        org.flexlb.enums.TaskPhase.RUNNING, "4", org.flexlb.enums.TaskPhase.PENDING));
        assertFalse(loop.runOnce());
        V2TestSupport.report(prefill, 2L, 1000L, 1000L, java.util.Map.of());
        assertTrue(loop.runOnce());
        assertEquals(6L, requests.stream().filter(r -> r.stage() == Stage.PUBLISHED).count());
        assertEquals(2L, queue.inflightCount());
        V2TestSupport.reportTasks(queue, java.util.Map.of(),
                java.util.Map.of("1", org.flexlb.enums.TaskPhase.RUNNING, "2", org.flexlb.enums.TaskPhase.PENDING));
        assertFalse(loop.runOnce());
        assertEquals(2L, queue.inflightCount());
        assertEquals(6L, requests.stream().filter(r -> r.stage() == Stage.PUBLISHED).count());
        assertEquals(0, decodeQueue.waitingCount());
    }

    @Test
    void prefillFullDoesNotReserveDecodeAndLaterRequestCanProceed() {
        Request blocked = addWaiting("1");
        Request later = addWaiting("2");
        doReturn(QueueBlockReason.PREFILL_INFLIGHT_FULL).when(prefillCheck)
                .tryAcquire(org.mockito.ArgumentMatchers.eq(blocked), org.mockito.ArgumentMatchers.anyInt());
        loop.runOnce();
        assertEquals(Stage.DECODE_WAITING, blocked.stage());
        assertEquals(Stage.PUBLISHED, later.stage());
        verify(selector, never()).select(org.mockito.ArgumentMatchers.eq(blocked.ctx()), any(), any());
        assertNull(decode.reservationHandle("1"));
    }

    @Test
    void decodeFailureReturnsPrefillSlotInSameRound() {
        Request request = addWaiting("1");
        doReturn(PlacementResult.blocked(RoleType.DECODE)).when(selector).select(any(), any(), any());
        assertFalse(loop.runOnce());
        var order = org.mockito.Mockito.inOrder(prefillCheck, selector);
        order.verify(prefillCheck)
                .tryAcquire(org.mockito.ArgumentMatchers.eq(request), org.mockito.ArgumentMatchers.anyInt());
        order.verify(selector).select(org.mockito.ArgumentMatchers.eq(request.ctx()), any(), any());
        order.verify(prefillCheck).release(request);
        assertEquals(Stage.DECODE_WAITING, request.stage());
        assertNull(request.decodeChoice());
    }

    @Test
    void requestIsReservedThenPublishedWithPrefillAndDecode() {
        Request request = addWaiting("1");
        queueCapacity.increaseTotalQueueCount(request.future());
        assertTrue(queueCapacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        request.holdQueueSlot(RoleType.PREFILL);

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
        assertEquals(Stage.PUBLISHED, first.stage());
        assertEquals(Stage.DECODE_WAITING, second.stage());
        verify(queueReporter).blocked(second, Stage.DECODE_WAITING, QueueBlockReason.DECODE_REQUESTS_FULL,
                decode.ipPort());
        assertEquals(1L, decode.routingView().placementUsage().occupiedRequests());

        decode.release(first.decodeChoice().reservation(), DecodeEndpoint.ReleaseReason.EXPIRED);
        assertTrue(loop.runOnce());

        assertEquals(Stage.PUBLISHED, second.stage());
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
        Request request = addReserved("1");

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
        Request request = addReserved("1");
        doReturn(PlacementResult.blocked(RoleType.DECODE)).when(selector).select(any(), any(), any());

        decode.close();
        loop.runOnce();

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

        assertEquals(Stage.PUBLISHED, high.stage());
        assertEquals(List.of("1"), high.future().join().getServerStatus().get(1).getPreemptRequestIds());
        verify(queueReporter).decodePreempted(80, 30, DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN, decode.ipPort());
        loop.runOnce();
        verify(queueReporter).decodePreempted(80, 30, DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN, decode.ipPort());
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
        doAnswer(call -> {
            high.cancel(CancelReason.CLIENT_CANCELLED);
            cleanup.releaseEnded(high);
            return mock(WorkerEndpoint.EndpointHold.class);
        }).when(prefillCheck).hold(high);
        loop.runOnce();
        assertEquals(Stage.CANCELLED, high.stage());

        verify(queueReporter, never()).decodePreempted(org.mockito.ArgumentMatchers.anyInt(),
                org.mockito.ArgumentMatchers.anyInt(), any(), any());
        high.cancel(CancelReason.CLIENT_CANCELLED);
        cleanup.releaseEnded(high);
        loop.runOnce();
        assertNull(decode.reservationHandle("2"));
        verify(queueReporter, never()).decodePreempted(org.mockito.ArgumentMatchers.anyInt(),
                org.mockito.ArgumentMatchers.anyInt(), any(), any());

        // 抢占撤销后，被抢占的请求可以再被别的请求抢占
        Request next = addWaiting("3", 80);
        assertTrue(loop.runOnce());
        assertEquals(Stage.PUBLISHED, next.stage());
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

        assertEquals(Stage.DECODE_WAITING, first.stage());
        assertNull(decode.reservationHandle("2"));
        assertNull(decode.reservationHandle("3"));
        assertEquals(0, decodeQueue.reservedCount());
    }

    @Test
    void publishedReservationNotReportedByEngineIsExpiredAfterConfirmTimeout() {
        config.getRequestLifecycle().getRequest().setStatusUpdateTimeoutMs(0L);
        Request request = addWaiting("1");
        loop.runOnce();
        assertEquals(Stage.PUBLISHED, request.stage());
        assertEquals(1, decodeQueue.publishedCount());
        assertNotNull(decode.reservationHandle("1"));
        assertTrue(reservations.contains("1"));

        loop.runOnce();

        assertEquals(0, decodeQueue.publishedCount());
        assertNull(decode.reservationHandle("1"));
        assertFalse(reservations.contains("1"));
        verify(reporter).reportStatusUpdateTimeout(RoleType.DECODE, decode.ipPort());
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
        verify(reporter, never()).reportStatusUpdateTimeout(RoleType.DECODE, decode.ipPort());
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
        assertEquals(Stage.PUBLISHED, later.stage());
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
        assertEquals(Stage.PUBLISHED, later.stage());
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

        assertEquals(Stage.PUBLISHED, other.stage());
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

    @Test
    void decisionsUsePreviousReservationOnDecodeThread() throws Exception {
        config.getDispatcher().setDispatchStarvationThresholdMs(60_000L);
        Request first = addWaiting("1");
        Request second = addWaiting("2");
        first.setPrefillDecidedAtMs(System.currentTimeMillis());
        second.setPrefillDecidedAtMs(System.currentTimeMillis());
        WorkerDirectory directory = mock(WorkerDirectory.class);
        when(directory.decodeRoutingSnapshot(any()))
                .thenAnswer(call -> List.of(decode.routingView(), emptyDecode.routingView()));
        when(directory.holdDecodeEndpoint(any())).thenAnswer(call -> {
            DecodeEndpoint.DecodeRoutingView view = call.getArgument(0);
            return view.address().equals(decode.routingView().address()) ? decode.tryHold() : emptyDecode.tryHold();
        });
        DecodeSelector realSelector = spy(new DecodeSelector(directory));
        doAnswer(call -> {
            assertEquals("flexlb-v2-decode-queue", Thread.currentThread().getName());
            return call.callRealMethod();
        }).when(realSelector).select(any(), any(), any());
        DecodeQueueLoop realLoop = new DecodeQueueLoop(decodeQueue, decodeLoopSignal, cleanup,
                new PlacementAvailability(), realSelector, router, mock(DecodePreemptionCoordinator.class),
                emptyDecodeFinder, configService, reporter, prefillCheck, reservations,
                mock(V2QueueReporter.class), endpointRegistry);
        realLoop.start();
        try {
            assertTrue(first.future().get(5, TimeUnit.SECONDS).isSuccess());
            assertTrue(second.future().get(5, TimeUnit.SECONDS).isSuccess());
            assertSame(decode, first.decodeChoice().endpoint());
            assertSame(emptyDecode, second.decodeChoice().endpoint());
            assertEquals(1, decode.routingView().totalLoad());
            assertEquals(1, emptyDecode.routingView().totalLoad());
        } finally {
            realLoop.close();
        }
    }

    @Test
    void laterRequestDoesNotPassStarvedHead() {
        config.getDispatcher().setDispatchStarvationThresholdMs(60_000L);
        Request head = addWaiting("1");
        Request later = addWaiting("2");
        head.setPrefillDecidedAtMs(System.currentTimeMillis());
        later.setPrefillDecidedAtMs(System.currentTimeMillis());
        doAnswer(call -> {
            return call.getArgument(0) == head.ctx() ? PlacementResult.blocked(RoleType.DECODE) : success();
        }).when(selector).select(any(), any(), any());
        head.setPrefillDecidedAtMs(System.currentTimeMillis() - 120_000L);

        loop.runOnce();

        assertEquals(Stage.DECODE_WAITING, head.stage());
        assertEquals(Stage.DECODE_WAITING, later.stage());
        assertEquals(0, decode.routingView().totalLoad());
    }

    @Test
    void requirementCalculationFailureFailsRequestWithoutReservation() {
        Request request = addWaiting("1");
        config.getRouter().getRoles().getDecode().getAvailability().setMaxKvUsagePercent(0L);

        assertTrue(loop.runOnce());

        assertEquals(Stage.FAILED, request.stage());
        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), request.future().join().getCode());
        assertNull(decode.reservationHandle("1"));
        assertFalse(reservations.contains("1"));
        verify(selector, never()).select(any(), any(), any());
    }

    private Request addWaiting(String requestId) {
        return addWaiting(requestId, 30);
    }

    @Test
    void unavailablePrefillWhileWaitingReturnsRequestToMainWithoutPublishing() {
        Request request = addWaiting("1");
        holdQueueQuota(request);
        doReturn(false).when(prefillCheck).isAvailable(request);
        loop.runOnce();
        assertReturnedToMain(request);
        verify(selector, never()).select(any(), any(), any());
    }

    @Test
    void unavailablePrefillReleasesExistingDecodeReservationBeforeReturningToMain() {
        Request request = addReserved("1");
        holdQueueQuota(request);
        reservations.add("1");
        doReturn(false).when(prefillCheck).isAvailable(request);

        loop.runOnce();
        assertReturnedToMain(request);
        assertNull(decode.reservationHandle("1"));
        assertFalse(reservations.contains("1"));
        loop.runOnce();
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    @Test
    void prefillRetiringAtPublishReleasesPermitAndReservation() {
        Request request = addWaiting("1");
        holdQueueQuota(request);
        doReturn(null).when(prefillCheck).hold(request);

        loop.runOnce();
        assertReturnedToMain(request);
        assertEquals(0, decode.routingView().totalLoad());
        assertFalse(reservations.contains("1"));
    }

    @Test
    void emptyDecodeShortcutAlsoChecksPrefillBeforePublishing() {
        Request request = addWaiting("1");
        holdQueueQuota(request);
        doReturn(PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED))).when(selector)
                .select(any(), any(), any());
        doAnswer(call -> SelectedRole.decode(emptyDecode.tryHold(), emptyDecodeStatus())).when(emptyDecodeFinder)
                .find(any(), any());
        doReturn(null).when(prefillCheck).hold(request);

        loop.runOnce();
        assertReturnedToMain(request);
        assertEquals(0, emptyDecode.routingView().totalLoad());
        assertFalse(reservations.contains("1"));
    }

    @Test
    void cancellationDuringPrefillCheckDoesNotRequeueOrDoubleReleaseQuota() {
        Request request = addWaiting("1");
        holdQueueQuota(request);
        doAnswer(call -> {
            request.cancel(CancelReason.CLIENT_CANCELLED);
            cleanup.releaseEnded(request);
            return null;
        }).when(prefillCheck).hold(request);

        loop.runOnce();
        assertEquals(Stage.CANCELLED, request.stage());
        assertEquals(0, mainQueue.size());
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
        assertEquals(0, decode.routingView().totalLoad());
        assertFalse(reservations.contains("1"));
    }

    @Test
    void orphanSweepCannotRemoveReservedOrPublishedV2Request() {
        Request request = addWaiting("1");
        loop.runOnce();
        assertTrue(reservations.contains("1"));
        decode.evictExpiredRequests(-1L, reservations::contains);
        assertNotNull(decode.reservationHandle("1"));

        loop.runOnce();
        assertEquals(Stage.PUBLISHED, request.stage());
        assertTrue(reservations.contains("1"));
        decode.evictExpiredRequests(-1L, reservations::contains);
        assertNotNull(decode.reservationHandle("1"));
    }

    private void holdQueueQuota(Request request) {
        queueCapacity.increaseTotalQueueCount(request.future());
        assertTrue(queueCapacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        request.holdQueueSlot(RoleType.PREFILL);
    }

    private void assertReturnedToMain(Request request) {
        assertEquals(Stage.MAIN_WAITING, request.stage());
        assertFalse(request.future().isDone());
        assertNull(request.prefillChoice());
        assertNull(request.decodeChoice());
        assertEquals(0L, request.prefillDecidedAtMs());
        assertEquals(1, mainQueue.size());
        assertEquals(1L, request.queueSeq());
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(0, queueCapacity.engineSubQueueWaiting(RoleType.PREFILL));
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
