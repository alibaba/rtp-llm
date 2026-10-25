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
import org.flexlb.balance.schedulev2.DecodeChoice;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.Request;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.RequestTable;
import org.flexlb.balance.schedulev2.Stage;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;

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
import static org.mockito.Mockito.when;

class DecodeQueueLoopTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final QueueCapacity queueCapacity = new QueueCapacity();
    private final RequestTable requestTable = new RequestTable();
    private final LoopSignal decodeLoopSignal = new LoopSignal();
    private final DecodeQueue decodeQueue = new DecodeQueue(false);
    private final RequestCleanup cleanup;
    private final DecodeSelector selector = mock(DecodeSelector.class);
    private final DecodeEndpoint decode =
            new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.2", 8080, 8081, "test"),
                    mock(EndpointEventHandler.class));
    private final ServerStatus decodeStatus = new ServerStatus();
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
                mock(DefaultRouter.class), new DecodePreemptionCoordinator(mock(EngineCancelChannel.class),
                        mock(RequestRegistry.class), mock(RequestSchedulerReporter.class)));
    }

    @AfterEach
    void closeEndpoint() {
        decode.close();
    }

    @Test
    void requestIsReservedThenPublishedWithPrefillAndDecode() {
        Request request = addWaiting("1");

        assertTrue(loop.runOnce());
        assertEquals(Stage.DECODE_RESERVED, request.stage());
        assertSame(decode, request.decodeChoice().endpoint());
        assertEquals(1, decodeQueue.reservedCount());

        assertTrue(loop.runOnce());
        assertEquals(Stage.PUBLISHED, request.stage());
        Response response = request.future().join();
        assertTrue(response.isSuccess());
        assertEquals(List.of(request.prefillChoice().serverStatus(), decodeStatus), response.getServerStatus());
        assertEquals(0, decodeQueue.reservedCount());
        assertEquals(0, requestTable.size());
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
