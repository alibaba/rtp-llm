package org.flexlb.balance.schedulev2.decode;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
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
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
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
                mock(DefaultRouter.class));
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
    void cancelledRequestReturnsDecodeReservation() {
        Request request = addWaiting("1");
        loop.runOnce();

        assertTrue(request.cancel(CancelReason.CLIENT_CANCELLED));
        cleanup.releaseEnded(request);

        assertNull(request.decodeChoice());
        assertEquals(0L, decode.routingView().inputKvReserved());
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

    private Request addWaiting(String requestId) {
        Request request = V2TestSupport.request(config, requestId, 30, 16L);
        request.setQueueSeq(Long.parseLong(requestId));
        V2TestSupport.choose(request, V2TestSupport.prefillEndpoint("p1:8000"), 0L, 0L);
        request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
        request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING);
        requestTable.add(request);
        decodeQueue.addWaiting(request);
        return request;
    }

    private PlacementResult<SelectedRole, RoleType> success() {
        return PlacementResult.success(SelectedRole.decode(decode.tryHold(), decodeStatus));
    }
}
