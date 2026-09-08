package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.optimizer.OptimizerClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FlexlbScheduleEngineIndexTest {

    private RouteService routeService;
    private FlexlbServiceImpl service;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        LBStatusConsistencyService consistencyService =
                mock(LBStatusConsistencyService.class);
        when(consistencyService.isNeedConsistency()).thenReturn(false);

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());

        ActiveRequestCounter activeRequestCounter = mock(ActiveRequestCounter.class);
        when(activeRequestCounter.acquire()).thenReturn(
                mock(ActiveRequestCounter.RequestToken.class));

        CacheAwareService cacheAwareService = mock(CacheAwareService.class);
        when(cacheAwareService.prepareBlockCacheKeys(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(null));

        service = new FlexlbServiceImpl(
                routeService,
                consistencyService,
                mock(EngineHealthReporter.class),
                activeRequestCounter,
                null,
                configService,
                mock(BatchSchedulerReporter.class),
                mock(ServerScheduleLatencyRecorder.class),
                mock(RequestSchedulerReporter.class),
                cacheAwareService,
                mock(OptimizerClient.class));
    }

    @Test
    void scheduleIncludesEngineIndexForMultiEngineWorker() {
        when(routeService.route(any(BalanceContext.class))).thenReturn(
                CompletableFuture.completedFuture(response(serverStatus(1, 2))));

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = schedule();

        assertTrue(response.getSuccess());
        assertEquals(1, response.getServerStatusCount());
        FlexlbScheduleProtocol.FlexlbServerStatusPB selected =
                response.getServerStatus(0);
        assertEquals("127.0.0.1", selected.getServerIp());
        assertEquals(8080, selected.getHttpPort());
        assertEquals(8081, selected.getGrpcPort());
        assertTrue(selected.hasEngineIndex());
        assertEquals(1, selected.getEngineIndex());
    }

    @Test
    void scheduleOmitsEngineIndexForSingleEngineWorker() {
        when(routeService.route(any(BalanceContext.class))).thenReturn(
                CompletableFuture.completedFuture(response(serverStatus(0, 1))));

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = schedule();

        assertTrue(response.getSuccess());
        assertEquals(1, response.getServerStatusCount());
        assertFalse(response.getServerStatus(0).hasEngineIndex());
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB schedule() {
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        service.schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId("5001")
                .setSeqLen(16)
                .addInputIds(1)
                .build(), observer);

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        org.mockito.Mockito.verify(observer).onNext(captor.capture());
        org.mockito.Mockito.verify(observer).onCompleted();
        return captor.getValue();
    }

    private static Response response(ServerStatus status) {
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        response.setServerStatus(List.of(status));
        return response;
    }

    private static ServerStatus serverStatus(int engineIndex, int multiEngineNum) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(RoleType.DECODE);
        status.setServerIp("127.0.0.1");
        status.setHttpPort(8080);
        status.setGrpcPort(8081);
        status.setDpRank(0);
        status.setGroup("test-group");
        status.setRequestId("5001");
        status.setSelectedEngineIndex(engineIndex, multiEngineNum);
        return status;
    }
}
