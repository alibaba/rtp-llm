package org.flexlb.api.server;

import io.grpc.stub.StreamObserver;
import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.api.protocol.FlexlbScheduleProtocol;
import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.master.MasterStatusService;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.ServiceMetricsReporter;
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

/**
 * Checks when scheduling responses include a worker's logical engine index.
 */
class FlexlbScheduleEngineIndexTest {

    private AbstractRequestScheduler routeService;
    private FlexlbServiceImpl service;

    @BeforeEach
    void setUp() {
        routeService = mock(AbstractRequestScheduler.class);
        MasterStatusService consistencyService = mock(MasterStatusService.class);
        when(consistencyService.isNeedConsistency()).thenReturn(false);

        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig routingConfig = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(routingConfig);

        CacheAwareService cacheAwareService = mock(CacheAwareService.class);

        service =
                FlexlbServiceTestSupport.create(
                        routeService,
                        mock(org.flexlb.balance.schedule.RequestRepository.class),
                        consistencyService,
                        mock(ServiceMetricsReporter.class),
                        mock(FlexlbGrpcForwarder.class),
                        configService,
                        mock(DeliveryMetricsReporter.class),
                        mock(ServerScheduleLatencyRecorder.class),
                        mock(RequestSchedulerReporter.class));
    }

    @Test
    void scheduleIncludesEngineIndexForMultiEngineWorker() {
        when(routeService.submit(any(RequestContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response(serverStatus(1, 2))));

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = schedule();

        assertTrue(response.getSuccess());
        assertEquals(1, response.getServerStatusCount());
        FlexlbScheduleProtocol.FlexlbServerStatusPB selected = response.getServerStatus(0);
        assertEquals("127.0.0.1", selected.getServerIp());
        assertEquals(8080, selected.getHttpPort());
        assertEquals(8081, selected.getGrpcPort());
        assertTrue(selected.hasEngineIndex());
        assertEquals(1, selected.getEngineIndex());
    }

    @Test
    void scheduleIncludesZeroEngineIndexForMultiEngineWorker() throws Exception {
        when(routeService.submit(any(RequestContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response(serverStatus(0, 2))));

        var result =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.parseFrom(schedule().toByteArray());

        assertTrue(result.getSuccess());
        assertTrue(result.getServerStatus(0).hasEngineIndex());
        assertEquals(0, result.getServerStatus(0).getEngineIndex());
    }

    @Test
    void scheduleOmitsEngineIndexForSingleEngineWorker() {
        when(routeService.submit(any(RequestContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response(serverStatus(0, 1))));

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = schedule();

        assertTrue(response.getSuccess());
        assertEquals(1, response.getServerStatusCount());
        assertFalse(response.getServerStatus(0).hasEngineIndex());
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB schedule() {
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        service.schedule(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(5001)
                        .setRequestIdStr("5001")
                        .setSeqLen(16)
                        .build(),
                observer);

        ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                ArgumentCaptor.forClass(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
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
