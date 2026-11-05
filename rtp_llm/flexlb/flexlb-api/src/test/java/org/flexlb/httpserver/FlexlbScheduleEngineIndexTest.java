package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
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
    private FlexlbConfig routingConfig;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        LBStatusConsistencyService consistencyService = mock(LBStatusConsistencyService.class);
        when(consistencyService.isNeedConsistency()).thenReturn(false);

        ConfigService configService = mock(ConfigService.class);
        routingConfig = new FlexlbConfig();
        routingConfig.setDispatcher(DispatcherConfig.nonBatch());
        when(configService.loadBalanceConfig()).thenReturn(routingConfig);

        CacheAwareService cacheAwareService = mock(CacheAwareService.class);

        service =
                new FlexlbServiceImpl(
                        routeService,
                        consistencyService,
                        mock(EngineHealthReporter.class),
                        mock(FlexlbGrpcForwarder.class),
                        configService,
                        mock(BatchSchedulerReporter.class),
                        mock(ServerScheduleLatencyRecorder.class),
                        mock(RequestSchedulerReporter.class),
                        cacheAwareService);
    }

    @Test
    void scheduleIncludesEngineIndexForMultiEngineWorker() {
        when(routeService.route(any(BalanceContext.class)))
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
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response(serverStatus(0, 2))));

        var result =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.parseFrom(schedule().toByteArray());

        assertTrue(result.getSuccess());
        assertTrue(result.getServerStatus(0).hasEngineIndex());
        assertEquals(0, result.getServerStatus(0).getEngineIndex());
    }

    @Test
    void scheduleOmitsEngineIndexForSingleEngineWorker() {
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(response(serverStatus(0, 1))));

        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = schedule();

        assertTrue(response.getSuccess());
        assertEquals(1, response.getServerStatusCount());
        assertFalse(response.getServerStatus(0).hasEngineIndex());
        assertFalse(response.hasPrefillDelegation());
    }

    @Test
    void schedulePreservesDelegationAndDecodeMetadataWithoutReportingEnqueue() throws Exception {
        Response route = response(serverStatus(1, 2));
        route.getServerStatus().getFirst().setPreemptRequestIds(List.of("victim"));
        route.setPrefillDelegation(new Response.PrefillDelegation("master_biz"));
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(
                        CompletableFuture.completedFuture(
                                Response.buildSuccessResponse(route, false)));
        var wire =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.parseFrom(schedule().toByteArray());
        assertTrue(wire.getSuccess());
        assertTrue(wire.hasPrefillDelegation());
        assertEquals("master_biz", wire.getPrefillDelegation().getMasterBizName());
        assertFalse(wire.getEnqueuedByMaster());
        assertEquals(1, wire.getServerStatus(0).getEngineIndex());
        assertEquals(List.of("victim"), wire.getServerStatus(0).getPreemptRequestIdsList());
        assertEquals("DECODE", wire.getServerStatus(0).getRole());
    }

    @Test
    void httpJsonPreservesTheSameDelegationContract() throws Exception {
        var route = response(serverStatus(0, 2));
        route.setPrefillDelegation(new Response.PrefillDelegation("master_biz"));
        String encoded = org.flexlb.util.JsonUtils.toString(Response.copyOf(route));
        var restored = org.flexlb.util.JsonUtils.toObject(encoded, Response.class);
        assertEquals(route.getPrefillDelegation(), restored.getPrefillDelegation());
        assertFalse(restored.isEnqueuedByMaster());
        assertEquals(0, restored.getServerStatus().getFirst().getEngineIndex());
        assertFalse(
                org.flexlb.util.JsonUtils.toString(response(serverStatus(0, 1)))
                        .contains("prefill_delegation"));
    }

    @Test
    void requestPreservesExplicitPrefillDelegationInGrpcAndHttp() throws Exception {
        var contextCaptor = ArgumentCaptor.forClass(BalanceContext.class);
        when(routeService.route(contextCaptor.capture()))
                .thenReturn(CompletableFuture.completedFuture(response(serverStatus(0, 1))));
        var observer = mock(StreamObserver.class);
        var wire =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(5001)
                        .setSeqLen(16)
                        .setSkipPrefill(true)
                        .build();
        service.schedule(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(wire.toByteArray()),
                observer);
        assertTrue(contextCaptor.getValue().getRequest().isSkipPrefill());
        var http =
                org.flexlb.util.JsonUtils.toObject(
                        "{\"request_id\":\"5001\",\"skip_prefill\":true}",
                        org.flexlb.dao.loadbalance.Request.class);
        assertTrue(http.isSkipPrefill());
        assertFalse(new org.flexlb.dao.loadbalance.Request().isSkipPrefill());
        assertFalse(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.getDefaultInstance()
                        .getSkipPrefill());
    }

    @Test
    void delegatedBatchDoesNotRequireAnEnginePayload() {
        routingConfig.setDispatcher(new DispatcherConfig());
        var navi = new org.flexlb.config.NaviSchedPrefillSchedulerConfig();
        navi.setMasterBizName("master_biz");
        routingConfig.setPrefillScheduler(navi);
        var route = response(serverStatus(0, 1));
        route.setPrefillDelegation(new Response.PrefillDelegation("master_biz"));
        when(routeService.route(any(BalanceContext.class)))
                .thenReturn(CompletableFuture.completedFuture(route));
        assertTrue(schedule(true).getSuccess());
        org.mockito.Mockito.verify(routeService).route(any(BalanceContext.class));
    }

    @Test
    void localBatchStillRequiresAnEnginePayload() {
        routingConfig.setDispatcher(new DispatcherConfig());
        var response = schedule();
        assertFalse(response.getSuccess());
        assertTrue(response.getErrorMessage().contains("generate_input"));
        org.mockito.Mockito.verifyNoInteractions(routeService);
    }

    @Test
    void delegationModeDoesNotLetAnUnmarkedBatchBypassPayloadValidation() {
        routingConfig.setDispatcher(new DispatcherConfig());
        var navi = new org.flexlb.config.NaviSchedPrefillSchedulerConfig();
        navi.setMasterBizName("master_biz");
        routingConfig.setPrefillScheduler(navi);
        assertFalse(schedule().getSuccess());
        org.mockito.Mockito.verifyNoInteractions(routeService);
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB schedule() {
        return schedule(false);
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB schedule(boolean skipPrefill) {
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);
        service.schedule(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(5001)
                        .setRequestIdStr("5001")
                        .setSeqLen(16)
                        .setSkipPrefill(skipPrefill)
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
