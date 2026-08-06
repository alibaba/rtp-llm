package org.flexlb.autotpm;

import org.flexlb.balance.endpoint.BatchDispatchExecutor;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchScheduler;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Closed-state parity guard for Stage 3 (blueprint §0 iron rule 1): with all
 * AUTO_TPM switches at their library defaults (off), typical scheduling
 * scenarios behave exactly as before Stage 3 landed —
 *
 * <ul>
 *   <li>normal submit → route → offer → completion: same success response,
 *       no preempt component ever touched (no Cancel RPC, no capability
 *       probe)</li>
 *   <li>route failure: the original error code passes through unchanged,
 *       both with the default null controller wiring and with a real
 *       controller wired but switched off</li>
 *   <li>non-capacity route errors never even consult the controller</li>
 *   <li>terminal paths leave no side effect (priority registry drains)</li>
 * </ul>
 */
class BaselineParityStage3Test {

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private BatchSchedulerReporter reporter;
    private BatchScheduler scheduler;
    private BatchDispatchExecutor dispatchExecutor;
    private EndpointRegistry endpointRegistry;
    private InflightStore inflightStore;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);

        // library defaults + the minimal operational knobs the batcher needs;
        // every AUTO_TPM switch stays untouched (default off)
        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setFlexlbBatchSizeMax(1);
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        when(configService.loadBalanceConfig()).thenReturn(config);

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return successRoute(ctx.getRequestId());
        });
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    return CompletableFuture.completedFuture(ackFor(request));
                });
        dispatchExecutor = new BatchDispatchExecutor(configService, null);
        inflightStore = new InflightStore(reporter, configService);
        endpointRegistry = new EndpointRegistry(configService, grpcClient, dispatchExecutor,
                inflightStore, reporter, null);
        scheduler = new BatchScheduler(configService, router,
                endpointRegistry, reporter, inflightStore,
                new FlexlbMetricHelper(null, MetricConstant.PATH_BATCH));

        WorkerStatus ws = new WorkerStatus();
        ws.setIp("10.0.0.1");
        ws.setPort(8080);
        ws.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, "10.0.0.1:8080", ws);
    }

    @AfterEach
    void tearDown() {
        inflightStore.shutdown();
        endpointRegistry.close();
        dispatchExecutor.shutdown();
    }

    // ---- guard: the library defaults really keep every AUTO_TPM switch off ----

    @Test
    void libraryDefaults_keepAllAutoTpmSwitchesOff() {
        FlexlbConfig defaults = new FlexlbConfig();
        assertFalse(defaults.isAutoTpmEnabled(), "master switch must default off");
        assertFalse(defaults.isAutoTpmDecodeRunningPreemptEnabled(), "preempt switch must default off");
    }

    // ---- normal submit → route → offer → completion, no preempt side effect ----

    @Test
    void defaultConfig_successPath_behavesAsPreStage3() throws Exception {
        CompletableFuture<Response> future = scheduler.submit(context(1));

        Response response = future.get(2, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        assertTrue(response.isEnqueuedByMaster());
        verify(grpcClient).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());

        // no preempt component touched: no Cancel RPC, no capability probe
        verify(grpcClient, never()).cancelAsync(anyString(), anyInt(), anyLong(), any(), anyLong());
        verify(grpcClient, never()).isCancelSupported(anyString(), anyInt());
        // terminal path side-effect free: the priority registry drained
        assertEquals(0, scheduler.priorityRegistry().size());
    }

    // ---- route failure: error code passthrough (default null-controller wiring) ----

    @Test
    void defaultConfig_routeFailure_errorCodeUnchanged_noPreemptComponentTouched() throws Exception {
        Response failure = Response.error(StrategyErrorType.NO_PREFILL_WORKER);
        when(router.route(any(BalanceContext.class))).thenReturn(failure);

        Response response = scheduler.submit(context(21)).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
        verify(grpcClient, never()).cancelAsync(anyString(), anyInt(), anyLong(), any(), anyLong());
        verify(grpcClient, never()).isCancelSupported(anyString(), anyInt());
        assertEquals(0, scheduler.priorityRegistry().size());
    }

    // ---- capacity failure with a wired but switched-off controller: still parity ----

    @Test
    void wiredControllerButSwitchesOff_capacityFailure_errorCodeUnchanged_noCancelRpc() throws Exception {
        // real controller wired in (Stage 3 wiring present), but all AUTO_TPM
        // switches remain at their defaults — the preempt branch must be a
        // complete no-op from the outside
        scheduler.setPressureController(new PriorityPressureController(configService,
                endpointRegistry, grpcClient, inflightStore, scheduler.priorityRegistry(),
                mock(FlexlbMetricHelper.class)));
        Response failure = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        when(router.route(any(BalanceContext.class))).thenReturn(failure);

        Response response = scheduler.submit(context(31)).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        verify(grpcClient, never()).cancelAsync(anyString(), anyInt(), anyLong(), any(), anyLong());
        verify(grpcClient, never()).isCancelSupported(anyString(), anyInt());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
        assertEquals(0, scheduler.priorityRegistry().size());
    }

    // ---- non-capacity route errors never even consult the controller ----

    @Test
    void nonCapacityRouteError_controllerNeverConsulted() throws Exception {
        PriorityPressureController controller = mock(PriorityPressureController.class);
        scheduler.setPressureController(controller);
        Response failure = Response.error(StrategyErrorType.NO_PREFILL_WORKER);
        when(router.route(any(BalanceContext.class))).thenReturn(failure);

        Response response = scheduler.submit(context(41)).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        verify(controller, never()).tryPreempt(any(BalanceContext.class));
    }

    // ==================== fixtures ====================

    private static EngineRpcService.EnqueueBatchResponsePB ackFor(
            EngineRpcService.EnqueueBatchRequestPB request) {
        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
        request.getDpSlotsList().stream()
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(ext -> ext.getInput().getRequestId())
                .forEach(reqId -> response.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(reqId)
                        .build()));
        return response.build();
    }

    private static BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));
        return ctx;
    }

    private static byte[] generateInputBytes(long requestId) {
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
    }

    private static Response successRoute(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)
        ));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("g1");
        status.setRequestId(requestId);
        return status;
    }
}
