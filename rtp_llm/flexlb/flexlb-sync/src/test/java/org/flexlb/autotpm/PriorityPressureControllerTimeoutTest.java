package org.flexlb.autotpm;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightState;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Never-optimistic timeout timeline of {@link PriorityPressureController}
 * (blueprint §1.7 wait-timeout branch + §1.10 WorkerStatus close-out),
 * driven against a REAL {@link DecodeEndpoint} / {@link InflightStore} so the
 * full hand-off between the bounded release wait and the calibrate close-out
 * path is exercised end to end:
 *
 * <ol>
 *   <li>a) release wait times out (layer-2 entry never disappears) →
 *       {@code tryPreempt} returns empty and the incoming request is NOT
 *       dispatched</li>
 *   <li>b) the victim's {@link InflightItem} stays non-terminal and the
 *       cancel intent survives the timeout</li>
 *   <li>c) the engine's finished report (errorCode=8100, cancelReason=2)
 *       later reaches {@code processFinishedTasks} → the victim is settled
 *       as AUTO_TPM_PREEMPTED (4290)</li>
 *   <li>d) once victim and incoming are each terminal, InflightStore
 *       accounting drains to zero (both remain as tombstones)</li>
 * </ol>
 */
class PriorityPressureControllerTimeoutTest {

    private static final String EP_IP = "10.0.0.1";
    private static final int EP_PORT = 8080;
    private static final int EP_GRPC_PORT = 8081;
    private static final long VICTIM_ID = 100L;
    private static final int VICTIM_PRIORITY = 30;
    private static final long INCOMING_ID = 200L;
    private static final int INCOMING_PRIORITY = 70;
    private static final long ENGINE_CANCELLED = 8100L;
    private static final int REASON_PRIORITY_PREEMPTED = 2;
    /** Short bounded-wait budget keeps the timeout timeline fast. */
    private static final long WAIT_TIMEOUT_MS = 30;

    private ConfigService configService;
    private EndpointRegistry endpointRegistry;
    private EngineGrpcClient grpcClient;
    private FlexlbMetricHelper metricHelper;
    private WorkerStatus status;
    private DecodeEndpoint decodeEp;
    private InflightStore inflightStore;
    private PriorityRegistry priorityRegistry;
    private PriorityPressureController controller;

    private CompletableFuture<Response> victimFuture;
    private InflightItem victimItem;
    private CompletableFuture<Response> incomingFuture;
    private InflightItem incomingItem;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        endpointRegistry = mock(EndpointRegistry.class);
        grpcClient = mock(EngineGrpcClient.class);
        metricHelper = mock(FlexlbMetricHelper.class);
        when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

        // real accounting + real endpoint: the timeline must cross the real
        // layer-2 state machine, not a stubbed hasEngineTask
        inflightStore = new InflightStore(mock(BatchSchedulerReporter.class), configService);
        status = new WorkerStatus();
        status.setIp(EP_IP);
        status.setPort(EP_PORT);
        status.setGrpcPort(EP_GRPC_PORT);
        decodeEp = new DecodeEndpoint(status, inflightStore);

        ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
        decodeEndpoints.put(decodeEp.ipPort(), decodeEp);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(decodeEndpoints);
        when(endpointRegistry.getDecode(decodeEp.ipPort())).thenReturn(decodeEp);
        when(grpcClient.isCancelSupported(EP_IP, EP_GRPC_PORT)).thenReturn(true);
        when(grpcClient.cancelAsync(eq(EP_IP), eq(EP_GRPC_PORT), eq(VICTIM_ID),
                eq(EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.CancelResponsePB.newBuilder()
                        .setFound(true)
                        .build()));

        // victim + incoming registered through putIfAbsent so activeCount tracks them
        victimFuture = new CompletableFuture<>();
        victimItem = new InflightItem(context(VICTIM_ID, VICTIM_PRIORITY), victimFuture, null);
        assertNull(inflightStore.putIfAbsent(String.valueOf(VICTIM_ID), victimItem));
        incomingFuture = new CompletableFuture<>();
        incomingItem = new InflightItem(context(INCOMING_ID, INCOMING_PRIORITY), incomingFuture, null);
        assertNull(inflightStore.putIfAbsent(String.valueOf(INCOMING_ID), incomingItem));

        priorityRegistry = new PriorityRegistry();
        priorityRegistry.register(VICTIM_ID, VICTIM_PRIORITY);
        priorityRegistry.register(INCOMING_ID, INCOMING_PRIORITY);

        // the engine has accepted the victim: layer-2 RUNNING via calibrate
        reportRunning(VICTIM_ID);
        assertTrue(decodeEp.hasEngineTask(VICTIM_ID));

        controller = new PriorityPressureController(configService, endpointRegistry, grpcClient,
                inflightStore, priorityRegistry, metricHelper);
    }

    @AfterEach
    void tearDown() {
        inflightStore.shutdown();
    }

    // ---- a) + b) wait timeout → empty, incoming not dispatched, victim non-terminal, intent kept ----

    @Test
    void tryPreempt_releaseWaitTimesOut_returnsEmpty_keepsVictimNonTerminalAndIntent() {
        Optional<PreemptResult> result = runTimeoutPreempt();

        // a) not optimistic: no capacity granted, the incoming request is not dispatched
        assertTrue(result.isEmpty(), "wait timeout must not grant capacity");
        assertFalse(incomingFuture.isDone(), "incoming must not be dispatched or settled by the controller");
        verify(metricHelper).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_TIMEOUT);

        // b) victim untouched, cancel intent survives for the close-out path
        assertFalse(victimItem.isTerminated(),
                "settlement belongs to the WorkerStatus close-out path after a wait timeout");
        assertFalse(victimFuture.isDone());
        assertTrue(controller.hasCancelIntent(VICTIM_ID));
        assertTrue(decodeEp.hasEngineTask(VICTIM_ID), "layer-2 entry stays until the finished report");
        assertEquals(2, inflightStore.activeCount(), "both items still active after the timeout");
    }

    // ---- c) WorkerStatus finished report (cancelReason=2) closes the victim out as 4290 ----

    @Test
    void workerStatusCloseOut_afterTimeout_settlesVictimAs4290() {
        runTimeoutPreempt();

        reportFinished(VICTIM_ID, ENGINE_CANCELLED, REASON_PRIORITY_PREEMPTED);

        assertTrue(victimItem.isTerminated());
        assertEquals(InflightState.FAILED, victimItem.state());
        Response settled = victimFuture.join();
        assertFalse(settled.isSuccess());
        assertEquals(4290, settled.getCode());
        assertEquals("AUTO_TPM_PREEMPTED", settled.getErrorMessage());
        // the finished report also released the layer-2 entry
        assertFalse(decodeEp.hasEngineTask(VICTIM_ID));
    }

    // ---- d) accounting drains to zero once both requests are terminal ----

    @Test
    void accounting_drainsToZero_afterVictimCloseOutAndIncomingTerminal() {
        assertEquals(2, inflightStore.activeCount());
        runTimeoutPreempt();

        // victim settled by the close-out path → one active item left
        reportFinished(VICTIM_ID, ENGINE_CANCELLED, REASON_PRIORITY_PREEMPTED);
        assertEquals(1, inflightStore.activeCount());

        // the incoming request reaches its own terminal (this round yielded no
        // capacity — the scheduler fails it with NO_AVAILABLE_WORKER)
        incomingItem.complete(Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
        assertTrue(incomingItem.isTerminated());

        assertEquals(0, inflightStore.activeCount(), "accounting must drain to zero");
        // both remain as tombstones (terminal state, still within TTL)
        assertEquals(2, inflightStore.totalSize());
        assertTrue(inflightStore.get(String.valueOf(VICTIM_ID)).isTerminated());
        assertTrue(inflightStore.get(String.valueOf(INCOMING_ID)).isTerminated());
    }

    // ==================== fixtures ====================

    /** Fire tryPreempt with a layer-2 entry that never disappears → wait timeout. */
    private Optional<PreemptResult> runTimeoutPreempt() {
        return controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));
    }

    private static FlexlbConfig enabledConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmEnabled(true);
        config.setAutoTpmDecodeRunningPreemptEnabled(true);
        config.setAutoTpmPreemptRateLimitPerMin(10);
        config.setAutoTpmEndpointPreemptQpsLimit(0);
        config.setAutoTpmCommitWaitReleaseTimeoutMs(WAIT_TIMEOUT_MS);
        config.setAutoTpmPreemptCriticalSectionMs(0);
        return config;
    }

    private static BalanceContext context(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setPriority(priority);
        return ctx;
    }

    /** Feed a RUNNING report so the request enters layer 2 via calibrate. */
    private void reportRunning(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(TaskPhase.RUNNING);
        status.getAvailableKvCacheTokens().set(10_000);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of(String.valueOf(requestId), task));
        decodeEp.onWorkerStatusUpdate(status, response);
    }

    /** Feed a finished report driving processFinishedTasks (close-out path). */
    private void reportFinished(long requestId, long errorCode, int cancelReason) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setErrorCode(errorCode);
        task.setCancelReason(cancelReason);
        task.setErrorMessage("cancelled");
        status.getAvailableKvCacheTokens().set(10_000);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of(String.valueOf(requestId), task));
        decodeEp.onWorkerStatusUpdate(status, response);
    }
}
