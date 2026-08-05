package org.flexlb.balance.endpoint;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightState;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.mockito.Mockito;
import static org.mockito.Mockito.mock;

/**
 * Full evidence-chain tests for the STALE round-based engine-task eviction
 * defence path on both {@link DecodeEndpoint} and {@link PrefillEndpoint}
 * (red-team audit: zero triggers in integration tests).
 *
 * <p>Boundary condition (verified against the code): an engine task is
 * evicted when {@code round - lastSeenRound >= STALE_EVICT_ROUNDS} (3);
 * strictly below the threshold it survives. The exact-boundary tests
 * already exist in {@code DecodeEndpointTest#calibrate_staleEngineTaskEvictedAfterMissingRounds}
 * and {@code PrefillEndpointTest#staleEngineTaskEvictedAfterMissingRounds};
 * this class adds the missing chain evidence:
 *
 * <ul>
 *   <li>selective omission — among several tasks only the one absent from
 *       consecutive reports is evicted, refreshed ones survive;</li>
 *   <li>eviction warn log emission;</li>
 *   <li>decode-side KV/counter linkage (layer-2 eviction must not touch the
 *       KV counters already released on acceptance — no double-release);</li>
 *   <li>A3 fix: STALE eviction now drives the bound {@link InflightItem} to a
 *       terminal state (FAILED) so the client future is settled in seconds,
 *       not the 300s TTL safety net.</li>
 * </ul>
 */
class EndpointStaleEvictionChainTest {

    private ch.qos.logback.classic.Logger syncLogger;
    private ListAppender<ILoggingEvent> logAppender;

    @BeforeEach
    void attachLogCapture() {
        syncLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("syncLogger");
        logAppender = new ListAppender<>();
        logAppender.start();
        syncLogger.addAppender(logAppender);
    }

    @AfterEach
    void detachLogCapture() {
        syncLogger.detachAppender(logAppender);
        logAppender.stop();
    }

    private long warnLogCount(String fragment) {
        return logAppender.list.stream()
                .filter(event -> event.getFormattedMessage().contains(fragment))
                .count();
    }

    // ==================== decode side ====================

    private WorkerStatus decodeStatus;
    private DecodeEndpoint decodeEndpoint;

    private void setUpDecode() {
        decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.1");
        decodeStatus.setPort(8080);
        decodeStatus.setGrpcPort(8081);
        decodeEndpoint = new DecodeEndpoint(decodeStatus, null);
    }

    private void decodeCalibrate(Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
        decodeStatus.getAvailableKvCacheTokens().set(10_000);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        decodeEndpoint.onWorkerStatusUpdate(decodeStatus, response);
    }

    private static TaskInfo decodeTask(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(TaskPhase.RUNNING);
        return task;
    }

    @Test
    void decodeSelectiveOmissionEvictsOnlyTheAbsentTask() {
        setUpDecode();
        decodeEndpoint.reserve(100L, 500, 800);
        decodeEndpoint.reserve(101L, 300, 400);

        // round 1: both accepted into layer 2 (KV reservations released)
        decodeCalibrate(Map.of("100", decodeTask(100L), "101", decodeTask(101L)), null);
        assertEquals(2, decodeEndpoint.decodeEngineTaskCount());
        assertEquals(0, decodeEndpoint.decodeInflightHardKvReserved());
        assertEquals(0, decodeEndpoint.decodeInflightExpectedKvReserved());

        // rounds 2..3: only 101 keeps being reported — below threshold, both survive
        decodeCalibrate(Map.of("101", decodeTask(101L)), null); // round 2
        decodeCalibrate(Map.of("101", decodeTask(101L)), null); // round 3: 3-1=2 < 3
        assertEquals(2, decodeEndpoint.decodeEngineTaskCount());
        assertEquals(0, warnLogCount("evicting as stale"));

        // round 4: 4-1=3 >= 3 — only the absentee is evicted
        decodeCalibrate(Map.of("101", decodeTask(101L)), null);
        assertEquals(1, decodeEndpoint.decodeEngineTaskCount());
        assertNull(decodeEndpoint.engineTaskPhase(100L), "absentee must be evicted");
        assertNotNull(decodeEndpoint.engineTaskPhase(101L), "refreshed task must survive");

        // eviction warn log fired exactly once, naming the absentee
        assertEquals(1, warnLogCount("evicting as stale"));
        assertEquals(1, warnLogCount("reqId=100"));

        // KV linkage: layer-2 eviction adjusts no KV counters (already released
        // on acceptance) — counters stay at zero, never negative
        assertEquals(0, decodeEndpoint.decodeInflightHardKvReserved());
        assertEquals(0, decodeEndpoint.decodeInflightExpectedKvReserved());
        assertEquals(1, decodeEndpoint.decodeTotalLoad());
    }

    @Test
    void decodeStaleEvictionTerminatesBoundInflightItem() {
        // A3 fix: STALE eviction now drives the bound InflightItem to a terminal
        // state (FAILED) so the client future is settled in seconds, not the
        // 300s TTL safety net.
        decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.1");
        decodeStatus.setPort(8080);
        decodeStatus.setGrpcPort(8081);
        FlexlbConfig lbConfig = new FlexlbConfig();
        lbConfig.setFlexlbInflightTtlMs(600_000);
        ConfigService configService = mock(ConfigService.class);
        Mockito.lenient().when(configService.loadBalanceConfig()).thenReturn(lbConfig);
        InflightStore store = new InflightStore(mock(BatchSchedulerReporter.class), configService);
        decodeEndpoint = new DecodeEndpoint(decodeStatus, store);

        try {
            CompletableFuture<Response> future = new CompletableFuture<>();
            Request request = new Request();
            request.setRequestId(100L);
            BalanceContext ctx = new BalanceContext();
            ctx.setRequest(request);
            InflightItem item = new InflightItem(ctx, future, null);

            decodeEndpoint.reserve(100L, 500, 500);
            item.setDecodeEp(decodeEndpoint);
            store.putIfAbsent(String.valueOf(100L), item);

            decodeCalibrate(Map.of("100", decodeTask(100L)), null); // round 1: accepted
            decodeCalibrate(null, null); // round 2
            decodeCalibrate(null, null); // round 3
            decodeCalibrate(null, null); // round 4: evicted as stale
            assertEquals(0, decodeEndpoint.decodeEngineTaskCount());

            assertEquals(InflightState.FAILED, item.state(),
                    "STALE eviction must drive the item terminal");
            assertTrue(future.isDone(), "the item future must be settled immediately");
        } finally {
            store.shutdown();
        }
    }

    // ==================== prefill side ====================

    private PrefillEndpoint prefillEndpoint;

    private void setUpPrefill() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchQueueMaxSize(100);
        config.setFlexlbBatchFixedWaitMs(300);
        config.setCostFormula("10 + 0.1*sum(computeTokens) + 5*batchSize");

        prefillEndpoint = new PrefillEndpoint(status, config,
                mock(EngineGrpcClient.class), mock(BatchDispatchExecutor.class),
                new BatchIdGenerator("127.0.0.1", 7001), () -> 0,
                mock(BatchSchedulerReporter.class), null);
    }

    @AfterEach
    void tearDownPrefill() {
        if (prefillEndpoint != null) {
            prefillEndpoint.close();
            prefillEndpoint = null;
        }
    }

    private void prefillCalibrate(Map<String, TaskInfo> finished, Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        prefillEndpoint.onWorkerStatusUpdate(prefillEndpoint.getStatus(), response);
    }

    private BatchItem prefillBatchItem(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(500);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(200);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, new CompletableFuture<>(), null,
                prefill, null, prefillEndpoint, null, System.currentTimeMillis());
    }

    private static TaskInfo prefillTask(long requestId, long batchId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setPhase(TaskPhase.RUNNING);
        return task;
    }

    @Test
    void prefillSelectiveOmissionEvictsOnlyTheAbsentBatch() {
        setUpPrefill();
        prefillEndpoint.commitBatch(1L, 100, List.of(prefillBatchItem(1L)));
        prefillEndpoint.commitBatch(2L, 100, List.of(prefillBatchItem(2L)));

        // round 1: both batches accepted into layer 2
        Map<String, TaskInfo> bothRunning = new HashMap<>();
        bothRunning.put("1", prefillTask(1L, 1L));
        bothRunning.put("2", prefillTask(2L, 2L));
        prefillCalibrate(Map.of(), bothRunning);
        assertEquals(2, prefillEndpoint.prefillEngineTaskCount());
        assertEquals(2, prefillEndpoint.prefillPendingRequestCount());

        // rounds 2..3: only batch 2 keeps being reported — both survive
        Map<String, TaskInfo> onlyBatch2 = Map.of("2", prefillTask(2L, 2L));
        prefillCalibrate(Map.of(), onlyBatch2); // round 2
        prefillCalibrate(Map.of(), onlyBatch2); // round 3: 3-1=2 < 3
        assertEquals(2, prefillEndpoint.prefillEngineTaskCount());
        assertEquals(0, warnLogCount("evicting as stale"));

        // round 4: 4-1=3 >= 3 — only the absentee batch is evicted
        prefillCalibrate(Map.of(), onlyBatch2);
        assertEquals(1, prefillEndpoint.prefillEngineTaskCount());

        // counter linkage: the evicted batch's requestCount is subtracted
        assertEquals(1, prefillEndpoint.prefillPendingRequestCount());

        // eviction warn log fired exactly once, naming the absentee key
        assertEquals(1, warnLogCount("evicting as stale"));
        assertEquals(1, warnLogCount("key=1"));
        assertTrue(logAppender.list.stream()
                        .noneMatch(e -> e.getFormattedMessage().contains("key=2")
                                && e.getFormattedMessage().contains("evicting as stale")),
                "the refreshed batch must not be evicted");
    }
}
