package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NaviBatchSchedulerConfig;
import com.google.protobuf.ByteString;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link NaviBatchScheduler} windowing and dispatch logic.
 * External dependencies are mocked; the optimizer runs for real on a small scale.
 */
class NaviBatchSchedulerTest {

    private static final double[] NAVI_PARAMS =
            {-4.0, 10.0, 1.4, 20.0, 0.1, 0.09, 1.4, 1.0, -4.0};

    private ConfigService configService;
    private EndpointRegistry endpointRegistry;
    private CacheAwareService cacheAwareService;
    private BatchSubmissionPort batchSubmissionPort;
    private ConfiguredLoadBalanceSelector decodeSelector;
    private NaviBatchScheduler scheduler;
    private FlexlbConfig flexlbConfig;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        endpointRegistry = mock(EndpointRegistry.class);
        cacheAwareService = mock(CacheAwareService.class);
        batchSubmissionPort = mock(BatchSubmissionPort.class);
        decodeSelector = mock(ConfiguredLoadBalanceSelector.class);

        NaviBatchSchedulerConfig batchCfg = new NaviBatchSchedulerConfig();
        batchCfg.setNaviBatchMaxCount(4);
        batchCfg.setNaviBatchWindowMs(100);
        batchCfg.setNaviBatchMaxLoopCount(5);

        flexlbConfig = mock(FlexlbConfig.class);
        when(flexlbConfig.isNaviBatch()).thenReturn(true);
        when(flexlbConfig.naviBatchScheduler()).thenReturn(batchCfg);
        when(flexlbConfig.getDispatcher())
                .thenReturn(SchedulingTestConfig.batchConfig().getDispatcher());
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        when(cacheAwareService.findMatchingEngines(any(), any(), any()))
                .thenReturn(Collections.emptyMap());

        scheduler = new NaviBatchScheduler(
                configService, endpointRegistry, cacheAwareService,
                batchSubmissionPort, decodeSelector);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    // ==================== Helper ====================

    private PrefillEndpoint mockEndpoint(String ip, int port) {
        return mockEndpoint(ip, port, emptyObservation());
    }

    private PrefillEndpoint mockEndpoint(String ip, int port,
                                         WorkerStatus.EngineObservation observation) {
        PrefillEndpoint ep = mock(PrefillEndpoint.class);
        WorkerStatus ws = mock(WorkerStatus.class);
        when(ws.isActiveGeneration()).thenReturn(true);
        when(ws.getGroup()).thenReturn("g1");
        // Generation id stays positive for endpoint topology snapshots.
        when(ws.getGenerationId()).thenReturn(1L);
        when(ws.committedEngineObservation()).thenReturn(observation);
        when(ep.getStatus()).thenReturn(ws);
        when(ep.getIp()).thenReturn(ip);
        when(ep.getHttpPort()).thenReturn(port);
        when(ep.ipPort()).thenReturn(ip + ":" + port);
        when(ep.getLoadMetric()).thenReturn(OptionalLong.of(0L));
        LearningPredictor predictor = mock(LearningPredictor.class);
        when(predictor.weightsSnapshot()).thenReturn(NAVI_PARAMS.clone());
        when(ep.getPredictor()).thenReturn(predictor);
        return ep;
    }

    /** Committed engine observation with no queued work (clean endpoint). */
    private static WorkerStatus.EngineObservation emptyObservation() {
        return observationWithWaiting(0L);
    }

    /** Committed engine observation reporting the given waiting count. */
    private static WorkerStatus.EngineObservation observationWithWaiting(
            long waitingQueryLen) {
        return new WorkerStatus.EngineObservation(
                RoleType.PREFILL, null, 0L, 0L, Map.of(), 0.0,
                0L, 0L, 0L, 0L, 0L, 0L, 0L, waitingQueryLen);
    }

    /**
     * Capture the formatted {@code flexlb_navi_queue_wait} INFO lines logged
     * while {@code action} runs, then detach the appender again. The appender
     * is detached before reading: the flush thread may still append follow-up
     * events (route-decision DEBUG lines fire after the futures complete),
     * and iterating a live appender list would throw CME.
     */
    private static List<String> captureQueueWaitLogs(Runnable action) {
        ch.qos.logback.classic.Logger logger =
                (ch.qos.logback.classic.Logger)
                        org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            action.run();
        } finally {
            logger.detachAppender(appender);
        }
        List<String> lines = new ArrayList<>();
        // AppenderBase#doAppend is synchronized; holding the same monitor
        // here closes the last race against a flush thread already inside
        // doAppend when the appender was detached.
        synchronized (appender) {
            for (ILoggingEvent event : appender.list) {
                String message = event.getFormattedMessage();
                if (message.contains("flexlb_navi_queue_wait")) {
                    lines.add(message);
                }
            }
        }
        return lines;
    }

    /** Extract one {@code key=value} field from a structured log line. */
    private static String fieldValue(String line, String key) {
        Matcher matcher = Pattern.compile(key + "=([^ ]+)").matcher(line);
        assertTrue(matcher.find(),
                "log line should contain " + key + ": " + line);
        return matcher.group(1);
    }

    private ConcurrentHashMap<String, PrefillEndpoint> endpointMap(String... keys) {
        ConcurrentHashMap<String, PrefillEndpoint> map = new ConcurrentHashMap<>();
        for (String key : keys) {
            String[] parts = key.split(":");
            map.put(key, mockEndpoint(parts[0], Integer.parseInt(parts[1])));
        }
        return map;
    }

    private BalanceContext makeContext(long seqLen) {
        Request request = new Request();
        request.setRequestId(System.nanoTime());
        request.setSeqLen(seqLen);
        BalanceContext ctx = new BalanceContext();
        ctx.setConfig(SchedulingTestConfig.batchConfig());
        ctx.setRequest(request);
        // Non-empty serialized input lets the optional BATCH dispatch path
        // engage; other tests degrade earlier (no admission / no decode).
        ctx.setGenerateInputPb(ByteString.copyFromUtf8("navi-test-input"));
        return ctx;
    }

    // ==================== Tests ====================

    @Test
    @DisplayName("count trigger: submitting maxCount requests flushes immediately")
    void countTriggerFlush() throws Exception {
        ConcurrentHashMap<String, PrefillEndpoint> empty = new ConcurrentHashMap<>();
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(empty);

        // Submit maxCount (4) to trigger immediate flush
        CompletableFuture<Response> future = null;
        for (int i = 0; i < 4; i++) {
            future = scheduler.submit(makeContext(1024));
        }
        Response resp = future.get(5, TimeUnit.SECONDS);
        assertNotNull(resp);
        assertFalse(resp.isSuccess(), "should fail with no worker");
    }

    @Test
    @DisplayName("two endpoints: requests are distributed by optimizer")
    void twoEndpointDistribution() throws Exception {
        ConcurrentHashMap<String, PrefillEndpoint> map =
                endpointMap("10.0.0.1:8080", "10.0.0.2:8080");
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(map);

        CompletableFuture<Response>[] futures = new CompletableFuture[4];
        for (int i = 0; i < 4; i++) {
            futures[i] = scheduler.submit(makeContext(2048));
        }
        for (int i = 0; i < 4; i++) {
            Response resp = futures[i].get(5, TimeUnit.SECONDS);
            assertNotNull(resp);
            assertTrue(resp.isSuccess());
            assertNotNull(resp.getServerStatus());
            assertFalse(resp.getServerStatus().isEmpty());
        }
    }

    @Test
    @DisplayName("shutdown drains buffer with error")
    void shutdownDrainsBuffer() throws Exception {
        ConcurrentHashMap<String, PrefillEndpoint> empty = new ConcurrentHashMap<>();
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(empty);

        // Submit 1 request (won't trigger count flush)
        CompletableFuture<Response> future = scheduler.submit(makeContext(1024));
        // Immediately shutdown
        scheduler.shutdown();
        Response resp = future.get(5, TimeUnit.SECONDS);
        assertNotNull(resp);
        assertFalse(resp.isSuccess(), "shutdown should drain with error");
    }

    @Test
    @DisplayName("null context → immediate error")
    void nullContext() {
        CompletableFuture<Response> future = scheduler.submit(null);
        assertTrue(future.isDone());
        assertFalse(future.join().isSuccess());
    }

    @Test
    @DisplayName("PD separation: response includes both prefill and decode ServerStatus")
    void pdSeparationIncludesDecode() throws Exception {
        // Mock a successful decode selection: ConfiguredLoadBalanceSelector
        // returns a SelectedRole whose serverStatus is a decode ServerStatus.
        ServerStatus decodeStatus = new ServerStatus();
        decodeStatus.setSuccess(true);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setServerIp("10.0.1.1");
        decodeStatus.setHttpPort(9090);
        decodeStatus.setGroup("g1");
        SelectedRole selectedDecode = mock(SelectedRole.class);
        when(selectedDecode.serverStatus()).thenReturn(decodeStatus);
        when(decodeSelector.select(any(BalanceContext.class), eq(RoleType.DECODE), any()))
                .thenReturn(selectedDecode);

        ConcurrentHashMap<String, PrefillEndpoint> map = endpointMap("10.0.0.1:8080");
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(map);

        // Submit 4 requests to trigger immediate flush
        CompletableFuture<Response>[] futures = new CompletableFuture[4];
        for (int i = 0; i < 4; i++) {
            futures[i] = scheduler.submit(makeContext(1024 + i * 100));
        }

        for (int i = 0; i < 4; i++) {
            Response resp = futures[i].get(5, TimeUnit.SECONDS);
            assertNotNull(resp, "response should not be null for request " + i);
            assertTrue(resp.isSuccess(), "request " + i + " should succeed");
            List<ServerStatus> statuses = resp.getServerStatus();
            assertNotNull(statuses, "serverStatus list should not be null for request " + i);
            assertEquals(2, statuses.size(),
                    "PD response should contain exactly 2 ServerStatus entries (prefill + decode) for request " + i);
            // First entry is prefill
            assertEquals(RoleType.PREFILL, statuses.get(0).getRole(),
                    "first ServerStatus should be PREFILL for request " + i);
            assertTrue(statuses.get(0).isSuccess());
            // Second entry is decode
            assertEquals(RoleType.DECODE, statuses.get(1).getRole(),
                    "second ServerStatus should be DECODE for request " + i);
            assertTrue(statuses.get(1).isSuccess());
            assertEquals("10.0.1.1", statuses.get(1).getServerIp());
            assertEquals(9090, statuses.get(1).getHttpPort());
        }
    }

    @Test
    @DisplayName("PD separation: decode unavailable → degrades to prefill-only success (P0-1)")
    void pdSeparationDecodeUnavailable() throws Exception {
        // Decode selection unavailable: the selector throws (e.g. no decode
        // strategy is configured). NaviBatchScheduler must degrade every
        // affected request to a prefill-only route decision.
        when(decodeSelector.select(any(BalanceContext.class), eq(RoleType.DECODE), any()))
                .thenThrow(new IllegalStateException(
                        "No load-balance strategy supports role=DECODE"));

        ConcurrentHashMap<String, PrefillEndpoint> map = endpointMap("10.0.0.1:8080");
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(map);

        CompletableFuture<Response>[] futures = new CompletableFuture[4];
        for (int i = 0; i < 4; i++) {
            futures[i] = scheduler.submit(makeContext(1024));
        }

        for (int i = 0; i < 4; i++) {
            Response resp = futures[i].get(5, TimeUnit.SECONDS);
            assertNotNull(resp);
            // P0-1: decode unavailable no longer rejects; the request
            // degrades to a prefill-only success so engine-side PD absorbs
            // the decode placement (v2 overload-tolerant semantics).
            assertTrue(resp.isSuccess(),
                    "request " + i + " should degrade to prefill-only success when decode is unavailable");
            List<ServerStatus> statuses = resp.getServerStatus();
            assertNotNull(statuses,
                    "degraded response should carry serverStatus for request " + i);
            assertEquals(1, statuses.size(),
                    "degraded response should contain only the prefill ServerStatus for request " + i);
            assertEquals(RoleType.PREFILL, statuses.get(0).getRole(),
                    "degraded response should carry the prefill role for request " + i);
            assertTrue(statuses.get(0).isSuccess(),
                    "prefill ServerStatus should be successful for request " + i);
        }
    }

    @Test
    @DisplayName("BATCH dispatch admission unavailable → degrades to route decisions")
    void batchDispatchAdmissionUnavailableDegrades() throws Exception {
        ConcurrentHashMap<String, PrefillEndpoint> map = endpointMap("10.0.0.1:8080");
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(map);

        // Transport admission is unavailable: tryPrepareSubmission() returns a rejected
        // attempt. Requests must still complete as route decisions.
        when(batchSubmissionPort.tryPrepareSubmission()).thenAnswer(invocation ->
                new org.flexlb.balance.delivery.CapacityBoundary.Attempt.Rejected<>(
                        new org.flexlb.balance.delivery.CapacityBoundary.Failed(
                                new IllegalStateException("dispatcher is shut down"))));

        CompletableFuture<Response>[] futures = new CompletableFuture[4];
        for (int i = 0; i < 4; i++) {
            futures[i] = scheduler.submit(makeContext(1024));
        }

        for (int i = 0; i < 4; i++) {
            Response resp = futures[i].get(5, TimeUnit.SECONDS);
            assertNotNull(resp);
            assertTrue(resp.isSuccess(),
                    "admission rejection must degrade to a route decision, not an error");
        }
    }

    @Test
    @DisplayName("BATCH dispatch delivered completion → enqueued_by_master response")
    void batchDispatchDeliveredMarksEnqueued() throws Exception {
        ConcurrentHashMap<String, PrefillEndpoint> map = endpointMap("10.0.0.1:8080");
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(map);

        // Batch dispatch needs a successful decode selection to build items.
        ServerStatus decodeStatus = new ServerStatus();
        decodeStatus.setSuccess(true);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setServerIp("10.0.1.1");
        decodeStatus.setHttpPort(9090);
        decodeStatus.setGroup("g1");
        SelectedRole selectedDecode = mock(SelectedRole.class);
        when(selectedDecode.serverStatus()).thenReturn(decodeStatus);
        when(decodeSelector.select(any(BalanceContext.class), eq(RoleType.DECODE), any()))
                .thenReturn(selectedDecode);

        // Successful prepare + immediate Delivered completion for every item.
        org.mockito.stubbing.Answer<Void> submitAnswer = invocation -> {
            BatchSubmissionPort.Command command = invocation.getArgument(0);
            java.util.function.BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> observer =
                    invocation.getArgument(1);
            for (ScheduledRequest item : command.exactItems()) {
                observer.accept(item,
                        SlotDeliveryPort.Completion.Delivered.INSTANCE);
            }
            return null;
        };
        BatchSubmissionPort.PreparedSubmission prepared =
                mock(BatchSubmissionPort.PreparedSubmission.class);
        org.mockito.Mockito.doAnswer(submitAnswer)
                .when(prepared).submitBatch(any(), any());
        when(batchSubmissionPort.tryPrepareSubmission()).thenReturn(
                new org.flexlb.balance.delivery.CapacityBoundary.Attempt.Accepted<>(prepared));

        CompletableFuture<Response>[] futures = new CompletableFuture[4];
        for (int i = 0; i < 4; i++) {
            futures[i] = scheduler.submit(makeContext(1024));
        }

        for (int i = 0; i < 4; i++) {
            Response resp = futures[i].get(5, TimeUnit.SECONDS);
            assertNotNull(resp);
            assertTrue(resp.isSuccess());
            assertTrue(resp.isEnqueuedByMaster(),
                    "delivered batch items must answer enqueued_by_master=true");
        }
    }

    // ==================== Queue-wait (engine observation) tests ====================

    @Test
    @DisplayName("queue wait units: waiting count × avg window tokens model-converted to ms (O(1))")
    void queueWaitModelConversion() throws Exception {
        // Helper level: waitingQueryLen=10 priced at one 2048-token average
        // window request through the single-batch drain model.
        WorkerStatus.EngineObservation observation = observationWithWaiting(10L);
        long modelMs = NaviBatchScheduler.engineQueueWaitEstimateMs(
                observation, NAVI_PARAMS.clone(), 2048L);

        double perRequestCost = NaviPrefillModel.calculateRequestLinearCost(
                NAVI_PARAMS, 2048L, 0L);
        double expectedMs = NaviPrefillModel.calculateLatencyAndDerivative(
                10L * perRequestCost, NAVI_PARAMS)[0];

        assertEquals(Math.round(expectedMs), modelMs,
                "queue wait must equal the single-batch drain model value");
        // Unit red line: the value is neither the raw waiting count nor the
        // queued token mass pressed into service as milliseconds.
        assertNotEquals(10L, modelMs);
        assertNotEquals(10L * 2048L, modelMs);
        // Boundary: no reported waiting count reads as zero wait.
        assertEquals(0L, NaviBatchScheduler.engineQueueWaitEstimateMs(
                observationWithWaiting(0L), NAVI_PARAMS.clone(), 2048L));

        // Scheduler level: a window of known seqLen requests fixes the
        // avg-token basis, and the formal observation log must carry the
        // O(1) conversion's result for the queued node.
        ConcurrentHashMap<String, PrefillEndpoint> map = new ConcurrentHashMap<>();
        map.put("10.0.0.1:8080", mockEndpoint(
                "10.0.0.1", 8080, observationWithWaiting(10L)));
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(map);

        List<String> logLines = captureQueueWaitLogs(() -> {
            CompletableFuture<Response>[] futures = new CompletableFuture[4];
            long[] seqLens = {1024L, 2048L, 3072L, 4096L};
            for (int i = 0; i < 4; i++) {
                futures[i] = scheduler.submit(makeContext(seqLens[i]));
            }
            for (int i = 0; i < 4; i++) {
                Response resp = futures[i].join();
                assertNotNull(resp);
                assertTrue(resp.isSuccess());
            }
        });
        assertEquals(1, logLines.size(),
                "exactly one queue-wait observation line per window");
        String line = logLines.get(0);
        // Window average: (1024+2048+3072+4096)/4 = 2560.
        assertEquals("2560", fieldValue(line, "avg_tokens"));
        assertEquals("10", fieldValue(line, "waiting"));
        double windowPerRequestCost = NaviPrefillModel.calculateRequestLinearCost(
                NAVI_PARAMS, 2560L, 0L);
        long expectedWindowMs = Math.round(
                NaviPrefillModel.calculateLatencyAndDerivative(
                        10L * windowPerRequestCost, NAVI_PARAMS)[0]);
        assertEquals(String.valueOf(expectedWindowMs),
                fieldValue(line, "engine_ms"),
                "engine_ms in the observation log must match the hand-computed "
                        + "single-batch drain estimate");
        assertEquals(fieldValue(line, "engine_ms"),
                fieldValue(line, "queue_wait_ms"),
                "ledger is empty here, so the merged wait equals engine_ms");
    }

    @Test
    @DisplayName("queue-aware placement: engine-reported queue steers requests to the clean node")
    void queueAwarePlacementPrefersCleanNode() throws Exception {
        // One endpoint reports a large waiting count (64 × ~2048-token
        // average requests), the other is clean; both share the same learned
        // weights, so the engine-reported queue is the only signal
        // separating them.
        ConcurrentHashMap<String, PrefillEndpoint> map = new ConcurrentHashMap<>();
        map.put("10.0.0.1:8080", mockEndpoint(
                "10.0.0.1", 8080, observationWithWaiting(64L)));
        map.put("10.0.0.2:8080", mockEndpoint("10.0.0.2", 8080, emptyObservation()));
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(map);

        // Three windows of maxCount(4) same-distribution requests. Per-window
        // PGD weight seeding is random, so aggregate across windows and assert
        // a robust majority instead of an exact split.
        int totalRequests = 12;
        CompletableFuture<Response>[] futures = new CompletableFuture[totalRequests];
        for (int i = 0; i < totalRequests; i++) {
            futures[i] = scheduler.submit(makeContext(2048));
        }
        int onClean = 0;
        for (int i = 0; i < totalRequests; i++) {
            Response resp = futures[i].get(5, TimeUnit.SECONDS);
            assertNotNull(resp);
            assertTrue(resp.isSuccess());
            assertNotNull(resp.getServerStatus());
            if ("10.0.0.2".equals(resp.getServerStatus().get(0).getServerIp())) {
                onClean++;
            }
        }
        assertTrue(onClean * 2 > totalRequests,
                "a majority of requests should land on the clean endpoint, got "
                        + onClean + "/" + totalRequests);
    }

    @Test
    @DisplayName("null engine observation: no NPE, wait falls back to ledger/zero")
    void nullEngineObservationIsSafe() throws Exception {
        // Helper level: a null observation converts to a zero estimate.
        assertEquals(0L, NaviBatchScheduler.engineQueueWaitEstimateMs(
                null, NAVI_PARAMS.clone(), 2048L));

        // Scheduler level: committedEngineObservation() returning null must
        // not throw — an NPE inside doOptimizeAndDispatch would surface as
        // NO_AVAILABLE_WORKER failures for the whole window.
        ConcurrentHashMap<String, PrefillEndpoint> map = new ConcurrentHashMap<>();
        map.put("10.0.0.1:8080", mockEndpoint("10.0.0.1", 8080, null));
        when(endpointRegistry.snapshotPrefillEndpoints()).thenReturn(map);

        CompletableFuture<Response>[] futures = new CompletableFuture[4];
        for (int i = 0; i < 4; i++) {
            futures[i] = scheduler.submit(makeContext(1024));
        }
        for (int i = 0; i < 4; i++) {
            Response resp = futures[i].get(5, TimeUnit.SECONDS);
            assertNotNull(resp);
            assertTrue(resp.isSuccess(),
                    "null engine observation must not break placement ("
                            + "waitMs falls back to the ledger signal)");
        }
    }
}
