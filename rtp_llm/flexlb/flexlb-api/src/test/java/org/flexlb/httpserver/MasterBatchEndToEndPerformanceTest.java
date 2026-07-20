package org.flexlb.httpserver;

import ch.qos.logback.classic.Level;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.ByteString;
import com.google.protobuf.StringValue;
import io.grpc.ManagedChannel;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockPrefillWorker;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.LoggerFactory;
import org.springframework.mock.env.MockEnvironment;

import java.io.BufferedReader;
import java.io.IOException;
import java.math.BigInteger;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.LockSupport;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/**
 * Loopback end-to-end regression for the Master batch scheduling data path.
 *
 * <p>The exercised path is:
 * client stub -> Netty Master gRPC server -> FlexlbServiceImpl -> RouteService
 * -> FlexlbBatchScheduler -> WorkerBatcher -> EngineGrpcClient -> Netty mock engine.
 * The worker capacities are fixed by the fixture, while worker selection uses the
 * production DefaultRouter and cost-based prefill/decode strategies. The 750-engine
 * selection topology has its own focused performance regression test.
 */
@Tag("performance-regression")
class MasterBatchEndToEndPerformanceTest extends FlexLBMockTestBase {

    private static final int WARMUP_REQUESTS = 64;
    private static final int REAL_REQUEST_TEMPLATE_COUNT = 128;
    private static final long TOKEN_ID_REMAP_SEED =
            Long.getLong("flexlb.perf.e2e.token-id-remap-seed", 0x5EED_F1E5L);
    private static final int REQUEST_COUNT =
            Integer.getInteger("flexlb.perf.e2e.requests", 8_192);
    private static final long MEASUREMENT_REQUEST_ID_BASE = 1_000_000L;
    private static final int ENGINE_MATRIX_DURATION_MS =
            Integer.getInteger("flexlb.perf.engine-matrix-duration-ms", 500);
    private static final int ENGINE_MATRIX_MIN_REQUESTS =
            Integer.getInteger("flexlb.perf.engine-matrix-min-requests", 1_024);
    private static final int[] STANDARD_ENGINE_MATRIX_TARGET_QPS =
            {1_000, 2_000, 5_000, 10_000};

    private static List<RealRequestTemplate> realRequestTemplates;

    private FlexlbGrpcServer masterServer;
    private ManagedChannel masterChannel;
    private FlexlbServiceGrpc.FlexlbServiceStub masterStub;
    private ServerScheduleLatencyRecorder latencyRecorder;
    private ActiveRequestCounter activeRequestCounter;
    private ch.qos.logback.classic.Logger flexlbLogger;
    private ch.qos.logback.classic.Logger mockRpcLogger;
    private ch.qos.logback.classic.Logger nettyLogger;
    private ch.qos.logback.classic.Logger grpcLogger;
    private Level previousFlexlbLogLevel;
    private Level previousMockRpcLogLevel;
    private Level previousNettyLogLevel;
    private Level previousGrpcLogLevel;
    private final Map<String, LongAdder> dispatchReasonCounts = new ConcurrentHashMap<>();

    @BeforeAll
    static void loadLogDerivedRequests() throws IOException {
        Path onlineLogs = findOnlineLogsDirectory();
        ObjectMapper mapper = new ObjectMapper();
        JsonNode accessLog = mapper.readTree(onlineLogs.resolve("sample_access.json").toFile());
        assertTrue(accessLog.path("sanitized").asBoolean(),
                "sample access fixture must be sanitized before it is committed");
        int[] loggedTokenCorpus = readTokenCorpus(accessLog.path("input_ids"));
        int[] obfuscatedTokenCorpus = obfuscatedCopy(loggedTokenCorpus, TOKEN_ID_REMAP_SEED);
        String model = accessLog.path("request_controls")
                .path("ds_header_attributes").path("model").asText("mock-model");
        JsonNode loggedGenerateConfig = accessLog.path("generate_config");
        List<TraceShape> shapes = readTraceShapes(
                mapper, onlineLogs.resolve("trace_30min.jsonl"));
        realRequestTemplates = buildRequestTemplates(
                shapes, obfuscatedTokenCorpus, model, loggedGenerateConfig,
                accessLog.path("output_token_len").asInt(1));

        assertEquals(REAL_REQUEST_TEMPLATE_COUNT, realRequestTemplates.size());
        assertTrue(realRequestTemplates.stream()
                .mapToInt(RealRequestTemplate::seqLen).distinct().count() >= 32,
                "log-derived requests must retain a varied input-length distribution");
        assertTrue(Arrays.stream(obfuscatedTokenCorpus).distinct().limit(100).count() == 100,
                "real input token corpus unexpectedly collapsed to synthetic IDs");
        assertTrue(tokenIdsDifferAtEveryPosition(loggedTokenCorpus, obfuscatedTokenCorpus),
                "every logged token ID must be obfuscated before replay");
        assertTrue(tokenIdSetsAreDisjoint(loggedTokenCorpus, obfuscatedTokenCorpus),
                "obfuscated requests must not contain any logged token ID value");
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.setFlexlbBatchAlgorithm("fixed_window");
        cfg.setFlexlbBatchFixedWaitMs(10L);
        cfg.setFlexlbBatchPredictThresholdMs(0L);
        cfg.setFlexlbBatchSizeMax(16);
        cfg.setFlexlbBatchQueueMaxSize(4_096);
        cfg.setFlexlbBatchMaxInflight(20_000);
        cfg.setFlexlbBatchDispatchPoolSize(32);
        cfg.setFlexlbBatchDispatchQueueSize(2_048);
        cfg.setFlexlbBatchFixedMaxInflightBatches(0);
        cfg.setPrefillQueueSizeThreshold(1_000_000L);
        return cfg;
    }

    @Override
    protected EngineWorkerStatus createEngineWorkerStatus() {
        return new EngineWorkerStatus(endpointRegistry);
    }

    @Override
    protected Router createRouter() {
        ResourceMeasureFactory resourceMeasureFactory = new ResourceMeasureFactory(List.of(
                new PrefillResourceMeasure(configService),
                new DecodeResourceMeasure(configService)));
        new CostBasedPrefillStrategy(
                engineWorkerStatus,
                new EmptyCacheAwareService(),
                resourceMeasureFactory,
                mock(EngineHealthReporter.class, withSettings().stubOnly()));
        new CostBasedDecodeStrategy(configService, engineWorkerStatus, resourceMeasureFactory);
        new RandomStrategy(engineWorkerStatus, configService, resourceMeasureFactory);
        return new DefaultRouter(
                configService,
                ignored -> GroupRoutingDecision.none(),
                endpointRegistry);
    }

    @BeforeEach
    void startMasterGrpcServer() throws Exception {
        suppressRequestPathLogs();
        doAnswer(invocation -> {
            String reason = invocation.getArgument(3);
            dispatchReasonCounts.computeIfAbsent(reason, ignored -> new LongAdder()).increment();
            return null;
        }).when(reporter).reportDispatchReason(
                anyString(), anyString(), anyString(), anyString());
        getDecodeEndpoint().getStatus().getAvailableKvCacheTokens().set(1_000_000_000L);
        getDecodeEndpoint().getStatus().getTotalKvCacheTokens().set(2_000_000_000L);
        getDecodeEndpoint().onWorkerStatusUpdate(
                getDecodeEndpoint().getStatus(), new WorkerStatusResponse());

        RouteService routeService = new RouteService(
                configService,
                (DefaultRouter) router,
                mock(QueueManager.class, withSettings().stubOnly()),
                scheduler,
                mock(RecentCacheKeyTraceReporter.class, withSettings().stubOnly()),
                endpointRegistry);

        LBStatusConsistencyService consistencyService =
                mock(LBStatusConsistencyService.class, withSettings().stubOnly());
        when(consistencyService.isNeedConsistency()).thenReturn(false);
        activeRequestCounter = new ActiveRequestCounter();
        latencyRecorder = new ServerScheduleLatencyRecorder();
        FlexlbServiceImpl service = new FlexlbServiceImpl(
                routeService,
                consistencyService,
                mock(EngineHealthReporter.class, withSettings().stubOnly()),
                activeRequestCounter,
                mock(FlexlbGrpcForwarder.class, withSettings().stubOnly()),
                configService,
                reporter,
                latencyRecorder);

        int grpcPort;
        try (ServerSocket socket = new ServerSocket(0)) {
            grpcPort = socket.getLocalPort();
        }
        // Set executor sizes directly on the config object (FlexlbGrpcServer reads
        // from FlexlbConfig, not from Environment properties).
        config.setFlexlbGrpcExecutorCoreSize(16);
        config.setFlexlbGrpcExecutorMaxSize(32);
        config.setFlexlbGrpcExecutorQueueSize(4096);

        MockEnvironment environment = new MockEnvironment()
                .withProperty("server.port", Integer.toString(grpcPort - 2));
        masterServer = new FlexlbGrpcServer(
                service,
                configService,
                environment,
                new NioEventLoopGroup(4),
                null,
                new GrpcServerTimingInterceptor());
        masterServer.start();

        masterChannel = NettyChannelBuilder
                .forAddress("127.0.0.1", grpcPort)
                .usePlaintext()
                .build();
        masterStub = FlexlbServiceGrpc.newStub(masterChannel)
                .withDeadlineAfter(20, TimeUnit.SECONDS);
    }

    @AfterEach
    void stopMasterGrpcServer() throws InterruptedException {
        if (masterChannel != null) {
            masterChannel.shutdownNow();
            masterChannel.awaitTermination(5, TimeUnit.SECONDS);
        }
        if (masterServer != null) {
            masterServer.shutdown();
        }
        if (dispatcher != null) {
            dispatcher.shutdown();
        }
        restoreRequestPathLogs();
    }

    @Test
    @Timeout(value = 45, unit = TimeUnit.SECONDS)
    void batchScheduleRemainsFastAcrossRealGrpcBoundaries() throws Exception {
        TrafficResult warmup = runTraffic(WARMUP_REQUESTS, 1L);
        assertSuccessful(warmup);
        awaitCompletionCount(WARMUP_REQUESTS);
        latencyRecorder.reset();
        mockPrefillWorker.resetRecords();

        TrafficResult result = runTraffic(REQUEST_COUNT, MEASUREMENT_REQUEST_ID_BASE);
        assertSuccessful(result);
        Map<String, Object> masterSnapshot = awaitCompletionCount(REQUEST_COUNT);
        BatchSummary batches = summarizeEngineBatches(
                MEASUREMENT_REQUEST_ID_BASE, allPrefillWorkers());

        double masterQps = number(masterSnapshot, "completion_qps").doubleValue();
        Map<String, Object> serverLatency = nestedMap(masterSnapshot, "server_total_ms");
        long serverP50Ms = number(serverLatency, "p50").longValue();
        long serverP90Ms = number(serverLatency, "p90").longValue();
        long serverP95Ms = number(serverLatency, "p95").longValue();
        long serverP99Ms = number(serverLatency, "p99").longValue();
        double serverMeanMs = number(serverLatency, "mean").doubleValue();

        System.out.printf(
                "FlexLB Master batch E2E: requests=%d client_qps=%.1f master_qps=%.1f "
                        + "client_p50=%.3fms client_p99=%.3fms master_p50=%dms master_p90=%dms "
                        + "master_p95=%dms master_p99=%dms master_avg=%.3fms "
                        + "engine_batches=%d avg_batch=%.2f max_batch=%d avg_input_tokens=%.1f%n",
                REQUEST_COUNT, result.qps(), masterQps, result.p50Ms(), result.p99Ms(),
                serverP50Ms, serverP90Ms, serverP95Ms, serverP99Ms, serverMeanMs,
                batches.batchCount(), batches.averageBatchSize(), batches.maxBatchSize(),
                batches.averageInputTokens());

        assertEquals(REQUEST_COUNT, number(masterSnapshot, "arrival_count").longValue());
        assertEquals(REQUEST_COUNT, number(masterSnapshot, "completion_count").longValue());
        assertEquals(REQUEST_COUNT, batches.requestIds().size(),
                "mock engine must receive every measured request exactly once");
        assertEquals(expectedRequestIds(MEASUREMENT_REQUEST_ID_BASE, REQUEST_COUNT),
                batches.requestIds());
        assertTrue(batches.batchCount() < REQUEST_COUNT,
                "fixed-window mode must coalesce requests before engine enqueue");
        assertTrue(batches.maxBatchSize() > 1,
                "at least one EnqueueBatch call must contain multiple tasks");
        assertTrue(batches.distinctInputLengths() >= 32,
                "engine traffic must retain the log-derived input-length distribution");
        assertEquals(0L, activeRequestCounter.getCount(),
                "all Master gRPC requests must release their active-request token");

        int processors = Runtime.getRuntime().availableProcessors();
        long defaultMinimumQps = Math.min(5_000L, Math.max(500L, processors * 250L));
        long minimumQps = Long.getLong("flexlb.perf.min-e2e-qps", defaultMinimumQps);
        long maximumServerP99Ms = Long.getLong("flexlb.perf.max-e2e-server-p99-ms", 250L);
        assertTrue(result.qps() >= minimumQps,
                () -> String.format("client E2E throughput %.1f QPS is below floor %d QPS",
                        result.qps(), minimumQps));
        assertTrue(masterQps >= minimumQps,
                () -> String.format("Master completion throughput %.1f QPS is below floor %d QPS",
                        masterQps, minimumQps));
        assertTrue(serverP99Ms <= maximumServerP99Ms,
                () -> String.format("Master server P99 %d ms exceeds ceiling %d ms",
                        serverP99Ms, maximumServerP99Ms));
    }

    @ParameterizedTest(name = "prefill={0}, decode={1}")
    @MethodSource("engineScales")
    @Timeout(value = 45, unit = TimeUnit.SECONDS)
    void masterMeetsRateSloAcrossEngineScaleMatrix(int prefillEngineCount,
                                                   int decodeEngineCount,
                                                   int[] targetQpsValues) throws Exception {
        while (allPrefillWorkers().size() < prefillEngineCount) {
            addPrefillWorker(MockWorkerBehavior.builder().build());
        }
        while (endpointRegistry.getEndpointCount(RoleType.DECODE) < decodeEngineCount) {
            addLogicalDecodeEndpoint(endpointRegistry.getEndpointCount(RoleType.DECODE));
        }
        assertEquals(prefillEngineCount, endpointRegistry.getEndpointCount(RoleType.PREFILL));
        assertEquals(decodeEngineCount, endpointRegistry.getEndpointCount(RoleType.DECODE));

        int warmupRequests = Math.max(WARMUP_REQUESTS, prefillEngineCount * 16);
        TrafficResult warmup = runTraffic(
                warmupRequests, 50_000_000L + prefillEngineCount * 10_000L);
        assertSuccessful(warmup);
        awaitCompletionCount(warmupRequests);
        resetMeasurementState();

        double minimumQpsRatio = Double.parseDouble(
                System.getProperty("flexlb.perf.engine-matrix-min-qps-ratio", "0.85"));
        long maximumServerP99Ms = Long.getLong(
                "flexlb.perf.engine-matrix-max-server-p99-ms", 250L);
        long maximumBatchWaitP99Ms = Long.getLong(
                "flexlb.perf.engine-matrix-max-batch-wait-p99-ms", 50L);

        for (int qpsIndex = 0; qpsIndex < targetQpsValues.length; qpsIndex++) {
            int targetQps = targetQpsValues[qpsIndex];
            int requestCount = Math.max(ENGINE_MATRIX_MIN_REQUESTS,
                    targetQps * ENGINE_MATRIX_DURATION_MS / 1_000);
            long firstRequestId = 10_000_000L
                    + prefillEngineCount * 1_000_000L
                    + qpsIndex * 100_000L;

            resetMeasurementState();
            TrafficResult result = runTraffic(requestCount, firstRequestId, targetQps);
            assertSuccessful(result);
            Map<String, Object> masterSnapshot = awaitCompletionCount(requestCount);
            BatchSummary batches = summarizeEngineBatches(firstRequestId, allPrefillWorkers());
            double masterQps = number(masterSnapshot, "completion_qps").doubleValue();
            Map<String, Object> serverLatency = nestedMap(masterSnapshot, "server_total_ms");
            long serverP50Ms = number(serverLatency, "p50").longValue();
            long serverP90Ms = number(serverLatency, "p90").longValue();
            long serverP95Ms = number(serverLatency, "p95").longValue();
            long serverP99Ms = number(serverLatency, "p99").longValue();
            double serverMeanMs = number(serverLatency, "mean").doubleValue();
            Map<String, Object> batchWait = nestedMap(masterSnapshot, "batch_wait_ms");
            long batchWaitP50Ms = number(batchWait, "p50").longValue();
            long batchWaitP90Ms = number(batchWait, "p90").longValue();
            long batchWaitP95Ms = number(batchWait, "p95").longValue();
            long batchWaitP99Ms = number(batchWait, "p99").longValue();
            double batchWaitMeanMs = number(batchWait, "mean").doubleValue();
            Map<String, Object> dispatchAck = nestedMap(masterSnapshot, "dispatch_ack_ms");
            long dispatchAckP99Ms = number(dispatchAck, "p99").longValue();
            double dispatchAckMeanMs = number(dispatchAck, "mean").doubleValue();
            long batchFullCount = dispatchReasonCount("batch_full");
            long windowTimeoutCount = dispatchReasonCount("fixed_window_timeout");
            long predictThresholdCount = dispatchReasonCount("predict_threshold");
            int activePrefillRoutes = activeScheduledEngineCount(result, RoleType.PREFILL);
            int activeDecodeRoutes = activeScheduledEngineCount(result, RoleType.DECODE);

            System.out.printf(
                    "FlexLB Master engine-scale E2E: prefill=%d decode=%d target_qps=%d "
                            + "requests=%d client_qps=%.1f master_qps=%.1f "
                            + "master_p50=%dms master_p90=%dms master_p95=%dms "
                            + "master_p99=%dms master_avg=%.3fms active_prefill_rpc=%d "
                            + "active_prefill_route=%d active_decode_route=%d "
                            + "batch_wait_p50=%dms batch_wait_p90=%dms "
                            + "batch_wait_p95=%dms batch_wait_p99=%dms "
                            + "batch_wait_avg=%.3fms dispatch_ack_p99=%dms "
                            + "dispatch_ack_avg=%.3fms engine_batches=%d "
                            + "batch_full=%d window_timeout=%d predict_threshold=%d "
                            + "avg_batch=%.2f max_batch=%d%n",
                    prefillEngineCount, decodeEngineCount, targetQps, requestCount,
                    result.qps(), masterQps, serverP50Ms, serverP90Ms, serverP95Ms,
                    serverP99Ms, serverMeanMs, batches.activeWorkerCount(),
                    activePrefillRoutes, activeDecodeRoutes,
                    batchWaitP50Ms, batchWaitP90Ms, batchWaitP95Ms, batchWaitP99Ms,
                    batchWaitMeanMs, dispatchAckP99Ms, dispatchAckMeanMs,
                    batches.batchCount(), batchFullCount, windowTimeoutCount,
                    predictThresholdCount, batches.averageBatchSize(),
                    batches.maxBatchSize());

            assertEquals(requestCount, number(masterSnapshot, "arrival_count").longValue());
            assertEquals(requestCount, number(masterSnapshot, "completion_count").longValue());
            assertEquals(requestCount, number(batchWait, "count").longValue(),
                    "every request must record Master batch queue wait");
            assertEquals(requestCount, number(dispatchAck, "count").longValue(),
                    "every request must record engine dispatch ACK latency");
            assertEquals(expectedRequestIds(firstRequestId, requestCount), batches.requestIds());
            assertEquals(prefillEngineCount, batches.activeWorkerCount(),
                    "every prefill engine must receive measured traffic");
            assertEquals(prefillEngineCount, activePrefillRoutes,
                    "every prefill engine must appear in measured routing decisions");
            assertEquals(decodeEngineCount, activeDecodeRoutes,
                    "every decode engine must appear in measured routing decisions");
            assertEquals(batches.batchCount(),
                    batchFullCount + windowTimeoutCount + predictThresholdCount,
                    "every engine batch must have one recorded dispatch reason");
            assertEquals(0L, activeRequestCounter.getCount());
            assertTrue(result.qps() >= targetQps * minimumQpsRatio,
                    () -> String.format(
                            "client throughput %.1f QPS missed %.0f%% of target %d QPS",
                            result.qps(), minimumQpsRatio * 100.0, targetQps));
            assertTrue(masterQps >= targetQps * minimumQpsRatio,
                    () -> String.format(
                            "Master throughput %.1f QPS missed %.0f%% of target %d QPS",
                            masterQps, minimumQpsRatio * 100.0, targetQps));
            assertTrue(serverP99Ms <= maximumServerP99Ms,
                    () -> String.format("Master server P99 %d ms exceeds ceiling %d ms",
                            serverP99Ms, maximumServerP99Ms));
            assertTrue(batchWaitP99Ms <= maximumBatchWaitP99Ms,
                    () -> String.format("Master batch wait P99 %d ms exceeds ceiling %d ms",
                            batchWaitP99Ms, maximumBatchWaitP99Ms));
            if (targetQps == 2_000) {
                assertTrue(batchWaitP95Ms > 0,
                        "2k QPS queueing scenario must observe non-zero batch wait");
                double arrivalsPerWindowPerPrefill = targetQps
                        * config.getFlexlbBatchFixedWaitMs() / 1_000.0 / prefillEngineCount;
                double expectedBatchSize = Math.min(
                        config.getFlexlbBatchSizeMax(), arrivalsPerWindowPerPrefill);
                double minimumAverageBatchSize = Math.max(1.0, expectedBatchSize * 0.8);
                assertTrue(batches.averageBatchSize() >= minimumAverageBatchSize,
                        () -> String.format(
                                "2k QPS average batch %.2f is below queueing floor %.2f "
                                        + "for %d prefill engines",
                                batches.averageBatchSize(), minimumAverageBatchSize,
                                prefillEngineCount));
                if (prefillEngineCount == 1) {
                    assertTrue(batchFullCount > windowTimeoutCount,
                            "one prefill at 2k QPS should primarily dispatch full batches");
                } else if (prefillEngineCount == 16) {
                    assertTrue(windowTimeoutCount > batchFullCount,
                            "sixteen prefills at 2k QPS should primarily dispatch on timeout");
                }
            }
        }
    }

    private static Stream<Arguments> engineScales() {
        return Stream.of(
                Arguments.of(1, 2, STANDARD_ENGINE_MATRIX_TARGET_QPS),
                Arguments.of(2, 4, new int[]{2_000}),
                Arguments.of(4, 8, STANDARD_ENGINE_MATRIX_TARGET_QPS),
                Arguments.of(8, 16, new int[]{2_000}),
                Arguments.of(16, 32, STANDARD_ENGINE_MATRIX_TARGET_QPS));
    }

    private TrafficResult runTraffic(int requestCount, long firstRequestId) throws Exception {
        return runTraffic(requestCount, firstRequestId, 0);
    }

    private TrafficResult runTraffic(int requestCount, long firstRequestId,
                                     int targetQps) throws Exception {
        List<CompletableFuture<TimedResponse>> futures = new ArrayList<>(requestCount);
        long trafficStartNanos = System.nanoTime();
        for (int index = 0; index < requestCount; index++) {
            if (targetQps > 0) {
                paceUntil(trafficStartNanos
                        + (long) index * TimeUnit.SECONDS.toNanos(1) / targetQps);
            }
            long requestId = firstRequestId + index;
            long requestStartNanos = System.nanoTime();
            CompletableFuture<TimedResponse> future = new CompletableFuture<>();
            masterStub.schedule(scheduleRequest(requestId, index), new StreamObserver<>() {
                @Override
                public void onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
                    future.complete(new TimedResponse(
                            response, System.nanoTime() - requestStartNanos));
                }

                @Override
                public void onError(Throwable throwable) {
                    future.completeExceptionally(throwable);
                }

                @Override
                public void onCompleted() {
                    // Unary response is completed in onNext.
                }
            });
            futures.add(future);
        }

        CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new))
                .get(30, TimeUnit.SECONDS);
        long elapsedNanos = System.nanoTime() - trafficStartNanos;
        long[] latencies = new long[requestCount];
        List<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responses =
                new ArrayList<>(requestCount);
        for (int index = 0; index < requestCount; index++) {
            TimedResponse response = futures.get(index).join();
            latencies[index] = response.latencyNanos();
            responses.add(response.response());
        }
        Arrays.sort(latencies);
        return new TrafficResult(
                requestCount * 1_000_000_000.0 / elapsedNanos,
                percentileNanos(latencies, 0.50) / 1_000_000.0,
                percentileNanos(latencies, 0.99) / 1_000_000.0,
                responses);
    }

    private static void paceUntil(long targetNanos) {
        long remainingNanos;
        while ((remainingNanos = targetNanos - System.nanoTime()) > 0L) {
            LockSupport.parkNanos(remainingNanos);
        }
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB scheduleRequest(long requestId,
                                                                                  int requestIndex) {
        RealRequestTemplate template = realRequestTemplates.get(
                Math.floorMod(requestIndex, realRequestTemplates.size()));
        EngineRpcService.GenerateInputPB generateInput = template.generateInput().toBuilder()
                .setRequestId(requestId)
                .setStartTime(System.currentTimeMillis())
                .setRequestInfo(template.generateInput().getRequestInfo().toBuilder()
                        .setRequestId(Long.toString(requestId))
                        .build())
                .build();
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateInput(ByteString.copyFrom(generateInput.toByteArray()))
                .addAllBlockCacheKeys(template.blockCacheKeys())
                .setSeqLen(template.seqLen())
                .setRequestTimeMs(System.currentTimeMillis())
                .setMaxNewTokens(template.maxNewTokens())
                .setNumBeams(1)
                .setModel(template.model())
                .setCacheKeyBlockSize(1_024L)
                .setScheduleMode(FlexlbScheduleProtocol.FlexlbScheduleModePB.FLEXLB_SCHEDULE_BATCH)
                .build();
    }

    private static void assertSuccessful(TrafficResult result) {
        for (FlexlbScheduleProtocol.FlexlbScheduleResponsePB response : result.responses()) {
            assertTrue(response.getSuccess(),
                    () -> "schedule failed: code=" + response.getCode()
                            + ", error=" + response.getErrorMessage());
            assertEquals(
                    FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_ACKNOWLEDGED,
                    response.getLifecycle().getState());
            assertEquals(2, response.getServerStatusCount());
        }
    }

    private static int activeScheduledEngineCount(TrafficResult result, RoleType roleType) {
        Set<String> endpoints = new HashSet<>();
        for (FlexlbScheduleProtocol.FlexlbScheduleResponsePB response : result.responses()) {
            for (FlexlbScheduleProtocol.FlexlbServerStatusPB status
                    : response.getServerStatusList()) {
                if (roleType.getCode().equals(status.getRole())) {
                    endpoints.add(status.getServerIp() + ":" + status.getHttpPort());
                }
            }
        }
        return endpoints.size();
    }

    private Map<String, Object> awaitCompletionCount(long expected) throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        Map<String, Object> snapshot;
        do {
            snapshot = latencyRecorder.snapshot();
            if (number(snapshot, "completion_count").longValue() >= expected) {
                return snapshot;
            }
            TimeUnit.MILLISECONDS.sleep(5);
        } while (System.nanoTime() < deadlineNanos);
        return latencyRecorder.snapshot();
    }

    private void resetMeasurementState() {
        latencyRecorder.reset();
        dispatchReasonCounts.clear();
        for (MockPrefillWorker worker : allPrefillWorkers()) {
            worker.resetRecords();
        }
        // Reset decode endpoint inflight KV state, simulating production's periodic
        // status sync. In the mock test there are no engine status reports, so
        // inflight KV reservations accumulate permanently across QPS levels. This
        // causes the weighted random selection (exp(-decayFactor * kvDelta)) to
        // degenerate to greedy minimum-load selection, starving some endpoints.
        // A status update resets reportedKvAvailable and confirmedRunningCount;
        // evictExpiredRequests(0) clears all inflight requests and their KV reservations.
        for (DecodeEndpoint ep : endpointRegistry.getDecodeEndpoints().values()) {
            WorkerStatusResponse response = new WorkerStatusResponse();
            response.setRunningTaskInfo(Map.of());
            response.setFinishedTaskInfo(Map.of());
            ep.onWorkerStatusUpdate(ep.getStatus(), response);
            ep.evictExpiredRequests(0);
        }
    }

    private long dispatchReasonCount(String reason) {
        LongAdder count = dispatchReasonCounts.get(reason);
        return count == null ? 0L : count.sum();
    }

    private BatchSummary summarizeEngineBatches(long firstRequestId,
                                                List<MockPrefillWorker> workers) {
        Set<Long> requestIds = new HashSet<>();
        Set<Integer> inputLengths = new HashSet<>();
        int maxBatchSize = 0;
        int totalRequests = 0;
        long totalInputTokens = 0;
        int batchCount = 0;
        int activeWorkerCount = 0;
        for (MockPrefillWorker worker : workers) {
            List<EngineRpcService.EnqueueBatchRequestPB> workerBatches =
                    worker.getRpcService().getEnqueuedRequests();
            if (!workerBatches.isEmpty()) {
                activeWorkerCount++;
            }
            batchCount += workerBatches.size();
            for (EngineRpcService.EnqueueBatchRequestPB batch : workerBatches) {
                int batchSize = 0;
                for (EngineRpcService.EnqueueBatchDpSlotPB slot : batch.getDpSlotsList()) {
                    for (EngineRpcService.EnqueueBatchExternalInputPB request
                            : slot.getRequestsList()) {
                        batchSize++;
                        totalRequests++;
                        long requestId = request.getInput().getRequestId();
                        assertTrue(requestIds.add(requestId),
                                "mock engine received a duplicate request_id");
                        int inputLength = request.getInput().getTokenIdsCount();
                        int requestIndex = Math.toIntExact(requestId - firstRequestId);
                        RealRequestTemplate template = realRequestTemplates.get(
                                Math.floorMod(requestIndex, realRequestTemplates.size()));
                        assertEquals(template.seqLen(), inputLength,
                                "engine input length must match the log-derived schedule request");
                        inputLengths.add(inputLength);
                        totalInputTokens += inputLength;
                    }
                }
                maxBatchSize = Math.max(maxBatchSize, batchSize);
            }
        }
        double averageBatchSize = batchCount == 0 ? 0.0 : totalRequests / (double) batchCount;
        double averageInputTokens = totalRequests == 0 ? 0.0 : totalInputTokens / (double) totalRequests;
        return new BatchSummary(batchCount, maxBatchSize, averageBatchSize,
                averageInputTokens, inputLengths.size(), activeWorkerCount, requestIds);
    }

    private static Path findOnlineLogsDirectory() throws IOException {
        Path current = Path.of("").toAbsolutePath();
        for (int depth = 0; depth < 6 && current != null; depth++) {
            Path candidate = current.resolve("tools/online_eval/data/online_logs");
            if (Files.isRegularFile(candidate.resolve("sample_access.json"))
                    && Files.isRegularFile(candidate.resolve("trace_30min.jsonl"))) {
                return candidate;
            }
            current = current.getParent();
        }
        throw new IOException("Cannot locate tools/online_eval/data/online_logs from "
                + Path.of("").toAbsolutePath());
    }

    private static int[] readTokenCorpus(JsonNode inputIds) throws IOException {
        if (!inputIds.isArray() || inputIds.isEmpty()) {
            throw new IOException("sample_access.json does not contain input_ids");
        }
        int[] result = new int[inputIds.size()];
        for (int index = 0; index < inputIds.size(); index++) {
            result[index] = inputIds.get(index).intValue();
        }
        return result;
    }

    private static int[] obfuscatedCopy(int[] source, long seed) {
        int[] sourceVocabulary = Arrays.stream(source).distinct().toArray();
        if (sourceVocabulary.length == 0) {
            throw new IllegalArgumentException("at least one token ID is required");
        }

        SplittableRandom random = new SplittableRandom(seed);
        for (int index = sourceVocabulary.length - 1; index > 0; index--) {
            int other = random.nextInt(index + 1);
            int value = sourceVocabulary[index];
            sourceVocabulary[index] = sourceVocabulary[other];
            sourceVocabulary[other] = value;
        }

        long pseudonymBase = (long) Arrays.stream(source).max().orElseThrow() + 1L;
        if (pseudonymBase + sourceVocabulary.length - 1L > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("not enough integer IDs for token obfuscation");
        }

        Map<Integer, Integer> tokenIdRemap = new HashMap<>(sourceVocabulary.length);
        for (int index = 0; index < sourceVocabulary.length; index++) {
            tokenIdRemap.put(sourceVocabulary[index], (int) (pseudonymBase + index));
        }

        int[] obfuscated = new int[source.length];
        for (int index = 0; index < source.length; index++) {
            obfuscated[index] = tokenIdRemap.get(source[index]);
        }
        return obfuscated;
    }

    private static boolean tokenIdsDifferAtEveryPosition(int[] source, int[] obfuscated) {
        if (source.length != obfuscated.length) {
            return false;
        }
        for (int index = 0; index < source.length; index++) {
            if (source[index] == obfuscated[index]) {
                return false;
            }
        }
        return true;
    }

    private static boolean tokenIdSetsAreDisjoint(int[] source, int[] obfuscated) {
        Set<Integer> sourceIds = new HashSet<>();
        for (int tokenId : source) {
            sourceIds.add(tokenId);
        }
        for (int tokenId : obfuscated) {
            if (sourceIds.contains(tokenId)) {
                return false;
            }
        }
        return true;
    }

    private static List<TraceShape> readTraceShapes(ObjectMapper mapper, Path tracePath)
            throws IOException {
        List<TraceShape> shapes = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(tracePath)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                JsonNode node = mapper.readTree(line);
                if (node.has("rid") || node.has("request_id")) {
                    throw new IOException("trace fixture contains an unsanitized request ID");
                }
                int inputLength = node.path("il").asInt();
                int outputLength = node.path("ol").asInt();
                if (inputLength <= 0 || outputLength <= 0) {
                    continue;
                }
                List<Long> blockKeys = new ArrayList<>();
                for (JsonNode blockKey : node.path("bh")) {
                    blockKeys.add(new BigInteger(blockKey.asText()).longValue());
                }
                shapes.add(new TraceShape(
                        inputLength,
                        outputLength,
                        List.copyOf(blockKeys)));
            }
        }
        if (shapes.size() < REAL_REQUEST_TEMPLATE_COUNT) {
            throw new IOException("Not enough usable requests in " + tracePath);
        }
        return shapes;
    }

    private static List<RealRequestTemplate> buildRequestTemplates(
            List<TraceShape> shapes,
            int[] realTokenCorpus,
            String model,
            JsonNode loggedGenerateConfig,
            int rawAccessOutputLength) {
        List<RealRequestTemplate> templates = new ArrayList<>(REAL_REQUEST_TEMPLATE_COUNT);
        for (int templateIndex = 0; templateIndex < REAL_REQUEST_TEMPLATE_COUNT; templateIndex++) {
            TraceShape shape;
            if (templateIndex == 0) {
                shape = new TraceShape(realTokenCorpus.length,
                        Math.max(1, rawAccessOutputLength), List.of());
            } else {
                int shapeIndex = (int) ((long) (templateIndex - 1) * shapes.size()
                        / (REAL_REQUEST_TEMPLATE_COUNT - 1));
                shape = shapes.get(shapeIndex);
            }
            int seqLen = Math.min(shape.inputLength(), realTokenCorpus.length);
            int corpusOffset = templateIndex == 0
                    ? 0 : Math.floorMod(templateIndex * 997, realTokenCorpus.length);
            int maxNewTokens = templateIndex == 0
                    ? loggedGenerateConfig.path("max_new_tokens")
                            .asInt(Math.max(1, shape.outputLength()))
                    : Math.max(1, shape.outputLength());

            EngineRpcService.GenerateConfigPB.Builder generateConfig =
                    EngineRpcService.GenerateConfigPB.newBuilder()
                            .setMaxNewTokens(maxNewTokens)
                            .setNumBeams(1)
                            .setNumReturnSequences(loggedGenerateConfig
                                    .path("num_return_sequences").asInt(1))
                            .setMinNewTokens(loggedGenerateConfig.path("min_new_tokens").asInt())
                            .setTopP((float) loggedGenerateConfig.path("top_p").asDouble(1.0))
                            .setTopK(loggedGenerateConfig.path("top_k").asInt())
                            .setTemperature((float) loggedGenerateConfig
                                    .path("temperature").asDouble(1.0))
                            .setRepetitionPenalty((float) loggedGenerateConfig
                                    .path("repetition_penalty").asDouble(1.0))
                            .setFrequencyPenalty((float) loggedGenerateConfig
                                    .path("frequency_penalty").asDouble())
                            .setPresencePenalty((float) loggedGenerateConfig
                                    .path("presence_penalty").asDouble())
                            .setReturnIncremental(true)
                            .setIsStreaming(true)
                            .setInThinkMode(loggedGenerateConfig
                                    .path("enable_thinking").asBoolean())
                            .setMaxThinkingTokens(loggedGenerateConfig
                                    .path("max_new_think_tokens").asInt())
                            .setTimeoutMs(loggedGenerateConfig.path("timeout_ms").asInt(120_000))
                            .setUniqueKey(String.format(
                                    "{\"rid\":\"log-template-%d\",\"input_len\":%d,\"output_len\":%d}",
                                    templateIndex, seqLen, shape.outputLength()));
            String responseFormat = loggedGenerateConfig.path("response_format").asText();
            if (!responseFormat.isBlank()) {
                generateConfig.setResponseFormat(StringValue.of(responseFormat));
            }

            EngineRpcService.GenerateInputPB.Builder input =
                    EngineRpcService.GenerateInputPB.newBuilder()
                            .setGenerateConfig(generateConfig)
                            .setClientId("flexlb_e2e_log_replay")
                            .setRequestInfo(EngineRpcService.RequestInfoPB.newBuilder()
                                    .setRequestId("log-template-" + templateIndex)
                                    .setTraceId("log-template-" + templateIndex)
                                    .setSourceRole("flexlb_e2e_ut")
                                    .build());
            for (int tokenIndex = 0; tokenIndex < seqLen; tokenIndex++) {
                input.addTokenIds(realTokenCorpus[
                        (corpusOffset + tokenIndex) % realTokenCorpus.length]);
            }
            templates.add(new RealRequestTemplate(
                    input.build(), shape.blockCacheKeys(), seqLen,
                    maxNewTokens, model));
        }
        return List.copyOf(templates);
    }

    private static Set<Long> expectedRequestIds(long firstRequestId, int requestCount) {
        Set<Long> expected = new HashSet<>(requestCount);
        for (int index = 0; index < requestCount; index++) {
            expected.add(firstRequestId + index);
        }
        return expected;
    }

    private void suppressRequestPathLogs() {
        flexlbLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        mockRpcLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("org.flexlb.mock.MockRpcService");
        nettyLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("io.netty");
        grpcLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("io.grpc");
        previousFlexlbLogLevel = flexlbLogger.getLevel();
        previousMockRpcLogLevel = mockRpcLogger.getLevel();
        previousNettyLogLevel = nettyLogger.getLevel();
        previousGrpcLogLevel = grpcLogger.getLevel();
        flexlbLogger.setLevel(Level.WARN);
        mockRpcLogger.setLevel(Level.WARN);
        nettyLogger.setLevel(Level.WARN);
        grpcLogger.setLevel(Level.WARN);
    }

    private void restoreRequestPathLogs() {
        if (flexlbLogger != null) {
            flexlbLogger.setLevel(previousFlexlbLogLevel);
        }
        if (mockRpcLogger != null) {
            mockRpcLogger.setLevel(previousMockRpcLogLevel);
        }
        if (nettyLogger != null) {
            nettyLogger.setLevel(previousNettyLogLevel);
        }
        if (grpcLogger != null) {
            grpcLogger.setLevel(previousGrpcLogLevel);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> nestedMap(Map<String, Object> source, String key) {
        return (Map<String, Object>) source.get(key);
    }

    private static Number number(Map<String, Object> source, String key) {
        return (Number) source.get(key);
    }

    private static long percentileNanos(long[] sortedValues, double percentile) {
        int index = Math.max(0, (int) Math.ceil(sortedValues.length * percentile) - 1);
        return sortedValues[index];
    }

    private record TimedResponse(FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                                 long latencyNanos) {
    }

    private record TrafficResult(double qps, double p50Ms, double p99Ms,
                                 List<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responses) {
    }

    private record BatchSummary(int batchCount, int maxBatchSize, double averageBatchSize,
                                double averageInputTokens, int distinctInputLengths,
                                int activeWorkerCount, Set<Long> requestIds) {
    }

    private record TraceShape(int inputLength, int outputLength,
                              List<Long> blockCacheKeys) {
    }

    private record RealRequestTemplate(EngineRpcService.GenerateInputPB generateInput,
                                       List<Long> blockCacheKeys,
                                       int seqLen,
                                       int maxNewTokens,
                                       String model) {
    }

    private static final class EmptyCacheAwareService implements CacheAwareService {
        @Override
        public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
                                                        RoleType roleType,
                                                        String group) {
            return Map.of();
        }

        @Override
        public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
            return null;
        }
    }
}
