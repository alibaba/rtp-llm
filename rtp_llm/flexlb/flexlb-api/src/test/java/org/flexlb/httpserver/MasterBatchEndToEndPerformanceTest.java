package org.flexlb.httpserver;

import ch.qos.logback.classic.Level;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.ByteString;
import com.google.protobuf.StringValue;
import io.grpc.CallOptions;
import io.grpc.ConnectivityState;
import io.grpc.Drainable;
import io.grpc.KnownLength;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.stub.ClientCalls;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.DefaultBatchDispatcherTestFactory;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.consistency.MasterElectService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockPrefillWorker;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.LoggerFactory;
import org.springframework.mock.env.MockEnvironment;
import reactor.netty.resources.LoopResources;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.LockSupport;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/**
 * Loopback end-to-end regression for the Master scheduling delivery path.
 *
 * <p>The exercised path is:
 * client call -> Netty Master gRPC server -> FlexlbServiceImpl -> RouteService
 * -> RequestScheduler -> WorkerBatcher. BATCH continues through EngineGrpcClient
 * to a Netty mock engine; NON_BATCH publishes the route decision to the frontend.
 * The worker capacities are fixed by the fixture, while worker selection uses the
 * production DefaultRouter with cost-based Prefill and Decode selection.
 */
@Tag("performance-regression")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class MasterBatchEndToEndPerformanceTest extends FlexLBMockTestBase {

    private static final int WARMUP_REQUESTS = 64;
    /** Per-request schedule deadline; a stalled Master fails fast instead of hanging. */
    private static final long REQUEST_DEADLINE_SECONDS = 20L;
    private static final int REAL_REQUEST_TEMPLATE_COUNT = 128;
    private static final int DISPATCH_THREADS = 32;
    private static final int DISPATCH_QUEUE_CAPACITY = 2_048;
    private static final long PACING_SPIN_THRESHOLD_NANOS =
            TimeUnit.MICROSECONDS.toNanos(100);
    private static final int MAX_RECORDED_DELIVERY_WAIT_MS = 1_000;
    private static final DeliveryMode DELIVERY_MODE = DeliveryMode.parse(
            System.getProperty("flexlb.perf.delivery-mode", "BATCH"));
    private static final int MASTER_GRPC_EXECUTOR_CORE_THREADS =
            Integer.getInteger("flexlb.perf.master-grpc-executor-core-threads", 16);
    private static final int MASTER_GRPC_EXECUTOR_MAX_THREADS =
            Integer.getInteger("flexlb.perf.master-grpc-executor-max-threads", 32);
    private static final int MASTER_CLIENT_CHANNELS =
            Integer.getInteger("flexlb.perf.master-client-channels", 4);
    private static final MethodDescriptor.Marshaller<byte[]>
            PRE_SERIALIZED_REQUEST_MARSHALLER = new MethodDescriptor.Marshaller<>() {
                @Override
                public InputStream stream(byte[] serializedRequest) {
                    return new KnownLengthByteArrayInputStream(serializedRequest);
                }

                @Override
                public byte[] parse(InputStream stream) {
                    try {
                        return stream.readAllBytes();
                    } catch (IOException failure) {
                        throw new IllegalStateException(
                                "failed to parse pre-serialized schedule request", failure);
                    }
                }
            };
    private static final MethodDescriptor<
            byte[], FlexlbScheduleProtocol.FlexlbScheduleResponsePB>
            PRE_SERIALIZED_SCHEDULE_METHOD = preSerializedScheduleMethod();
    private static final long TOKEN_ID_REMAP_SEED =
            Long.getLong("flexlb.perf.e2e.token-id-remap-seed", 0x5EED_F1E5L);
    private static final int REQUEST_COUNT =
            Integer.getInteger("flexlb.perf.e2e.requests", 8_192);
    private static final long MEASUREMENT_REQUEST_ID_BASE = 1_000_000L;
    private static final int ENGINE_MATRIX_DURATION_MS =
            Integer.getInteger("flexlb.perf.engine-matrix-duration-ms", 500);
    private static final int ENGINE_MATRIX_MIN_REQUESTS =
            Integer.getInteger("flexlb.perf.engine-matrix-min-requests", 1_024);
    private static final int ENGINE_MATRIX_WARMUP_REQUESTS_PER_PREFILL =
            Integer.getInteger(
                    "flexlb.perf.engine-matrix-warmup-requests-per-prefill", 2);
    private static final int ENGINE_MATRIX_FIRST_PREFILL_GRPC_PORT =
            Integer.getInteger(
                    "flexlb.perf.engine-matrix-first-prefill-grpc-port", 22_001);
    /**
     * Measured requests per engine. The matrix asserts that every prefill and
     * decode engine appears in the measured routing decisions, so the window
     * has to be long enough to reach the whole fleet.
     */
    private static final int ENGINE_MATRIX_REQUESTS_PER_ENGINE =
            Integer.getInteger("flexlb.perf.engine-matrix-requests-per-engine", 16);
    /**
     * Comma-separated {@code prefillxdecode} topologies replacing the default
     * matrix, e.g. {@code 100x100,200x200,500x500,750x500} for production
     * fleet sizes.
     */
    private static final String ENGINE_MATRIX_TOPOLOGIES =
            System.getProperty("flexlb.perf.engine-matrix-topologies", "");
    private static final int[] STANDARD_ENGINE_MATRIX_TARGET_QPS =
            parseTargetQps(System.getProperty(
                    "flexlb.perf.engine-matrix-target-qps", "1000,2000,5000,10000"));
    private static final MasterElectService STANDALONE_MASTER_ELECT_SERVICE =
            new MasterElectService() {
                @Override
                public void start() {
                }

                @Override
                public void offline() {
                }

                @Override
                public void destroy() {
                }

                @Override
                public boolean isNeedConsistency() {
                    return false;
                }

                @Override
                public boolean isMaster() {
                    return false;
                }

                @Override
                public void refreshMasterHost(boolean forceSync) {
                }
            };
    private static final RequestSchedulerReporter NO_OP_REQUEST_REPORTER =
            new RequestSchedulerReporter(new NoOpFlexMonitor());

    private static List<RealRequestTemplate> realRequestTemplates;

    private FlexlbGrpcServer masterServer;
    private NioEventLoopGroup masterServerEventLoopGroup;
    private List<ManagedChannel> masterChannels = List.of();
    private ServerScheduleLatencyRecorder latencyRecorder;
    private ActiveRequestCounter activeRequestCounter;
    private static ch.qos.logback.classic.Logger flexlbLogger;
    private static ch.qos.logback.classic.Logger syncLogger;
    private static ch.qos.logback.classic.Logger mockWorkerLogger;
    private static ch.qos.logback.classic.Logger mockRpcLogger;
    private static ch.qos.logback.classic.Logger nettyLogger;
    private static ch.qos.logback.classic.Logger grpcLogger;
    private static ch.qos.logback.classic.Logger pvLogger;
    private static ch.qos.logback.classic.Logger prefillStrategyLogger;
    private static Level previousFlexlbLogLevel;
    private static Level previousSyncLogLevel;
    private static Level previousMockWorkerLogLevel;
    private static Level previousMockRpcLogLevel;
    private static Level previousNettyLogLevel;
    private static Level previousGrpcLogLevel;
    private static Level previousPvLogLevel;
    private static Level previousPrefillStrategyLogLevel;
    private final Map<String, LongAdder> dispatchReasonCounts = new ConcurrentHashMap<>();
    private final MillisecondHistogram deliveryWaitHistogram =
            new MillisecondHistogram(MAX_RECORDED_DELIVERY_WAIT_MS);

    @BeforeAll
    static void loadLogDerivedRequests() throws IOException {
        suppressRequestPathLogs();
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

    @AfterAll
    static void restoreLogsAfterTests() {
        restoreRequestPathLogs();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.fixedWindowDecision().setMaxCollectionWaitMs(10L);
        cfg.fixedWindowDecision().setMaxRequests(16);
        if (DELIVERY_MODE == DeliveryMode.NON_BATCH) {
            // This scheduling-only fixture does not replay terminal worker status,
            // so route leases cannot be retired. Leave the per-worker admission
            // limit disabled; the matrix still exercises the production binding
            // and selection path without accumulating artificial backpressure.
            cfg.setDispatcher(DispatcherConfig.nonBatch());
        }
        // The unpaced burst test requires every measured request to enter the
        // scheduler. Keep the test-only admission limits consistent with its
        // configurable request count instead of racing a smaller queue cap.
        cfg.queueScheduler().getCapacity().setMaxWaitingRequestsPerPrefillWorker(
                Math.max(4_096, REQUEST_COUNT));
        cfg.queueScheduler().getCapacity().setMaxOutstandingRequestsGlobal(
                Math.max(20_000, REQUEST_COUNT));
        cfg.queueScheduler().getLifecycle()
                .setMaxDeliveredNotAcceptedRequestsGlobal(
                        Math.max(20_000, REQUEST_COUNT));
        cfg.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(1_000_000L);
        return cfg;
    }

    @Override
    protected WorkerDirectory createWorkerDirectory() {
        return new WorkerDirectory(endpointRegistry);
    }

    @Override
    protected BatchSchedulerReporter createBatchSchedulerReporter() {
        return new CountingBatchSchedulerReporter(
                dispatchReasonCounts, deliveryWaitHistogram);
    }

    @Override
    protected RequestSchedulerReporter createRequestSchedulerReporter() {
        return NO_OP_REQUEST_REPORTER;
    }

    @Override
    protected DefaultBatchDispatcher createDispatcher() {
        return DefaultBatchDispatcherTestFactory.create(
                grpcClient, configService,
                DISPATCH_THREADS, DISPATCH_QUEUE_CAPACITY);
    }

    @Override
    protected DefaultRouter createRouter() {
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of());
        CostBasedPrefillStrategy prefillSelector =
                new CostBasedPrefillStrategy(
                        engineWorkerStatus,
                        cache,
                        mock(EngineHealthReporter.class));
        ModelMetaConfig modelMeta = mock(
                ModelMetaConfig.class, withSettings().stubOnly());
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.DECODE, RoleType.PREFILL));
        return new DefaultRouter(
                prefillSelector,
                new CostBasedDecodeStrategy(engineWorkerStatus),
                new RandomStrategy(engineWorkerStatus),
                configService,
                modelMeta);
    }

    @BeforeEach
    void startMasterGrpcServer() throws Exception {
        publishDecodeCapacity(1_000_000_000L, 2_000_000_000L);

        RouteService routeService = new RouteService(
                configService,
                (DefaultRouter) router,
                scheduler,
                new RecentCacheKeyTraceReporter());

        activeRequestCounter = new ActiveRequestCounter();
        latencyRecorder = new ServerScheduleLatencyRecorder();
        EngineHealthReporter engineHealthReporter = createNoOpEngineHealthReporter();
        FlexlbServiceImpl service = new FlexlbServiceImpl(
                routeService,
                STANDALONE_MASTER_ELECT_SERVICE,
                engineHealthReporter,
                activeRequestCounter,
                mock(FlexlbGrpcForwarder.class, withSettings().stubOnly()),
                configService,
                reporter,
                latencyRecorder,
                NO_OP_REQUEST_REPORTER);

        int grpcPort;
        try (ServerSocket socket = new ServerSocket(0)) {
            grpcPort = socket.getLocalPort();
        }
        MockEnvironment environment = new MockEnvironment()
                .withProperty("server.port", Integer.toString(grpcPort - 2))
                .withProperty("FLEXLB_GRPC_EXECUTOR_CORE_SIZE",
                        Integer.toString(MASTER_GRPC_EXECUTOR_CORE_THREADS))
                .withProperty("FLEXLB_GRPC_EXECUTOR_MAX_SIZE",
                        Integer.toString(MASTER_GRPC_EXECUTOR_MAX_THREADS))
                .withProperty("FLEXLB_GRPC_EXECUTOR_QUEUE_SIZE",
                        Integer.toString(Math.max(4_096, REQUEST_COUNT)));
        masterServerEventLoopGroup = new NioEventLoopGroup(4);
        masterServer = new FlexlbGrpcServer(
                service,
                configService,
                environment,
                masterServerEventLoopGroup,
                null,
                new GrpcServerTimingInterceptor(),
                new GrpcQosHeaderInterceptor());
        masterServer.start();

        if (MASTER_CLIENT_CHANNELS <= 0) {
            throw new IllegalArgumentException(
                    "flexlb.perf.master-client-channels must be positive");
        }
        List<ManagedChannel> channels = new ArrayList<>(MASTER_CLIENT_CHANNELS);
        try {
            for (int index = 0; index < MASTER_CLIENT_CHANNELS; index++) {
                ManagedChannel channel = NettyChannelBuilder
                        .forAddress("127.0.0.1", grpcPort)
                        .usePlaintext()
                        .build();
                channels.add(channel);
                awaitChannelReady(channel);
            }
        } catch (Exception | Error failure) {
            for (ManagedChannel channel : channels) {
                channel.shutdownNow();
            }
            for (ManagedChannel channel : channels) {
                try {
                    channel.awaitTermination(5, TimeUnit.SECONDS);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    failure.addSuppressed(interrupted);
                    break;
                }
            }
            throw failure;
        }
        masterChannels = List.copyOf(channels);
    }

    private EngineHealthReporter createNoOpEngineHealthReporter() {
        CacheMetricsReporter constructorOnlyCacheMetricsReporter =
                mock(CacheMetricsReporter.class, withSettings().stubOnly());
        LoopResources constructorOnlyLoopResources =
                useNative -> grpcClientEventLoopGroup();
        return new EngineHealthReporter(
                new NoOpFlexMonitor(),
                constructorOnlyCacheMetricsReporter,
                grpcClient,
                constructorOnlyLoopResources,
                engineWorkerStatus);
    }

    @AfterEach
    void stopMasterGrpcServer() throws InterruptedException {
        for (ManagedChannel channel : masterChannels) {
            channel.shutdownNow();
        }
        for (ManagedChannel channel : masterChannels) {
            channel.awaitTermination(5, TimeUnit.SECONDS);
        }
        masterChannels = List.of();
        if (masterServer != null) {
            masterServer.shutdown();
        }
        if (masterServerEventLoopGroup != null) {
            assertTrue(masterServerEventLoopGroup.terminationFuture()
                            .await(5, TimeUnit.SECONDS),
                    "Master gRPC event-loop did not terminate");
        }
    }

    @Test
    @Order(1)
    @Timeout(value = 45, unit = TimeUnit.SECONDS)
    void batchScheduleRemainsFastAcrossRealGrpcBoundaries() throws Exception {
        assumeTrue(DELIVERY_MODE == DeliveryMode.BATCH,
                "single-worker burst exercises Master-owned BATCH delivery only");
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
                "FlexLB Master delivery E2E: delivery=%s requests=%d "
                        + "client_qps=%.1f master_qps=%.1f "
                        + "client_p50=%.3fms client_p99=%.3fms master_p50=%dms master_p90=%dms "
                        + "master_p95=%dms master_p99=%dms master_avg=%.3fms "
                        + "engine_batches=%d avg_batch=%.2f max_batch=%d avg_input_tokens=%.1f%n",
                DELIVERY_MODE, REQUEST_COUNT, result.qps(), masterQps,
                result.p50Ms(), result.p99Ms(),
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
        awaitNoActiveRequests();

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

    @ParameterizedTest(name = "prefill={0}, decode={1}, target_qps={2}")
    @Order(2)
    @MethodSource("engineScales")
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void masterMeetsRateSloAcrossEngineScaleMatrix(int prefillEngineCount,
                                                   int decodeEngineCount,
                                                   int targetQps) throws Exception {
        assertTrue(config.isFixedWindowDecision(),
                "engine-scale perf must exercise FIXED_WINDOW decisions");
        provisionPrefillEndpoints(prefillEngineCount);
        while (endpointRegistry.getEndpointCount(RoleType.DECODE) < decodeEngineCount) {
            addLogicalDecodeEndpoint(endpointRegistry.getEndpointCount(RoleType.DECODE));
        }
        assertEquals(prefillEngineCount, endpointRegistry.getEndpointCount(RoleType.PREFILL));
        assertEquals(decodeEngineCount, endpointRegistry.getEndpointCount(RoleType.DECODE));

        int warmupRequests = Math.max(
                WARMUP_REQUESTS,
                prefillEngineCount * ENGINE_MATRIX_WARMUP_REQUESTS_PER_PREFILL);
        TrafficResult warmup = runTraffic(
                warmupRequests,
                50_000_000L + prefillEngineCount * 10_000L + targetQps,
                targetQps);
        assertSuccessful(warmup);
        awaitCompletionCount(warmupRequests);
        awaitDeliveryWaitCount(warmupRequests);
        resetMeasurementState();

        double minimumQpsRatio = Double.parseDouble(
                System.getProperty("flexlb.perf.engine-matrix-min-qps-ratio", "0.85"));
        double maximumClientP99Ms = Double.parseDouble(System.getProperty(
                "flexlb.perf.engine-matrix-max-client-p99-ms", "250"));
        long maximumServerP99Ms = Long.getLong(
                "flexlb.perf.engine-matrix-max-server-p99-ms", 250L);
        long maximumBatchWaitP99Ms = Long.getLong(
                "flexlb.perf.engine-matrix-max-batch-wait-p99-ms", 50L);

        int fleetCoverageRequests = Math.max(prefillEngineCount, decodeEngineCount)
                * ENGINE_MATRIX_REQUESTS_PER_ENGINE;
        int requestCount = Math.max(
                Math.max(ENGINE_MATRIX_MIN_REQUESTS, fleetCoverageRequests),
                targetQps * ENGINE_MATRIX_DURATION_MS / 1_000);
        long firstRequestId = 10_000_000L
                + prefillEngineCount * 1_000_000L
                + targetQps * 100L;

        resetMeasurementState();
        TrafficResult result = runTraffic(requestCount, firstRequestId, targetQps);
        assertSuccessful(result);
        Map<String, Object> masterSnapshot = awaitCompletionCount(requestCount);
        awaitDeliveryWaitCount(requestCount);
        MillisecondHistogram.Snapshot deliveryWait = deliveryWaitHistogram.snapshot();
        BatchSummary batches = summarizeEngineBatches(firstRequestId, allPrefillWorkers());
        double masterQps = number(masterSnapshot, "completion_qps").doubleValue();
        Map<String, Object> serverLatency = nestedMap(masterSnapshot, "server_total_ms");
        long serverP99Ms = number(serverLatency, "p99").longValue();
        Map<String, Object> batchWait = nestedMap(masterSnapshot, "batch_wait_ms");
        long batchWaitP95Ms = number(batchWait, "p95").longValue();
        long batchWaitP99Ms = number(batchWait, "p99").longValue();
        Map<String, Object> dispatchAck = nestedMap(masterSnapshot, "dispatch_ack_ms");
        Map<String, Object> grpcQueue = nestedMap(masterSnapshot, "grpc_queue_ms");
        Map<String, Object> routeSubmit = nestedMap(masterSnapshot, "route_submit_ms");
        Map<String, Object> ackResponse = nestedMap(masterSnapshot, "ack_response_ms");
        long batchFullCount = dispatchReasonCount("batch_full");
        long windowTimeoutCount = dispatchReasonCount("fixed_window_timeout");
        long predictedExecutionCapCount = dispatchReasonCount("predicted_execution_cap");
        long totalDispatchReasons = batchFullCount
                + windowTimeoutCount + predictedExecutionCapCount;
        int activePrefillRoutes = activeScheduledEngineCount(result, RoleType.PREFILL);
        int activeDecodeRoutes = activeScheduledEngineCount(result, RoleType.DECODE);
        boolean batchDelivery = DELIVERY_MODE == DeliveryMode.BATCH;
        String batchFullLabel = batchDelivery ? Long.toString(batchFullCount) : "N/A";
        String windowTimeoutLabel = batchDelivery ? Long.toString(windowTimeoutCount) : "N/A";
        String predictedCapLabel = batchDelivery
                ? Long.toString(predictedExecutionCapCount) : "N/A";
        String averageBatchLabel = batchDelivery
                ? String.format("%.2f", batches.averageBatchSize()) : "N/A";
        String maximumBatchLabel = batchDelivery
                ? Integer.toString(batches.maxBatchSize()) : "N/A";

        System.out.printf(
                "FlexLB Master engine-scale E2E: delivery=%s decision=FIXED_WINDOW "
                        + "prefill=%d decode=%d "
                        + "target_qps=%d requests=%d client_qps=%.1f master_qps=%.1f "
                        + "client_p50=%.3fms client_p90=%.3fms client_p95=%.3fms "
                        + "client_p99=%.3fms "
                        + "master_p50=%s master_p90=%s master_p95=%s "
                        + "master_p99=%s active_prefill_rpc=%d "
                        + "active_prefill_route=%d active_decode_route=%d "
                        + "batch_wait_count=%d batch_wait_p50=%s batch_wait_p90=%s "
                        + "batch_wait_p95=%s batch_wait_p99=%s "
                        + "delivery_wait_count=%d delivery_wait_p50=%s "
                        + "delivery_wait_p90=%s delivery_wait_p95=%s "
                        + "delivery_wait_p99=%s "
                        + "dispatch_ack_count=%d dispatch_ack_p99=%s "
                        + "engine_batches=%d batch_full=%s window_timeout=%s "
                        + "predicted_execution_cap=%s avg_batch=%s max_batch=%s "
                        + "grpc_queue_p99=%s route_submit_p99=%s "
                        + "ack_response_p99=%s%n",
                DELIVERY_MODE, prefillEngineCount, decodeEngineCount, targetQps,
                requestCount, result.qps(), masterQps,
                result.p50Ms(), result.p90Ms(), result.p95Ms(), result.p99Ms(),
                latencyBucketLabel(serverLatency, "p50"),
                latencyBucketLabel(serverLatency, "p90"),
                latencyBucketLabel(serverLatency, "p95"),
                latencyBucketLabel(serverLatency, "p99"),
                batches.activeWorkerCount(),
                activePrefillRoutes, activeDecodeRoutes,
                number(batchWait, "count").longValue(),
                latencyBucketLabel(batchWait, "p50"),
                latencyBucketLabel(batchWait, "p90"),
                latencyBucketLabel(batchWait, "p95"),
                latencyBucketLabel(batchWait, "p99"),
                deliveryWait.count(), millisecondBucketLabel(deliveryWait.count(), deliveryWait.p50()),
                millisecondBucketLabel(deliveryWait.count(), deliveryWait.p90()),
                millisecondBucketLabel(deliveryWait.count(), deliveryWait.p95()),
                millisecondBucketLabel(deliveryWait.count(), deliveryWait.p99()),
                number(dispatchAck, "count").longValue(),
                latencyBucketLabel(dispatchAck, "p99"),
                batches.batchCount(), batchFullLabel, windowTimeoutLabel,
                predictedCapLabel, averageBatchLabel, maximumBatchLabel,
                latencyBucketLabel(grpcQueue, "p99"),
                latencyBucketLabel(routeSubmit, "p99"),
                latencyBucketLabel(ackResponse, "p99"));

        assertEquals(requestCount, number(masterSnapshot, "arrival_count").longValue());
        assertEquals(requestCount, number(masterSnapshot, "completion_count").longValue());
        assertEquals(requestCount, number(serverLatency, "count").longValue(),
                "every request must record Master server latency");
        assertEquals(requestCount, number(grpcQueue, "count").longValue(),
                "every request must record gRPC queue latency");
        assertEquals(requestCount, number(routeSubmit, "count").longValue(),
                "every request must record route-submit latency");
        assertEquals(requestCount, number(ackResponse, "count").longValue(),
                "every request must record response-publication latency");
        assertEquals(requestCount, deliveryWait.count(),
                "every request must record enqueue-to-delivery wait");
        assertEquals(prefillEngineCount, activePrefillRoutes,
                "every prefill engine must appear in measured routing decisions");
        assertEquals(decodeEngineCount, activeDecodeRoutes,
                "every decode engine must appear in measured routing decisions");
        awaitNoActiveRequests();
        assertTrue(result.qps() >= targetQps * minimumQpsRatio,
                () -> String.format(
                        "client throughput %.1f QPS missed %.0f%% of target %d QPS",
                        result.qps(), minimumQpsRatio * 100.0, targetQps));
        assertTrue(masterQps >= targetQps * minimumQpsRatio,
                () -> String.format(
                        "Master throughput %.1f QPS missed %.0f%% of target %d QPS",
                        masterQps, minimumQpsRatio * 100.0, targetQps));
        assertTrue(result.p99Ms() <= maximumClientP99Ms,
                () -> String.format("client E2E P99 %.3f ms exceeds ceiling %.3f ms",
                        result.p99Ms(), maximumClientP99Ms));
        double sparseWindowArrivalsPerPrefill = targetQps
                * config.fixedWindowDecision().getMaxCollectionWaitMs()
                / 1_000.0 / prefillEngineCount;
        if (sparseWindowArrivalsPerPrefill < 0.5) {
            double minimumObservedWindowMs =
                    config.fixedWindowDecision().getMaxCollectionWaitMs() * 0.5;
            assertTrue(result.p50Ms() >= minimumObservedWindowMs,
                    () -> String.format(
                            "client E2E P50 %.3f ms did not observe the %d ms fixed window",
                            result.p50Ms(),
                            config.fixedWindowDecision().getMaxCollectionWaitMs()));
            assertTrue(deliveryWait.p50() >= minimumObservedWindowMs,
                    () -> String.format(
                            "delivery wait P50 %d ms did not observe the %d ms fixed window",
                            deliveryWait.p50(),
                            config.fixedWindowDecision().getMaxCollectionWaitMs()));
        }
        assertTrue(serverP99Ms <= maximumServerP99Ms,
                () -> String.format("Master server P99 %d ms exceeds ceiling %d ms",
                        serverP99Ms, maximumServerP99Ms));
        assertTrue(deliveryWait.p99() <= maximumBatchWaitP99Ms,
                () -> String.format("delivery wait P99 %d ms exceeds ceiling %d ms",
                        deliveryWait.p99(), maximumBatchWaitP99Ms));

        if (DELIVERY_MODE == DeliveryMode.BATCH) {
            assertEquals(requestCount, number(batchWait, "count").longValue(),
                    "every request must record Master batch queue wait");
            assertEquals(requestCount, number(dispatchAck, "count").longValue(),
                    "every request must record engine dispatch ACK latency");
            assertEquals(expectedRequestIds(firstRequestId, requestCount), batches.requestIds());
            assertEquals(prefillEngineCount, batches.activeWorkerCount(),
                    "every prefill engine must receive measured traffic");
            assertEquals(batches.batchCount(), totalDispatchReasons,
                    "every engine batch must have one recorded dispatch reason");
            assertTrue(batchWaitP99Ms <= maximumBatchWaitP99Ms,
                    () -> String.format("Master batch wait P99 %d ms exceeds ceiling %d ms",
                            batchWaitP99Ms, maximumBatchWaitP99Ms));
            if (targetQps == 2_000) {
                assertTrue(batchWaitP95Ms > 0,
                        "2k QPS queueing scenario must observe non-zero batch wait");
                double arrivalsPerWindowPerPrefill = targetQps
                        * config.fixedWindowDecision().getMaxCollectionWaitMs()
                        / 1_000.0 / prefillEngineCount;
                double expectedBatchSize = Math.min(
                        config.fixedWindowDecision().getMaxRequests(), arrivalsPerWindowPerPrefill);
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
        } else {
            assertEquals(0L, number(batchWait, "count").longValue(),
                    "NON_BATCH must not report a batch queue-wait histogram");
            assertEquals(0L, number(dispatchAck, "count").longValue(),
                    "NON_BATCH must not report an EnqueueBatch ACK histogram");
            assertEquals(0, batches.batchCount(),
                    "NON_BATCH must not call EnqueueBatch");
            assertEquals(0, batches.activeWorkerCount(),
                    "NON_BATCH must not contact a Prefill RPC server");
            assertTrue(batches.requestIds().isEmpty(),
                    "NON_BATCH must leave engine batch records empty");
            assertEquals(0L, totalDispatchReasons,
                    "NON_BATCH must not report a batch dispatch reason");
        }
    }

    private static Stream<Arguments> engineScales() {
        List<Arguments> arguments = new ArrayList<>();
        if (!ENGINE_MATRIX_TOPOLOGIES.isBlank()) {
            for (String rawTopology : ENGINE_MATRIX_TOPOLOGIES.split(",")) {
                String topology = rawTopology.trim();
                if (topology.isEmpty()) {
                    continue;
                }
                String[] engines = topology.split("x");
                addEngineScale(arguments,
                        Integer.parseInt(engines[0]),
                        Integer.parseInt(engines[1]),
                        STANDARD_ENGINE_MATRIX_TARGET_QPS);
            }
            return arguments.stream();
        }
        addEngineScale(arguments, 1, 2, STANDARD_ENGINE_MATRIX_TARGET_QPS);
        addEngineScale(arguments, 2, 4, new int[]{2_000});
        addEngineScale(arguments, 4, 8, STANDARD_ENGINE_MATRIX_TARGET_QPS);
        addEngineScale(arguments, 8, 16, new int[]{2_000});
        addEngineScale(arguments, 16, 32, STANDARD_ENGINE_MATRIX_TARGET_QPS);
        return arguments.stream();
    }

    private static void addEngineScale(
            List<Arguments> arguments,
            int prefillEngineCount,
            int decodeEngineCount,
            int[] targetQpsValues) {
        for (int targetQps : targetQpsValues) {
            arguments.add(Arguments.of(
                    prefillEngineCount, decodeEngineCount, targetQps));
        }
    }

    private static int[] parseTargetQps(String configured) {
        int[] targetQpsValues = Arrays.stream(configured.split(","))
                .map(String::trim)
                .filter(value -> !value.isEmpty())
                .mapToInt(Integer::parseInt)
                .toArray();
        if (targetQpsValues.length == 0
                || Arrays.stream(targetQpsValues).anyMatch(value -> value <= 0)) {
            throw new IllegalArgumentException(
                    "engine matrix target QPS values must be positive");
        }
        return targetQpsValues;
    }

    private TrafficResult runTraffic(int requestCount, long firstRequestId) throws Exception {
        return runTraffic(requestCount, firstRequestId, 0);
    }

    private void provisionPrefillEndpoints(int required) throws IOException {
        while (endpointRegistry.getEndpointCount(RoleType.PREFILL) < required) {
            int additionalIndex = endpointRegistry.getEndpointCount(RoleType.PREFILL) - 1;
            if (DELIVERY_MODE == DeliveryMode.BATCH) {
                addPrefillWorker(
                        MockWorkerBehavior.builder().build(),
                        ENGINE_MATRIX_FIRST_PREFILL_GRPC_PORT
                                + additionalIndex * 2);
            } else {
                addLogicalPrefillEndpoint(additionalIndex + 1);
            }
        }
    }

    private TrafficResult runTraffic(int requestCount, long firstRequestId,
                                     int targetQps) throws Exception {
        byte[][] serializedRequests = new byte[requestCount][];
        List<CompletableFuture<TimedResponse>> futures = new ArrayList<>(requestCount);
        for (int index = 0; index < requestCount; index++) {
            // Keep only the real wire payload once the measured window starts.
            serializedRequests[index] =
                    scheduleRequest(firstRequestId + index, index).toByteArray();
            futures.add(new CompletableFuture<>());
        }

        long trafficStartNanos = System.nanoTime();
        long issueIntervalNanos = targetQps > 0
                ? TimeUnit.SECONDS.toNanos(1) / targetQps
                : 0L;
        long nextIssueNanos = trafficStartNanos;
        for (int index = 0; index < requestCount; index++) {
            if (targetQps > 0) {
                paceUntil(nextIssueNanos);
            }
            long issueStartedNanos = System.nanoTime();
            issueRequest(futures.get(index), serializedRequests[index], index);
            if (targetQps > 0) {
                // Preserve the configured open-loop rate after a scheduling or GC
                // pause. Replaying missed slots as an immediate burst measures the
                // load generator's catch-up policy, not steady Master capacity.
                nextIssueNanos = Math.max(
                        nextIssueNanos + issueIntervalNanos,
                        issueStartedNanos + issueIntervalNanos);
            }
        }

        try {
            CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new))
                    .get(30, TimeUnit.SECONDS);
        } catch (Exception failure) {
            long completed = futures.stream().filter(CompletableFuture::isDone).count();
            long successful = futures.stream()
                    .filter(CompletableFuture::isDone)
                    .filter(future -> !future.isCompletedExceptionally())
                    .filter(future -> !future.isCancelled())
                    .count();
            System.out.printf(
                    "FlexLB Master traffic failure: requests=%d completed=%d successful=%d "
                            + "exceptional=%d scheduler_inflight=%d queued=%d "
                            + "active_grpc=%d engine_received=%d%n",
                    requestCount, completed, successful, completed - successful,
                    scheduler.getInflightSize(), scheduler.getQueuedRequestCount(),
                    activeRequestCounter.getCount(), receivedEngineRequestCount());
            for (int index = 0; index < futures.size(); index++) {
                CompletableFuture<TimedResponse> future = futures.get(index);
                if (!future.isCompletedExceptionally() && !future.isCancelled()) {
                    continue;
                }
                long requestId = firstRequestId + index;
                System.out.printf(
                        "FlexLB Master exceptional request: request_id=%d state=%s%n",
                        requestId, scheduler.getRequestState(requestId, 0L));
            }
            throw failure;
        }
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
                percentileNanos(latencies, 0.90) / 1_000_000.0,
                percentileNanos(latencies, 0.95) / 1_000_000.0,
                percentileNanos(latencies, 0.99) / 1_000_000.0,
                responses);
    }

    private void issueRequest(
            CompletableFuture<TimedResponse> future,
            byte[] serializedRequest,
            int requestIndex) {
        long requestStartNanos = System.nanoTime();
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                new StreamObserver<>() {
                    @Override
                    public void onNext(
                            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
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
                };
        try {
            ManagedChannel channel = masterChannels.get(
                    Math.floorMod(requestIndex, masterChannels.size()));
            ClientCalls.asyncUnaryCall(
                    channel.newCall(
                            PRE_SERIALIZED_SCHEDULE_METHOD,
                            CallOptions.DEFAULT.withDeadlineAfter(
                                    REQUEST_DEADLINE_SECONDS, TimeUnit.SECONDS)),
                    serializedRequest,
                    observer);
        } catch (Throwable failure) {
            future.completeExceptionally(failure);
        }
    }

    private static MethodDescriptor<byte[], FlexlbScheduleProtocol.FlexlbScheduleResponsePB>
            preSerializedScheduleMethod() {
        MethodDescriptor<FlexlbScheduleProtocol.FlexlbScheduleRequestPB,
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB> scheduleMethod =
                FlexlbServiceGrpc.getScheduleMethod();
        return scheduleMethod.toBuilder(
                PRE_SERIALIZED_REQUEST_MARSHALLER,
                scheduleMethod.getResponseMarshaller()).build();
    }

    private static final class KnownLengthByteArrayInputStream
            extends ByteArrayInputStream implements KnownLength, Drainable {
        private KnownLengthByteArrayInputStream(byte[] serializedRequest) {
            super(serializedRequest);
        }

        @Override
        public int drainTo(OutputStream target) throws IOException {
            int remaining = count - pos;
            target.write(buf, pos, remaining);
            pos = count;
            return remaining;
        }
    }

    private long receivedEngineRequestCount() {
        long received = 0L;
        for (MockPrefillWorker worker : allPrefillWorkers()) {
            for (EngineRpcService.EnqueueBatchRequestPB batch
                    : worker.getRpcService().getEnqueuedRequests()) {
                for (EngineRpcService.EnqueueBatchDpSlotPB slot : batch.getDpSlotsList()) {
                    received += slot.getRequestsCount();
                }
            }
        }
        return received;
    }

    /**
     * Connect the channel before any traffic, so a burst is written to the wire
     * instead of being buffered against a channel that is still resolving.
     */
    private static void awaitChannelReady(ManagedChannel channel) throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        ConnectivityState state = channel.getState(true);
        while (state != ConnectivityState.READY && System.nanoTime() < deadlineNanos) {
            CountDownLatch changed = new CountDownLatch(1);
            channel.notifyWhenStateChanged(state, changed::countDown);
            changed.await(1, TimeUnit.SECONDS);
            state = channel.getState(true);
        }
        assertEquals(ConnectivityState.READY, state,
                "Master channel must be connected before traffic starts");
    }

    private static void paceUntil(long targetNanos) {
        long remainingNanos;
        while ((remainingNanos = targetNanos - System.nanoTime()) > 0L) {
            if (remainingNanos > PACING_SPIN_THRESHOLD_NANOS) {
                LockSupport.parkNanos(
                        remainingNanos - PACING_SPIN_THRESHOLD_NANOS);
            } else {
                Thread.onSpinWait();
            }
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
                .setGenerateTimeout(TimeUnit.MINUTES.toMillis(5))
                .setMaxNewTokens(template.maxNewTokens())
                .setNumBeams(1)
                .setModel(template.model())
                .setCacheKeyBlockSize(1_024L)
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
            assertEquals(DELIVERY_MODE.enqueuedByMaster(),
                    response.getEnqueuedByMaster(),
                    "response delivery ownership must match the configured mode");
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

    private void awaitNoActiveRequests() throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (activeRequestCounter.getCount() != 0L
                && System.nanoTime() < deadlineNanos) {
            TimeUnit.MILLISECONDS.sleep(1);
        }
        assertEquals(0L, activeRequestCounter.getCount(),
                "all Master gRPC requests must release their active-request token");
    }

    private void awaitDeliveryWaitCount(long expectedCount) throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (deliveryWaitHistogram.count() < expectedCount
                && System.nanoTime() < deadlineNanos) {
            TimeUnit.MILLISECONDS.sleep(1);
        }
        assertEquals(expectedCount, deliveryWaitHistogram.count(),
                "enqueue-to-delivery telemetry must cover every request");
    }

    private void resetMeasurementState() {
        latencyRecorder.reset();
        dispatchReasonCounts.clear();
        deliveryWaitHistogram.reset();
        for (MockPrefillWorker worker : allPrefillWorkers()) {
            worker.resetRecords();
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

    private static void suppressRequestPathLogs() {
        flexlbLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        syncLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("syncLogger");
        mockWorkerLogger = (ch.qos.logback.classic.Logger)
                LoggerFactory.getLogger("org.flexlb.mock.MockWorker");
        mockRpcLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("org.flexlb.mock.MockRpcService");
        nettyLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("io.netty");
        grpcLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("io.grpc");
        pvLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        prefillStrategyLogger = (ch.qos.logback.classic.Logger)
                LoggerFactory.getLogger(CostBasedPrefillStrategy.class);
        previousFlexlbLogLevel = flexlbLogger.getLevel();
        previousSyncLogLevel = syncLogger.getLevel();
        previousMockWorkerLogLevel = mockWorkerLogger.getLevel();
        previousMockRpcLogLevel = mockRpcLogger.getLevel();
        previousNettyLogLevel = nettyLogger.getLevel();
        previousGrpcLogLevel = grpcLogger.getLevel();
        previousPvLogLevel = pvLogger.getLevel();
        previousPrefillStrategyLogLevel = prefillStrategyLogger.getLevel();
        // Lazy channel creation logs one WARN with the growing channel pool per
        // endpoint. Topology setup intentionally creates up to 16 endpoints in
        // a burst, so keep that one-time diagnostic I/O outside this data-path
        // performance measurement.
        flexlbLogger.setLevel(Level.ERROR);
        syncLogger.setLevel(Level.WARN);
        mockWorkerLogger.setLevel(Level.WARN);
        mockRpcLogger.setLevel(Level.WARN);
        nettyLogger.setLevel(Level.WARN);
        grpcLogger.setLevel(Level.WARN);
        pvLogger.setLevel(Level.WARN);
        prefillStrategyLogger.setLevel(Level.WARN);
    }

    private static void restoreRequestPathLogs() {
        if (flexlbLogger != null) {
            flexlbLogger.setLevel(previousFlexlbLogLevel);
        }
        if (syncLogger != null) {
            syncLogger.setLevel(previousSyncLogLevel);
        }
        if (mockWorkerLogger != null) {
            mockWorkerLogger.setLevel(previousMockWorkerLogLevel);
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
        if (pvLogger != null) {
            pvLogger.setLevel(previousPvLogLevel);
        }
        if (prefillStrategyLogger != null) {
            prefillStrategyLogger.setLevel(previousPrefillStrategyLogLevel);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> nestedMap(Map<String, Object> source, String key) {
        return (Map<String, Object>) source.get(key);
    }

    private static Number number(Map<String, Object> source, String key) {
        return (Number) source.get(key);
    }

    private static String latencyBucketLabel(Map<String, Object> histogram, String percentile) {
        if (number(histogram, "count").longValue() == 0L) {
            return "N/A";
        }
        long millis = number(histogram, percentile).longValue();
        return millis == 0L ? "<1ms" : millis + "ms";
    }

    private static String millisecondBucketLabel(long count, long millis) {
        if (count == 0L) {
            return "N/A";
        }
        return millis == 0L ? "<1ms" : millis + "ms";
    }

    private static long percentileNanos(long[] sortedValues, double percentile) {
        int index = Math.max(0, (int) Math.ceil(sortedValues.length * percentile) - 1);
        return sortedValues[index];
    }

    private record TimedResponse(FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                                 long latencyNanos) {
    }

    private record TrafficResult(double qps, double p50Ms, double p90Ms,
                                 double p95Ms, double p99Ms,
                                 List<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responses) {
    }

    private record BatchSummary(int batchCount, int maxBatchSize, double averageBatchSize,
                                double averageInputTokens, int distinctInputLengths,
                                int activeWorkerCount, Set<Long> requestIds) {
    }

    private enum DeliveryMode {
        BATCH,
        NON_BATCH;

        private static DeliveryMode parse(String configured) {
            String value = configured == null ? "" : configured.trim();
            return switch (value) {
                case "BATCH" -> BATCH;
                case "NON_BATCH" -> NON_BATCH;
                default -> throw new IllegalArgumentException(
                        "flexlb.perf.delivery-mode must be BATCH or NON_BATCH, got: "
                                + configured);
            };
        }

        private boolean enqueuedByMaster() {
            return this == BATCH;
        }
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

    private static final class CountingBatchSchedulerReporter extends BatchSchedulerReporter {
        private final Map<String, LongAdder> dispatchReasonCounts;
        private final MillisecondHistogram deliveryWaitHistogram;

        private CountingBatchSchedulerReporter(
                Map<String, LongAdder> dispatchReasonCounts,
                MillisecondHistogram deliveryWaitHistogram) {
            super(new NoOpFlexMonitor());
            this.dispatchReasonCounts = dispatchReasonCounts;
            this.deliveryWaitHistogram = deliveryWaitHistogram;
        }

        @Override
        public void reportDispatchReason(String role, String engineIp, String reason) {
            dispatchReasonCounts.computeIfAbsent(reason, ignored -> new LongAdder()).increment();
        }

        @Override
        public void reportBatchWaitTimeMs(
                String role, String engineIp, long waitMs, int priority) {
            deliveryWaitHistogram.record(waitMs);
        }
    }

    private static final class MillisecondHistogram {
        private final int maximumBucketMs;
        private final AtomicLongArray buckets;
        private final AtomicLong count = new AtomicLong();

        private MillisecondHistogram(int maximumBucketMs) {
            this.maximumBucketMs = maximumBucketMs;
            this.buckets = new AtomicLongArray(maximumBucketMs + 2);
        }

        private void record(long millis) {
            int bucket = (int) Math.min(maximumBucketMs + 1L, Math.max(0L, millis));
            buckets.incrementAndGet(bucket);
            count.incrementAndGet();
        }

        private long count() {
            return count.get();
        }

        private Snapshot snapshot() {
            long sampleCount = count();
            return new Snapshot(
                    sampleCount,
                    percentile(sampleCount, 0.50),
                    percentile(sampleCount, 0.90),
                    percentile(sampleCount, 0.95),
                    percentile(sampleCount, 0.99));
        }

        private long percentile(long sampleCount, double quantile) {
            if (sampleCount == 0L) {
                return 0L;
            }
            long target = Math.max(1L, (long) Math.ceil(sampleCount * quantile));
            long cumulative = 0L;
            for (int bucket = 0; bucket < buckets.length(); bucket++) {
                cumulative += buckets.get(bucket);
                if (cumulative >= target) {
                    return bucket;
                }
            }
            throw new IllegalStateException("delivery-wait histogram sample count drifted");
        }

        private void reset() {
            count.set(0L);
            for (int bucket = 0; bucket < buckets.length(); bucket++) {
                buckets.set(bucket, 0L);
            }
        }

        private record Snapshot(long count, long p50, long p90, long p95, long p99) {
        }
    }
}
