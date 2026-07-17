package org.flexlb.mock;

import io.netty.channel.nio.NioEventLoopGroup;
import java.io.IOException;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Base class for mock-worker integration tests.
 *
 * <p>Sets up a real {@link FlexlbBatchScheduler} backed by a real
 * {@link EngineGrpcClient} that creates real Netty gRPC channels to
 * mock workers.  No Spring Boot context, no model loading, no GPU.
 *
 * <p>Subclasses call {@link #setupWorkers} in {@code @BeforeEach} (or
 * rely on the default), then use {@link #submitRequest(long)} and
 * {@link #cancelRequest(long)} to drive the scheduler.
 *
 * <p>Architecture:
 * <pre>
 * Real FlexlbBatchScheduler (direct construction)
 *   ├── Real DefaultBatchDispatcher
 *   │     └── Real EngineGrpcClient (real Netty channels)
 *   │           ↕  real gRPC (Netty)
 *   │     MockPrefillWorker (gRPC server, no model)
 *   │     MockDecodeWorker  (gRPC server, no model)
 *   ├── Real EndpointRegistry
 *   └── Mock Router (returns mock worker addresses)
 * </pre>
 */
public abstract class FlexLBMockTestBase {

    private static final Logger log = LoggerFactory.getLogger(FlexLBMockTestBase.class);

    // ==================== Managed resources ====================

    protected MockPrefillWorker mockPrefillWorker;
    protected MockDecodeWorker mockDecodeWorker;
    protected FlexlbBatchScheduler scheduler;
    protected EndpointRegistry endpointRegistry;
    protected FlexlbConfig config;
    protected ConfigService configService;
    protected Router router;
    protected EngineGrpcClient grpcClient;
    protected DefaultBatchDispatcher dispatcher;
    protected BatchSchedulerReporter reporter;
    protected EngineWorkerStatus engineWorkerStatus;

    private NioEventLoopGroup eventLoopGroup;
    private ThreadPoolExecutor grpcExecutor;

    // Additional prefill workers started by tests (for multi-worker scenarios)
    private final List<MockPrefillWorker> additionalPrefillWorkers = new ArrayList<>();
    private final List<String> additionalPrefillIpPorts = new ArrayList<>();

    // ==================== Worker addresses (set by setupWorkers) ====================

    protected String prefillIp;
    protected int prefillHttpPort;
    protected int prefillGrpcPort;
    protected String prefillIpPort;

    protected String decodeIp;
    protected int decodeHttpPort;
    protected int decodeGrpcPort;
    protected String decodeIpPort;

    // ==================== Lifecycle ====================

    /**
     * Start mock workers and wire up the scheduler.  Override
     * {@link #createPrefillBehavior()} and {@link #createDecodeBehavior()}
     * to customize worker behavior.
     */
    @BeforeEach
    public void setUpBase() throws Exception {
        // 1. Start mock workers
        mockPrefillWorker = new MockPrefillWorker(createPrefillBehavior());
        mockPrefillWorker.start(0);
        prefillGrpcPort = mockPrefillWorker.getPort();
        prefillHttpPort = prefillGrpcPort - 1;
        prefillIp = "127.0.0.1";
        prefillIpPort = prefillIp + ":" + prefillHttpPort;

        mockDecodeWorker = new MockDecodeWorker(createDecodeBehavior());
        mockDecodeWorker.start(0);
        decodeGrpcPort = mockDecodeWorker.getPort();
        decodeHttpPort = decodeGrpcPort - 1;
        decodeIp = "127.0.0.1";
        decodeIpPort = decodeIp + ":" + decodeHttpPort;

        log.info("Mock workers started: prefill=127.0.0.1:{}(grpc={}), decode=127.0.0.1:{}(grpc={})",
                prefillHttpPort, prefillGrpcPort, decodeHttpPort, decodeGrpcPort);

        // 2. Create config
        config = createConfig();
        configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);

        // 3. Create gRPC infrastructure
        eventLoopGroup = new NioEventLoopGroup(2);
        grpcExecutor = new ThreadPoolExecutor(
                2, 4, 60L, TimeUnit.SECONDS, new LinkedBlockingQueue<>(128));

        CustomNameResolver nameResolver = (listener) -> { /* no-op */ };
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        EngineLocalView engineLocalView = mock(EngineLocalView.class);
        GlobalCacheIndex globalCacheIndex = mock(GlobalCacheIndex.class);

        grpcClient = new EngineGrpcClient(
                nameResolver, grpcExecutor, eventLoopGroup,
                engineLocalView, globalCacheIndex, grpcReporter);

        // 4. Create real dispatcher
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);

        // 5. Mock reporter (metrics no-op)
        reporter = mock(BatchSchedulerReporter.class);

        // 6. Mock engineWorkerStatus
        engineWorkerStatus = mock(EngineWorkerStatus.class);

        // 7. Create real EndpointRegistry (scheduler=null for now, replaced below)
        endpointRegistry = new EndpointRegistry(configService, null, reporter);

        // 8. Build WorkerStatus for prefill and decode mock workers
        WorkerStatus prefillWs = new WorkerStatus();
        prefillWs.setIp(prefillIp);
        prefillWs.setPort(prefillHttpPort);
        prefillWs.setGrpcPort(prefillGrpcPort);
        prefillWs.setRole(RoleType.PREFILL);
        prefillWs.setAlive(true);
        prefillWs.setGroup("test-group");
        prefillWs.setDpRank(0);
        prefillWs.setAvailableKvCacheTokens(new java.util.concurrent.atomic.AtomicLong(1_000_000L));
        prefillWs.setTotalKvCacheTokens(new java.util.concurrent.atomic.AtomicLong(2_000_000L));

        WorkerStatus decodeWs = new WorkerStatus();
        decodeWs.setIp(decodeIp);
        decodeWs.setPort(decodeHttpPort);
        decodeWs.setGrpcPort(decodeGrpcPort);
        decodeWs.setRole(RoleType.DECODE);
        decodeWs.setAlive(true);
        decodeWs.setGroup("test-group");
        decodeWs.setDpRank(0);
        decodeWs.setAvailableKvCacheTokens(new java.util.concurrent.atomic.AtomicLong(1_000_000L));
        decodeWs.setTotalKvCacheTokens(new java.util.concurrent.atomic.AtomicLong(2_000_000L));

        // 9. Register decode endpoint (no scheduler dependency)
        endpointRegistry.ensureDecodeEndpoint(decodeIpPort, decodeWs);

        // 10. Mock Router to return mock worker addresses
        router = mock(Router.class);
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return successRoute(ctx.getRequestId());
        });

        // 11. Create real scheduler
        scheduler = new FlexlbBatchScheduler(
                configService, router, grpcClient, engineWorkerStatus,
                endpointRegistry, dispatcher, reporter, null);

        // 12. Register prefill endpoint with the real scheduler as BatchDecisionHandler
        PrefillEndpoint prefillEp = new PrefillEndpoint(prefillWs, config, scheduler, reporter);
        endpointRegistry.putPrefill(prefillIpPort, prefillEp);

        // 13. Register in EngineWorkerStatus static map for completeness
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(prefillIpPort, prefillWs);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(decodeIpPort, decodeWs);
    }

    @AfterEach
    public void tearDownBase() {
        // Stop additional prefill workers started by tests
        for (MockPrefillWorker worker : additionalPrefillWorkers) {
            worker.stop();
        }
        additionalPrefillWorkers.clear();
        for (String ipPort : additionalPrefillIpPorts) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().remove(ipPort);
        }
        additionalPrefillIpPorts.clear();

        if (scheduler != null) {
            scheduler.shutdown();
        }
        if (mockPrefillWorker != null) {
            mockPrefillWorker.stop();
        }
        if (mockDecodeWorker != null) {
            mockDecodeWorker.stop();
        }
        if (grpcClient != null) {
            grpcClient.shutdownChannelPool();
        }
        if (grpcExecutor != null) {
            grpcExecutor.shutdownNow();
        }
        if (eventLoopGroup != null) {
            eventLoopGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        }
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
    }

    // ==================== Override points ====================

    /**
     * Override to configure prefill worker behavior.
     * Default: immediate response, no failures.
     */
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder().build();
    }

    /**
     * Override to configure decode worker behavior.
     * Default: immediate response, no failures.
     */
    protected MockWorkerBehavior createDecodeBehavior() {
        return MockWorkerBehavior.builder().build();
    }

    /**
     * Override to customize the FlexlbConfig.
     * Default: batch enabled, size_max=1, immediate dispatch.
     */
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);        // single request triggers dispatch
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchFillThreshold(1.0);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    // ==================== Helper: submit / cancel ====================

    /**
     * Submit a request with the given ID and default seq_len=128.
     */
    protected CompletableFuture<Response> submitRequest(long requestId) {
        return scheduler.submit(createBalanceContext(requestId));
    }

    /**
     * Submit a request with the given ID and seq_len.
     */
    protected CompletableFuture<Response> submitRequest(long requestId, long seqLen) {
        return scheduler.submit(createBalanceContext(requestId, seqLen));
    }

    /**
     * Cancel a request by ID.
     */
    protected void cancelRequest(long requestId) {
        scheduler.cancel(requestId);
    }

    /**
     * Trigger inflight TTL cleanup manually (simulates @Scheduled in production).
     */
    protected void triggerTtlCleanup() {
        scheduler.cleanupInflight();
        endpointRegistry.evictExpiredAll(config.getFlexlbInflightTtlMs());
    }

    // ==================== Helper: endpoint accessors ====================

    protected PrefillEndpoint getPrefillEndpoint() {
        return endpointRegistry.getPrefill(prefillIpPort);
    }

    protected DecodeEndpoint getDecodeEndpoint() {
        return endpointRegistry.getDecode(decodeIpPort);
    }

    // ==================== Helper: multi-worker support ====================

    /**
     * Start an additional mock prefill worker and register it in the EndpointRegistry.
     *
     * <p>The worker is automatically stopped in {@code @AfterEach}.  Tests can use
     * {@link #workerIpPort(MockWorker)} to get the worker's {@code ip:httpPort} key
     * for routing and endpoint lookups.
     *
     * @param behavior behavior configuration for the new worker
     * @return the started {@link MockPrefillWorker}
     */
    protected MockPrefillWorker addPrefillWorker(MockWorkerBehavior behavior) throws IOException {
        MockPrefillWorker worker = new MockPrefillWorker(behavior);
        worker.start(0);
        int grpcPort = worker.getPort();
        int httpPort = grpcPort - 1;
        String ip = "127.0.0.1";
        String ipPort = ip + ":" + httpPort;

        WorkerStatus ws = new WorkerStatus();
        ws.setIp(ip);
        ws.setPort(httpPort);
        ws.setGrpcPort(grpcPort);
        ws.setRole(RoleType.PREFILL);
        ws.setAlive(true);
        ws.setGroup("test-group");
        ws.setDpRank(0);
        ws.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
        ws.setTotalKvCacheTokens(new AtomicLong(2_000_000L));

        PrefillEndpoint ep = new PrefillEndpoint(ws, config, scheduler, reporter);
        endpointRegistry.putPrefill(ipPort, ep);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(ipPort, ws);

        additionalPrefillWorkers.add(worker);
        additionalPrefillIpPorts.add(ipPort);

        log.info("Additional prefill worker started: {} (grpc={})", ipPort, grpcPort);
        return worker;
    }

    /**
     * Get the {@code ip:httpPort} string for a mock worker (for routing/endpoint lookup).
     */
    protected static String workerIpPort(MockWorker worker) {
        return "127.0.0.1:" + worker.getHttpPort();
    }

    // ==================== Internal: BalanceContext construction ====================

    protected BalanceContext createBalanceContext(long requestId) {
        return createBalanceContext(requestId, 128);
    }

    protected BalanceContext createBalanceContext(long requestId, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("mock-model");

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(config);
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
                        .setGroupTimeout(com.google.protobuf.Int32Value.of(77))
                        .build())
                .build();
        return input.toByteArray();
    }

    private Response successRoute(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                serverStatus(RoleType.PREFILL, prefillIp, prefillHttpPort, prefillGrpcPort, requestId),
                serverStatus(RoleType.DECODE, decodeIp, decodeHttpPort, decodeGrpcPort, requestId)
        ));
        return response;
    }

    private static ServerStatus serverStatus(RoleType role, String ip, int httpPort, int grpcPort, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("test-group");
        status.setRequestId(requestId);
        return status;
    }
}
