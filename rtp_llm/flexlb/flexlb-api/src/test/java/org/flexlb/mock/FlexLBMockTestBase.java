package org.flexlb.mock;

import com.google.protobuf.Int32Value;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.priority.UnsupportedEngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.cache.EngineCacheInvalidator;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
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
 * <p>Sets up a real {@link PriorityScheduler} backed by a real
 * {@link EngineGrpcClient} that creates real Netty gRPC channels to
 * mock workers.  No Spring Boot context, no model loading, no GPU.
 *
 * <p>Subclasses call {@link #setupWorkers} in {@code @BeforeEach} (or
 * rely on the default), then use {@link #submitRequest(long)} and
 *
 * <p>Architecture:
 * <pre>
 * Real PriorityScheduler (direct construction)
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
    protected PriorityScheduler scheduler;
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
    private final List<String> additionalDecodeIpPorts = new ArrayList<>();

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
        EngineCacheInvalidator engineCacheInvalidator = mock(EngineCacheInvalidator.class);

        grpcClient = new EngineGrpcClient(
                nameResolver, grpcExecutor, eventLoopGroup,
                engineCacheInvalidator, grpcReporter);

        // 4. Create real dispatcher
        dispatcher = createDispatcher();

        // 5. Reporter (metrics no-op by default)
        reporter = createBatchSchedulerReporter();

        // 6. Create real EndpointRegistry (scheduler=null for now, replaced below)
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);

        // 7. Engine status is mocked by default; E2E subclasses can use the real registry-backed view.
        engineWorkerStatus = createEngineWorkerStatus();

        // 8. Build WorkerStatus for prefill and decode mock workers
        WorkerStatus prefillWs = new WorkerStatus();
        prefillWs.setIp(prefillIp);
        prefillWs.setPort(prefillHttpPort);
        prefillWs.setGrpcPort(prefillGrpcPort);
        prefillWs.setRole(RoleType.PREFILL);
        prefillWs.setAlive(true);
        prefillWs.setGroup("test-group");
        prefillWs.setDpRank(0);
        prefillWs.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
        prefillWs.setTotalKvCacheTokens(new AtomicLong(2_000_000L));

        WorkerStatus decodeWs = new WorkerStatus();
        decodeWs.setIp(decodeIp);
        decodeWs.setPort(decodeHttpPort);
        decodeWs.setGrpcPort(decodeGrpcPort);
        decodeWs.setRole(RoleType.DECODE);
        decodeWs.setAlive(true);
        decodeWs.setGroup("test-group");
        decodeWs.setDpRank(0);
        decodeWs.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
        decodeWs.setTotalKvCacheTokens(new AtomicLong(2_000_000L));

        // 9. Register decode endpoint (no scheduler dependency)
        endpointRegistry.ensureEndpoint(RoleType.DECODE, decodeIpPort, decodeWs);

        // 10. Fixed routing by default; E2E subclasses can install the production router.
        router = createRouter();

        // 11. Create real scheduler
        scheduler = new PriorityScheduler(
                configService, router,
                endpointRegistry, dispatcher, reporter, null, null,
                new UnsupportedEngineCancelChannel());

        // 12. Register prefill endpoint with the real scheduler as DecisionGroupHandler
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, prefillIpPort, prefillWs);

        // 13. Register in EngineWorkerStatus static map for completeness
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(prefillIpPort, prefillWs);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(decodeIpPort, decodeWs);
    }

    @AfterEach
    public void tearDownBase() {
        // Stop scheduler-owned work before tearing down its dispatcher or workers.
        if (scheduler != null) {
            scheduler.shutdown();
        }
        if (dispatcher != null) {
            dispatcher.shutdown();
        }

        // Stop additional prefill workers started by tests.
        for (MockPrefillWorker worker : additionalPrefillWorkers) {
            worker.stop();
        }
        additionalPrefillWorkers.clear();
        for (String ipPort : additionalPrefillIpPorts) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().remove(ipPort);
        }
        additionalPrefillIpPorts.clear();
        for (String ipPort : additionalDecodeIpPorts) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().remove(ipPort);
        }
        additionalDecodeIpPorts.clear();

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
            try {
                grpcExecutor.awaitTermination(2, TimeUnit.SECONDS);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        if (eventLoopGroup != null) {
            eventLoopGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS)
                    .syncUninterruptibly();
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

    protected EngineWorkerStatus createEngineWorkerStatus() {
        return mock(EngineWorkerStatus.class);
    }

    protected Router createRouter() {
        Router fixedRouter = mock(Router.class);
        when(fixedRouter.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            reserveDecode(ctx);
            return successRoute(ctx.getRequestId());
        });
        return fixedRouter;
    }

    protected BatchSchedulerReporter createBatchSchedulerReporter() {
        return mock(BatchSchedulerReporter.class);
    }

    /** Override when an integration fixture needs deterministic dispatcher sizing. */
    protected DefaultBatchDispatcher createDispatcher() {
        return new DefaultBatchDispatcher(grpcClient, configService, null);
    }

    /** Mirror the Decode ownership side effect performed by production routing strategies. */
    protected void reserveDecode(BalanceContext ctx) {
        DecodeEndpoint decodeEndpoint = getDecodeEndpoint();
        long seqLen = ctx.getRequest().getSeqLen();
        long expectedKvTokens = config.decodeKvReservationTokens(
                seqLen,
                ctx.getRequest().getMaxNewTokens(),
                decodeEndpoint.realKvTotal());
        decodeEndpoint.reserve(ctx.getRequestId(), Math.max(0L, seqLen),
                expectedKvTokens, ctx.getPriority());
    }

    /**
     * Override to customize the FlexlbConfig.
     * Default: BATCH mode, size_max=1, immediate dispatch.
     */
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.batchDispatcher().setMaxRequests(1); // single request triggers dispatch
        cfg.batchDispatcher().setMaxCollectionWaitMs(300);
        cfg.batchDispatcher().setEnqueueRpcTimeoutMs(5_000L);
        cfg.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(300_000L);
        return cfg;
    }

    // ==================== Helper: submit ====================

    /**
     * Submit a request with the given ID and default seq_len=128.
     */
    protected CompletableFuture<Response> submitRequest(String requestId) {
        return scheduler.submit(createBalanceContext(requestId));
    }

    /**
     * Submit a request with the given ID and seq_len.
     */
    protected CompletableFuture<Response> submitRequest(String requestId, long seqLen) {
        return scheduler.submit(createBalanceContext(requestId, seqLen));
    }

    /**
     * Trigger inflight TTL cleanup manually (simulates @Scheduled in production).
     */
    protected void triggerTtlCleanup() {
        scheduler.cleanupInflight();
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

        endpointRegistry.ensureEndpoint(RoleType.PREFILL, ipPort, ws);
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

    /**
     * Return every live prefill mock used by the current fixture.
     */
    protected List<MockPrefillWorker> allPrefillWorkers() {
        List<MockPrefillWorker> workers = new ArrayList<>(additionalPrefillWorkers.size() + 1);
        workers.add(mockPrefillWorker);
        workers.addAll(additionalPrefillWorkers);
        return List.copyOf(workers);
    }

    /**
     * Add a decode endpoint without starting a server. The Master schedule path only
     * selects decode metadata; it does not contact decode before returning the ACK.
     */
    protected DecodeEndpoint addLogicalDecodeEndpoint(int workerIndex) {
        if (workerIndex <= 0 || workerIndex > 254) {
            throw new IllegalArgumentException("logical decode worker index must be in [1, 254]");
        }
        String ip = "192.0.2." + workerIndex;
        int httpPort = 61_000;
        int grpcPort = httpPort + 1;
        String ipPort = ip + ":" + httpPort;

        WorkerStatus ws = new WorkerStatus();
        ws.setIp(ip);
        ws.setPort(httpPort);
        ws.setGrpcPort(grpcPort);
        ws.setRole(RoleType.DECODE);
        ws.setAlive(true);
        ws.setGroup("test-group");
        ws.setDpRank(workerIndex);
        ws.setAvailableKvCacheTokens(new AtomicLong(1_000_000_000L));
        ws.setTotalKvCacheTokens(new AtomicLong(2_000_000_000L));

        DecodeEndpoint endpoint = (DecodeEndpoint) endpointRegistry.ensureEndpoint(RoleType.DECODE, ipPort, ws);
        endpoint.onWorkerStatusUpdate(ws, new WorkerStatusResponse());
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(ipPort, ws);
        additionalDecodeIpPorts.add(ipPort);
        return endpoint;
    }

    // ==================== Internal: BalanceContext construction ====================

    protected BalanceContext createBalanceContext(String requestId) {
        return createBalanceContext(requestId, 128);
    }

    protected BalanceContext createBalanceContext(String requestId, long seqLen) {
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

    private static byte[] generateInputBytes(String requestId) {
        EngineRpcService.GenerateInputPB input = RequestIdFixtures.write(EngineRpcService.GenerateInputPB.newBuilder(), requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .setGroupTimeout(Int32Value.of(77))
                        .build())
                .build();
        return input.toByteArray();
    }

    private Response successRoute(String requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                serverStatus(RoleType.PREFILL, prefillIp, prefillHttpPort, prefillGrpcPort, requestId),
                serverStatus(RoleType.DECODE, decodeIp, decodeHttpPort, decodeGrpcPort, requestId)
        ));
        return response;
    }

    private static ServerStatus serverStatus(RoleType role, String ip, int httpPort, int grpcPort, String requestId) {
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
