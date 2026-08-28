package org.flexlb.mock;

import com.google.protobuf.ByteString;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.QueueRoutingResult;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.RequestSchedulerTestRuntime;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.InternalRuntimeSettings;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
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

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Base class for mock-worker integration tests.
 *
 * <p>Sets up a real {@link RequestScheduler} backed by a real
 * {@link EngineGrpcClient} that creates real Netty gRPC channels to
 * mock workers.  No Spring Boot context, no model loading, no GPU.
 *
 * <p>Subclasses call {@link #setupWorkers} in {@code @BeforeEach} (or
 * rely on the default), then use {@link #submitRequest(long)} and
 *
 * <p>Architecture:
 * <pre>
 * Real RequestScheduler (test-only composition root)
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
    protected RequestScheduler scheduler;
    protected EndpointRegistry endpointRegistry;
    protected FlexlbConfig config;
    protected ConfigService configService;
    protected Router router;
    protected EngineGrpcClient grpcClient;
    protected DefaultBatchDispatcher dispatcher;
    protected BatchSchedulerReporter reporter;
    protected WorkerDirectory engineWorkerStatus;
    private RequestSchedulerTestRuntime schedulerRuntime;

    private NioEventLoopGroup eventLoopGroup;
    private ThreadPoolExecutor grpcExecutor;

    /** Addressable logical endpoints: one subnet per 254 endpoints. */
    private static final int LOGICAL_WORKER_INDEX_LIMIT = 254 * 254;

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
        configService = new ConfigService() {
            @Override
            public FlexlbConfig loadBalanceConfig() {
                return config;
            }
        };

        // 3. Create gRPC infrastructure
        InternalRuntimeSettings runtime = config.getInternalRuntime();
        eventLoopGroup = new NioEventLoopGroup(runtime.getGrpcClientEventLoopThreads());
        grpcExecutor = new ThreadPoolExecutor(
                runtime.getGrpcClientExecutorThreads(), runtime.getGrpcClientExecutorThreads(),
                60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(runtime.getGrpcClientExecutorQueueCapacity()));

        CustomNameResolver nameResolver = (listener) -> { /* no-op */ };
        GrpcReporter grpcReporter = new GrpcReporter(new NoOpFlexMonitor());
        grpcClient = new EngineGrpcClient(
                nameResolver, grpcExecutor, eventLoopGroup,
                grpcReporter, 1_000);

        // 4. Create real dispatcher
        dispatcher = createDispatcher();

        // 5. Reporter (metrics no-op by default)
        reporter = createBatchSchedulerReporter();

        // 6. Compose the real request lifecycle and endpoint runtime. The
        // router is bound after endpoints and the registry-backed status view
        // exist, matching production's constructor graph without reopening it.
        schedulerRuntime = new RequestSchedulerTestRuntime(
                configService,
                dispatcher,
                reporter,
                createRequestSchedulerReporter(),
                new UnsupportedCancelStub());
        endpointRegistry = schedulerRuntime.endpointRegistry();
        scheduler = schedulerRuntime.scheduler();

        // 7. Engine status is mocked by default; E2E subclasses can use the real registry-backed view.
        engineWorkerStatus = createWorkerDirectory();

        // 8. Build WorkerStatus for prefill and decode mock workers
        WorkerStatus prefillWs = discoveredWorkerStatus(
                RoleType.PREFILL,
                prefillIp,
                prefillHttpPort,
                prefillGrpcPort,
                0L,
                1_000_000L,
                2_000_000L);

        WorkerStatus decodeWs = discoveredWorkerStatus(
                RoleType.DECODE,
                decodeIp,
                decodeHttpPort,
                decodeGrpcPort,
                0L,
                1_000_000L,
                2_000_000L);

        // 9. Register decode endpoint (no scheduler dependency)
        endpointRegistry.registerPreinitializedEndpoint(RoleType.DECODE, decodeIpPort, decodeWs);

        // 10. Register Prefill against the already-composed exact lifecycle.
        endpointRegistry.registerPreinitializedEndpoint(RoleType.PREFILL, prefillIpPort, prefillWs);

        // 11. Fixed routing by default; E2E subclasses can install production routing.
        router = createRouter();
        schedulerRuntime.bindRouter(router);

        // 12. Publish discovered status in the fixture's directory.
        engineWorkerStatus.statusMap(RoleType.PREFILL).clear();
        engineWorkerStatus.statusMap(RoleType.DECODE).clear();
        engineWorkerStatus.statusMap(RoleType.PREFILL).put(prefillIpPort, prefillWs);
        engineWorkerStatus.statusMap(RoleType.DECODE).put(decodeIpPort, decodeWs);
    }

    @AfterEach
    public void tearDownBase() {
        // Stop scheduler-owned work before tearing down its dispatcher or workers.
        if (schedulerRuntime != null) {
            schedulerRuntime.close();
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
            engineWorkerStatus.statusMap(RoleType.PREFILL).remove(ipPort);
        }
        additionalPrefillIpPorts.clear();
        for (String ipPort : additionalDecodeIpPorts) {
            engineWorkerStatus.statusMap(RoleType.DECODE).remove(ipPort);
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
        engineWorkerStatus.statusMap(RoleType.PREFILL).clear();
        engineWorkerStatus.statusMap(RoleType.DECODE).clear();
    }

    // ==================== Override points ====================

    protected final NioEventLoopGroup grpcClientEventLoopGroup() {
        return eventLoopGroup;
    }

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

    protected WorkerDirectory createWorkerDirectory() {
        return new WorkerDirectory(endpointRegistry);
    }

    protected Router createRouter() {
        Router fixedRouter = mock(Router.class);
        when(fixedRouter.routeForQueue(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return schedulerRuntime.admittedRoute(
                    ctx, successRoute(ctx.getRequestId()));
        });
        return fixedRouter;
    }

    /** Build the exact pinned queue admission for a fixture response. */
    protected final QueueRoutingResult admittedRoute(
            BalanceContext context, Response response) {
        return schedulerRuntime.admittedRoute(context, response);
    }

    protected BatchSchedulerReporter createBatchSchedulerReporter() {
        return mock(BatchSchedulerReporter.class);
    }

    protected RequestSchedulerReporter createRequestSchedulerReporter() {
        return mock(RequestSchedulerReporter.class);
    }

    /** Override when an integration fixture needs deterministic dispatcher sizing. */
    protected DefaultBatchDispatcher createDispatcher() {
        return new DefaultBatchDispatcher(grpcClient, configService, null);
    }

    /**
     * Override to customize the FlexlbConfig.
     * Default: BATCH mode, size_max=1, immediate dispatch.
     */
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.fixedWindowDecision().setMaxRequests(1); // single request triggers dispatch
        cfg.fixedWindowDecision().setMaxCollectionWaitMs(300);
        DispatcherConfig dispatcher = assertInstanceOf(
                DispatcherConfig.class, cfg.getDispatcher());
        dispatcher.setEnqueueRpcTimeoutMs(5_000L);
        cfg.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(300_000L);
        return cfg;
    }

    // ==================== Helper: submit ====================

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
        return addPrefillWorker(behavior, 0);
    }

    /** Start an additional Prefill worker on an exact gRPC port. */
    protected MockPrefillWorker addPrefillWorker(
            MockWorkerBehavior behavior,
            int grpcPort) throws IOException {
        MockPrefillWorker worker = new MockPrefillWorker(behavior);
        worker.start(grpcPort);
        int actualGrpcPort = worker.getPort();
        int httpPort = actualGrpcPort - 1;
        String ip = "127.0.0.1";
        String ipPort = ip + ":" + httpPort;

        WorkerStatus ws = discoveredWorkerStatus(
                RoleType.PREFILL,
                ip,
                httpPort,
                actualGrpcPort,
                0L,
                1_000_000L,
                2_000_000L);

        endpointRegistry.registerPreinitializedEndpoint(RoleType.PREFILL, ipPort, ws);
        engineWorkerStatus.statusMap(RoleType.PREFILL).put(ipPort, ws);

        additionalPrefillWorkers.add(worker);
        additionalPrefillIpPorts.add(ipPort);

        log.info("Additional prefill worker started: {} (grpc={})", ipPort, actualGrpcPort);
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
     * Add a Prefill endpoint without starting a server. Frontend-delivery tests only
     * return its route metadata; the Master does not contact Prefill before replying.
     */
    protected PrefillEndpoint addLogicalPrefillEndpoint(int workerIndex) {
        if (workerIndex <= 0 || workerIndex > LOGICAL_WORKER_INDEX_LIMIT) {
            throw new IllegalArgumentException("logical prefill worker index must be in [1, "
                    + LOGICAL_WORKER_INDEX_LIMIT + "]");
        }
        String ip = "198.19." + (workerIndex - 1) / 254 + "."
                + ((workerIndex - 1) % 254 + 1);
        int httpPort = 60_000;
        int grpcPort = httpPort + 1;
        String ipPort = ip + ":" + httpPort;

        WorkerStatus ws = discoveredWorkerStatus(
                RoleType.PREFILL,
                ip,
                httpPort,
                grpcPort,
                workerIndex,
                1_000_000_000L,
                2_000_000_000L);

        PrefillEndpoint endpoint = (PrefillEndpoint) endpointRegistry
                .registerPreinitializedEndpoint(RoleType.PREFILL, ipPort, ws);
        engineWorkerStatus
                .statusMap(RoleType.PREFILL).put(ipPort, ws);
        additionalPrefillIpPorts.add(ipPort);
        return endpoint;
    }

    /**
     * Add a decode endpoint without starting a server. The Master schedule path only
     * selects decode metadata; it does not contact decode before returning the ACK.
     */
    protected DecodeEndpoint addLogicalDecodeEndpoint(int workerIndex) {
        if (workerIndex <= 0 || workerIndex > LOGICAL_WORKER_INDEX_LIMIT) {
            throw new IllegalArgumentException("logical decode worker index must be in [1, "
                    + LOGICAL_WORKER_INDEX_LIMIT + "]");
        }
        // RFC 2544 benchmarking block, spread over subnets so a production-sized
        // decode fleet still gets one distinct address per endpoint.
        String ip = "198.18." + (workerIndex - 1) / 254 + "." + ((workerIndex - 1) % 254 + 1);
        int httpPort = 61_000;
        int grpcPort = httpPort + 1;
        String ipPort = ip + ":" + httpPort;

        WorkerStatus ws = discoveredWorkerStatus(
                RoleType.DECODE,
                ip,
                httpPort,
                grpcPort,
                workerIndex,
                1_000_000_000L,
                2_000_000_000L);

        DecodeEndpoint endpoint = (DecodeEndpoint) endpointRegistry.registerPreinitializedEndpoint(RoleType.DECODE, ipPort, ws);
        engineWorkerStatus.statusMap(RoleType.DECODE).put(ipPort, ws);
        additionalDecodeIpPorts.add(ipPort);
        return endpoint;
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
        ctx.setGenerateInputPb(ByteString.copyFrom(generateInputBytes(requestId)));
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

    /** Apply one strictly newer response through the production transaction. */
    protected final void applyWorkerStatusResponse(
            WorkerStatus status, WorkerStatusResponse response) {
        schedulerRuntime.applyStatus(status, response);
    }

    /** Apply one already immutable gRPC status observation. */
    protected final void applyWorkerStatusObservation(
            WorkerStatus status,
            WorkerStatus.StatusObservation observation) {
        schedulerRuntime.applyStatus(status, observation);
    }

    /** Publish a synthetic Decode capacity observation for integration setup. */
    protected final void publishDecodeCapacity(long available, long total) {
        WorkerStatus status = getDecodeEndpoint().getStatus();
        WorkerStatusResponse response = workerStatusResponse(
                RoleType.DECODE,
                status.getDpRank(),
                available,
                total,
                status.appliedStatusCursor().statusVersion() + 1L);
        schedulerRuntime.applyStatus(status, response);
    }

    private static WorkerStatus discoveredWorkerStatus(
            RoleType role,
            String ip,
            int httpPort,
            int grpcPort,
            long dpRank,
            long availableKv,
            long totalKv) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                role, "test-group", ip, httpPort, grpcPort, null);
        WorkerStatusResponse initial = workerStatusResponse(
                role, dpRank, availableKv, totalKv, 1L);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(initial));
            status.publishPreparedStatus(prepared);
        } finally {
            status.lock.unlock();
        }
        return status;
    }

    private static WorkerStatusResponse workerStatusResponse(
            RoleType role,
            long dpRank,
            long availableKv,
            long totalKv,
            long statusVersion) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setDpRank(dpRank);
        response.setAvailableKvCacheTokens(availableKv);
        response.setTotalKvCacheTokens(totalKv);
        response.setStatusVersion(statusVersion);
        response.setLatestFinishedVersion(0L);
        return response;
    }

    /** Test-local fail-closed cancel transport for fixtures without preemption. */
    private static final class UnsupportedCancelStub
            implements EngineCancelChannel {
        @Override
        public boolean isSupported(DecodeEndpoint endpoint) {
            return false;
        }

        @Override
        public CompletableFuture<CancelOutcome> cancel(
                CancelTarget target, long requestId, long timeoutMs) {
            return CompletableFuture.completedFuture(
                    CancelOutcome.unsupported());
        }
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
