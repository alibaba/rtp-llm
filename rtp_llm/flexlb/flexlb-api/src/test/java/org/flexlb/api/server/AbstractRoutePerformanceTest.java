package org.flexlb.api.server;

import ch.qos.logback.classic.Level;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.api.metrics.ApiMetricsReporter;
import org.flexlb.api.protocol.FlexlbScheduleProtocol;
import org.flexlb.api.protocol.FlexlbServiceGrpc;
import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.VitStrategy;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.CacheMatchSource;
import org.flexlb.cache.VitWorkerCacheManager;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.cache.CacheMatchConfiguration;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.master.MasterStatusView;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.util.metrics.NoOpFlexMonitor;
import org.flexlb.worker.cache.WorkerCacheUpdateResult;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.rpc.protocol.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.mock.env.MockEnvironment;

import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/**
 * Checks 750P/750D RPCs with ample capacity or independently saturated Prefill and Decode roles.
 * Real Netty clients call the production service and routing code; worker completion is simulated.
 */
@Tag("route-performance")
abstract class AbstractRoutePerformanceTest extends FlexLBMockTestBase {
    private FlexlbGrpcServer server;
    private NioEventLoopGroup serverLoops;

    private final List<ManagedChannel> channels = new ArrayList<>();
    private final List<FlexlbServiceGrpc.FlexlbServiceStub> stubs = new ArrayList<>();
    private final ScheduledExecutorService cleanup = Executors.newSingleThreadScheduledExecutor();
    private final AtomicReference<Throwable> cleanupError = new AtomicReference<>();
    private final AtomicInteger activeRoutes = new AtomicInteger();
    private final AtomicInteger peakActiveRoutes = new AtomicInteger();
    private final AtomicLong nextId = new AtomicLong(1000000L);
    private final Set<String> usedPrefill = ConcurrentHashMap.newKeySet();
    private final Set<String> usedDecode = ConcurrentHashMap.newKeySet();
    private final ObjectMapper mapper = new ObjectMapper();
    private final ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg =
                new ConfigService(
                                "{\"requestLifecycle\":{\"inactivityTimeoutMs\": 60000},\"scheduler\":{\"type\":\""
                                        + mode()
                                        + "\"}}")
                        .loadBalanceConfig();
        cfg.getRequestLifecycle().setInactivityTimeoutMs(60000L);
        cfg.getGrpcServer().setExecutorCoreSize(16);
        cfg.getGrpcServer().setExecutorMaxSize(32);
        cfg.getGrpcServer().setExecutorQueueSize(65536);
        cfg.getGrpcServer().setMaxReceivingRequests(8192);
        if (scenario() != Scenario.AVAILABLE) {
            cfg.getRequestLifecycle().setDispatchTimeoutMs(250L);
            if (scenario() == Scenario.PREFILL_LIMITED) {
                cfg.getRouter().getRoles().getPrefill().getAdmission().setMaxInflightRequests(1);
            } else {
                cfg.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            }
        }
        return cfg;
    }

    protected abstract String mode();

    protected Scenario scenario() {
        return Scenario.AVAILABLE;
    }

    @Override
    protected DeliveryMetricsReporter createDeliveryMetricsReporter() {
        return new DeliveryMetricsReporter(new NoOpFlexMonitor());
    }

    @Override
    protected RequestSchedulerReporter createRequestSchedulerReporter() {
        return new RequestSchedulerReporter(new NoOpFlexMonitor());
    }

    @Override
    protected RequestRouteSelector createRouter() {
        ModelMetaConfig meta = mock(ModelMetaConfig.class, withSettings().stubOnly());
        when(meta.requiredRoles()).thenReturn(List.of(RoleType.DECODE, RoleType.PREFILL));
        when(meta.getServiceRoutes()).thenReturn(List.of());
        CacheMetricsReporter cacheMetrics = new CacheMetricsReporter();
        org.springframework.test.util.ReflectionTestUtils.setField(
                cacheMetrics, "monitor", new NoOpFlexMonitor());
        var comparison =
                new org.flexlb.cache.localstandby.LocalStandbyComparisonService(
                        new CacheMatchConfiguration(meta, configService), null, cacheMetrics);
        CacheAwareService cache =
                new CacheAwareService(cacheMetrics, null, comparison, null) {
                    @Override
                    public CacheMatchResult findMatchingEngines(
                            org.flexlb.cache.CacheMatchQuery query) {
                        return cacheMatches(query);
                    }

                    @Override
                    public WorkerCacheUpdateResult updateFromWorkerStatus(
                            org.flexlb.worker.status.WorkerStatus worker) {
                        return null;
                    }

                    @Override
                    public void removeEngineBlockCache(String address) {}
                };
        return new RequestRouteSelector(
                endpointRegistry,
                new org.flexlb.balance.schedule.SchedulerConfiguration()
                        .prefillStrategy(configService, endpointRegistry, cache, cacheMetrics),
                new DecodeStrategy(
                        configService,
                        new org.flexlb.balance.schedule.SchedulerConfiguration()
                                .decodeAdmissionCapacity(configService),
                        endpointRegistry),
                new VitStrategy(mock(VitWorkerCacheManager.class), endpointRegistry),
                new EncoderStrategy(configService, endpointRegistry),
                meta);
    }

    protected CacheMatchResult cacheMatches(org.flexlb.cache.CacheMatchQuery query) {
        return CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC);
    }

    protected int requestPriority(long requestId) {
        return 0;
    }

    protected void prepareTraffic(int qps) {}

    @BeforeAll
    static void quietLogs() {
        ((ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("ROOT"))
                .setLevel(Level.WARN);
        for (String n : List.of("pvLogger", "flexlbLogger", "syncLogger", "org.flexlb"))
            ((ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger(n))
                    .setLevel(Level.WARN);
        ((ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("pvLogger"))
                .setLevel(Level.OFF);
    }

    @BeforeEach
    void startMaster() throws Exception {
        Object theoryReporter =
                org.springframework.test.util.ReflectionTestUtils.getField(scheduler, "cacheTrace");
        org.springframework.test.util.ReflectionTestUtils.setField(
                theoryReporter, "configService", configService);
        publishDecodeCapacity(1_000_000_000L, 2_000_000_000L);
        for (int i = 1; i < 750; i++) {
            addLogicalPrefillEndpoint(i);
            addLogicalDecodeEndpoint(i);
        }
        assertEquals(750, endpointRegistry.counts(RoleType.PREFILL).published());
        assertEquals(750, endpointRegistry.counts(RoleType.DECODE).published());
        MasterStatusView standalone =
                new MasterStatusView() {
                    public boolean isNeedConsistency() {
                        return false;
                    }

                    public boolean isMaster() {
                        return false;
                    }
                };
        var service =
                new FlexlbServiceImpl(
                        new org.flexlb.balance.request.RequestRoutingService(
                                scheduler,
                                requestRegistry(),
                                (org.flexlb.balance.strategy.RequestRouteSelector)
                                        org.springframework.test.util.ReflectionTestUtils.getField(
                                                scheduler, "router"),
                                new CacheAwareService(null, null, null, null) {
                                    @Override
                                    public void updateFromRoutedRequest(
                                            String requestId,
                                            long blockSize,
                                            List<Long> cacheKeys,
                                            List<org.flexlb.worker.status.ServerStatus> workers) {
                                        // Cache persistence is outside this routing benchmark.
                                    }
                                }),
                        config,
                        requestRegistry(),
                        standalone,
                        new ApiMetricsReporter(new NoOpFlexMonitor()),
                        mock(FlexlbGrpcForwarder.class),
                        recorder,
                        createRequestSchedulerReporter());
        int port;
        try (ServerSocket socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        serverLoops = new NioEventLoopGroup(4);
        server =
                new FlexlbGrpcServer(
                        service,
                        configService,
                        new MockEnvironment()
                                .withProperty("server.port", Integer.toString(port - 2)),
                        serverLoops,
                        null,
                        new GrpcServerTimingInterceptor(),
                        new GrpcRequestHeadersInterceptor());
        server.start();
        for (int i = 0; i < 4; i++) {
            ManagedChannel channel =
                    NettyChannelBuilder.forAddress("127.0.0.1", port).usePlaintext().build();
            channels.add(channel);
            stubs.add(FlexlbServiceGrpc.newStub(channel));
        }
        System.out.printf(
                "E2E_ENV mode=%s workers=750P/750D java=%s cpus=%d heap=%d%n",
                mode(),
                System.getProperty("java.version"),
                Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().maxMemory());
    }

    @AfterEach
    void stopMaster() throws Exception {
        for (ManagedChannel channel : channels) channel.shutdownNow();
        for (ManagedChannel channel : channels) channel.awaitTermination(5, TimeUnit.SECONDS);
        if (server != null) server.shutdown();
        if (serverLoops != null)
            serverLoops.shutdownGracefully(0, 1, TimeUnit.SECONDS).syncUninterruptibly();
        cleanup.shutdown();
        cleanup.awaitTermination(30, TimeUnit.SECONDS);
        assertNull(cleanupError.get(), "simulated worker completion failed");
    }

    @ParameterizedTest
    @ValueSource(ints = {3000, 10000})
    @Timeout(90)
    void routeRpcAt750PrefillAndDecode(int qps) throws Exception {
        java.nio.file.Files.deleteIfExists(resultPath(qps));
        for (ManagedChannel channel : channels) {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (channel.getState(true) != ConnectivityState.READY) {
                var state = channel.getState(true);
                if (state == ConnectivityState.READY) break;
                var changed = new CompletableFuture<Void>();
                channel.notifyWhenStateChanged(state, () -> changed.complete(null));
                changed.get(Math.max(1, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
            }
        }
        assertEquals(0, traffic(64, 1, false), "priming requests must be admitted");
        awaitWorkerCompletions();
        traffic(qps, 10, false);
        awaitWorkerCompletions();
        recorder.reset();
        usedPrefill.clear();
        usedDecode.clear();
        peakActiveRoutes.set(0);
        assertAll(
                () -> traffic(qps, 10, true),
                () -> {
                    awaitWorkerCompletions();
                    if (scenario() != Scenario.AVAILABLE) {
                        assertEquals(
                                0, traffic(64, 1, false), "capacity must recover after overload");
                        awaitWorkerCompletions();
                    }
                    awaitEmptyResourceLedgers();
                    System.out.printf(
                            "E2E_RECOVERY mode=%s scenario=%s qps=%d ledgers_empty=true%n",
                            mode(), scenario(), qps);
                });
    }

    private int traffic(int qps, int seconds, boolean measured) throws Exception {
        prepareTraffic(qps);
        int count = qps * seconds;
        long[] rt = new long[count], lag = new long[count], plannedRt = new long[count];
        CountDownLatch done = new CountDownLatch(count);
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger rejected = new AtomicInteger();
        Map<Integer, AtomicInteger> rejectionCodes = new ConcurrentHashMap<>();
        AtomicReference<String> firstError = new AtomicReference<>();
        AtomicLong lastCompletion = new AtomicLong();
        List<Integer> tokens = new ArrayList<>(1024);
        for (int t = 0; t < 1024; t++) tokens.add(t + 101);
        var resourceSampler = new ResourceSampler();
        if (measured) resourceSampler.start();
        try {
            long phaseStartMs = System.currentTimeMillis();
            long base = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(100);
            long firstIssued = 0L, lastIssued = 0L;
            for (int i = 0; i < count; i++) {
                long due = base + (long) i * 1000000000L / qps;
                for (long left; (left = due - System.nanoTime()) > 0; ) {
                    if (left > 100000) LockSupport.parkNanos(left - 50000);
                    else Thread.onSpinWait();
                }
                long id = nextId.incrementAndGet();
                int index = i;
                var input =
                        EngineRpcService.GenerateInputPB.newBuilder()
                                .setRequestId(id)
                                .addAllTokenIds(tokens)
                                .setGenerateConfig(
                                        EngineRpcService.GenerateConfigPB.newBuilder()
                                                .setMaxNewTokens(8))
                                .build();
                var request =
                        FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                                .setRequestId(id)
                                .setGenerateInput(input.toByteString())
                                .setSeqLen(1024)
                                .setMaxNewTokens(8)
                                .setNumBeams(1)
                                .setModel("mock-model")
                                .setPriority(requestPriority(id))
                                .setRequestTimeMs(System.currentTimeMillis())
                                .setGenerateTimeout(60000L)
                                .build();
                long sent = System.nanoTime();
                if (i == 0) firstIssued = sent;
                lastIssued = sent;
                lag[i] = sent - due;
                stubs.get(i % stubs.size())
                        .withDeadlineAfter(20, TimeUnit.SECONDS)
                        .schedule(
                                request,
                                new StreamObserver<>() {
                                    public void onNext(
                                            FlexlbScheduleProtocol.FlexlbScheduleResponsePB
                                                    response) {
                                        long now = System.nanoTime();
                                        rt[index] = now - sent;
                                        plannedRt[index] = now - due;
                                        lastCompletion.accumulateAndGet(now, Math::max);
                                        if (!response.getSuccess()
                                                && expectedCapacityFailure(response)) {
                                            rejected.incrementAndGet();
                                            rejectionCodes
                                                    .computeIfAbsent(
                                                            response.getCode(),
                                                            ignored -> new AtomicInteger())
                                                    .incrementAndGet();
                                        } else if (!response.getSuccess()) {
                                            failures.incrementAndGet();
                                            firstError.compareAndSet(
                                                    null,
                                                    response.getCode()
                                                            + " "
                                                            + response.getErrorMessage());
                                        } else {
                                            peakActiveRoutes.accumulateAndGet(
                                                    activeRoutes.incrementAndGet(), Math::max);
                                            cleanup.schedule(
                                                    () -> completeWorkerRequest(id),
                                                    completionDelayMs(),
                                                    TimeUnit.MILLISECONDS);
                                        }
                                    }

                                    public void onError(Throwable error) {
                                        failures.incrementAndGet();
                                        firstError.compareAndSet(null, error.toString());
                                        lastCompletion.accumulateAndGet(
                                                System.nanoTime(), Math::max);
                                        done.countDown();
                                    }

                                    public void onCompleted() {
                                        done.countDown();
                                    }
                                });
            }
            assertTrue(done.await(30, TimeUnit.SECONDS), "RPC completion timeout");
            if (measured) {
                resourceSampler.close();
                awaitWorkerCompletions();
                Map<String, Object> result = new LinkedHashMap<>();
                result.put("pid", ProcessHandle.current().pid());
                result.put("start_epoch_ms", phaseStartMs);
                result.put("end_epoch_ms", System.currentTimeMillis());
                result.put("mode", mode());
                result.put("scenario", scenario().name());
                result.put(
                        "prefill_selection",
                        config.getRouter().getRoles().getPrefill().getSelection().getType().name());
                result.put(
                        "request_order",
                        config.getScheduler().getOrdering().getPolicy().getType().name());
                result.put("priority_enabled", config.isPriorityOrdering());
                result.put("rejections", rejected.get());
                result.put("rejection_codes", rejectionCodes);
                result.put("completion_delay_ms", completionDelayMs());
                result.put(
                        "prefill_max_inflight_requests",
                        config.getRouter()
                                .getRoles()
                                .getPrefill()
                                .getAdmission()
                                .getMaxInflightRequests());
                result.put(
                        "decode_max_requests",
                        config.getRouter()
                                .getRoles()
                                .getDecode()
                                .getAvailability()
                                .getMaxEngineRequests());
                result.put("peak_active_routes", peakActiveRoutes.get());
                double elapsedSeconds = (lastCompletion.get() - base) / 1e9;
                result.put(
                        "successful_qps",
                        (count - failures.get() - rejected.get()) / elapsedSeconds);
                result.put("target_qps", qps);
                result.put("requests", count);
                result.put("errors", failures.get());
                result.put("first_error", firstError.get());
                result.put(
                        "completed_qps",
                        (count - failures.get()) / ((lastCompletion.get() - base) / 1e9));
                result.put("issued_qps", (count - 1) / ((lastIssued - firstIssued) / 1e9));
                var latency = stats(rt);
                result.put("client_rpc_ms", latency);
                result.put("planned_to_response_ms", stats(plannedRt));
                result.put("client_issue_lag_ms", stats(lag));
                result.put("server", recorder.snapshot());
                result.put("selected_prefill_count", usedPrefill.size());
                result.put("selected_decode_count", usedDecode.size());
                result.put("resources", resourceSampler.summary());
                java.nio.file.Path output = resultPath(qps);
                java.nio.file.Files.createDirectories(output.getParent());
                mapper.writerWithDefaultPrettyPrinter().writeValue(output.toFile(), result);
                System.out.println("E2E_RESULT " + mapper.writeValueAsString(result));
                var limitResource =
                        getClass().getResourceAsStream("/route-performance-limits.json");
                assertNotNull(limitResource, "missing performance limits");
                com.fasterxml.jackson.databind.JsonNode limits;
                try (limitResource) {
                    var allLimits = mapper.readTree(limitResource);
                    limits =
                            scenario() == Scenario.AVAILABLE
                                    ? allLimits.path(Integer.toString(qps))
                                    : allLimits
                                            .path("pressure")
                                            .path(mode())
                                            .path(Integer.toString(qps));
                }
                assertFalse(limits.isMissingNode(), "missing limits for target QPS " + qps);
                for (String metric :
                        List.of(
                                "minQpsRatio",
                                "p50Ms",
                                "avgMs",
                                "cpuAvgPercent",
                                "cpuPeakPercent",
                                "rssPeakBytes")) {
                    assertTrue(
                            limits.path(metric).asDouble() > 0,
                            "missing or nonpositive limit " + metric);
                }
                assertAll(
                        () -> assertEquals(0, failures.get(), firstError.get()),
                        () ->
                                assertTrue(
                                        (double) result.get("completed_qps")
                                                >= qps * limits.path("minQpsRatio").asDouble(),
                                        "completed QPS below configured target ratio"),
                        () ->
                                assertTrue(
                                        (double) latency.get("p50")
                                                <= limits.path("p50Ms").asDouble(),
                                        "client p50 RT expected <= "
                                                + limits.path("p50Ms").asDouble()
                                                + " ms, actual "
                                                + latency.get("p50")
                                                + " ms"),
                        () ->
                                assertTrue(
                                        (double) latency.get("avg")
                                                <= limits.path("avgMs").asDouble(),
                                        "client average RT expected <= "
                                                + limits.path("avgMs").asDouble()
                                                + " ms, actual "
                                                + latency.get("avg")
                                                + " ms"),
                        () -> assertEquals(750, usedPrefill.size(), "Prefill selection coverage"),
                        () -> assertEquals(750, usedDecode.size(), "Decode selection coverage"),
                        () ->
                                assertEquals(
                                        count,
                                        ((Number) recorder.snapshot().get("completion_count"))
                                                .intValue(),
                                        "server completion coverage"),
                        () -> resourceSampler.check(limits),
                        () -> {
                            if (scenario() != Scenario.AVAILABLE) {
                                assertAll(
                                        () ->
                                                assertTrue(
                                                        rejected.get()
                                                                >= count
                                                                        * limits.path(
                                                                                        "minRejectionRatio")
                                                                                .asDouble(),
                                                        "fixture did not create sustained capacity shortage"),
                                        () ->
                                                assertTrue(
                                                        (double) result.get("successful_qps")
                                                                >= limits.path("minSuccessQps")
                                                                        .asDouble(),
                                                        "successful QPS expected >= "
                                                                + limits.path("minSuccessQps")
                                                                        .asDouble()
                                                                + ", actual "
                                                                + result.get("successful_qps")),
                                        () ->
                                                assertTrue(
                                                        (double) result.get("successful_qps")
                                                                <= limits.path("maxSuccessQps")
                                                                        .asDouble(),
                                                        "successful QPS expected <= "
                                                                + limits.path("maxSuccessQps")
                                                                        .asDouble()
                                                                + ", actual "
                                                                + result.get("successful_qps")),
                                        () ->
                                                assertTrue(
                                                        peakActiveRoutes.get() >= 675,
                                                        "overload must fill the finite 750-slot role"));
                            }
                        });
            }
            assertEquals(0, failures.get(), firstError.get());
            return rejected.get();
        } finally {
            resourceSampler.close();
        }
    }

    private boolean expectedCapacityFailure(
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
        int code = response.getCode();
        if (scenario() == Scenario.AVAILABLE) return false;
        if (mode().equals("QUEUE")) return code == StrategyErrorType.QUEUE_TIMEOUT.getErrorCode();
        if (scenario() == Scenario.DECODE_LIMITED
                && code == StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode()) {
            return response.getAdmissionRejectReason()
                    == FlexlbScheduleProtocol.ScheduleFailureReasonPB.SAME_PRIORITY_AHEAD;
        }
        int roleError =
                scenario() == Scenario.PREFILL_LIMITED
                        ? StrategyErrorType.NO_PREFILL_WORKER.getErrorCode()
                        : StrategyErrorType.NO_DECODE_WORKER.getErrorCode();
        return code == roleError || code == StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode();
    }

    private void completeWorkerRequest(long id) {
        try {
            RequestContext ctx = requestRegistry().findActive(Long.toString(id));
            assertNotNull(ctx, "successful routing must retain its request until completion");
            var route = ctx.snapshot().route();
            assertNotNull(route);
            if (scenario() == Scenario.PREFILL_LIMITED) {
                assertTrue(
                        route.prefillEndpoint().selectionSnapshot().outstandingRequests() <= 1,
                        "Prefill capacity was exceeded");
            } else if (scenario() == Scenario.DECODE_LIMITED) {
                assertTrue(
                        route.decodeEndpoint().selectionSnapshot().usage().occupiedRequests() <= 1,
                        "Decode capacity was exceeded");
            }
            usedPrefill.add(route.prefillEndpoint().ipPort());
            usedDecode.add(route.decodeEndpoint().ipPort());
            publishWorkerCompletion(route.prefillEndpoint(), Long.toString(id));
            publishWorkerCompletion(route.decodeEndpoint(), Long.toString(id));
        } catch (Throwable error) {
            cleanupError.compareAndSet(null, error);
        } finally {
            activeRoutes.decrementAndGet();
        }
    }

    private void publishWorkerCompletion(
            org.flexlb.worker.endpoint.WorkerEndpoint endpoint, String requestId) {
        var status = endpoint.getStatus();
        var cursor = status.appliedStatusCursor();
        var task =
                new org.flexlb.worker.status.WorkerStatus.TaskObservation(
                        requestId,
                        0L,
                        0L,
                        1024L,
                        0L,
                        0L,
                        System.currentTimeMillis(),
                        status.getDpRank(),
                        0L,
                        "",
                        null,
                        0L,
                        org.flexlb.worker.status.PriorityPreemptionProgress.NONE,
                        null);
        applyWorkerStatusObservation(
                status,
                status.bindStatusObservation(
                        status.committedEngineObservation(),
                        true,
                        cursor.statusVersion() + 1L,
                        cursor.latestFinishedTaskVersion() + 1L,
                        Map.of(requestId, task)));
    }

    private long completionDelayMs() {
        return scenario() == Scenario.AVAILABLE ? 0L : 1000L;
    }

    private void awaitWorkerCompletions() throws Exception {
        cleanup.schedule(() -> {}, completionDelayMs(), TimeUnit.MILLISECONDS)
                .get(30, TimeUnit.SECONDS);
        assertNull(cleanupError.get(), "simulated worker completion failed");
    }

    private Map<String, Object> stats(long[] values) {
        long[] sorted = values.clone();
        Arrays.sort(sorted);
        double avg = Arrays.stream(sorted).average().orElse(0) / 1e6;
        return Map.of(
                "avg",
                avg,
                "p50",
                sorted[(int) Math.ceil(sorted.length * .5) - 1] / 1e6,
                "p99",
                sorted[(int) Math.ceil(sorted.length * .99) - 1] / 1e6,
                "max",
                sorted[sorted.length - 1] / 1e6);
    }

    private java.nio.file.Path resultPath(int qps) {
        String policy = config.getRouter().getRoles().getPrefill().getSelection().getType().name();
        String order = config.getScheduler().getOrdering().getPolicy().getType().name();
        String suffix =
                policy.equals("COST_BASED") && order.equals("FIFO")
                        ? ""
                        : "-" + policy + "-" + order;
        return java.nio.file.Path.of(
                "target",
                "route-performance",
                mode() + "-" + scenario() + suffix + "-" + qps + ".json");
    }

    private void awaitEmptyResourceLedgers() {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (true) {
            boolean prefillEmpty =
                    endpointRegistry
                            .<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                    RoleType.PREFILL, null)
                            .stream()
                            .allMatch(
                                    endpoint -> endpoint.selectionSnapshot().computeTokens() == 0);
            boolean decodeEmpty =
                    endpointRegistry
                            .<org.flexlb.worker.endpoint.DecodeEndpoint>endpoints(
                                    RoleType.DECODE, null)
                            .stream()
                            .allMatch(
                                    endpoint ->
                                            endpoint.selectionSnapshot().usage().occupiedRequests()
                                                    == 0);
            if (prefillEmpty && decodeEmpty) return;
            assertTrue(
                    System.nanoTime() < deadline,
                    "worker resources did not return to zero after traffic");
            LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(1));
        }
    }

    protected enum Scenario {
        AVAILABLE,
        PREFILL_LIMITED,
        DECODE_LIMITED
    }

    /**
     * Samples process CPU and resident memory during one measured traffic window.
     * CPU 100 percent represents one fully occupied logical processor.
     */
    private static final class ResourceSampler implements AutoCloseable {
        private final com.sun.management.OperatingSystemMXBean operatingSystem =
                (com.sun.management.OperatingSystemMXBean)
                        java.lang.management.ManagementFactory.getOperatingSystemMXBean();
        private final ScheduledExecutorService sampler =
                Executors.newSingleThreadScheduledExecutor();
        private final List<Double> cpu = new ArrayList<>();

        private long lastTime;
        private long lastCpu;
        private long rssPeakBytes;

        private Throwable failure;

        @Override
        public void close() throws InterruptedException {
            sampler.shutdown();
            assertTrue(
                    sampler.awaitTermination(5, TimeUnit.SECONDS), "resource sampler did not stop");
        }

        void start() {
            lastTime = System.nanoTime();
            lastCpu = operatingSystem.getProcessCpuTime();
            assertTrue(lastCpu >= 0, "process CPU time is unavailable");
            sampler.scheduleAtFixedRate(this::sample, 1, 1, TimeUnit.SECONDS);
        }

        void check(com.fasterxml.jackson.databind.JsonNode limits) {
            assertNull(failure, "process resource sampling failed");
            assertTrue(cpu.size() >= 8, "insufficient resource samples");
            assertTrue(
                    (double) summary().get("cpu_avg_percent") > 0
                            && (double) summary().get("cpu_avg_percent")
                                    <= limits.path("cpuAvgPercent").asDouble(),
                    "average process CPU exceeds " + limits.path("cpuAvgPercent").asDouble() + "%");
            assertTrue(
                    (double) summary().get("cpu_peak_percent")
                            <= limits.path("cpuPeakPercent").asDouble(),
                    "peak process CPU exceeds limit");
            assertTrue(
                    rssPeakBytes > 0 && rssPeakBytes <= limits.path("rssPeakBytes").asLong(),
                    "process RSS exceeds configured limit or is unavailable");
        }

        Map<String, Object> summary() {
            return Map.of(
                    "samples",
                    cpu.size(),
                    "cpu_avg_percent",
                    cpu.stream().mapToDouble(Double::doubleValue).average().orElse(0),
                    "cpu_peak_percent",
                    cpu.stream().mapToDouble(Double::doubleValue).max().orElse(0),
                    "rss_peak_bytes",
                    rssPeakBytes);
        }

        private void sample() {
            try {
                long now = System.nanoTime();
                long processCpu = operatingSystem.getProcessCpuTime();
                cpu.add(100.0 * (processCpu - lastCpu) / (now - lastTime));
                lastTime = now;
                lastCpu = processCpu;
                rssPeakBytes = Math.max(rssPeakBytes, residentBytes());
            } catch (Throwable error) {
                failure = error;
            }
        }

        private long residentBytes() throws java.io.IOException, InterruptedException {
            if (operatingSystem.getName().startsWith("Mac")) {
                var process =
                        new ProcessBuilder(
                                        "/bin/ps",
                                        "-o",
                                        "rss=",
                                        "-p",
                                        Long.toString(ProcessHandle.current().pid()))
                                .start();
                try {
                    if (!process.waitFor(1, TimeUnit.SECONDS) || process.exitValue() != 0)
                        throw new java.io.IOException("ps RSS sampling failed");
                    String rss =
                            new String(
                                            process.getInputStream().readAllBytes(),
                                            java.nio.charset.StandardCharsets.UTF_8)
                                    .trim();
                    return Math.multiplyExact(Long.parseLong(rss), 1024L);
                } finally {
                    if (process.isAlive()) process.destroyForcibly();
                }
            }
            String rss =
                    java.nio.file.Files.readAllLines(java.nio.file.Path.of("/proc/self/status"))
                            .stream()
                            .filter(line -> line.startsWith("VmRSS:"))
                            .findFirst()
                            .orElseThrow();
            return Math.multiplyExact(Long.parseLong(rss.split("\\s+")[1]), 1024L);
        }
    }
}
