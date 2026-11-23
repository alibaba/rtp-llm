package org.flexlb.service;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService.CacheStatusPB;
import org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalCacheStatusPB;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.flexlb.service.config.merger.FlexlbConfigMerger;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Checks ViT cache residency ranking and invalidation when worker generations change.
 */
class VitCacheDirectoryTest {
    private VitCacheDirectory directory;
    private VitCacheSelector selector;

    private final Map<String, WorkerStatus> live = new HashMap<>();

    private WorkerStatus a;
    private WorkerStatus b;
    private EngineGrpcClient grpc;
    private LBStatusConsistencyService consistency;
    private WorkerDirectory workers;

    @BeforeEach
    void setUp() {
        a = worker("10.0.0.1", "one");
        b = worker("10.0.0.2", "two");
        live.put(a.getLogicalIpPort(), a);
        live.put(b.getLogicalIpPort(), b);
        workers = mock(WorkerDirectory.class);
        when(workers.vitRoutingSnapshot()).thenAnswer(call -> Map.copyOf(live));
        when(workers.captureEndpoint(eq(RoleType.VIT), any()))
                .thenAnswer(
                        call -> {
                            WorkerStatus status = live.get(call.getArgument(1));
                            return status == null
                                    ? null
                                    : new WorkerEndpoint(status).tryPinGeneration();
                        });
        grpc = mock(EngineGrpcClient.class);
        consistency = mock(LBStatusConsistencyService.class);
        directory = new VitCacheDirectory(workers, grpc, consistency);
        selector = new VitCacheSelector(directory, workers);
    }

    @AfterEach
    void tearDown() {
        directory.stop();
    }

    @Test
    void vitOnlyWithoutAffinityKeysSelectsByLoadAndRevalidatesTheWorker() {
        a.recordVitLoad(OptionalLong.of(8), a.vitRoutedInputs(), System.nanoTime());
        b.recordVitLoad(OptionalLong.of(1), b.vitRoutedInputs(), System.nanoTime());
        var route = context();
        assertFalse(selector.select(route, null).isSuccess());

        route.getRequest().setVitRouteOnly(true);
        var cacheDirectory = mock(VitCacheDirectory.class);
        var loadSelector = new VitCacheSelector(cacheDirectory, workers);
        var selected = loadSelector.select(route, null);
        assertTrue(selected.isSuccess());
        assertEquals(b.getLogicalIpPort(), selected.getLogicalIpPort());
        assertEquals(a.getLogicalIpPort(), loadSelector.select(route, "one").getLogicalIpPort());
        assertEquals(0L, b.vitRoutedInputs());
        verifyNoInteractions(cacheDirectory);

        route.getRequest().setSelectedVit(selected);
        assertTrue(loadSelector.validate(route, "two").isSuccess());
    }

    @Test
    void loadOnlyModeIgnoresAffinityAndChargesEachMediaInputOnce() {
        var config =
                FlexlbConfigMerger.merge(
                        SchedulingTestConfig.newConfig(),
                        """
                {"router":{"roles":{"vit":{"cacheAffinityEnabled":false}}}}
                """,
                        "FLEXLB_CONFIG");
        snapshot(a, "a1", "image");
        a.recordVitLoad(OptionalLong.of(8), a.vitRoutedInputs(), System.nanoTime());
        b.recordVitLoad(OptionalLong.of(1), b.vitRoutedInputs(), System.nanoTime());
        assertEquals(
                a.getLogicalIpPort(), selector.select(context("image"), null).getLogicalIpPort());

        var route = context(config, "image", "image");
        route.getRequest().setVitRouteOnly(true);
        var selected = selector.select(route, null);
        assertEquals(b.getLogicalIpPort(), selected.getLogicalIpPort());
        assertEquals(2L, b.vitRoutedInputs());
        assertEquals(0L, a.vitRoutedInputs());

        route.getRequest().setSelectedVit(selected);
        assertTrue(selector.validate(route, "two").isSuccess());
        assertEquals(2L, b.vitRoutedInputs());
    }

    @Test
    void loadOnlyModeRanksAgainAfterLosingTheAssignmentRace() throws Exception {
        var config = new FlexlbConfig();
        config.getRouter().getRoles().getVit().setCacheAffinityEnabled(false);
        a.recordVitLoad(OptionalLong.of(0), 0L, System.nanoTime());
        b.recordVitLoad(OptionalLong.of(1), 0L, System.nanoTime());
        CountDownLatch bothSelectedA = new CountDownLatch(2);
        var endpoint = new WorkerEndpoint(a);
        when(workers.captureEndpoint(RoleType.VIT, a.getLogicalIpPort()))
                .thenAnswer(
                        call -> {
                            bothSelectedA.countDown();
                            assertTrue(bothSelectedA.await(5, TimeUnit.SECONDS));
                            return endpoint.tryPinGeneration();
                        });

        try (var executor = Executors.newFixedThreadPool(2)) {
            var first = context(config, "image", "image");
            first.getRequest().setVitRouteOnly(true);
            var second = context(config, "image", "image");
            second.getRequest().setVitRouteOnly(true);
            var one = executor.submit(() -> selector.select(first, null).getLogicalIpPort());
            var two = executor.submit(() -> selector.select(second, null).getLogicalIpPort());
            assertEquals(
                    Set.of(a.getLogicalIpPort(), b.getLogicalIpPort()),
                    Set.of(one.get(6, TimeUnit.SECONDS), two.get(6, TimeUnit.SECONDS)));
        }
        assertEquals(2L, a.vitRoutedInputs());
        assertEquals(2L, b.vitRoutedInputs());
        verify(workers, times(2)).captureEndpoint(RoleType.VIT, a.getLogicalIpPort());
        verify(workers).captureEndpoint(RoleType.VIT, b.getLogicalIpPort());
    }

    @Test
    void vitOnlyRoutesAccountForMediaAssignedSinceThePollStarted() {
        snapshot(a, "a1");
        snapshot(b, "b1");
        long aPollCursor = a.vitRoutedInputs();
        a.recordVitLoad(OptionalLong.of(0), aPollCursor, System.nanoTime());
        b.recordVitLoad(OptionalLong.of(1), b.vitRoutedInputs(), System.nanoTime());

        var route = context("cold", "cold");
        route.getRequest().setVitRouteOnly(true);
        assertEquals(a.getIp(), selector.select(route, null).getServerIp());
        assertEquals(2L, a.vitRoutedInputs());

        // A cold key still follows its pending placement while an unrelated key
        // follows current load. A poll started before assignment keeps the charge.
        a.recordVitLoad(OptionalLong.of(0), aPollCursor, System.nanoTime());
        assertEquals(a.getIp(), selector.select(context("cold"), null).getServerIp());
        assertEquals(b.getIp(), selector.select(context("independent"), null).getServerIp());
        a.recordVitLoad(OptionalLong.of(0), a.vitRoutedInputs(), System.nanoTime());
        assertEquals(a.getIp(), selector.select(context("another"), null).getServerIp());
    }

    @Test
    void selectsAndRevalidatesAnExactLogicalEngineAtASharedAddress() {
        var first =
                WorkerStatus.createDiscovered(
                        RoleType.VIT, "one", a.getIp(), 8000, 8001, "", null, 0, 2);
        var second =
                WorkerStatus.createDiscovered(
                        RoleType.VIT, "one", a.getIp(), 8000, 8001, "", null, 1, 2);
        live.clear();
        live.put(first.getLogicalIpPort(), first);
        live.put(second.getLogicalIpPort(), second);
        snapshot(first, "first");
        snapshot(second, "second", "image");
        var request = context("image");
        var selected = selector.select(request, "one");
        assertEquals(1, selected.getEngineIndex());
        assertEquals(second.getGenerationId(), selected.getWorkerGeneration());
        request.getRequest().setSelectedVit(selected);
        assertTrue(selector.validate(request, "one").isSuccess());

        selected.setEngineIndex(0);
        assertFalse(selector.validate(request, "one").isSuccess());
    }

    @Test
    void capturesOnlyTheWinningGenerationAndRejectsAReplacementBeforeAccounting() {
        snapshot(a, "a1", "image");
        snapshot(b, "b1");
        var route = context("image");
        route.getRequest().setVitRouteOnly(true);
        var replacement = worker(a.getIp(), "one");
        when(workers.captureEndpoint(RoleType.VIT, a.getLogicalIpPort()))
                .thenReturn(new WorkerEndpoint(replacement).tryPinGeneration());
        assertFalse(selector.select(route, null).isSuccess());
        assertEquals(0L, a.vitRoutedInputs());
        assertEquals(0L, replacement.vitRoutedInputs());
        verify(workers).captureEndpoint(RoleType.VIT, a.getLogicalIpPort());
        verify(workers, org.mockito.Mockito.never())
                .captureEndpoint(RoleType.VIT, b.getLogicalIpPort());
    }

    @Test
    void workerRestartInvalidatesColdKeyPlacements() {
        snapshot(a, "a1");
        snapshot(b, "b1");
        selector.select(context("cold1", "cold2"), "one");
        selector.select(context("other"), "two");
        snapshot(a, "a2");
        assertEquals(
                b.getIp(), selector.select(context("cold1", "cold2", "other"), null).getServerIp());
    }

    @Test
    void prefersGpuThenCpuThenHashOnlyAndReplacesResidency() {
        List<String> keys = List.of("image");
        tierSnapshot(a, keys, keys, List.of());
        tierSnapshot(b, keys, List.of(), keys);
        assertEquals(a.getIp(), selector.select(context("image"), null).getServerIp());
        tierSnapshot(a, keys, List.of(), List.of());
        assertEquals(b.getIp(), selector.select(context("image"), null).getServerIp());
        // A snapshot without the optional tier fields clears previously advertised residency.
        snapshot(b, "instance", "image");
        tierSnapshot(a, keys, List.of(), keys);
        assertEquals(a.getIp(), selector.select(context("image"), null).getServerIp());
    }

    @Test
    void hashCoveragePrecedesEmbeddingCoverageAndEmbeddingCoveragePrecedesTier() {
        List<String> keys = List.of("image1", "image2");
        tierSnapshot(a, keys, List.of(), List.of());
        tierSnapshot(b, List.of("image1"), keys, List.of());
        assertEquals(a.getIp(), selector.select(context("image1", "image2"), null).getServerIp());
        tierSnapshot(a, keys, List.of(), keys);
        tierSnapshot(b, keys, List.of("image1"), List.of());
        assertEquals(a.getIp(), selector.select(context("image1", "image2"), null).getServerIp());
        tierSnapshot(b, keys, List.of("image1"), List.of("image2"));
        assertEquals(b.getIp(), selector.select(context("image1", "image2"), null).getServerIp());
    }

    @Test
    void embeddingWithoutHashBeatsPendingPlacementButRespectsGroupAndHealth() {
        String selected = selector.select(context("cold"), null).getServerIp();
        WorkerStatus other = selected.equals(a.getIp()) ? b : a;
        WorkerStatus pendingWorker = selected.equals(a.getIp()) ? a : b;
        tierSnapshot(other, List.of(), List.of(), List.of("cold"));
        assertEquals(other.getIp(), selector.select(context("cold"), null).getServerIp());
        assertEquals(
                pendingWorker.getIp(),
                selector.select(context("cold"), pendingWorker.getGroup()).getServerIp());
        live.remove(other.getLogicalIpPort());
        assertEquals(pendingWorker.getIp(), selector.select(context("cold"), null).getServerIp());
    }

    @Test
    void rejectsInvalidTierSnapshotsWithoutDiscardingKnownResidency() {
        tierSnapshot(a, List.of(), List.of("image"), List.of());
        tierSnapshot(b, List.of(), List.of(), List.of("image"));
        tierSnapshot(a, List.of(), List.of("image"), List.of("image"));
        assertEquals(a.getIp(), selector.select(context("image"), null).getServerIp());
        tierSnapshot(a, List.of(), List.of(), List.of(""));
        assertEquals(a.getIp(), selector.select(context("image"), null).getServerIp());
        List<String> manyKeys = IntStream.range(0, 100_000).mapToObj(i -> "key" + i).toList();
        tierSnapshot(a, manyKeys, List.of(), List.of("extra"));
        assertEquals(a.getIp(), selector.select(context("image"), null).getServerIp());
    }

    @Test
    void refreshUsesGrpcPortAndPreservesTierFields() {
        var snapshot =
                MultimodalCacheStatusPB.newBuilder()
                        .setWorkerInstance("a1")
                        .addKeys("hash")
                        .addGpuEmbeddingKeys("gpu")
                        .addCpuEmbeddingKeys("cpu")
                        .build();
        var request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
        when(grpc.getMultimodalCacheStatusAsync(a.getIp(), 8001, request, 2000))
                .thenReturn(
                        CompletableFuture.completedFuture(
                                CacheStatusPB.newBuilder().setMultimodalCache(snapshot).build()));
        when(grpc.getMultimodalCacheStatusAsync(b.getIp(), 8001, request, 2000))
                .thenReturn(CompletableFuture.completedFuture(CacheStatusPB.getDefaultInstance()));
        directory.refresh();
        directory.refresh();
        verify(grpc, times(1)).getMultimodalCacheStatusAsync(a.getIp(), 8001, request, 2000);
        verify(grpc, times(1)).getMultimodalCacheStatusAsync(b.getIp(), 8001, request, 2000);
        for (String key : List.of("hash", "gpu", "cpu")) {
            assertEquals(a.getIp(), selector.select(context(key), null).getServerIp());
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void failedOrUnsupportedWorkersRetryOnlyAfterInterval() throws Exception {
        when(grpc.getMultimodalCacheStatusAsync(eq(a.getIp()), eq(8001), any(), eq(2000L)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        when(grpc.getMultimodalCacheStatusAsync(eq(b.getIp()), eq(8001), any(), eq(2000L)))
                .thenReturn(CompletableFuture.completedFuture(CacheStatusPB.getDefaultInstance()));
        directory.refresh();
        directory.refresh();
        verify(grpc, times(2)).getMultimodalCacheStatusAsync(any(), eq(8001), any(), eq(2000L));
        var field = VitCacheDirectory.class.getDeclaredField("attempts");
        field.setAccessible(true);
        var attempts = (Map<String, Long>) field.get(directory);
        attempts.replaceAll((key, value) -> value - VitCacheDirectory.SYNC_INTERVAL_MS);
        directory.refresh();
        verify(grpc, times(4)).getMultimodalCacheStatusAsync(any(), eq(8001), any(), eq(2000L));
    }

    @Test
    void expiringSnapshotDoesNotResetRecentFailedAttempt() throws Exception {
        snapshot(a, "a1", "image");
        when(grpc.getMultimodalCacheStatusAsync(any(), eq(8001), any(), eq(2000L)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        directory.refresh();
        var prune = VitCacheDirectory.class.getDeclaredMethod("prune", Map.class, long.class);
        prune.setAccessible(true);
        prune.invoke(
                directory,
                live,
                System.currentTimeMillis() + 2 * VitCacheDirectory.SYNC_INTERVAL_MS + 1);
        directory.refresh();
        verify(grpc, times(2)).getMultimodalCacheStatusAsync(any(), eq(8001), any(), eq(2000L));
    }

    @Test
    void absentGrpcDirectoryKeepsKnownKeysButExplicitEmptyDirectoryClearsThem() {
        tierSnapshot(a, List.of("image"), List.of("image"), List.of());
        snapshot(b, "b1", "image");
        when(grpc.getMultimodalCacheStatusAsync(any(), eq(8001), any(), eq(2000L)))
                .thenReturn(CompletableFuture.completedFuture(CacheStatusPB.getDefaultInstance()));
        directory.refresh();
        assertEquals(a.getIp(), selector.select(context("image"), null).getServerIp());
        snapshot(a, "a1");
        assertEquals(b.getIp(), selector.select(context("image"), null).getServerIp());
    }

    @Test
    void grpcTransportAcceptsLargeDirectoryAndReusesClientChannel() throws Exception {
        List<String> keys =
                IntStream.range(0, 100_000).mapToObj(i -> String.format("%064x", i)).toList();
        var response =
                CacheStatusPB.newBuilder()
                        .setMultimodalCache(
                                MultimodalCacheStatusPB.newBuilder()
                                        .setWorkerInstance("large-worker")
                                        .addAllKeys(keys)
                                        .addAllGpuEmbeddingKeys(keys))
                        .build();
        assertTrue(response.getSerializedSize() > 8 * 1024 * 1024);
        assertTrue(response.getSerializedSize() < 16 * 1024 * 1024);
        var received = new AtomicReference<CacheVersionPB>();
        Server server =
                NettyServerBuilder.forPort(0)
                        .addService(
                                new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
                                    @Override
                                    public void getCacheStatus(
                                            CacheVersionPB request,
                                            StreamObserver<CacheStatusPB> observer) {
                                        received.set(request);
                                        observer.onNext(response);
                                        observer.onCompleted();
                                    }
                                })
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void getCacheStatus(
                                            CacheVersionPB request,
                                            StreamObserver<CacheStatusPB> observer) {
                                        observer.onNext(response);
                                        observer.onCompleted();
                                    }
                                })
                        .build()
                        .start();
        var executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);
        var eventLoop = new NioEventLoopGroup(1);
        var channelFactory = new GrpcChannelFactory(executor, eventLoop, 1000);
        var client =
                new EngineGrpcClient(
                        mock(EngineAddressResolver.class),
                        channelFactory,
                        mock(GrpcReporter.class),
                        mock(CacheMatchConfiguration.class));
        try {
            var request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
            for (int i = 0; i < 2; i++) {
                var actual =
                        client.getMultimodalCacheStatusAsync(
                                        "127.0.0.1", server.getPort(), request, 5000)
                                .join();
                assertEquals(response, actual);
                assertTrue(received.get().getNeedCacheKeys());
            }
            assertEquals(
                    response,
                    client.getCacheStatusAsync("127.0.0.1", server.getPort(), request, 5000)
                            .join());
            directory.replace(a, response.getMultimodalCache());
            assertEquals(
                    a.getIp(),
                    selector.select(context(keys.get(keys.size() - 1)), null).getServerIp());
        } finally {
            client.shutdown();
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            eventLoop.shutdownGracefully(0, 5, TimeUnit.SECONDS).sync();
            executor.shutdownNow();
        }
    }

    @Test
    void prefersMostHitsAndReplacesEvictedKeys() {
        snapshot(a, "a1", "image1", "image2");
        snapshot(b, "b1", "image1");
        assertEquals(a.getIp(), selector.select(context("image1", "image2"), null).getServerIp());
        snapshot(a, "a1");
        assertEquals(b.getIp(), selector.select(context("image1"), null).getServerIp());
    }

    @Test
    void cacheAffinityNeverOverridesGroupOrHealth() {
        snapshot(a, "a1", "image");
        assertEquals(b.getIp(), selector.select(context("image"), "two").getServerIp());
        live.remove(a.getLogicalIpPort());
        assertEquals(b.getIp(), selector.select(context("image"), null).getServerIp());
        live.remove(b.getLogicalIpPort());
        assertFalse(selector.select(context("image"), null).isSuccess());
    }

    @Test
    void concurrentColdRequestsSharePlacementButNotConfirmedOwnership() throws Exception {
        var executor = Executors.newFixedThreadPool(8);
        try {
            List<Future<String>> selections =
                    IntStream.range(0, 64)
                            .mapToObj(
                                    i ->
                                            executor.submit(
                                                    () ->
                                                            selector.select(context("cold"), null)
                                                                    .getServerIp()))
                            .toList();
            String selected = selections.get(0).get();
            for (Future<String> selection : selections) {
                assertEquals(selected, selection.get());
            }
            WorkerStatus other = selected.equals(a.getIp()) ? b : a;
            snapshot(other, "real", "cold");
            assertEquals(other.getIp(), selector.select(context("cold"), null).getServerIp());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void cacheAffinityPinsTheEngineThatOwnsTheImageBehindASharedAddress() {
        live.clear();
        WorkerStatus first =
                WorkerStatus.createDiscovered(
                        RoleType.VIT, "one", "10.0.0.1", 8000, 8001, "", null, 0, 2);
        WorkerStatus second =
                WorkerStatus.createDiscovered(
                        RoleType.VIT, "one", "10.0.0.1", 8000, 8002, "", null, 1, 2);
        live.put(first.getLogicalIpPort(), first);
        live.put(second.getLogicalIpPort(), second);
        snapshot(first, "first");
        snapshot(second, "second", "image");

        var ctx = context("image");
        var selected = selector.select(ctx, "one");
        assertEquals(1, selected.getEngineIndex());
        ctx.getRequest().setSelectedVit(selected);
        try (var pinned = selector.selectPinned(ctx, "one")) {
            org.junit.jupiter.api.Assertions.assertNotNull(pinned);
            assertEquals(second.getGenerationId(), pinned.serverStatus().getWorkerGeneration());
        }
        live.remove(second.getLogicalIpPort());
        assertFalse(selector.validate(ctx, "one").isSuccess());
    }

    @Test
    void tracesWhyAnEarlierVitSelectionWasRejected() {
        snapshot(a, "a1", "image");
        var ctx = context("image");
        var selected = selector.select(ctx, "one");
        ctx.getRequest().setSelectedVit(selected);
        Span span = mock(Span.class);
        when(span.storeInContext(any())).thenCallRealMethod();
        ctx.setTraceContext(Context.root().with(span));
        FlexlbTrace.configure(OpenTelemetry.noop(), "test");
        try {
            assertTrue(selector.validate(ctx, "one").isSuccess());
            verify(span, never()).setAttribute(eq(FlexlbTrace.VIT_VALIDATION_REASON), any());

            assertFalse(selector.validate(ctx, "two").isSuccess());
            verify(span).setAttribute(FlexlbTrace.VIT_VALIDATION_REASON, "policy_group_mismatch");

            selected.setGrpcPort(9000);
            assertFalse(selector.validate(ctx, "one").isSuccess());
            verify(span).setAttribute(FlexlbTrace.VIT_VALIDATION_REASON, "grpc_port_mismatch");

            selected.setGrpcPort(a.getGrpcPort());
            selected.setWorkerGeneration(0);
            assertFalse(selector.validate(ctx, "one").isSuccess());
            verify(span).setAttribute(FlexlbTrace.VIT_VALIDATION_REASON, "generation_mismatch");

            live.remove(a.getLogicalIpPort());
            assertFalse(selector.validate(ctx, "one").isSuccess());
            verify(span).setAttribute(FlexlbTrace.VIT_VALIDATION_REASON, "endpoint_unavailable");
        } finally {
            FlexlbTrace.configure(null, "");
        }
    }

    @Test
    void rejectsStaleSnapshotAndInvalidSelectedWorker() {
        snapshot(a, "a1", "image");
        WorkerStatus replacement = worker(a.getIp(), "one");
        live.put(a.getLogicalIpPort(), replacement);
        snapshot(a, "old", "image");
        snapshot(b, "b1", "image");
        assertEquals(b.getIp(), selector.select(context("image"), null).getServerIp());
        var ctx = context("image");
        var selected = selector.select(ctx, "two");
        ctx.getRequest().setSelectedVit(selected);
        assertTrue(selector.validate(ctx, "two").isSuccess());
        assertFalse(selector.validate(ctx, "one").isSuccess());
        live.remove(b.getLogicalIpPort());
        assertFalse(selector.validate(ctx, null).isSuccess());
    }

    @Test
    void snapshotFailureKeepsKnownKeysAndLeaderChangeRefreshesImmediately() {
        snapshot(a, "a1", "image");
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        directory.refresh();
        verifyNoInteractions(grpc);
        when(grpc.getMultimodalCacheStatusAsync(any(), eq(8001), any(), eq(2000L)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        when(consistency.isMaster()).thenReturn(true);
        directory.refresh();
        assertEquals(a.getIp(), selector.select(context("image"), null).getServerIp());
        directory.refresh();
        verify(grpc, times(2)).getMultimodalCacheStatusAsync(any(), eq(8001), any(), eq(2000L));
        when(consistency.isMaster()).thenReturn(false);
        directory.refresh();
        when(consistency.isMaster()).thenReturn(true);
        directory.refresh();
        verify(grpc, times(4)).getMultimodalCacheStatusAsync(any(), eq(8001), any(), eq(2000L));
    }

    @Test
    void selectionKeepsGenerationThroughCopyAndPin() throws Exception {
        snapshot(a, "epoch-a", "image");
        var ctx = context("image");
        var selected = selector.select(ctx, "one");
        var copied = org.flexlb.dao.loadbalance.ServerStatus.copyOf(selected);
        var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        assertFalse(
                mapper.writeValueAsString(new org.flexlb.dao.loadbalance.ServerStatus())
                        .contains("worker_generation"));
        copied =
                mapper.readValue(
                        mapper.writeValueAsBytes(copied),
                        org.flexlb.dao.loadbalance.ServerStatus.class);
        assertEquals(a.getGenerationId(), copied.getWorkerGeneration());
        ctx.getRequest().setSelectedVit(copied);
        copied.setWorkerGeneration(0);
        assertFalse(selector.validate(ctx, "one").isSuccess());
        copied.setWorkerGeneration(a.getGenerationId());
        try (var pinned = selector.selectPinned(ctx, "one")) {
            org.junit.jupiter.api.Assertions.assertNotNull(pinned);
            assertEquals(a.getGenerationId(), pinned.serverStatus().getWorkerGeneration());
        }
        live.put(a.getLogicalIpPort(), worker(a.getIp(), "one"));
        assertFalse(selector.validate(ctx, "one").isSuccess());
        org.junit.jupiter.api.Assertions.assertNull(selector.selectPinned(ctx, "one"));
    }

    private WorkerStatus worker(String ip, String group) {
        return WorkerStatus.createDiscovered(RoleType.VIT, group, ip, 8000, 8001, "");
    }

    private BalanceContext context(String... keys) {
        return context(new FlexlbConfig(), keys);
    }

    private BalanceContext context(FlexlbConfig config, String... keys) {
        Request request = new Request();
        request.setRequestId("123");
        request.setCacheAffinityKeys(List.of(keys));
        request.setGenerateTimeout(30000);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        return context;
    }

    private void snapshot(WorkerStatus worker, String epoch, String... keys) {
        directory.replace(
                worker,
                MultimodalCacheStatusPB.newBuilder()
                        .setWorkerInstance(epoch)
                        .addAllKeys(List.of(keys))
                        .build());
    }

    private void tierSnapshot(
            WorkerStatus worker, List<String> hashes, List<String> gpu, List<String> cpu) {
        directory.replace(
                worker,
                MultimodalCacheStatusPB.newBuilder()
                        .setWorkerInstance("instance")
                        .addAllKeys(hashes)
                        .addAllGpuEmbeddingKeys(gpu)
                        .addAllCpuEmbeddingKeys(cpu)
                        .build());
    }
}
