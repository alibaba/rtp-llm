package org.flexlb.service;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.balance.endpoint.EndpointRegistry.VitRoutingEntry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.constant.MetricConstant;
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
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * Exercises ViT cache snapshots, worker replacement, and concurrent routing decisions that
 * combine reported load, locally dispatched images, and cache hits.
 */
class VitCacheDirectoryTest {
    private final FlexMonitor monitor = mock(FlexMonitor.class);

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
        when(workers.statusSnapshot(RoleType.VIT)).thenAnswer(call -> Map.copyOf(live));
        when(workers.endpointAddressSnapshot(RoleType.VIT))
                .thenAnswer(call -> new java.util.ArrayList<>(live.keySet()));
        when(workers.vitRoutingSnapshot())
                .thenAnswer(
                        call ->
                                live.entrySet().stream()
                                        .map(
                                                entry ->
                                                        new VitRoutingEntry(
                                                                entry.getKey(),
                                                                new WorkerEndpoint(
                                                                        entry.getValue())))
                                        .toList());
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
        selector = new VitCacheSelector(directory, workers, monitor);
    }

    @AfterEach
    void tearDown() {
        selector.stop();
        directory.stop();
    }

    @Test
    void defaultsToLeastMediaWorkEvenWhenBusyWorkerHasCache() {
        var ctx = context("image");
        ctx.getConfig().getRouter().getRoles().getVit().setCacheAffinityEnabled(false);
        snapshot(a, "a", "image");
        load(a, 100L);
        load(b, 2L);
        assertEquals(b.getIp(), selector.select(ctx, null).getServerIp());
        assertEquals(a.getIp(), selector.select(ctx, "one").getServerIp());
    }

    @Test
    void leastLoadInspectsCacheAndPinsOnlyTheWinner() {
        var cacheDirectory = mock(VitCacheDirectory.class);
        var fastSelector = new VitCacheSelector(cacheDirectory, workers, monitor);
        try {
            var ctx = firstVitRoute(false, "image");
            load(a, 100L);
            load(b, 2L);
            clearInvocations(workers);
            assertEquals(b.getIp(), fastSelector.select(ctx, null).getServerIp());
            verify(cacheDirectory).snapshotFor(b);
            verifyNoMoreInteractions(cacheDirectory);
            verify(workers, times(1)).captureEndpoint(RoleType.VIT, b.getLogicalIpPort());
            verify(workers, never()).captureEndpoint(RoleType.VIT, a.getLogicalIpPort());
            verify(workers, never()).statusSnapshot(any());
            verify(workers, never()).endpointAddressSnapshot(any());
        } finally {
            fastSelector.stop();
        }
    }

    @Test
    void selectionDoesNotWaitForDirectoryWriterOrSelectorMonitor() throws Exception {
        snapshot(a, "a1", "image");
        load(a, 0L);
        load(b, 100L);
        var executor = Executors.newSingleThreadExecutor();
        try {
            synchronized (directory) {
                synchronized (selector) {
                    for (boolean affinity : List.of(false, true)) {
                        var ctx = context("image");
                        ctx.getConfig()
                                .getRouter()
                                .getRoles()
                                .getVit()
                                .setCacheAffinityEnabled(affinity);
                        assertEquals(
                                a.getIp(),
                                executor.submit(() -> selector.select(ctx, null))
                                        .get(2, TimeUnit.SECONDS)
                                        .getServerIp());
                    }
                }
            }
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void stalledWinnerCaptureDoesNotSerializeLeastLoadRequests() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var captures = new AtomicInteger();
        when(workers.captureEndpoint(RoleType.VIT, a.getLogicalIpPort()))
                .thenAnswer(
                        call -> {
                            if (captures.incrementAndGet() == 1) {
                                entered.countDown();
                                assertTrue(release.await(5, TimeUnit.SECONDS));
                            }
                            return new WorkerEndpoint(a).tryPinGeneration();
                        });
        var ctx = context();
        ctx.getConfig().getRouter().getRoles().getVit().setCacheAffinityEnabled(false);
        var executor = Executors.newFixedThreadPool(2);
        try {
            var first = executor.submit(() -> selector.select(ctx, "one"));
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var second = executor.submit(() -> selector.select(ctx, "one"));
            assertTrue(second.get(2, TimeUnit.SECONDS).isSuccess());
            release.countDown();
            assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        } finally {
            release.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void winnerReplacementAfterSnapshotIsRejectedInBothModes() {
        var oldFleet = workers.vitRoutingSnapshot();
        snapshot(a, "old", "image");
        when(workers.vitRoutingSnapshot()).thenReturn(oldFleet);
        live.put(a.getLogicalIpPort(), worker(a.getIp(), "one"));
        for (boolean affinity : List.of(false, true)) {
            var ctx = context("image");
            ctx.getConfig().getRouter().getRoles().getVit().setCacheAffinityEnabled(affinity);
            assertFalse(selector.select(ctx, "one").isSuccess());
        }
    }

    @Test
    void cacheReadsKeepTheirSnapshotAndDoNotPinEveryWorker() {
        snapshot(a, "a1", "image");
        clearInvocations(workers);
        var before = directory.candidates(null, List.of("image"));
        verify(workers, never()).captureEndpoint(any(), any());
        snapshot(a, "a2");
        var after = directory.candidates(null, List.of("image"));
        assertEquals(1, before.get(a.getLogicalIpPort()).hashHits());
        assertTrue(before.get(a.getLogicalIpPort()).hashKeys().contains("image"));
        assertEquals(0, after.get(a.getLogicalIpPort()).hashHits());
        assertThrows(
                UnsupportedOperationException.class,
                () -> before.get(a.getLogicalIpPort()).hashKeys().add("other"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void backgroundCleanupDropsRestartedPendingPlacement(boolean affinity) {
        snapshot(a, "a1");
        selector.select(firstVitRoute(affinity, "cold"), "one");
        snapshot(a, "a2");
        selector.prunePending();
        load(a, 100L);
        load(b, 0L);
        assertEquals(b.getIp(), selector.select(context("cold"), null).getServerIp());
    }

    @Test
    void defaultAffinityAccepts32ExtraInputsAndFallsBackAt33() {
        var ctx = context("image");
        load(b, 100L);
        snapshot(a, "a", "image");
        load(a, 132L);
        assertEquals(a.getIp(), selector.select(ctx, null).getServerIp());
        load(a, 133L);
        assertEquals(b.getIp(), selector.select(ctx, null).getServerIp());
        load(a, 132L);
        snapshot(b, "b", "image");
        assertEquals(b.getIp(), selector.select(ctx, null).getServerIp());
    }

    @Test
    void loadGapIsRelativeAndFallsBackToTheActualMinimum() {
        var ctx = context("x", "y");
        ctx.getConfig().getRouter().getRoles().getVit().setMaxExtraInflightInputs(2L);
        var middle = worker("10.0.0.3", "one");
        live.put(middle.getLogicalIpPort(), middle);
        snapshot(a, "a", "x", "y");
        snapshot(middle, "middle", "x");
        load(b, 100L);
        load(middle, 101L);
        load(a, 102L);
        assertEquals(a.getIp(), selector.select(ctx, null).getServerIp());
        load(a, 103L);
        assertEquals(b.getIp(), selector.select(ctx, null).getServerIp());
        load(middle, 100L);
        assertEquals(middle.getIp(), selector.select(ctx, null).getServerIp());
    }

    @Test
    void zeroGapCountsLocallyRoutedInputsAndPreservesAffinityOnTies() {
        var ctx = firstVitRoute(true, "x", "x");
        ctx.getConfig().getRouter().getRoles().getVit().setMaxExtraInflightInputs(0L);
        snapshot(a, "a", "x");
        load(a, 0L);
        load(b, 0L);
        assertEquals(a.getIp(), selector.select(ctx, null).getServerIp());
        assertEquals(b.getIp(), selector.select(ctx, null).getServerIp());
        assertEquals(a.getIp(), selector.select(ctx, null).getServerIp());
    }

    @Test
    void boundedAffinityUsesAvailableLoadAndFallsBackToCacheWhenAllLoadsAreUnknown() {
        var ctx = context("x");
        ctx.getConfig().getRouter().getRoles().getVit().setMaxExtraInflightInputs(2L);
        snapshot(a, "a", "x");
        assertEquals(a.getIp(), selector.select(ctx, null).getServerIp());
        load(b, 100L);
        assertEquals(b.getIp(), selector.select(ctx, null).getServerIp());
    }

    @Test
    void theoreticalHitsCountEachInputAtMostOnceBeforeRecordingPending() {
        tierSnapshot(a, List.of("both"), List.of("both", "gpu"), List.of("cpu"));
        var request = firstVitRoute(true, "both", "both", "gpu", "cpu", "cold", "cold");
        selector.recordVitOnlyInputs(request.getRequest().getCacheAffinityKeys());
        assertEquals(a.getIp(), selector.select(request, "one").getServerIp());
        var hits = FlexMetricTags.of("kind", "hit");
        verify(monitor)
                .report(
                        MetricConstant.VIT_CACHE_AFFINITY_IMAGES,
                        FlexMetricTags.of("kind", "total"),
                        6);
        verify(monitor).prepare(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, hits);
        verify(monitor).report(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, hits, 4);
        assertEquals(a.getIp(), selector.select(request, "one").getServerIp());
        verify(monitor).report(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, hits, 6);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void onlyTheFinalAffinityEnabledVitOnlySelectionContributesHits() {
        snapshot(a, "a1", "hot");
        load(a, 100L);
        load(b, 0L);
        var routed = selector.select(firstVitRoute(true, "hot"), null);
        assertEquals(b.getIp(), routed.getServerIp());
        assertEquals(a.getIp(), selector.select(firstVitRoute(false, "hot"), "one").getServerIp());
        var fullRoute = context("hot");
        assertEquals(a.getIp(), selector.select(fullRoute, "one").getServerIp());
        fullRoute.getRequest().setSelectedVit(routed);
        assertTrue(selector.validate(fullRoute, "two").isSuccess());
        verifyNoInteractions(monitor);
    }

    @Test
    void leastLoadRoutesRememberColdKeysForSubsequentAffinityRoutes() {
        load(a, 0L);
        load(b, 2L);
        assertEquals(
                a.getIp(),
                selector.select(firstVitRoute(false, "x", "x", "y"), null).getServerIp());
        // All three inputs count towards load. With affinity off, even x goes to the lighter
        // worker.
        assertEquals(b.getIp(), selector.select(firstVitRoute(false, "x"), null).getServerIp());

        var observed = directory.candidates(null, List.of("x", "y")).get(a.getLogicalIpPort());
        assertEquals(0, observed.hashHits());
        assertEquals(0, observed.embeddingHits());
        load(a, 30L);
        load(b, 0L);
        assertEquals(a.getIp(), selector.select(firstVitRoute(true, "y"), null).getServerIp());
        load(a, 0L);
        load(b, 30L);
        assertEquals(b.getIp(), selector.select(firstVitRoute(true, "x"), null).getServerIp());
        snapshot(b, "b1", "y");
        assertEquals(b.getIp(), selector.select(firstVitRoute(true, "y"), null).getServerIp());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void mediaDispatchedDuringPollRemainCountedUntilAFollowingPoll(boolean affinity) {
        load(a, 0L);
        load(b, 3L);
        try (var earlyPoll = a.tryBeginStatusPoll()) {
            var images = firstVitRoute(affinity, "a", "b", "c", "d");
            assertEquals(a.getIp(), selector.select(images, null).getServerIp());
            a.recordVitLoad(earlyPoll, 0L);
        }
        assertEquals(0L, a.vitLoad().inflightInputs());
        assertEquals(b.getIp(), selector.select(firstVitRoute(affinity, "e"), null).getServerIp());
        load(a, 0L);
        assertEquals(a.getIp(), selector.select(firstVitRoute(affinity, "f"), null).getServerIp());
    }

    @Test
    void validatingPinnedWorkerDoesNotChargeTheImagesAgain() {
        load(a, 0L);
        load(b, 3L);
        var first = firstVitRoute(false, "1", "2");
        var selected = selector.select(first, null);
        assertEquals(a.getIp(), selected.getServerIp());
        first.getRequest().setVitRouteOnly(false);
        first.getRequest().setSelectedVit(selected);
        try (var pinned = selector.selectPinned(first, null)) {
            assertNotNull(pinned);
        }
        assertEquals(
                a.getIp(), selector.select(firstVitRoute(false, "3", "4"), null).getServerIp());
        assertEquals(b.getIp(), selector.select(firstVitRoute(false, "5"), null).getServerIp());
    }

    @Test
    void simultaneousSelectionsSpreadAcrossWorkersOnOneLoadSnapshot() throws Exception {
        load(a, 0L);
        load(b, 1L);
        int requests = 12;
        var atWinner = new CountDownLatch(requests);
        when(workers.captureEndpoint(eq(RoleType.VIT), eq(a.getLogicalIpPort())))
                .thenAnswer(
                        call -> {
                            atWinner.countDown();
                            assertTrue(atWinner.await(5, TimeUnit.SECONDS));
                            return new WorkerEndpoint(a).tryPinGeneration();
                        });
        try (var pool = Executors.newFixedThreadPool(requests)) {
            var results =
                    IntStream.range(0, requests)
                            .mapToObj(
                                    i ->
                                            pool.submit(
                                                    () -> {
                                                        var ctx =
                                                                firstVitRoute(
                                                                        false,
                                                                        "image-" + i,
                                                                        "other-" + i);
                                                        ctx.getRequest()
                                                                .setRequestId(Integer.toString(i));
                                                        return selector.select(ctx, null)
                                                                .getServerIp();
                                                    }))
                            .toList();
            int sentToA = 0;
            for (var result : results) {
                String ip = result.get(10, TimeUnit.SECONDS);
                if (a.getIp().equals(ip)) {
                    sentToA++;
                } else {
                    assertEquals(b.getIp(), ip);
                }
            }
            assertEquals(requests / 2, sentToA);
        }
    }

    @Test
    void unknownAndInvalidLoadsAreNotIdleAndClosedPollCannotOverwriteLoad() {
        var ctx = context();
        load(a, 10L);
        assertEquals(a.getIp(), selector.select(ctx, null).getServerIp());
        load(b, -1L);
        assertEquals(a.getIp(), selector.select(ctx, null).getServerIp());
        var closed = b.tryBeginStatusPoll();
        closed.close();
        b.recordVitLoad(closed, 0L);
        assertEquals(a.getIp(), selector.select(ctx, null).getServerIp());
        load(b, 0L);
        assertEquals(b.getIp(), selector.select(ctx, null).getServerIp());
    }

    @Test
    void staleLoadIsUnknownWithoutChangingWorkerHealth() throws InterruptedException {
        var ctx = context();
        ctx.getConfig().getRouter().getRoles().getVit().setLoadMaxAgeMs(1);
        load(a, 0L);
        Thread.sleep(20);
        load(b, 20L);
        // Increase the age budget after expiring A, leaving B fresh.
        ctx.getConfig().getRouter().getRoles().getVit().setLoadMaxAgeMs(10);
        assertEquals(b.getIp(), selector.select(ctx, null).getServerIp());
        assertTrue(live.containsKey(a.getLogicalIpPort()));
    }

    @Test
    void routesBeforeMediaHashesExistAndRevalidatesSelectedWorker() {
        var ctx = context();
        ctx.getRequest().setVitRouteOnly(true);
        var selected = selector.select(ctx, "two");
        assertTrue(selected.isSuccess());
        assertEquals(b.getIp(), selected.getServerIp());
        ctx.getRequest().setSelectedVit(selected);
        assertTrue(selector.validate(ctx, "two").isSuccess());
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

    private WorkerStatus worker(String ip, String group) {
        return WorkerStatus.createDiscovered(RoleType.VIT, group, ip, 8000, 8001, "");
    }

    private void load(WorkerStatus worker, Long count) {
        try (var poll = worker.tryBeginStatusPoll()) {
            worker.recordVitLoad(poll, count);
        }
    }

    private BalanceContext firstVitRoute(boolean affinity, String... keys) {
        var ctx = context(keys);
        ctx.getConfig().getRouter().getRoles().getVit().setCacheAffinityEnabled(affinity);
        ctx.getRequest().setVitRouteOnly(true);
        return ctx;
    }

    private BalanceContext context(String... keys) {
        Request request = new Request();
        request.setRequestId("123");
        request.setCacheAffinityKeys(List.of(keys));
        request.setGenerateTimeout(30000);
        BalanceContext context = new BalanceContext(new FlexlbConfig());
        context.getConfig().getRouter().getRoles().getVit().setCacheAffinityEnabled(true);
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
