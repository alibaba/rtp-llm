package org.flexlb.service;

import io.grpc.Server;
import io.grpc.Status;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService.CacheStatusPB;
import org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalCacheStatusPB;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class VitCacheDirectoryTest {
    private VitCacheDirectory directory;
    private VitCacheSelector selector;
    private final Map<String, WorkerStatus> live = new HashMap<>();
    private WorkerStatus a;
    private WorkerStatus b;
    private EngineGrpcClient grpc;
    private LBStatusConsistencyService consistency;

    @BeforeEach
    void setUp() {
        a = worker("10.0.0.1", "one");
        b = worker("10.0.0.2", "two");
        live.put(a.getIpPort(), a);
        live.put(b.getIpPort(), b);
        var workers = mock(WorkerDirectory.class);
        when(workers.statusSnapshot(RoleType.VIT)).thenAnswer(call -> Map.copyOf(live));
        when(workers.endpointAddressSnapshot(RoleType.VIT)).thenAnswer(call -> new java.util.ArrayList<>(live.keySet()));
        when(workers.captureEndpoint(eq(RoleType.VIT), any())).thenAnswer(call -> {
            WorkerStatus status = live.get(call.getArgument(1));
            return status == null ? null : new WorkerEndpoint(status).tryPinGeneration();
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

    private WorkerStatus worker(String ip, String group) {
        return WorkerStatus.createDiscovered(RoleType.VIT, group, ip, 8000, 8001, "");
    }

    private BalanceContext context(String... keys) {
        Request request = new Request();
        request.setRequestId(123);
        request.setCacheAffinityKeys(List.of(keys));
        request.setGenerateTimeout(30000);
        BalanceContext context = new BalanceContext(new FlexlbConfig());
        context.setRequest(request);
        return context;
    }

    private void snapshot(WorkerStatus worker, String epoch, String... keys) {
        directory.replace(worker, MultimodalCacheStatusPB.newBuilder()
                .setWorkerInstance(epoch).addAllKeys(List.of(keys)).build());
    }

    private void tierSnapshot(WorkerStatus worker, List<String> hashes, List<String> gpu, List<String> cpu) {
        directory.replace(worker, MultimodalCacheStatusPB.newBuilder()
                .setWorkerInstance("instance").addAllKeys(hashes)
                .addAllGpuEmbeddingKeys(gpu).addAllCpuEmbeddingKeys(cpu).build());
    }

    @Test
    void workerRestartInvalidatesColdKeyPlacements() {
        snapshot(a, "a1");
        snapshot(b, "b1");
        selector.select(context("cold1", "cold2"), "one");
        selector.select(context("other"), "two");
        snapshot(a, "a2");
        assertEquals(b.getIp(), selector.select(context("cold1", "cold2", "other"), null).getServerIp());
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
        assertEquals(pendingWorker.getIp(),
                selector.select(context("cold"), pendingWorker.getGroup()).getServerIp());
        live.remove(other.getIpPort());
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
        var snapshot = MultimodalCacheStatusPB.newBuilder().setWorkerInstance("a1")
                .addKeys("hash")
                .addGpuEmbeddingKeys("gpu").addCpuEmbeddingKeys("cpu").build();
        var request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
        when(grpc.getMultimodalCacheStatusAsync(a.getIp(), 8001, request, 2000))
                .thenReturn(CompletableFuture.completedFuture(
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
        prune.invoke(directory, live, System.currentTimeMillis() + 2 * VitCacheDirectory.SYNC_INTERVAL_MS + 1);
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
        List<String> keys = IntStream.range(0, 100_000)
                .mapToObj(i -> String.format("%064x", i)).toList();
        var response = CacheStatusPB.newBuilder().setMultimodalCache(
                MultimodalCacheStatusPB.newBuilder().setWorkerInstance("large-worker")
                        .addAllKeys(keys).addAllGpuEmbeddingKeys(keys)).build();
        assertTrue(response.getSerializedSize() > 8 * 1024 * 1024);
        var received = new AtomicReference<CacheVersionPB>();
        Server server = NettyServerBuilder.forPort(0)
                .addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
                    @Override
                    public void getCacheStatus(CacheVersionPB request, StreamObserver<CacheStatusPB> observer) {
                        received.set(request);
                        observer.onNext(response);
                        observer.onCompleted();
                    }
                })
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getCacheStatus(CacheVersionPB request, StreamObserver<CacheStatusPB> observer) {
                        observer.onNext(response);
                        observer.onCompleted();
                    }
                }).build().start();
        var executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);
        var eventLoop = new NioEventLoopGroup(1);
        var client = new EngineGrpcClient(mock(CustomNameResolver.class), executor, eventLoop,
                mock(GrpcReporter.class), 1000, 5000);
        try {
            var request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
            for (int i = 0; i < 2; i++) {
                var actual = client.getMultimodalCacheStatusAsync("127.0.0.1", server.getPort(), request, 5000).join();
                assertEquals(response, actual);
                assertTrue(received.get().getNeedCacheKeys());
            }
            CompletionException error = assertThrows(CompletionException.class,
                    () -> client.getCacheStatusAsync("127.0.0.1", server.getPort(), request, 5000).join());
            assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(error).getCode());
            directory.replace(a, response.getMultimodalCache());
            assertEquals(a.getIp(), selector.select(context(keys.get(keys.size() - 1)), null).getServerIp());
        } finally {
            client.shutdownChannelPool();
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
        live.remove(a.getIpPort());
        assertEquals(b.getIp(), selector.select(context("image"), null).getServerIp());
        live.remove(b.getIpPort());
        assertFalse(selector.select(context("image"), null).isSuccess());
    }

    @Test
    void concurrentColdRequestsSharePlacementButNotConfirmedOwnership() throws Exception {
        var executor = Executors.newFixedThreadPool(8);
        try {
            List<Future<String>> selections = IntStream.range(0, 64)
                    .mapToObj(i -> executor.submit(() -> selector.select(context("cold"), null).getServerIp()))
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
        live.put(a.getIpPort(), replacement);
        snapshot(a, "old", "image");
        snapshot(b, "b1", "image");
        assertEquals(b.getIp(), selector.select(context("image"), null).getServerIp());
        var ctx = context("image");
        var selected = selector.select(ctx, "two");
        ctx.getRequest().setSelectedVit(selected);
        assertTrue(selector.validate(ctx, "two").isSuccess());
        assertFalse(selector.validate(ctx, "one").isSuccess());
        live.remove(b.getIpPort());
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
        assertFalse(mapper.writeValueAsString(new org.flexlb.dao.loadbalance.ServerStatus())
                .contains("worker_generation"));
        copied = mapper.readValue(mapper.writeValueAsBytes(copied),
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
        live.put(a.getIpPort(), worker(a.getIp(), "one"));
        assertFalse(selector.validate(ctx, "one").isSuccess());
        org.junit.jupiter.api.Assertions.assertNull(selector.selectPinned(ctx, "one"));
    }

}
