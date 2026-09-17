package org.flexlb.service;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService.CacheStatusPB;
import org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalCacheStatusPB;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class VitCacheDirectoryTest {
    private VitCacheDirectory directory;
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
        var workers = mock(EngineWorkerStatus.class);
        when(workers.selectModelWorkerStatus(eq(RoleType.VIT), any())).thenAnswer(call -> {
            String group = call.getArgument(1);
            if (group == null) {
                return live;
            }
            Map<String, WorkerStatus> filtered = new HashMap<>();
            live.forEach((address, worker) -> {
                if (group.equals(worker.getGroup())) {
                    filtered.put(address, worker);
                }
            });
            return filtered;
        });
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        grpc = mock(EngineGrpcClient.class);
        consistency = mock(LBStatusConsistencyService.class);
        directory = new VitCacheDirectory(workers, config, new ResourceMeasureFactory(List.of()), grpc, consistency);
    }

    @AfterEach
    void tearDown() {
        directory.stop();
    }

    private WorkerStatus worker(String ip, String group) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(8000);
        worker.setAlive(true);
        worker.setGroup(group);
        return worker;
    }

    private BalanceContext context(String... keys) {
        Request request = new Request();
        request.setRequestId(123);
        request.setMediaKeys(List.of(keys));
        request.setGenerateTimeout(30000);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        return context;
    }

    private void snapshot(WorkerStatus worker, String epoch, String... keys) {
        directory.replace(worker, MultimodalCacheStatusPB.newBuilder()
                .setFeatureHashVersion(1).setWorkerInstance(epoch).addAllKeys(List.of(keys)).build());
    }

    private void tierSnapshot(WorkerStatus worker, List<String> hashes, List<String> gpu, List<String> cpu) {
        directory.replace(worker, MultimodalCacheStatusPB.newBuilder()
                .setFeatureHashVersion(1).setWorkerInstance("instance").addAllKeys(hashes)
                .addAllGpuEmbeddingKeys(gpu).addAllCpuEmbeddingKeys(cpu).build());
    }

    @Test
    void prefersGpuThenCpuThenHashOnlyAndReplacesResidency() {
        List<String> keys = List.of("image");
        tierSnapshot(a, keys, keys, List.of());
        tierSnapshot(b, keys, List.of(), keys);
        assertEquals(a.getIp(), directory.select(context("image"), null).getServerIp());
        tierSnapshot(a, keys, List.of(), List.of());
        assertEquals(b.getIp(), directory.select(context("image"), null).getServerIp());
        // A snapshot without the optional tier fields clears previously advertised residency.
        snapshot(b, "instance", "image");
        tierSnapshot(a, keys, List.of(), keys);
        assertEquals(a.getIp(), directory.select(context("image"), null).getServerIp());
    }

    @Test
    void hashCoveragePrecedesEmbeddingCoverageAndEmbeddingCoveragePrecedesTier() {
        List<String> keys = List.of("image1", "image2");
        tierSnapshot(a, keys, List.of(), List.of());
        tierSnapshot(b, List.of("image1"), keys, List.of());
        assertEquals(a.getIp(), directory.select(context("image1", "image2"), null).getServerIp());
        tierSnapshot(a, keys, List.of(), keys);
        tierSnapshot(b, keys, List.of("image1"), List.of());
        assertEquals(a.getIp(), directory.select(context("image1", "image2"), null).getServerIp());
        tierSnapshot(b, keys, List.of("image1"), List.of("image2"));
        assertEquals(b.getIp(), directory.select(context("image1", "image2"), null).getServerIp());
    }

    @Test
    void embeddingWithoutHashBeatsPendingPlacementButRespectsGroupAndHealth() {
        String selected = directory.select(context("cold"), null).getServerIp();
        WorkerStatus other = selected.equals(a.getIp()) ? b : a;
        WorkerStatus pendingWorker = selected.equals(a.getIp()) ? a : b;
        tierSnapshot(other, List.of(), List.of(), List.of("cold"));
        assertEquals(other.getIp(), directory.select(context("cold"), null).getServerIp());
        assertEquals(pendingWorker.getIp(),
                directory.select(context("cold"), pendingWorker.getGroup()).getServerIp());
        other.setAlive(false);
        assertEquals(pendingWorker.getIp(), directory.select(context("cold"), null).getServerIp());
    }

    @Test
    void rejectsInvalidTierSnapshotsWithoutDiscardingKnownResidency() {
        tierSnapshot(a, List.of(), List.of("image"), List.of());
        tierSnapshot(b, List.of(), List.of(), List.of("image"));
        tierSnapshot(a, List.of(), List.of("image"), List.of("image"));
        assertEquals(a.getIp(), directory.select(context("image"), null).getServerIp());
        tierSnapshot(a, List.of(), List.of(), List.of(""));
        assertEquals(a.getIp(), directory.select(context("image"), null).getServerIp());
        List<String> manyKeys = IntStream.range(0, 100_000).mapToObj(i -> "key" + i).toList();
        tierSnapshot(a, manyKeys, List.of(), List.of("extra"));
        assertEquals(a.getIp(), directory.select(context("image"), null).getServerIp());
    }

    @Test
    void refreshUsesGrpcPortAndPreservesTierFields() {
        var snapshot = MultimodalCacheStatusPB.newBuilder().setWorkerInstance("a1")
                .setFeatureHashVersion(1).addKeys("hash")
                .addGpuEmbeddingKeys("gpu").addCpuEmbeddingKeys("cpu").build();
        var request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
        when(grpc.getMultimodalCacheStatus(a.getIp(), 8001, request, 2000))
                .thenReturn(CacheStatusPB.newBuilder().setMultimodalCache(snapshot).build());
        when(grpc.getMultimodalCacheStatus(b.getIp(), 8001, request, 2000))
                .thenReturn(CacheStatusPB.getDefaultInstance());
        directory.refresh();
        directory.refresh();
        verify(grpc, times(1)).getMultimodalCacheStatus(a.getIp(), 8001, request, 2000);
        verify(grpc, times(1)).getMultimodalCacheStatus(b.getIp(), 8001, request, 2000);
        for (String key : List.of("hash", "gpu", "cpu")) {
            assertEquals(a.getIp(), directory.select(context(key), null).getServerIp());
        }
    }

    @Test
    void absentGrpcDirectoryKeepsKnownKeysButExplicitEmptyDirectoryClearsThem() {
        tierSnapshot(a, List.of("image"), List.of("image"), List.of());
        snapshot(b, "b1", "image");
        when(grpc.getMultimodalCacheStatus(any(), eq(8001), any(), eq(2000L)))
                .thenReturn(CacheStatusPB.getDefaultInstance());
        directory.refresh();
        assertEquals(a.getIp(), directory.select(context("image"), null).getServerIp());
        snapshot(a, "a1");
        assertEquals(b.getIp(), directory.select(context("image"), null).getServerIp());
    }

    @Test
    void grpcTransportAcceptsLargeDirectoryAndReusesClientChannel() throws Exception {
        List<String> keys = IntStream.range(0, 100_000)
                .mapToObj(i -> String.format("%064x", i)).toList();
        var response = CacheStatusPB.newBuilder().setMultimodalCache(
                MultimodalCacheStatusPB.newBuilder().setWorkerInstance("large-worker")
                        .setFeatureHashVersion(1).addAllKeys(keys).addAllGpuEmbeddingKeys(keys)).build();
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
                }).build().start();
        var executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);
        var eventLoop = new NioEventLoopGroup(1);
        var client = new EngineGrpcClient(mock(CustomNameResolver.class), executor, eventLoop,
                null, null, mock(GrpcReporter.class));
        try {
            var request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
            for (int i = 0; i < 2; i++) {
                var actual = client.getMultimodalCacheStatus("127.0.0.1", server.getPort(), request, 5000);
                assertEquals(response, actual);
                assertTrue(received.get().getNeedCacheKeys());
            }
            directory.replace(a, response.getMultimodalCache());
            assertEquals(a.getIp(), directory.select(context(keys.get(keys.size() - 1)), null).getServerIp());
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
        assertEquals(a.getIp(), directory.select(context("image1", "image2"), null).getServerIp());
        snapshot(a, "a1");
        assertEquals(b.getIp(), directory.select(context("image1"), null).getServerIp());
    }

    @Test
    void cacheAffinityNeverOverridesGroupOrHealth() {
        snapshot(a, "a1", "image");
        assertEquals(b.getIp(), directory.select(context("image"), "two").getServerIp());
        a.setAlive(false);
        assertEquals(b.getIp(), directory.select(context("image"), null).getServerIp());
        b.setAlive(false);
        assertFalse(directory.select(context("image"), null).isSuccess());
    }

    @Test
    void concurrentColdRequestsSharePlacementButNotConfirmedOwnership() throws Exception {
        var executor = Executors.newFixedThreadPool(8);
        try {
            List<Future<String>> selections = IntStream.range(0, 64)
                    .mapToObj(i -> executor.submit(() -> directory.select(context("cold"), null).getServerIp()))
                    .toList();
            String selected = selections.get(0).get();
            for (Future<String> selection : selections) {
                assertEquals(selected, selection.get());
            }
            WorkerStatus other = selected.equals(a.getIp()) ? b : a;
            snapshot(other, "real", "cold");
            assertEquals(other.getIp(), directory.select(context("cold"), null).getServerIp());
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
        assertEquals(b.getIp(), directory.select(context("image"), null).getServerIp());
        var ctx = context("image");
        var selected = directory.select(ctx, "two");
        ctx.getRequest().setSelectedVit(selected);
        assertTrue(directory.validate(ctx, "two").isSuccess());
        assertFalse(directory.validate(ctx, "one").isSuccess());
        b.setAlive(false);
        assertFalse(directory.validate(ctx, null).isSuccess());
    }

    @Test
    void snapshotFailureKeepsKnownKeysAndLeaderChangeRefreshesImmediately() {
        snapshot(a, "a1", "image");
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        directory.refresh();
        verifyNoInteractions(grpc);
        when(grpc.getMultimodalCacheStatus(any(), eq(8001), any(), eq(2000L)))
                .thenThrow(new RuntimeException("unavailable"));
        when(consistency.isMaster()).thenReturn(true);
        directory.refresh();
        assertEquals(a.getIp(), directory.select(context("image"), null).getServerIp());
        directory.refresh();
        verify(grpc, times(2)).getMultimodalCacheStatus(any(), eq(8001), any(), eq(2000L));
        when(consistency.isMaster()).thenReturn(false);
        directory.refresh();
        when(consistency.isMaster()).thenReturn(true);
        directory.refresh();
        verify(grpc, times(4)).getMultimodalCacheStatus(any(), eq(8001), any(), eq(2000L));
    }
}
