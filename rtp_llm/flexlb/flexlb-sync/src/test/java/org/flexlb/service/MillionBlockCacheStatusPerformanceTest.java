package org.flexlb.service;

import ch.qos.logback.classic.Level;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.cache.match.localsync.DynamicCacheIntervalService;
import org.flexlb.cache.match.localsync.EngineLocalView;
import org.flexlb.cache.match.localsync.GlobalCacheIndex;
import org.flexlb.cache.match.localsync.KvCacheManager;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.engine.grpc.EngineRpcService.CacheStatusPB;
import org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Loopback uses real Netty/protobuf and the production converter and cache indexes.
 */
@Tag("performance-regression")
class MillionBlockCacheStatusPerformanceTest {
    private static final String ADDRESS = "127.0.0.1:8000@0";
    private static final Map<ch.qos.logback.classic.Logger, Level> PREVIOUS_LEVELS =
            new HashMap<>();

    private final WorkerStatus owner =
            WorkerStatus.createDiscovered(
                    org.flexlb.dao.route.RoleType.PREFILL,
                    "group",
                    "127.0.0.1",
                    8000,
                    8001,
                    "site");

    @BeforeAll
    static void suppressTransportDebugLogging() {
        for (String name : new String[] {"io.grpc.netty", "io.netty"}) {
            var logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(name);
            PREVIOUS_LEVELS.put(logger, logger.getLevel());
            logger.setLevel(Level.WARN);
        }
    }

    @AfterAll
    static void restoreLogging() {
        PREVIOUS_LEVELS.forEach(ch.qos.logback.classic.Logger::setLevel);
        PREVIOUS_LEVELS.clear();
    }

    @ParameterizedTest
    @ValueSource(ints = {1_000_000, 2_000_000, 3_000_000, 4_000_000})
    @Timeout(240)
    void millionBlockResponsesThroughTransportAndCacheApplication(int blocks) throws Exception {
        int selectedBlocks = Integer.getInteger("cache.perf.blocks", blocks);
        org.junit.jupiter.api.Assumptions.assumeTrue(selectedBlocks == blocks);
        System.out.printf(
                "Cache status benchmark: blocks=%d processors=%d maxHeapMiB=%.0f%n",
                blocks,
                Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().maxMemory() / (1024.0 * 1024));
        var response = new AtomicReference<CacheStatusPB>();
        var server =
                NettyServerBuilder.forPort(0)
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void getCacheStatus(
                                            CacheVersionPB request,
                                            StreamObserver<CacheStatusPB> observer) {
                                        observer.onNext(response.get());
                                        observer.onCompleted();
                                    }
                                })
                        .build()
                        .start();
        var callbacks = (ThreadPoolExecutor) Executors.newFixedThreadPool(8);
        var eventLoop = new NioEventLoopGroup(2);
        var client =
                new EngineGrpcClient(
                        mock(EngineAddressResolver.class),
                        new GrpcChannelFactory(callbacks, eventLoop, 1000),
                        mock(GrpcReporter.class),
                        mock(CacheMatchConfiguration.class),
                        30_000);
        try {
            for (boolean legacy : new boolean[] {false, true}) {
                response.set(snapshot(legacy, 0, blocks));
                // Exclude channel establishment/JIT from the repeated transport measurements.
                receive(client, server.getPort());
                for (int iteration = 0; iteration < 5; iteration++) {
                    long start = System.nanoTime();
                    var received = receive(client, server.getPort());
                    long rpc = System.nanoTime();
                    var converted = EngineStatusConverter.convertToCacheStatus(received);
                    long convertedAt = System.nanoTime();
                    assertEquals(blocks, converted.getCachedKeys().size());
                    System.out.printf(
                            "Million cache transport: blocks=%d format=%s bytes=%d rpcMs=%.3f convertMs=%.3f heapUsedMiB=%.1f%n",
                            blocks,
                            legacy ? "legacy-map" : "packed-list",
                            received.getSerializedSize(),
                            millis(rpc - start),
                            millis(convertedAt - rpc),
                            heapMiB());
                }
                var concurrent = new ArrayList<CompletableFuture<CacheStatusPB>>();
                long burstStart = System.nanoTime();
                for (int worker = 0; worker < 4; worker++) {
                    concurrent.add(
                            client.getCacheStatusAsync(
                                    "127.0.0.1",
                                    server.getPort(),
                                    CacheVersionPB.newBuilder().setNeedCacheKeys(true).build(),
                                    30_000));
                }
                for (var future : concurrent) {
                    var received = future.join();
                    assertEquals(
                            blocks,
                            legacy
                                    ? received.getCacheKeysCount()
                                    : received.getCacheKeyListCount());
                }
                System.out.printf(
                        "Million cache transport burst: blocks=%d format=%s responses=4 elapsedMs=%.3f heapUsedMiB=%.1f%n",
                        blocks,
                        legacy ? "legacy-map" : "packed-list",
                        millis(System.nanoTime() - burstStart),
                        heapMiB());
            }
            // Conversion yields the same owned Set for both wire formats; application is measured
            // separately.
            for (int threads : new int[] {1, 8, 16}) {
                var config =
                        ConfigService.parse(
                                """
                        {"requestLifecycle":{"request":{"timeoutMs":1000}},
                         "workerRegistry":{"executors":{"cacheDiffParallelism":%d}}}
                        """
                                        .formatted(threads));
                var service = mock(ConfigService.class);
                when(service.loadBalanceConfig()).thenReturn(config);
                try (var context = new AnnotationConfigApplicationContext()) {
                    context.getBeanFactory().registerSingleton("configService", service);
                    context.getBeanFactory()
                            .registerSingleton(
                                    "cacheMetricsReporter", mock(CacheMetricsReporter.class));
                    context.registerBean(
                            DynamicCacheIntervalService.class,
                            () -> mock(DynamicCacheIntervalService.class));
                    context.registerBean(
                            WorkerStatusProvider.class, () -> mock(WorkerStatusProvider.class));
                    context.register(
                            EngineLocalView.class, GlobalCacheIndex.class, KvCacheManager.class);
                    context.refresh();
                    var manager = context.getBean(KvCacheManager.class);
                    var local = context.getBean(EngineLocalView.class);
                    var global = context.getBean(GlobalCacheIndex.class);
                    var initial =
                            EngineStatusConverter.convertToCacheStatus(snapshot(false, 0, blocks))
                                    .getCachedKeys();
                    long cold = System.nanoTime();
                    manager.updateEngineCache(worker(), "PREFILL", initial, null);
                    System.out.printf(
                            "Million cache apply: blocks=%d threads=%d case=cold elapsedMs=%.3f heapUsedMiB=%.1f%n",
                            blocks, threads, millis(System.nanoTime() - cold), heapMiB());
                    for (int changed :
                            new int[] {0, blocks / 500, blocks * 2 / 100, blocks * 3 / 100}) {
                        var next =
                                EngineStatusConverter.convertToCacheStatus(
                                                snapshot(false, changed, blocks))
                                        .getCachedKeys();
                        for (int iteration = 0; iteration < 4; iteration++) {
                            // Alternate snapshots so every measured iteration has the advertised
                            // churn.
                            var keys = iteration % 2 == 0 ? next : initial;
                            long gcBefore = gcMillis();
                            long start = System.nanoTime();
                            manager.updateEngineCache(worker(), "PREFILL", keys, null);
                            long elapsed = System.nanoTime() - start;
                            assertEquals(keys, local.getEngineCacheBlocks(ADDRESS));
                            assertEquals(blocks, global.totalBlocks());
                            assertEquals(blocks, global.totalMappings());
                            if (iteration > 0) {
                                System.out.printf(
                                        "Million cache apply: blocks=%d threads=%d effective=%d changed=%d applyIncludingDiffMs=%.3f gcMs=%d heapUsedMiB=%.1f%n",
                                        blocks,
                                        threads,
                                        Math.min(
                                                threads,
                                                Runtime.getRuntime().availableProcessors()),
                                        changed,
                                        millis(elapsed),
                                        gcMillis() - gcBefore,
                                        heapMiB());
                            }
                        }
                    }
                }
            }
        } finally {
            client.shutdown();
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            eventLoop.shutdownGracefully(0, 5, TimeUnit.SECONDS).sync();
            callbacks.shutdownNow();
        }
    }

    private static CacheStatusPB receive(EngineGrpcClient client, int port) {
        return client.getCacheStatusAsync(
                        "127.0.0.1",
                        port,
                        CacheVersionPB.newBuilder().setNeedCacheKeys(true).build(),
                        30_000)
                .join();
    }

    private static CacheStatusPB snapshot(boolean legacy, int offset, int blocks) {
        var builder = CacheStatusPB.newBuilder().setVersion(offset + 1);
        // Mixed signed 64-bit hashes avoid unrealistically small varints and sequential hash
        // buckets.
        LongStream.range(offset, (long) offset + blocks)
                .map(MillionBlockCacheStatusPerformanceTest::hash)
                .forEach(
                        key -> {
                            if (legacy) {
                                builder.putCacheKeys(key, true);
                            } else {
                                builder.addCacheKeyList(key);
                            }
                        });
        return builder.build();
    }

    private static long hash(long value) {
        long mixed = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        mixed = (mixed ^ (mixed >>> 27)) * 0x94d049bb133111ebL;
        return mixed ^ (mixed >>> 31);
    }

    private static double millis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static double heapMiB() {
        var runtime = Runtime.getRuntime();
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024);
    }

    private static long gcMillis() {
        return ManagementFactory.getGarbageCollectorMXBeans().stream()
                .mapToLong(bean -> Math.max(0, bean.getCollectionTime()))
                .sum();
    }

    private WorkerStatus worker() {
        return owner;
    }
}
