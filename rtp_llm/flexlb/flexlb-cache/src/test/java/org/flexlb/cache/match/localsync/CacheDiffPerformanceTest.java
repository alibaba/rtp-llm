package org.flexlb.cache.match.localsync;

import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Tag("performance-regression")
class CacheDiffPerformanceTest {
    @Test
    @Timeout(60)
    void concurrentSnapshotsRemainExactAcrossConfiguredParallelism() throws Exception {
        var snapshot = LongStream.range(25_000, 75_000).boxed().collect(Collectors.toSet());
        for (int threads : new int[]{1, 8, 16}) {
            var config = ConfigService.parse("""
                    {"requestLifecycle":{"request":{"timeoutMs":1000}},
                     "workerRegistry":{"executors":{"cacheDiffParallelism":%d}}}
                    """.formatted(threads));
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            try (var context = new AnnotationConfigApplicationContext();
                 var callers = Executors.newFixedThreadPool(8)) {
                context.getBeanFactory().registerSingleton("configService", service);
                context.registerBean(DynamicCacheIntervalService.class, () -> mock(DynamicCacheIntervalService.class));
                context.register(EngineLocalView.class);
                context.refresh();
                var local = context.getBean(EngineLocalView.class);
                for (int worker = 0; worker < 8; worker++) {
                    String address = "worker-" + worker;
                    LongStream.range(0, 50_000).forEach(key -> local.addOrUpdateCacheBlock(address, key));
                }
                for (int iteration = 0; iteration < 5; iteration++) {
                    var tasks = new ArrayList<Callable<Void>>();
                    for (int worker = 0; worker < 8; worker++) {
                        String address = "worker-" + worker;
                        tasks.add(() -> {
                            var diff = local.calculateDiff(address, snapshot);
                            assertEquals(25_000, diff.getAddedBlocks().size());
                            assertEquals(25_000, diff.getRemovedBlocks().size());
                            return null;
                        });
                    }
                    long start = System.nanoTime();
                    for (var result : callers.invokeAll(tasks)) {
                        result.get();
                    }
                    if (iteration > 0) {
                        System.out.printf("Cache diff performance: configured=%d effective=%d snapshots=8 keys=50000 elapsedMs=%.3f%n",
                                threads, Math.min(threads, Runtime.getRuntime().availableProcessors()),
                                (System.nanoTime() - start) / 1_000_000.0);
                    }
                }
            }
        }
    }

    @Test
    @Timeout(600)
    void hundredWorkersWithDisjointKeysAndRandomChurn() throws Exception {
        int blocks = Integer.getInteger("cache.perf.keysPerWorker", 30_000);
        var config = ConfigService.parse("""
                {"requestLifecycle":{"request":{"timeoutMs":1000}},
                 "workerRegistry":{"executors":{"cacheDiffParallelism":16}}}
                """);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        try (var context = new AnnotationConfigApplicationContext();
             var callers = Executors.newFixedThreadPool(32)) {
            context.getBeanFactory().registerSingleton("configService", service);
            context.registerBean(DynamicCacheIntervalService.class, () -> mock(DynamicCacheIntervalService.class));
            context.getBeanFactory().registerSingleton("cacheMetricsReporter", mock(org.flexlb.cache.telemetry.CacheMetricsReporter.class));
                    context.getBeanFactory().registerSingleton("workerStatusProvider", mock(org.flexlb.dao.master.WorkerStatusProvider.class));
                    context.register(EngineLocalView.class, GlobalCacheIndex.class, KvCacheManager.class);
            context.refresh();
            var manager = context.getBean(KvCacheManager.class);
            var global = context.getBean(GlobalCacheIndex.class);
            var local = context.getBean(EngineLocalView.class);
            var initial = new ArrayList<org.flexlb.cache.primitive.LongKeySet>();
            for (int worker = 0; worker < 100; worker++) {
                var keys = new org.flexlb.cache.primitive.LongKeySet(blocks);
                for (int key = 0; key < blocks; key++) {
                    keys.add(fleetKey(worker, key));
                }
                initial.add(keys);
                manager.updateEngineCache(workerStatus("fleet-" + worker), "PREFILL", keys, null);
            }
            for (int percent : new int[]{2, 3}) {
                var changed = new ArrayList<org.flexlb.cache.primitive.LongKeySet>();
                for (int worker = 0; worker < 100; worker++) {
                    var keys = new org.flexlb.cache.primitive.LongKeySet(initial.get(worker));
                    var chosen = new java.util.HashSet<Integer>();
                    var random = new java.util.Random(817L + worker);
                    while (chosen.size() < blocks * percent / 100) {
                        chosen.add(random.nextInt(blocks));
                    }
                    for (int key : chosen) {
                        keys.remove(fleetKey(worker, key));
                        keys.add(fleetKey(worker, blocks + key));
                    }
                    changed.add(keys);
                }
                for (int round = 0; round < 4; round++) {
                    var tasks = new ArrayList<Callable<Void>>();
                    var snapshots = round % 2 == 0 ? changed : initial;
                    for (int worker = 0; worker < 100; worker++) {
                        int workerIndex = worker;
                        tasks.add(() -> {
                            var keys = snapshots.get(workerIndex);
                            String address = "fleet-" + workerIndex;
                            manager.updateEngineCache(workerStatus(address), "PREFILL", keys, null);
                            assertEquals(keys, local.getEngineCacheBlocks(address));
                            long sample = keys.iterator().nextLong();
                            assertEquals(java.util.Map.of(address, 1), manager.findMatchingEngines(
                                    java.util.List.of(sample), java.util.List.of(address)));
                            return null;
                        });
                    }
                    long start = System.nanoTime();
                    for (var result : callers.invokeAll(tasks)) {
                        result.get();
                    }
                    long elapsed = System.nanoTime() - start;
                    assertEquals(100L * blocks, global.totalBlocks());
                    assertEquals(100L * blocks, global.totalMappings());
                    if (round > 0) {
                        System.out.printf("Fleet cache apply: workers=100 keysPerWorker=%d randomChurnPercent=%d round=%d elapsedMs=%.3f%n",
                                blocks, percent, round, elapsed / 1_000_000.0);
                    }
                }
            }
        }
    }

    private static long fleetKey(int worker, int key) {
        long value = ((long) worker << 32) | Integer.toUnsignedLong(key);
        value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }
    private static org.flexlb.dao.master.WorkerStatus workerStatus(String address) {
        var status = mock(org.flexlb.dao.master.WorkerStatus.class);
        var identity = mock(org.flexlb.dao.master.WorkerIdentity.class);
        when(identity.getLogicalIpPort()).thenReturn(address);
        when(status.getWorkerIdentity()).thenReturn(identity);
        when(status.getMetricIpPort()).thenReturn(address);
        return status;
    }
}
