package org.flexlb.cache.match.localsync;

import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class EngineLocalViewExecutorTest {
    @ParameterizedTest
    @ValueSource(ints = {1, 16})
    void springUsesConfiguredPoolForExactDiffAndClosesIt(int threads) throws Exception {
        var config =
                ConfigService.parse(
                        """
                {"requestLifecycle":{"request":{"timeoutMs":1000}},
                 "workerRegistry":{"executors":{"cacheDiffParallelism":%d}}}
                """
                                .formatted(threads));
        var configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        ForkJoinPool pool;
        try (var context = new AnnotationConfigApplicationContext()) {
            context.getBeanFactory().registerSingleton("configService", configService);
            context.getBeanFactory()
                    .registerSingleton("cacheMetricsReporter", mock(CacheMetricsReporter.class));
            context.registerBean(
                    DynamicCacheIntervalService.class,
                    () -> mock(DynamicCacheIntervalService.class));
            context.register(EngineLocalView.class);
            context.refresh();
            var local = context.getBean(EngineLocalView.class);
            var field = EngineLocalView.class.getDeclaredField("customPool");
            field.setAccessible(true);
            LongStream.range(0, 10_000).forEach(key -> local.addOrUpdateCacheBlock("worker", key));
            var diff = local.calculateDiff("worker", keys(5_000, 15_000));
            pool = (ForkJoinPool) field.get(local);
            assertEquals(
                    Math.min(threads, Runtime.getRuntime().availableProcessors()),
                    pool.getParallelism());
            assertEquals(keys(10_000, 15_000), diff.getAddedBlocks());
            assertEquals(keys(0, 5_000), diff.getRemovedBlocks());
        }
        assertTrue(pool.isShutdown());
    }

    @org.junit.jupiter.api.Test
    @org.junit.jupiter.api.Timeout(15)
    void failedDiffWaitsForItsOtherReaderBeforeReleasingTheLocalView() throws Exception {
        var local = new EngineLocalView();
        var pool = new ForkJoinPool(2);
        var reading = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        org.springframework.test.util.ReflectionTestUtils.setField(local, "customPool", pool);
        org.springframework.test.util.ReflectionTestUtils.setField(
                local, "dynamicIntervalManager", mock(DynamicCacheIntervalService.class));
        local.addOrUpdateCacheBlock("worker", 1L);
        Set<Long> failingSnapshot =
                new java.util.AbstractSet<>() {
                    @Override
                    public int size() {
                        return 1;
                    }

                    @Override
                    public java.util.Iterator<Long> iterator() {
                        throw new IllegalStateException("snapshot reader failed");
                    }

                    @Override
                    public boolean contains(Object key) {
                        reading.countDown();
                        try {
                            assertTrue(release.await(5, java.util.concurrent.TimeUnit.SECONDS));
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                            throw new AssertionError(interrupted);
                        }
                        return false;
                    }
                };
        try (var caller = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            var result =
                    caller.submit(
                            () ->
                                    org.junit.jupiter.api.Assertions.assertThrows(
                                            IllegalStateException.class,
                                            () -> local.calculateDiff("worker", failingSnapshot)));
            try {
                assertTrue(reading.await(5, java.util.concurrent.TimeUnit.SECONDS));
                org.junit.jupiter.api.Assertions.assertFalse(result.isDone());
            } finally {
                release.countDown();
            }
            result.get(5, java.util.concurrent.TimeUnit.SECONDS);
        } finally {
            release.countDown();
            local.shutdown();
        }
    }

    private static Set<Long> keys(long start, long end) {
        return LongStream.range(start, end).boxed().collect(Collectors.toSet());
    }
}
