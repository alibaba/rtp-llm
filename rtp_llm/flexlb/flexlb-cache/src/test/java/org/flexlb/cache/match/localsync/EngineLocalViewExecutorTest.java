package org.flexlb.cache.match.localsync;

import org.flexlb.cache.match.localsync.DynamicCacheIntervalService;
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
        var config = ConfigService.parse("""
                {"requestLifecycle":{"request":{"timeoutMs":1000}},
                 "workerRegistry":{"executors":{"cacheDiffParallelism":%d}}}
                """.formatted(threads));
        var configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        ForkJoinPool pool;
        try (var context = new AnnotationConfigApplicationContext()) {
            context.getBeanFactory().registerSingleton("configService", configService);
            context.registerBean(DynamicCacheIntervalService.class, () -> mock(DynamicCacheIntervalService.class));
            context.register(EngineLocalView.class);
            context.refresh();
            var local = context.getBean(EngineLocalView.class);
            local.calculateDiff("warmup", Set.of());
            var field = EngineLocalView.class.getDeclaredField("customPool");
            field.setAccessible(true);
            pool = (ForkJoinPool) field.get(local);
            assertEquals(Math.min(threads, Runtime.getRuntime().availableProcessors()), pool.getParallelism());
            LongStream.range(0, 10_000).forEach(key -> local.addOrUpdateCacheBlock("worker", key));
            var diff = local.calculateDiff("worker", keys(5_000, 15_000));
            assertEquals(keys(10_000, 15_000), diff.getAddedBlocks());
            assertEquals(keys(0, 5_000), diff.getRemovedBlocks());
        }
        assertTrue(pool.isShutdown());
    }

    private static Set<Long> keys(long start, long end) {
        return LongStream.range(start, end).boxed().collect(Collectors.toSet());
    }
}
