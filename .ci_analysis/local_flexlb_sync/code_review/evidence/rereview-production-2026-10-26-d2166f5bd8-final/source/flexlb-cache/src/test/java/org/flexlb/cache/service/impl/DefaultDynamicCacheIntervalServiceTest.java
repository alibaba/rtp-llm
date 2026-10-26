package org.flexlb.cache.service.impl;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.WorkerRegistryConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DefaultDynamicCacheIntervalServiceTest {

    @Test
    void initializesAtMinimumWhenConfiguredRangeIsAboveDefault() {
        DefaultDynamicCacheIntervalService service = service(30, 250, 500);

        assertEquals(250, service.getCurrentIntervalMs());
    }

    @Test
    void initializesAtMaximumWhenConfiguredRangeIsBelowDefault() {
        DefaultDynamicCacheIntervalService service = service(30, 20, 75);

        assertEquals(75, service.getCurrentIntervalMs());
    }

    @Test
    void instancesDoNotShareIntervalState() {
        DefaultDynamicCacheIntervalService first = service(30, 50, 300);
        DefaultDynamicCacheIntervalService second = service(30, 50, 300);

        first.updateDiffStatistics(300);
        first.updateDiffStatistics(300);
        first.updateDiffStatistics(300);

        assertEquals(70, first.getCurrentIntervalMs());
        assertEquals(100, second.getCurrentIntervalMs());
    }

    @Test
    void intervalUpdatesRemainWithinConfiguredBounds() {
        DefaultDynamicCacheIntervalService service = service(30, 50, 200);

        for (int i = 0; i < 100; i++) {
            service.updateDiffStatistics(300);
            assertTrue(service.getCurrentIntervalMs() >= 50);
            assertTrue(service.getCurrentIntervalMs() <= 200);
        }
        assertEquals(50, service.getCurrentIntervalMs());

        for (int i = 0; i < 100; i++) {
            service.updateDiffStatistics(0);
            assertTrue(service.getCurrentIntervalMs() >= 50);
            assertTrue(service.getCurrentIntervalMs() <= 200);
        }
        assertEquals(200, service.getCurrentIntervalMs());
    }

    private static DefaultDynamicCacheIntervalService service(
            int targetDiffSize, long minIntervalMs, long maxIntervalMs) {
        WorkerRegistryConfig.CacheStatusConfig cacheStatus =
                new WorkerRegistryConfig.CacheStatusConfig();
        cacheStatus.setTargetDiffSize(targetDiffSize);
        cacheStatus.setMinRefreshIntervalMs(minIntervalMs);
        cacheStatus.setMaxRefreshIntervalMs(maxIntervalMs);
        FlexlbConfig config = new FlexlbConfig();
        config.getWorkerRegistry().setCacheStatus(cacheStatus);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        return new DefaultDynamicCacheIntervalService(configService);
    }
}
