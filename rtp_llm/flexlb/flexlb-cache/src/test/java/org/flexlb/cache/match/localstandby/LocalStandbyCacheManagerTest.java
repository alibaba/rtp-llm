package org.flexlb.cache.match.localstandby;

import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies that standby predictions apply only to the routed worker generation and block size.
 */
class LocalStandbyCacheManagerTest {
    private static final String ADDRESS = "127.0.0.1:8080";
    private static final List<Long> KEYS = List.of(11L, 22L);

    @Test
    void reportsMappingCountAndOneRejectionPerCapacityLimitedUpdate() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        KvcmCacheMatchingConfig kvcm = new KvcmCacheMatchingConfig();
        kvcm.getLocalStandby().setMaximumEntries(1);
        config.setCacheMatching(kvcm);
        when(configService.loadBalanceConfig()).thenReturn(config);
        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        LocalStandbyCacheManager manager = new LocalStandbyCacheManager(configService, metrics);
        try {
            manager.observe(worker(1));
            Request request = new Request();
            request.setBlockCacheKeys(List.of(11L, 22L, 33L));
            request.setCacheKeyBlockSize(16L);
            ServerStatus selected = new ServerStatus();
            selected.setServerIp("127.0.0.1");
            selected.setHttpPort(8080);
            selected.setRole(RoleType.PREFILL);
            selected.setWorkerGeneration(1);
            selected.setSuccess(true);

            manager.reportMappingCount();
            verify(metrics).reportLocalStandbyMappingCount(0L);
            manager.recordRoutedRequest(request, List.of(selected));
            verify(metrics, timeout(1000)).reportLocalStandbyCapacityRejected();

            assertEquals(Map.of(ADDRESS, 1), manager.findMatchingEngines(KEYS, RoleType.PREFILL, List.of(ADDRESS), 16));
            verify(metrics).reportLocalStandbyBlockSize(RoleType.PREFILL, 16L);
            manager.reportMappingCount();
            verify(metrics).reportLocalStandbyMappingCount(1L);
        } finally {
            manager.shutdown();
        }
    }

    @Test
    void replacementCannotInheritPredictedBlocks() throws InterruptedException {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheMatching(new KvcmCacheMatchingConfig());
        when(configService.loadBalanceConfig()).thenReturn(config);
        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        LocalStandbyCacheManager manager = new LocalStandbyCacheManager(configService, metrics);
        try {
            WorkerStatus original = worker(1);
            manager.observe(original);
            Request request = mock(Request.class);
            when(request.getBlockCacheKeys()).thenReturn(KEYS);
            when(request.getCacheKeyBlockSize()).thenReturn(16L);
            when(request.getSeqLen()).thenReturn(32L);
            ServerStatus selected = new ServerStatus();
            selected.setServerIp("127.0.0.1");
            selected.setHttpPort(8080);
            selected.setRole(RoleType.PREFILL);
            selected.setWorkerGeneration(1);
            selected.setSuccess(true);
            selected.setKvcmMatchedBlocks(1);

            manager.recordRoutedRequest(request, List.of(selected));
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            Map<String, Integer> match = manager.findMatchingEngines(KEYS, RoleType.PREFILL, List.of(ADDRESS), 16);
            while (!match.equals(Map.of(ADDRESS, 2)) && System.nanoTime() < deadline) {
                Thread.sleep(5);
                match = manager.findMatchingEngines(KEYS, RoleType.PREFILL, List.of(ADDRESS), 16);
            }
            assertEquals(Map.of(ADDRESS, 2), match);
            verify(metrics, timeout(1000)).reportLocalStandbyPrediction(RoleType.PREFILL, ADDRESS, 0, 32);
            assertEquals(Map.of(), manager.findMatchingEngines(KEYS, RoleType.PREFILL, List.of(ADDRESS), 32));

            manager.observe(worker(2));
            manager.remove(original);
            manager.recordRoutedRequest(request, List.of(selected));
            assertEquals(Map.of(ADDRESS, 0), manager.findMatchingEngines(KEYS, RoleType.PREFILL, List.of(ADDRESS), 16));
        } finally {
            manager.shutdown();
        }
    }

    private static WorkerStatus worker(long generation) {
        WorkerStatus worker = mock(WorkerStatus.class);
        CacheStatus cache = new CacheStatus();
        cache.setBlockSize(16);
        when(worker.getIpPort()).thenReturn(ADDRESS);
        when(worker.getGenerationId()).thenReturn(generation);
        when(worker.getRole()).thenReturn(RoleType.PREFILL);
        when(worker.getCacheStatus()).thenReturn(cache);
        when(worker.isActiveGeneration()).thenReturn(true);
        return worker;
    }
}
