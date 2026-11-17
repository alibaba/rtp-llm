package org.flexlb.cache.match.localstandby;

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
import static org.mockito.Mockito.when;

/**
 * Verifies that standby predictions apply only to the routed worker generation and block size.
 */
class LocalStandbyCacheManagerTest {
    private static final String ADDRESS = "127.0.0.1:8080";
    private static final List<Long> KEYS = List.of(11L, 22L);

    @Test
    void replacementCannotInheritPredictedBlocks() throws InterruptedException {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheMatching(new KvcmCacheMatchingConfig());
        when(configService.loadBalanceConfig()).thenReturn(config);
        LocalStandbyCacheManager manager = new LocalStandbyCacheManager(configService);
        try {
            WorkerStatus original = worker(1);
            manager.observe(original);
            Request request = mock(Request.class);
            when(request.getBlockCacheKeys()).thenReturn(KEYS);
            when(request.getCacheKeyBlockSize()).thenReturn(16L);
            ServerStatus selected = new ServerStatus();
            selected.setServerIp("127.0.0.1");
            selected.setHttpPort(8080);
            selected.setRole(RoleType.PREFILL);
            selected.setWorkerGeneration(1);
            selected.setSuccess(true);

            manager.recordRoutedRequest(request, List.of(selected));
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            Map<String, Integer> match = manager.findMatchingEngines(KEYS, RoleType.PREFILL, List.of(ADDRESS), 16);
            while (!match.equals(Map.of(ADDRESS, 2)) && System.nanoTime() < deadline) {
                Thread.sleep(5);
                match = manager.findMatchingEngines(KEYS, RoleType.PREFILL, List.of(ADDRESS), 16);
            }
            assertEquals(Map.of(ADDRESS, 2), match);
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
