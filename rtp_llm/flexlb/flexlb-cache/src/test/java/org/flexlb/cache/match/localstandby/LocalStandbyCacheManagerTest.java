package org.flexlb.cache.match.localstandby;

import java.util.List;
import java.util.Map;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.*;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

class LocalStandbyCacheManagerTest {
    @Test void onlyContiguousPredictionsFromTheCurrentIncarnationMatch() throws Exception {
        var config = new FlexlbConfig(); config.setCacheMatching(new KvcmCacheMatchingConfig());
        var topology = new KvcmConfig(); topology.setAddress("meta"); topology.setNamespace("deployment");
        var route = new ServiceRoute(); route.setKvcm(topology);
        var manager = new LocalStandbyCacheManager(new CacheMatchConfiguration(route, config), mock(CacheMetricsReporter.class));
        WorkerStatus first = worker();
        var keys = List.of(11L, 22L); var candidates = List.of(first.getIpPort());
        try {
            manager.observe(first);
            assertEquals(Map.of(first.getIpPort(), 0), manager.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16));
            var request = new Request(); request.setBlockCacheKeys(keys); request.setCacheKeyBlockSize(16);
            var selected = new ServerStatus(); selected.setRole(RoleType.PREFILL); selected.setSuccess(true);
            selected.setWorkerGeneration(first.getGenerationId()); selected.setServerIp(first.getIp()); selected.setHttpPort(first.getPort());
            manager.updateFromRoutedRequest(request, List.of(selected));
            long until = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(2);
            while (!Map.of(first.getIpPort(), 2).equals(manager.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16))) {
                assertTrue(System.nanoTime() < until, "Prediction was not published"); Thread.sleep(5);
            }
            assertEquals(Map.of(first.getIpPort(), 1), manager.findMatchingEngines(List.of(11L, 33L, 22L), RoleType.PREFILL, candidates, 16));
            assertTrue(manager.findMatchingEngines(keys, RoleType.PREFILL, candidates, 32).isEmpty());
            WorkerStatus replacement = worker(); manager.observe(replacement);
            manager.remove(first); // A late retirement must not remove the replacement.
            assertEquals(Map.of(first.getIpPort(), 0), manager.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16));
        } finally { manager.shutdown(); }
    }

    private static WorkerStatus worker() {
        var worker = WorkerStatus.createDiscovered(RoleType.PREFILL, "group", "127.0.0.1", 8088, 8089, "site");
        var status = new CacheStatus(); status.setBlockSize(16);
        worker.cacheMutationLock.lock();
        try { worker.publishCacheStatus(status); } finally { worker.cacheMutationLock.unlock(); }
        return worker;
    }
}
