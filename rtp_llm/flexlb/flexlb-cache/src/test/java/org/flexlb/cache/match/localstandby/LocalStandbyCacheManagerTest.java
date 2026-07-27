package org.flexlb.cache.match.localstandby;

import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class LocalStandbyCacheManagerTest {

    @Test
    void matchesOnlyContiguousPrefixForEachWorker() {
        WorkerStatusProvider workerStatusProvider = mock(WorkerStatusProvider.class);
        WorkerStatus worker1 = worker("10.0.0.1", 8080);
        WorkerStatus worker2 = worker("10.0.0.2", 8080);
        when(workerStatusProvider.getWorkerStatuses(RoleType.PREFILL, "default"))
                .thenReturn(List.of(worker1, worker2));
        LocalStandbyCacheManager manager = new LocalStandbyCacheManager(
                new CacheMatchConfiguration(modelMetaConfig(300_000)),
                workerStatusProvider,
                mock(CacheMetricsReporter.class));

        manager.addRoutedRequestBlocks(worker1.getIpPort(), List.of(11L, 22L, 33L));
        manager.addRoutedRequestBlocks(worker2.getIpPort(), List.of(11L, 33L));

        Map<String, Integer> matches =
                manager.findMatchingEngines(List.of(11L, 22L, 33L), RoleType.PREFILL, "default");

        assertEquals(3, matches.get(worker1.getIpPort()));
        assertEquals(1, matches.get(worker2.getIpPort()));
        assertEquals(
                Map.of(worker1.getIpPort(), 1, worker2.getIpPort(), 1),
                manager.findMatchingEngines(List.of(11L), RoleType.PREFILL, "default"));
        manager.shutdown();
    }

    @Test
    void expiresEntriesAfterConfiguredTtl() throws InterruptedException {
        WorkerStatusProvider workerStatusProvider = mock(WorkerStatusProvider.class);
        WorkerStatus worker = worker("10.0.0.1", 8080);
        when(workerStatusProvider.getWorkerStatuses(RoleType.PREFILL, "default"))
                .thenReturn(List.of(worker));
        LocalStandbyCacheManager manager = new LocalStandbyCacheManager(
                new CacheMatchConfiguration(modelMetaConfig(20)),
                workerStatusProvider,
                mock(CacheMetricsReporter.class));
        manager.addRoutedRequestBlocks(worker.getIpPort(), List.of(11L));

        Thread.sleep(30);

        assertEquals(
                0,
                manager.findMatchingEngines(List.of(11L), RoleType.PREFILL, "default")
                        .get(worker.getIpPort()));
        assertEquals(0, manager.mappingCount());
        manager.shutdown();
    }

    @Test
    void derivesMaximumEntriesFromReportedHbmCapacity() {
        WorkerStatusProvider workerStatusProvider = mock(WorkerStatusProvider.class);
        WorkerStatus worker = worker("10.0.0.1", 8080);
        worker.setAlive(true);
        worker.setCacheStatus(CacheStatus.builder()
                .totalKvCache(10_000)
                .blockSize(100)
                .build());
        when(workerStatusProvider.getWorkerStatuses(RoleType.PREFILL, "default"))
                .thenReturn(List.of(worker));
        LocalStandbyCacheManager manager = new LocalStandbyCacheManager(
                new CacheMatchConfiguration(
                        modelMetaConfig(300_000, 1_000, 1.2)),
                workerStatusProvider,
                mock(CacheMetricsReporter.class));

        manager.refreshMaximumEntries();
        manager.addRoutedRequestBlocks(worker.getIpPort(), List.of(11L));

        assertEquals(120, manager.maximumEntryCount());
        manager.shutdown();
    }

    @Test
    void usesConfiguredStandbyBlockSizeToDeriveMaximumEntries() {
        WorkerStatusProvider workerStatusProvider = mock(WorkerStatusProvider.class);
        WorkerStatus worker = worker("10.0.0.1", 8080);
        worker.setAlive(true);
        worker.setCacheStatus(CacheStatus.builder()
                .totalKvCache(10_000)
                .blockSize(100)
                .build());
        when(workerStatusProvider.getWorkerStatuses(RoleType.PREFILL, "default"))
                .thenReturn(List.of(worker));
        LocalStandbyCacheManager manager = new LocalStandbyCacheManager(
                new CacheMatchConfiguration(modelMetaConfig(300_000, 1_000, 1.2, 200)),
                workerStatusProvider,
                mock(CacheMetricsReporter.class));

        manager.refreshMaximumEntries();

        assertEquals(60, manager.maximumEntryCount());
        manager.shutdown();
    }

    @Test
    void rejectsNewMappingsAtCapacityAndAllowsExistingMappingsToRefresh() {
        WorkerStatusProvider workerStatusProvider = mock(WorkerStatusProvider.class);
        CacheMetricsReporter cacheMetricsReporter = mock(CacheMetricsReporter.class);
        WorkerStatus worker = worker("10.0.0.1", 8080);
        when(workerStatusProvider.getWorkerStatuses(RoleType.PREFILL, "default"))
                .thenReturn(List.of(worker));
        LocalStandbyCacheManager manager = new LocalStandbyCacheManager(
                new CacheMatchConfiguration(modelMetaConfig(300_000, 1, 1.2)),
                workerStatusProvider,
                cacheMetricsReporter);

        manager.addRoutedRequestBlocks(worker.getIpPort(), List.of(11L));
        manager.addRoutedRequestBlocks(worker.getIpPort(), List.of(11L));
        manager.addRoutedRequestBlocks(worker.getIpPort(), List.of(22L));

        assertEquals(1, manager.mappingCount());
        assertEquals(1, manager.findMatchingEngines(
                List.of(11L), RoleType.PREFILL, "default").get(worker.getIpPort()));
        assertEquals(0, manager.findMatchingEngines(
                List.of(22L), RoleType.PREFILL, "default").get(worker.getIpPort()));
        manager.reportMappingCount();
        verify(cacheMetricsReporter).reportLocalStandbyCapacityRejected();
        verify(cacheMetricsReporter).reportLocalStandbyMappingCount(1);
        manager.shutdown();
    }

    private WorkerStatus worker(String ip, int port) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp(ip);
        workerStatus.setPort(port);
        return workerStatus;
    }

    private ModelMetaConfig modelMetaConfig(long expirationMs) {
        return modelMetaConfig(
                expirationMs,
                LocalStandbyConfig.DEFAULT_MAXIMUM_ENTRIES,
                LocalStandbyConfig.DEFAULT_CAPACITY_MULTIPLIER);
    }

    private ModelMetaConfig modelMetaConfig(long expirationMs, long maximumEntries, double capacityMultiplier) {
        return modelMetaConfig(expirationMs, maximumEntries, capacityMultiplier, 0);
    }

    private ModelMetaConfig modelMetaConfig(long expirationMs, long maximumEntries, double capacityMultiplier, long blockSize) {
        LocalStandbyConfig standby = new LocalStandbyConfig();
        standby.setEntryTtlMs(expirationMs);
        standby.setMaximumEntries(maximumEntries);
        standby.setCapacityMultiplier(capacityMultiplier);
        standby.setBlockSize(blockSize);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setLocalStandby(standby);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);
        GroupRoleEndPoint roleEndpoint = new GroupRoleEndPoint();
        roleEndpoint.setGroup("default");
        roleEndpoint.setPrefillEndpoint(new Endpoint());
        route.setRoleEndpoints(List.of(roleEndpoint));

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return config;
    }
}
