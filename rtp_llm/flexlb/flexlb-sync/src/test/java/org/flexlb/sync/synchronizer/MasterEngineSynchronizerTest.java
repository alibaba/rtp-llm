package org.flexlb.sync.synchronizer;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class MasterEngineSynchronizerTest {
    @Test
    void embeddingRefreshPublishesDiscoveryWithoutLlmStatusProbes() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.getWorkerRegistry().setEngineType(EngineType.EMBEDDING);
        config.getWorkerRegistry().getHealth().setStatusPollIntervalMs(60_000);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.modelName()).thenReturn("embedding");
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PDFUSION));
        WorkerAddressService addresses = mock(WorkerAddressService.class);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        EndpointRegistry endpoints = mock(EndpointRegistry.class);
        WorkerDirectory directory = new WorkerDirectory(endpoints);
        WorkerHost host = new WorkerHost("10.0.0.1", 8000);
        when(addresses.getEngineWorkerList("embedding", RoleType.PDFUSION)).thenReturn(List.of(host));
        MasterEngineSynchronizer synchronizer = new MasterEngineSynchronizer(
                addresses, mock(EngineHealthReporter.class), directory, grpc, model,
                mock(CacheAwareService.class), mock(DynamicCacheIntervalService.class), service);
        try {
            ScheduledThreadPoolExecutor scheduler = (ScheduledThreadPoolExecutor)
                    ReflectionTestUtils.getField(synchronizer, "scheduler");
            scheduler.shutdown();
            assertTrue(scheduler.awaitTermination(5, TimeUnit.SECONDS));
            ThreadPoolExecutor executor = (ThreadPoolExecutor) ReflectionTestUtils.getField(synchronizer, "engineSyncExecutor");
            executor.setCorePoolSize(1);
            executor.setMaximumPoolSize(1);
            synchronizer.syncEngineStatus();
            executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
            Map<String, WorkerStatus> snapshot = directory.statusSnapshot(RoleType.PDFUSION);
            assertEquals(1, snapshot.size());
            assertTrue(snapshot.containsKey(host.getIpPort()));
            assertFalse(snapshot.get(host.getIpPort()).pollHealth().reportedAlive());
            assertEquals(0, directory.routingCapacity(RoleType.PDFUSION));
            assertTrue(synchronizer.isReady());
            assertThrows(UnsupportedOperationException.class, snapshot::clear);
            when(addresses.getEngineWorkerList("embedding", RoleType.PDFUSION)).thenReturn(List.of());
            synchronizer.syncEngineStatus();
            executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
            assertTrue(directory.statusSnapshot(RoleType.PDFUSION).isEmpty());
            assertFalse(synchronizer.isReady());
            assertEquals(1, snapshot.size(), "previous membership snapshots stay immutable");
            assertFalse(snapshot.get(host.getIpPort()).isActiveGeneration());
            verifyNoInteractions(grpc);
        } finally {
            synchronizer.destroy();
        }
    }
}
