package org.flexlb.sync.synchronizer;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.ScheduledThreadPoolExecutor;
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
        WorkerDirectory directory = mock(WorkerDirectory.class);
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
            synchronizer.syncEngineStatus();
            List<WorkerHost> snapshot = synchronizer.embeddingWorkerSnapshot(RoleType.PDFUSION);
            assertEquals(List.of(host), snapshot);
            assertTrue(synchronizer.isReady());
            assertThrows(UnsupportedOperationException.class, () -> snapshot.add(host));
            when(addresses.getEngineWorkerList("embedding", RoleType.PDFUSION)).thenReturn(List.of());
            synchronizer.syncEngineStatus();
            assertTrue(synchronizer.embeddingWorkerSnapshot(RoleType.PDFUSION).isEmpty());
            assertFalse(synchronizer.isReady());
            assertEquals(List.of(host), snapshot, "previous snapshots stay immutable");
            verifyNoInteractions(grpc, directory);
        } finally {
            synchronizer.destroy();
        }
    }
}
