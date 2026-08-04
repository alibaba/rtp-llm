package org.flexlb.sync.synchronizer;

import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.constant.CommonConstants;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class MasterEngineSynchronizerTest {

    @AfterEach
    void shutdownExecutors() {
        if (AbstractEngineStatusSynchronizer.engineSyncExecutor != null) {
            AbstractEngineStatusSynchronizer.engineSyncExecutor.shutdownNow();
        }
        if (AbstractEngineStatusSynchronizer.statusCheckExecutor != null) {
            AbstractEngineStatusSynchronizer.statusCheckExecutor.shutdownNow();
        }
        AbstractEngineStatusSynchronizer.engineSyncExecutor = null;
        AbstractEngineStatusSynchronizer.statusCheckExecutor = null;
    }

    @Test
    void continuesSubmittingOtherRolesWhenOneEngineSyncTaskIsRejected() {
        String serviceId = CommonConstants.FUNCTION + ".test-model";
        ServiceRoute route = mock(ServiceRoute.class);
        when(route.getServiceId()).thenReturn(serviceId);
        when(route.getAllRoleTypes()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        when(route.getRoleEndpoints(any(RoleType.class))).thenReturn(List.of(mock(Endpoint.class)));

        ModelMetaConfig modelMetaConfig = mock(ModelMetaConfig.class);
        when(modelMetaConfig.getServiceRoutes()).thenReturn(List.of(route));
        when(modelMetaConfig.getServiceRoute(serviceId)).thenReturn(route);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        MasterEngineSynchronizer synchronizer = new MasterEngineSynchronizer(
                mock(WorkerAddressService.class),
                mock(EngineHealthReporter.class),
                mock(EngineWorkerStatus.class),
                mock(EngineGrpcService.class),
                modelMetaConfig,
                mock(CacheAwareService.class),
                configService,
                false);
        synchronizer.destroy();

        ExecutorService engineSyncExecutor = mock(ExecutorService.class);
        AbstractEngineStatusSynchronizer.engineSyncExecutor = engineSyncExecutor;
        when(engineSyncExecutor.submit(any(Runnable.class)))
                .thenThrow(new RejectedExecutionException("queue full"))
                .thenReturn(null);

        synchronizer.syncEngineStatus();

        verify(engineSyncExecutor, times(2)).submit(any(Runnable.class));
    }
}
