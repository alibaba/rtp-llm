package org.flexlb.sync;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.address.WorkerAddressService.WorkerDiscoveryResult;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class WorkerAddressServiceTest {

    @Mock
    private EngineHealthReporter engineHealthReporter;

    @Mock
    private ModelMetaConfig modelMetaConfig;

    @Mock
    private ServiceDiscovery serviceDiscovery;

    @Test
    @SuppressWarnings("unchecked")
    void testGetHosts_Timeout() throws Exception {
        // Arrange
        String modelName = "TestModel";
        String address = "TestAddress";
        ExecutorService timeoutExecutor = mock(ExecutorService.class);
        Future<List<WorkerHost>> future = mock(Future.class);
        when(timeoutExecutor.submit(Mockito.<Callable<List<WorkerHost>>>any())).thenReturn(future);
        when(future.get(500, TimeUnit.MILLISECONDS)).thenThrow(new TimeoutException("slow"));
        WorkerAddressService workerAddressService = new WorkerAddressService(
                engineHealthReporter,
                modelMetaConfig,
                serviceDiscovery,
                timeoutExecutor
        );

        // Act
        WorkerDiscoveryResult discoveryResult = workerAddressService.getServiceHostsResult(modelName, address);

        // Assert
        Assertions.assertTrue(discoveryResult.hosts().isEmpty());
        Assertions.assertFalse(discoveryResult.reliable());
        verify(future).cancel(true);
        verify(engineHealthReporter).reportStatusCheckerFail(
                eq(modelName),
                eq(BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT),
                isNull(),
                isNull()
        );
    }

    @Test
    @SuppressWarnings("unchecked")
    void testGetHosts_Success() throws Exception {
        // Arrange
        String modelName = "TestModel";
        String address = "TestAddress";
        List<WorkerHost> expectedHosts = List.of(new WorkerHost("127.0.0.1", 8080, 8081, 8082, "site1", "group1"));
        ExecutorService executor = mock(ExecutorService.class);
        Future<List<WorkerHost>> future = mock(Future.class);
        when(executor.submit(Mockito.<Callable<List<WorkerHost>>>any())).thenReturn(future);
        when(future.get(500, TimeUnit.MILLISECONDS)).thenReturn(expectedHosts);
        WorkerAddressService workerAddressService = new WorkerAddressService(
                engineHealthReporter,
                modelMetaConfig,
                serviceDiscovery,
                executor
        );

        // Act
        WorkerDiscoveryResult discoveryResult = workerAddressService.getServiceHostsResult(modelName, address);

        // Assert
        Assertions.assertEquals(expectedHosts, discoveryResult.hosts());
        Assertions.assertTrue(discoveryResult.reliable());
    }

    @Test
    void testGetHosts_Rejected() {
        // Arrange
        String modelName = "TestModel";
        String address = "TestAddress";
        ExecutorService rejectingExecutor = mock(ExecutorService.class);
        when(rejectingExecutor.submit(Mockito.<Callable<List<WorkerHost>>>any()))
                .thenThrow(new RejectedExecutionException("full"));
        WorkerAddressService service = new WorkerAddressService(
                engineHealthReporter,
                modelMetaConfig,
                serviceDiscovery,
                rejectingExecutor
        );

        // Act
        WorkerDiscoveryResult discoveryResult = service.getServiceHostsResult(modelName, address);

        // Assert
        Assertions.assertTrue(discoveryResult.hosts().isEmpty());
        Assertions.assertFalse(discoveryResult.reliable());
        verify(engineHealthReporter).reportStatusCheckerFail(
                eq(modelName),
                eq(BalanceStatusEnum.SERVICE_DISCOVERY_ERROR),
                isNull(),
                isNull()
        );
    }
}
