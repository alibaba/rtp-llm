package org.flexlb.sync;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class WorkerAddressServiceTest {

    @Mock
    private WorkerAddressService.ServiceDiscoveryRunner serviceDiscoveryRunner;

    @Mock
    private ExecutorService serviceDiscoveryExecutor;

    @Mock
    private EngineHealthReporter engineHealthReporter;

    @InjectMocks
    private WorkerAddressService workerAddressService;

    @Test
    @SuppressWarnings("unchecked")
    void testGetHosts_Timeout() throws Exception {
        // Arrange
        String modelName = "TestModel";
        String address = "TestAddress";
        Future<List<WorkerHost>> future = mock(Future.class);
        Mockito.doNothing().when(engineHealthReporter).reportStatusCheckerFail(any(), any(), null);

        // Act
        List<WorkerHost> actualHosts = workerAddressService.getServiceHosts(modelName, address);

        // Assertions
        Assertions.assertTrue(actualHosts.isEmpty());
        verify(serviceDiscoveryRunner, never()).call();
        verify(serviceDiscoveryExecutor, never()).submit(any(WorkerAddressService.ServiceDiscoveryRunner.class));
        verify(future, never()).get(500, TimeUnit.MILLISECONDS);
    }
}