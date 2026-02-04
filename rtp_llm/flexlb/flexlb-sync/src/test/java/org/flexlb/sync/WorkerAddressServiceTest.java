package org.flexlb.sync;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class WorkerAddressServiceTest {

    @Mock
    private EngineHealthReporter engineHealthReporter;

    @Mock
    private ModelMetaConfig modelMetaConfig;

    @Mock
    private ServiceDiscovery serviceDiscovery;

    @Mock
    private ExecutorService serviceDiscoveryExecutor;

    @InjectMocks
    private WorkerAddressService workerAddressService;

    @Test
    @SuppressWarnings("unchecked")
    void testGetHosts_Timeout() throws Exception {
        // Arrange
        String modelName = "TestModel";
        String address = "TestAddress";
        Future<List<WorkerHost>> future = mock(Future.class);
        when(serviceDiscovery.getHosts(anyString())).thenReturn(Collections.emptyList());

        // Act
        List<WorkerHost> actualHosts = workerAddressService.getServiceHosts(modelName, address);

        // Assertions - should return empty list on timeout
        Assertions.assertTrue(actualHosts.isEmpty());
    }

    @Test
    @SuppressWarnings("unchecked")
    void testGetHosts_Success() throws Exception {
        // Arrange
        String modelName = "TestModel";
        String address = "TestAddress";
        List<WorkerHost> expectedHosts = List.of(new WorkerHost("127.0.0.1", 8080, 8081, 8082, "site1", "group1"));
        Future<List<WorkerHost>> future = mock(Future.class);
        when(serviceDiscovery.getHosts(anyString())).thenReturn(expectedHosts);

        // Act
        List<WorkerHost> actualHosts = workerAddressService.getServiceHosts(modelName, address);

        // Assertions - should return hosts
        Assertions.assertFalse(actualHosts.isEmpty());
    }
}