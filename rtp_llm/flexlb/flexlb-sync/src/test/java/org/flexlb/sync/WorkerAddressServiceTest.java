package org.flexlb.sync;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;

import static org.mockito.ArgumentMatchers.any;
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
    void testGetHosts_Empty() {
        // Arrange
        String modelName = "TestModel";
        Endpoint endpoint = endpoint("TestAddress");
        when(serviceDiscovery.getHosts(any(Endpoint.class))).thenReturn(Collections.emptyList());

        // Act
        List<WorkerHost> actualHosts = workerAddressService.getServiceHosts(modelName, endpoint);

        // Assertions - should return empty list on timeout
        Assertions.assertTrue(actualHosts.isEmpty());
    }

    @Test
    void testGetHosts_Success() {
        // Arrange
        String modelName = "TestModel";
        Endpoint endpoint = endpoint("TestAddress");
        List<WorkerHost> expectedHosts = List.of(new WorkerHost("127.0.0.1", 8080, 8081, 8082, "site1", "group1"));
        when(serviceDiscovery.getHosts(any(Endpoint.class))).thenReturn(expectedHosts);

        // Act
        List<WorkerHost> actualHosts = workerAddressService.getServiceHosts(modelName, endpoint);

        // Assertions - should return hosts
        Assertions.assertFalse(actualHosts.isEmpty());
    }

    private Endpoint endpoint(String address) {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        return endpoint;
    }
}
