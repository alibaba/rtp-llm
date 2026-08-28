package org.flexlb.service.address;

import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.mockito.Mockito.anyString;
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
    private ConfigService configService;

    private WorkerAddressService workerAddressService;

    @BeforeEach
    void setUp() {
        Mockito.lenient().when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        workerAddressService = new WorkerAddressService(engineHealthReporter, modelMetaConfig,
                serviceDiscovery, configService);
    }

    @AfterEach
    void tearDown() {
        workerAddressService.destroy();
    }

    @Test
    void discoveryFailureReturnsNoWorkers() {
        String modelName = "TestModel";
        String address = "TestAddress";
        when(modelMetaConfig.endpointsWithGroup(modelName, RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group1", endpoint(address))));
        when(serviceDiscovery.getHosts(anyString()))
                .thenThrow(new IllegalStateException("discovery unavailable"));

        List<WorkerHost> actualHosts = workerAddressService.getEngineWorkerList(
                modelName, RoleType.PREFILL);

        Assertions.assertTrue(actualHosts.isEmpty());
    }

    @Test
    void discoveredWorkersAreConvertedThroughThePublicBoundary() {
        String modelName = "TestModel";
        String address = "TestAddress";
        List<WorkerHost> expectedHosts = List.of(new WorkerHost("127.0.0.1", 8080, 8081, 8082, "site1", "group1"));
        when(modelMetaConfig.endpointsWithGroup(modelName, RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("group1", endpoint(address))));
        when(serviceDiscovery.getHosts(anyString())).thenReturn(expectedHosts);

        List<WorkerHost> actualHosts = workerAddressService.getEngineWorkerList(
                modelName, RoleType.PREFILL);

        Assertions.assertFalse(actualHosts.isEmpty());
    }

    private static Endpoint endpoint(String address) {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol(BackendServiceProtocolEnum.GRPC.getName());
        return endpoint;
    }
}
