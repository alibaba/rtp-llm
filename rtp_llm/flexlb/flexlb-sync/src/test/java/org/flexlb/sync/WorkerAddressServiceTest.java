package org.flexlb.sync;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.exception.ServiceDiscoveryException;
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
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.mockito.Mockito.anyString;
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
    void testGetHosts_EmptySuccess() throws Exception {
        // Arrange - discovery succeeds but the service genuinely has no hosts
        String modelName = "TestModel";
        String address = "TestAddress";
        when(serviceDiscovery.getHosts(anyString())).thenReturn(Collections.emptyList());

        // Act
        List<WorkerHost> actualHosts = workerAddressService.getServiceHosts(modelName, address);

        // Assertions - a successful empty result is returned as-is
        Assertions.assertTrue(actualHosts.isEmpty());
    }

    @Test
    void testGetHosts_ErrorThrows() throws Exception {
        // Arrange - the discovery client itself fails
        when(serviceDiscovery.getHosts(anyString())).thenThrow(new RuntimeException("vipserver down"));

        // Assertions - failure must be distinguishable from an empty fleet
        Assertions.assertThrows(ServiceDiscoveryException.class,
                () -> workerAddressService.getServiceHosts("TestModel", "TestAddress"));
    }

    @Test
    void testGetHosts_TimeoutThrows() throws Exception {
        // Arrange - discovery hangs past the 500ms budget
        when(serviceDiscovery.getHosts(anyString())).thenAnswer(invocation -> {
            TimeUnit.MILLISECONDS.sleep(800);
            return Collections.emptyList();
        });

        Assertions.assertThrows(ServiceDiscoveryException.class,
                () -> workerAddressService.getServiceHosts("TestModel", "TestAddress"));
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

    @Test
    void testGetEngineWorkerList_OneGroupDiscoveryFailure_AbortsWholeRoleRefresh() throws Exception {
        // Multi-group role where one group's discovery fails. getEngineWorkerList has no per-group
        // try/catch, so a single group's outage aborts the whole round (ServiceDiscoveryException
        // propagates) rather than returning the healthy group's hosts. EngineSyncRunner's catch then
        // refreshes the staleness clock so no healthy worker is evicted during the outage. Pin this
        // contract so the partial-discovery behavior cannot silently regress to "merge what we got".
        RoleType roleType = RoleType.PDFUSION;

        Endpoint healthyEndpoint = new Endpoint();
        healthyEndpoint.setAddress("healthy-address");
        GroupRoleEndPoint healthyGroup = new GroupRoleEndPoint();
        healthyGroup.setGroup("group-healthy");
        healthyGroup.setPdFusionEndpoint(healthyEndpoint);

        Endpoint failingEndpoint = new Endpoint();
        failingEndpoint.setAddress("failing-address");
        GroupRoleEndPoint failingGroup = new GroupRoleEndPoint();
        failingGroup.setGroup("group-failing");
        failingGroup.setPdFusionEndpoint(failingEndpoint);

        ServiceRoute route = new ServiceRoute();
        route.setRoleEndpoints(List.of(healthyGroup, failingGroup));

        when(modelMetaConfig.getServiceRoute(anyString())).thenReturn(route);
        when(serviceDiscovery.getHosts("healthy-address"))
                .thenReturn(List.of(new WorkerHost("10.0.0.1", 8080, 8081, 8082, "site1", "group-healthy")));
        when(serviceDiscovery.getHosts("failing-address"))
                .thenThrow(new RuntimeException("vipserver down"));

        Assertions.assertThrows(ServiceDiscoveryException.class,
                () -> workerAddressService.getEngineWorkerList("TestModel", roleType));
    }
}