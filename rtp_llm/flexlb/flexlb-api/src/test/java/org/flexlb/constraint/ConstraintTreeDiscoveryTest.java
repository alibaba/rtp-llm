package org.flexlb.constraint;

import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class ConstraintTreeDiscoveryTest {
    @Test
    void bootstrapSeesUnreadyWorkersWithoutChangingInferenceDiscovery() {
        var config = mock(ModelMetaConfig.class);
        var route = mock(ServiceRoute.class);
        when(config.getServiceRoute(any())).thenReturn(route);
        var endpoint = new Endpoint();
        endpoint.setAddress("test-pool");
        endpoint.setProtocol("http");
        when(route.getAllEndpointsWithGroup(RoleType.PDFUSION))
                .thenReturn(List.of(Pair.of("default", endpoint)));
        var discovery = mock(ServiceDiscovery.class);
        when(discovery.getHosts("test-pool")).thenReturn(List.of());
        when(discovery.getAllHosts("test-pool"))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 18000, "local")));
        var addresses = new WorkerAddressService(mock(EngineHealthReporter.class), config, discovery);
        assertTrue(addresses.getEngineWorkerList("test-model", RoleType.PDFUSION).isEmpty());
        var targets = addresses.getAllEngineWorkerList("test-model", RoleType.PDFUSION);
        assertEquals(1, targets.size());
        assertEquals(18005, targets.getFirst().getHttpServerPort());
        assertTrue(addresses.getEngineWorkerList("test-model", RoleType.PDFUSION).isEmpty());
        verify(discovery, times(2)).getHosts("test-pool");
        verify(discovery).getAllHosts("test-pool");
    }
}
