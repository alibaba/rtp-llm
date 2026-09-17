package org.flexlb.engine.grpc.nameresolver;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineAddressResolverTest {

    @Test
    @SuppressWarnings("unchecked")
    void notifiesEveryRegisteredListener() {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress("worker-service");
        endpoint.setProtocol(BackendServiceProtocolEnum.GRPC.getName());

        GroupRoleEndPoint roleEndpoint = new GroupRoleEndPoint();
        roleEndpoint.setGroup("default");
        roleEndpoint.setPdFusionEndpoint(endpoint);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setRoleEndpoints(List.of(roleEndpoint));

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(route.getServiceId(), route);

        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of(workerHost("10.0.0.1", 8080)));

        EngineAddressResolver resolver =
                new EngineAddressResolver(serviceDiscovery, modelMetaConfig);
        ArgumentCaptor<ServiceHostListener> discoveryListener =
                ArgumentCaptor.forClass(ServiceHostListener.class);
        verify(serviceDiscovery).listen(eq(endpoint), discoveryListener.capture());

        EngineAddressResolver.Listener grpcListener = mock(EngineAddressResolver.Listener.class);
        EngineAddressResolver.Listener cacheListener = mock(EngineAddressResolver.Listener.class);
        resolver.subscribe(grpcListener);
        resolver.subscribe(cacheListener);
        resolver.subscribe(grpcListener);

        ArgumentCaptor<List<WorkerHost>> grpcHosts = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<List<WorkerHost>> cacheHosts = ArgumentCaptor.forClass(List.class);
        verify(grpcListener).onAddressUpdate(grpcHosts.capture());
        verify(cacheListener).onAddressUpdate(cacheHosts.capture());

        assertWorkerHost(grpcHosts.getValue().get(0), "10.0.0.1", 8081, 18002);
        assertWorkerHost(cacheHosts.getValue().get(0), "10.0.0.1", 8081, 18002);

        discoveryListener.getValue().onHostsChanged(
                List.of(workerHost("10.0.0.2", 8080)));

        verify(grpcListener, times(2)).onAddressUpdate(grpcHosts.capture());
        verify(cacheListener, times(2)).onAddressUpdate(cacheHosts.capture());

        assertWorkerHost(grpcHosts.getValue().get(0), "10.0.0.2", 8081, 18002);
        assertWorkerHost(cacheHosts.getValue().get(0), "10.0.0.2", 8081, 18002);
    }

    private void assertWorkerHost(WorkerHost host, String ip, int grpcPort, int workerStatusPort) {
        assertInstanceOf(WorkerHost.class, host);
        assertEquals(ip, host.getIp());
        assertEquals(grpcPort, host.getGrpcPort());
        assertEquals(workerStatusPort, host.getWorkerStatusPort());
    }

    private WorkerHost workerHost(String ip, int httpPort) {
        return new WorkerHost(ip, httpPort, httpPort + 1, httpPort + 5,
                18002, "", "", "");
    }
}
