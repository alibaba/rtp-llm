package org.flexlb.engine.grpc.nameresolver;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineAddressResolverTest {

    @Test
    void notifiesEveryRegisteredListener() {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress("worker-service");

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
                .thenReturn(List.of(WorkerHost.of("10.0.0.1", 8080)));

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

        List<String> initialHosts = List.of("10.0.0.1:8080");
        verify(grpcListener).onAddressUpdate(initialHosts);
        verify(cacheListener).onAddressUpdate(initialHosts);

        discoveryListener.getValue().onHostsChanged(
                List.of(WorkerHost.of("10.0.0.2", 8080)));

        List<String> updatedHosts = List.of("10.0.0.2:8080");
        verify(grpcListener).onAddressUpdate(updatedHosts);
        verify(cacheListener).onAddressUpdate(updatedHosts);
    }
}
