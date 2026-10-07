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
import static org.mockito.ArgumentMatchers.argThat;
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

    @Test
    void emptyPollsAndNullOrEmptyPushesRetainAddressesUntilNonEmptyRecovery() {
        Endpoint first = new Endpoint();
        first.setAddress("vip-a");
        Endpoint second = new Endpoint();
        second.setAddress("vip-b");
        ServiceRoute route = mock(ServiceRoute.class);
        when(route.getAllEndpoints()).thenReturn(List.of(first, second));
        ModelMetaConfig config = mock(ModelMetaConfig.class);
        when(config.getServiceRoutes()).thenReturn(List.of(route));
        ServiceDiscovery discovery = mock(ServiceDiscovery.class);
        WorkerHost a = workerHost("10.0.0.1", 8080);
        WorkerHost b = workerHost("10.0.0.2", 8080);
        WorkerHost c = workerHost("10.0.0.3", 8080);
        when(discovery.getHosts(first)).thenReturn(List.of(a), List.of())
                .thenThrow(new IllegalStateException("unreachable"));
        when(discovery.getHosts(second)).thenReturn(List.of(b), List.of(c));
        EngineAddressResolver resolver = new EngineAddressResolver(discovery, config);
        ArgumentCaptor<ServiceHostListener> callback = ArgumentCaptor.forClass(ServiceHostListener.class);
        verify(discovery).listen(eq(first), callback.capture());
        EngineAddressResolver.Listener listener = mock(EngineAddressResolver.Listener.class);
        resolver.subscribe(listener);
        callback.getValue().onHostsChanged(null);
        callback.getValue().onHostsChanged(List.of());
        verify(listener).onAddressUpdate(argThat(
                hosts -> hosts.size() == 2 && hosts.containsAll(List.of(a, b))));

        resolver.periodicHostUpdate();
        resolver.periodicHostUpdate();
        EngineAddressResolver.Listener latest = mock(EngineAddressResolver.Listener.class);
        resolver.subscribe(latest);
        verify(latest).onAddressUpdate(argThat(
                hosts -> hosts.size() == 2 && hosts.containsAll(List.of(a, c))));

        callback.getValue().onHostsChanged(List.of(b));
        verify(listener).onAddressUpdate(argThat(
                hosts -> hosts.size() == 2 && hosts.containsAll(List.of(b, c))));
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
