package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RoutingServiceDiscoveryTest {

    @Test
    void doesNotValidateOnRuntimeQueries() {
        RecordingProvider provider = new RecordingProvider();
        RoutingServiceDiscovery discovery = new RoutingServiceDiscovery(List.of(provider));
        Endpoint endpoint = endpoint();

        discovery.validate(endpoint);
        discovery.getHosts(endpoint);
        discovery.listen(endpoint, hosts -> {
        });

        assertEquals(1, provider.validationCount.get());
    }

    @Test
    void normalizesDiscoveredHostsWithEndpointProtocolAndGroup() {
        WorkerHost discoveredHost =
                WorkerHost.of("10.0.0.1", 8081, "site-a", "deployment-a");
        RecordingProvider provider = new RecordingProvider(List.of(discoveredHost));
        RoutingServiceDiscovery discovery = new RoutingServiceDiscovery(List.of(provider));
        Endpoint endpoint = endpoint();
        endpoint.setProtocol("grpc");
        endpoint.setGroup("group-a");

        WorkerHost normalizedHost = discovery.getHosts(endpoint).getFirst();

        assertEquals("10.0.0.1", normalizedHost.getIp());
        assertEquals(8080, normalizedHost.getHttpPort());
        assertEquals(8081, normalizedHost.getGrpcPort());
        assertEquals(8085, normalizedHost.getHttpServerPort());
        assertEquals("site-a", normalizedHost.getSite());
        assertEquals("group-a", normalizedHost.getGroup());
        assertEquals("deployment-a", normalizedHost.getDeploymentName());
    }

    private Endpoint endpoint() {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(ServiceDiscoveryType.STATIC_ENV);
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress("static-service");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }

    private static class RecordingProvider implements ServiceDiscoveryProvider {

        private final AtomicInteger validationCount = new AtomicInteger();
        private final List<WorkerHost> hosts;

        private RecordingProvider() {
            this(List.of());
        }

        private RecordingProvider(List<WorkerHost> hosts) {
            this.hosts = hosts;
        }

        @Override
        public ServiceDiscoveryType getType() {
            return ServiceDiscoveryType.STATIC_ENV;
        }

        @Override
        public void validate(Endpoint endpoint) {
            validationCount.incrementAndGet();
        }

        @Override
        public List<WorkerHost> getHosts(Endpoint endpoint) {
            return hosts;
        }

        @Override
        public void listen(Endpoint endpoint, ServiceHostListener listener) {
            listener.onHostsChanged(hosts);
        }

        @Override
        public void shutdown() {
        }
    }
}
