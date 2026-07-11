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
            return List.of();
        }

        @Override
        public void listen(Endpoint endpoint, ServiceHostListener listener) {
            listener.onHostsChanged(List.of());
        }

        @Override
        public void shutdown() {
        }
    }
}
