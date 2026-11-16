package org.flexlb.util.discovery;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks provider selection and preserves provider addresses for queries and subscriptions.
 */
class RoutingServiceDiscoveryTest {

    @Test
    void discoveryPreservesProviderPortWithoutWorkerPortRules() {
        var host = DiscoveredHost.of("127.0.0.1", 65535);
        var discovery = new RoutingServiceDiscovery(List.of(new RecordingProvider(List.of(host))));
        var endpoint = endpoint();
        endpoint.setProtocol("grpc");
        org.junit.jupiter.api.Assertions.assertSame(host, discovery.getHosts(endpoint).getFirst());
        assertEquals(1, discovery.getHosts(endpoint).size());
        discovery.listen(
                endpoint,
                hosts -> org.junit.jupiter.api.Assertions.assertSame(host, hosts.getFirst()));
    }

    @Test
    void rejectsInvalidDiscoveryConfigurationOnQueriesAndSubscriptions() {
        RoutingServiceDiscovery discovery =
                new RoutingServiceDiscovery(List.of(new RecordingProvider()));
        DiscoveryEndpoint missingDiscovery = endpoint();
        missingDiscovery.setDiscovery(null);
        DiscoveryEndpoint missingType = endpoint();
        missingType.getDiscovery().setType(null);
        DiscoveryEndpoint missingProvider = endpoint();
        missingProvider.getDiscovery().setType(ServiceDiscoveryType.VIPSERVER);

        for (DiscoveryEndpoint endpoint :
                new DiscoveryEndpoint[] {null, missingDiscovery, missingType, missingProvider}) {
            assertThrows(IllegalArgumentException.class, () -> discovery.getHosts(endpoint));
            assertThrows(
                    IllegalArgumentException.class, () -> discovery.listen(endpoint, hosts -> {}));
        }
    }

    @Test
    void doesNotValidateOnRuntimeQueries() {
        RecordingProvider provider = new RecordingProvider();
        RoutingServiceDiscovery discovery = new RoutingServiceDiscovery(List.of(provider));
        DiscoveryEndpoint endpoint = endpoint();

        discovery.validate(endpoint);
        discovery.getHosts(endpoint);
        discovery.listen(endpoint, hosts -> {});

        assertEquals(1, provider.validationCount.get());
    }

    @Test
    void exposesCurrentFlexlbRuntimePolicyToDiscoveryProviders() {
        AtomicReference<DiscoveryClientSettings> current =
                new AtomicReference<>(runtimeConfig(700));
        RecordingProvider provider = new RecordingProvider();
        RoutingServiceDiscovery discovery =
                new RoutingServiceDiscovery(List.of(provider), current::get);
        DiscoveryEndpoint endpoint = endpoint();

        discovery.validate(endpoint);
        assertEquals(700, endpoint.getDiscovery().getConnectTimeoutMs());

        current.set(runtimeConfig(900));
        assertEquals(900, endpoint.getDiscovery().getConnectTimeoutMs());
    }

    private DiscoveryEndpoint endpoint() {
        DiscoverySettings discovery = new DiscoverySettings();
        discovery.setType(ServiceDiscoveryType.STATIC_ENV);
        DiscoveryEndpoint endpoint = new DiscoveryEndpoint();
        endpoint.setAddress("static-service");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }

    private DiscoveryClientSettings runtimeConfig(int connectTimeoutMs) {
        DiscoveryClientSettings config = new DiscoveryClientSettings();
        config.setConnectTimeoutMs(connectTimeoutMs);
        return config;
    }

    /**
     * Records discovery lookups and returns test-controlled worker lists.
     */
    private static class RecordingProvider implements ServiceDiscoveryProvider {

        private final AtomicInteger validationCount = new AtomicInteger();
        private final List<DiscoveredHost> hosts;

        private RecordingProvider() {
            this(List.of());
        }

        private RecordingProvider(List<DiscoveredHost> hosts) {
            this.hosts = hosts;
        }

        @Override
        public ServiceDiscoveryType getType() {
            return ServiceDiscoveryType.STATIC_ENV;
        }

        @Override
        public void validate(DiscoveryEndpoint endpoint) {
            validationCount.incrementAndGet();
        }

        @Override
        public List<DiscoveredHost> getHosts(DiscoveryEndpoint endpoint) {
            return hosts;
        }

        @Override
        public void listen(DiscoveryEndpoint endpoint, ServiceHostListener listener) {
            listener.onHostsChanged(hosts);
        }

        @Override
        public void shutdown() {}
    }
}
