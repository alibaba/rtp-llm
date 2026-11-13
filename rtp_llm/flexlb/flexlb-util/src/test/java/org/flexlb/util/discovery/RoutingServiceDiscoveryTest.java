package org.flexlb.util.discovery;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks routing discovery across provider results and configured groups.
 */
class RoutingServiceDiscoveryTest {

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
    void normalizesDiscoveredHostsWithEndpointProtocolAndGroup() {
        DiscoveredHost discoveredHost =
                DiscoveredHost.of("10.0.0.1", 8081, "site-a", "deployment-a");
        RecordingProvider provider = new RecordingProvider(List.of(discoveredHost));
        RoutingServiceDiscovery discovery = new RoutingServiceDiscovery(List.of(provider));
        DiscoveryEndpoint endpoint = endpoint();
        endpoint.setProtocol("grpc");
        endpoint.setGroup("group-a");

        DiscoveredHost normalizedHost = discovery.getHosts(endpoint).getFirst();

        assertEquals("10.0.0.1", normalizedHost.getIp());
        assertEquals(8080, normalizedHost.getHttpPort());
        assertEquals(8081, normalizedHost.getGrpcPort());
        assertEquals(8081, normalizedHost.getWorkerStatusPort());
        assertEquals(8085, normalizedHost.getHttpServerPort());
        assertEquals("site-a", normalizedHost.getSite());
        assertEquals("group-a", normalizedHost.getGroup());
        assertEquals("deployment-a", normalizedHost.getDeploymentName());
        assertEquals(0, normalizedHost.getEngineIndex());
        assertEquals(1, normalizedHost.getMultiEngineNum());
        assertEquals("10.0.0.1:8080", normalizedHost.getPhysicalIpPort());
        assertEquals("10.0.0.1:8080@0", normalizedHost.getLogicalIpPort());
    }

    @Test
    void usesWorkerStatusPortConfiguredOnEndpoint() {
        DiscoveredHost discoveredHost = DiscoveredHost.of("10.0.0.1", 8080);
        RecordingProvider provider = new RecordingProvider(List.of(discoveredHost));
        RoutingServiceDiscovery discovery = new RoutingServiceDiscovery(List.of(provider));
        DiscoveryEndpoint endpoint = endpoint();
        endpoint.setProtocol("http");
        endpoint.setWorkerStatusPort(18002);

        DiscoveredHost normalizedHost = discovery.getHosts(endpoint).getFirst();

        assertEquals(8080, normalizedHost.getHttpPort());
        assertEquals(8081, normalizedHost.getGrpcPort());
        assertEquals(18002, normalizedHost.getWorkerStatusPort());
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

    @Test
    void expandsDiscoveredHostIntoIndexedLogicalWorkers() {
        DiscoveredHost discoveredHost = DiscoveredHost.of("10.0.0.1", 8080);
        RecordingProvider provider = new RecordingProvider(List.of(discoveredHost));
        RoutingServiceDiscovery discovery = new RoutingServiceDiscovery(List.of(provider));
        DiscoveryEndpoint endpoint = endpoint();
        endpoint.setProtocol("http");
        endpoint.setWorkerStatusPort(18002);
        endpoint.setMultiEngineNum(2);

        List<DiscoveredHost> hosts = discovery.getHosts(endpoint);

        assertEquals(2, hosts.size());
        assertIterableEquals(
                List.of(0, 1), hosts.stream().map(DiscoveredHost::getEngineIndex).toList());
        assertIterableEquals(
                List.of(18002, 18003),
                hosts.stream().map(DiscoveredHost::getWorkerStatusPort).toList());
        assertIterableEquals(
                List.of("10.0.0.1:8080@0", "10.0.0.1:8080@1"),
                hosts.stream().map(DiscoveredHost::getLogicalIpPort).toList());
        assertIterableEquals(
                List.of("10.0.0.1:8080", "10.0.0.1:8080"),
                hosts.stream().map(DiscoveredHost::getPhysicalIpPort).toList());
        assertIterableEquals(
                List.of(2, 2), hosts.stream().map(DiscoveredHost::getMultiEngineNum).toList());
    }

    @Test
    void rejectsInvalidDerivedPortsBeforePublishingHosts() {
        DiscoveryEndpoint endpoint = endpoint();
        endpoint.setProtocol("grpc");
        for (int discoveredPort : List.of(0, 1, 65535, Integer.MAX_VALUE)) {
            RoutingServiceDiscovery discovery =
                    new RoutingServiceDiscovery(
                            List.of(
                                    new RecordingProvider(
                                            List.of(
                                                    DiscoveredHost.of(
                                                            "10.0.0.1", discoveredPort)))));
            assertThrows(IllegalArgumentException.class, () -> discovery.getHosts(endpoint));
        }
        RoutingServiceDiscovery discovery =
                new RoutingServiceDiscovery(
                        List.of(
                                new RecordingProvider(
                                        List.of(DiscoveredHost.of("10.0.0.1", 8081)))));
        endpoint.setWorkerStatusPort(65535);
        endpoint.setMultiEngineNum(2);
        assertThrows(IllegalArgumentException.class, () -> discovery.getHosts(endpoint));
        endpoint.setWorkerStatusPort(Integer.MAX_VALUE);
        assertThrows(IllegalArgumentException.class, () -> discovery.getHosts(endpoint));
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
