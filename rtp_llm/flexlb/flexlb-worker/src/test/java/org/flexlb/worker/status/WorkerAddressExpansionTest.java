package org.flexlb.worker.status;

import org.flexlb.config.service.WorkerDiscoveryEndpoint;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.util.discovery.DiscoveryClientSettings;
import org.flexlb.util.discovery.DiscoverySettings;
import org.flexlb.util.discovery.RoutingServiceDiscovery;
import org.flexlb.util.discovery.ServiceDiscoveryProvider;
import org.flexlb.util.discovery.ServiceDiscoveryType;
import org.flexlb.util.discovery.ServiceHostListener;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks Worker-specific port derivation and logical Engine expansion after discovery.
 */
class WorkerAddressExpansionTest {
    @Test
    void normalizesDiscoveredHostsWithEndpointProtocolAndGroup() {
        DiscoveredHost discoveredHost =
                DiscoveredHost.of("10.0.0.1", 8081, "site-a", "deployment-a");
        RecordingProvider provider = new RecordingProvider(List.of(discoveredHost));
        RoutingServiceDiscovery discovery = new RoutingServiceDiscovery(List.of(provider));
        WorkerDiscoveryEndpoint endpoint = endpoint();
        endpoint.setProtocol("grpc");
        endpoint.setGroup("group-a");

        WorkerHost normalizedHost =
                WorkerAddressService.expandWorkerHosts(discovery.getHosts(endpoint), endpoint)
                        .getFirst();

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
        WorkerDiscoveryEndpoint endpoint = endpoint();
        endpoint.setProtocol("http");
        endpoint.setWorkerStatusPort(18002);

        WorkerHost normalizedHost =
                WorkerAddressService.expandWorkerHosts(discovery.getHosts(endpoint), endpoint)
                        .getFirst();

        assertEquals(8080, normalizedHost.getHttpPort());
        assertEquals(8081, normalizedHost.getGrpcPort());
        assertEquals(18002, normalizedHost.getWorkerStatusPort());
    }

    @Test
    void expandsDiscoveredHostIntoIndexedLogicalWorkers() {
        DiscoveredHost discoveredHost = DiscoveredHost.of("10.0.0.1", 8080);
        RecordingProvider provider = new RecordingProvider(List.of(discoveredHost));
        RoutingServiceDiscovery discovery = new RoutingServiceDiscovery(List.of(provider));
        WorkerDiscoveryEndpoint endpoint = endpoint();
        endpoint.setProtocol("http");
        endpoint.setWorkerStatusPort(18002);
        endpoint.setMultiEngineNum(2);

        List<WorkerHost> hosts =
                WorkerAddressService.expandWorkerHosts(discovery.getHosts(endpoint), endpoint);

        assertEquals(2, hosts.size());
        assertIterableEquals(
                List.of(0, 1), hosts.stream().map(WorkerHost::getEngineIndex).toList());
        assertIterableEquals(
                List.of(18002, 18003),
                hosts.stream().map(WorkerHost::getWorkerStatusPort).toList());
        assertIterableEquals(
                List.of("10.0.0.1:8080@0", "10.0.0.1:8080@1"),
                hosts.stream().map(WorkerHost::getLogicalIpPort).toList());
        assertIterableEquals(
                List.of("10.0.0.1:8080", "10.0.0.1:8080"),
                hosts.stream().map(WorkerHost::getPhysicalIpPort).toList());
        assertIterableEquals(
                List.of(2, 2), hosts.stream().map(WorkerHost::getMultiEngineNum).toList());
    }

    @Test
    void rejectsInvalidDerivedPortsBeforePublishingHosts() {
        WorkerDiscoveryEndpoint endpoint = endpoint();
        endpoint.setProtocol("grpc");
        for (int discoveredPort : List.of(0, 1, 65535, Integer.MAX_VALUE)) {
            RoutingServiceDiscovery discovery =
                    new RoutingServiceDiscovery(
                            List.of(
                                    new RecordingProvider(
                                            List.of(
                                                    DiscoveredHost.of(
                                                            "10.0.0.1", discoveredPort)))));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            WorkerAddressService.expandWorkerHosts(
                                    discovery.getHosts(endpoint), endpoint));
        }
        RoutingServiceDiscovery discovery =
                new RoutingServiceDiscovery(
                        List.of(
                                new RecordingProvider(
                                        List.of(DiscoveredHost.of("10.0.0.1", 8081)))));
        endpoint.setWorkerStatusPort(65535);
        endpoint.setMultiEngineNum(2);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        WorkerAddressService.expandWorkerHosts(
                                discovery.getHosts(endpoint), endpoint));
        endpoint.setWorkerStatusPort(Integer.MAX_VALUE);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        WorkerAddressService.expandWorkerHosts(
                                discovery.getHosts(endpoint), endpoint));
    }

    private WorkerDiscoveryEndpoint endpoint() {
        DiscoverySettings discovery = new DiscoverySettings();
        discovery.setType(ServiceDiscoveryType.STATIC_ENV);
        WorkerDiscoveryEndpoint endpoint = new WorkerDiscoveryEndpoint();
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
        public void validate(org.flexlb.util.discovery.DiscoveryEndpoint endpoint) {
            validationCount.incrementAndGet();
        }

        @Override
        public List<DiscoveredHost> getHosts(org.flexlb.util.discovery.DiscoveryEndpoint endpoint) {
            return hosts;
        }

        @Override
        public void listen(
                org.flexlb.util.discovery.DiscoveryEndpoint endpoint,
                ServiceHostListener listener) {
            listener.onHostsChanged(hosts);
        }

        @Override
        public void shutdown() {}
    }
}
