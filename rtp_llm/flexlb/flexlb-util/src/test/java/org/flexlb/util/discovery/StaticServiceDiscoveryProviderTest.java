package org.flexlb.util.discovery;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks static discovery parsing and returned worker addresses.
 */
class StaticServiceDiscoveryProviderTest {

    private final StaticServiceDiscoveryProvider provider = new StaticServiceDiscoveryProvider();

    @Test
    void readsHostsFromEndpointDiscoveryConfiguration() {
        DiscoveryEndpoint endpoint = endpointWithHosts("127.0.0.1:8080", "127.0.0.2:8081");
        provider.validate(endpoint);

        List<DiscoveredHost> hosts = provider.getHosts(endpoint);

        assertEquals(2, hosts.size());
        assertEquals("127.0.0.1:8080", hosts.get(0).getIpPort());
        assertEquals("127.0.0.2:8081", hosts.get(1).getIpPort());
    }

    @Test
    void invokesListenerOnceWithConfiguredHosts() {
        DiscoveryEndpoint endpoint = endpointWithHosts("127.0.0.1:8080");
        AtomicReference<List<DiscoveredHost>> listenerHosts = new AtomicReference<>();
        provider.validate(endpoint);

        provider.listen(endpoint, listenerHosts::set);

        assertEquals("127.0.0.1:8080", listenerHosts.get().getFirst().getIpPort());
    }

    @Test
    void rejectsMissingHosts() {
        DiscoveryEndpoint endpoint = endpointWithHosts();

        assertThrows(IllegalArgumentException.class, () -> provider.validate(endpoint));
    }

    @Test
    void rejectsInvalidHost() {
        DiscoveryEndpoint endpoint = endpointWithHosts("127.0.0.1");

        assertThrows(IllegalArgumentException.class, () -> provider.validate(endpoint));
    }

    private DiscoveryEndpoint endpointWithHosts(String... hosts) {
        DiscoverySettings discovery = new DiscoverySettings();
        discovery.setType(ServiceDiscoveryType.STATIC_ENV);
        discovery.setHosts(List.of(hosts));
        DiscoveryEndpoint endpoint = new DiscoveryEndpoint();
        endpoint.setAddress("static-service");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }
}
