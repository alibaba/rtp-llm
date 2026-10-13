package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class StaticServiceDiscoveryProviderTest {

    private final StaticServiceDiscoveryProvider provider = new StaticServiceDiscoveryProvider();

    @Test
    void readsHostsFromEndpointDiscoveryConfiguration() {
        Endpoint endpoint = endpointWithHosts("127.0.0.1:8080", "127.0.0.2:8081");
        provider.validate(endpoint);

        List<WorkerHost> hosts = provider.getHosts(endpoint);

        assertEquals(2, hosts.size());
        assertEquals("127.0.0.1:8080", hosts.get(0).getIpPort());
        assertEquals("127.0.0.2:8081", hosts.get(1).getIpPort());
    }

    @Test
    void invokesListenerOnceWithConfiguredHosts() {
        Endpoint endpoint = endpointWithHosts("127.0.0.1:8080");
        AtomicReference<List<WorkerHost>> listenerHosts = new AtomicReference<>();
        provider.validate(endpoint);

        provider.listen(endpoint, listenerHosts::set);

        assertEquals("127.0.0.1:8080", listenerHosts.get().getFirst().getIpPort());
    }

    @Test
    void rejectsMissingHosts() {
        Endpoint endpoint = endpointWithHosts();

        assertThrows(IllegalArgumentException.class, () -> provider.validate(endpoint));
    }

    @Test
    void rejectsInvalidHost() {
        Endpoint endpoint = endpointWithHosts("127.0.0.1");

        assertThrows(IllegalArgumentException.class, () -> provider.validate(endpoint));
    }

    private Endpoint endpointWithHosts(String... hosts) {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(ServiceDiscoveryType.STATIC_ENV);
        discovery.setHosts(List.of(hosts));
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress("static-service");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }
}
