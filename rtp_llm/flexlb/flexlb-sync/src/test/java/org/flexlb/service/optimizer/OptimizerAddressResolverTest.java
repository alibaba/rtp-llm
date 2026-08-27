package org.flexlb.service.optimizer;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class OptimizerAddressResolverTest {

    private static final String DOMAIN = "optimizer.test.domain.com";
    private static final int PORT = 8082;
    private static final long POLL_INTERVAL_MS = 10L;

    @Test
    void vipserver_should_refresh_by_polling_without_installing_private_listener() throws Exception {
        assertDynamicDiscoveryRefreshes(ServiceDiscoveryType.VIPSERVER);
    }

    @Test
    void dashscope_should_refresh_by_polling_without_installing_private_listener() throws Exception {
        assertDynamicDiscoveryRefreshes(ServiceDiscoveryType.DASHSCOPE);
    }

    @Test
    void configured_port_should_override_discovered_ports() throws Exception {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(ServiceDiscoveryType.STATIC_ENV);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080), WorkerHost.of("127.0.0.2", 9000)));
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);

        resolver.start();
        assertEquals(List.of("127.0.0.1:8082", "127.0.0.2:8082"), resolver.getAddresses());
        resolver.shutdown();
    }

    @Test
    void dynamic_discovery_should_run_initial_refresh_on_scheduler() {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(ServiceDiscoveryType.VIPSERVER);
        AtomicReference<String> refreshThread = new AtomicReference<>();
        when(serviceDiscovery.getHosts(endpoint)).thenAnswer(invocation -> {
            refreshThread.compareAndSet(null, Thread.currentThread().getName());
            return List.of(WorkerHost.of("1.1.1.1", 8000));
        });
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);

        resolver.start();

        verify(serviceDiscovery, timeout(1000).atLeastOnce()).getHosts(endpoint);
        assertTrue(refreshThread.get().startsWith("optimizer-discovery-refresh"));
        resolver.shutdown();
    }

    @Test
    void static_env_should_pull_once_without_polling_or_listener() throws Exception {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(ServiceDiscoveryType.STATIC_ENV);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);

        resolver.start();
        assertEquals(List.of("127.0.0.1:8082"), resolver.getAddresses());
        verify(serviceDiscovery, timeout(100).times(1)).getHosts(endpoint);
        verify(serviceDiscovery, never()).validate(endpoint);
        verify(serviceDiscovery, never()).listen(any(), any());
        resolver.shutdown();
    }

    @Test
    void should_recover_from_initial_pull_failure_in_background() throws Exception {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(ServiceDiscoveryType.VIPSERVER);
        when(serviceDiscovery.getHosts(endpoint))
                .thenThrow(new RuntimeException("temporary failure"))
                .thenReturn(List.of(WorkerHost.of("4.4.4.4", 7000)));
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);

        resolver.start();
        assertTrue(resolver.getAddresses().isEmpty());
        verify(serviceDiscovery, timeout(1000).atLeast(2)).getHosts(endpoint);
        awaitAddresses(resolver, List.of("4.4.4.4:8082"));
        resolver.shutdown();
    }

    @Test
    void successful_empty_refresh_should_clear_cached_addresses() throws Exception {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(ServiceDiscoveryType.DASHSCOPE);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of(WorkerHost.of("1.1.1.1", 8000)))
                .thenReturn(List.of());
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);

        resolver.start();
        awaitAddresses(resolver, List.of("1.1.1.1:8082"));
        awaitAddresses(resolver, List.of());
        resolver.shutdown();
    }

    @Test
    void invalidRefreshHostsShouldKeepThePreviousSnapshot() throws Exception {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(ServiceDiscoveryType.DASHSCOPE);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of(WorkerHost.of("1.1.1.1", 8000)))
                .thenReturn(List.of(WorkerHost.of("", 8000)));
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);

        resolver.start();
        awaitAddresses(resolver, List.of("1.1.1.1:8082"));
        verify(serviceDiscovery, timeout(1000).atLeast(2)).getHosts(endpoint);
        assertEquals(List.of("1.1.1.1:8082"), resolver.getAddresses());
        resolver.shutdown();
    }

    @Test
    void get_addresses_should_only_read_snapshot() throws Exception {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(ServiceDiscoveryType.VIPSERVER);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of(WorkerHost.of("1.1.1.1", 8000)));
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);
        resolver.start();
        awaitAddresses(resolver, List.of("1.1.1.1:8082"));
        resolver.shutdown();
        clearInvocations(serviceDiscovery);

        assertEquals(List.of("1.1.1.1:8082"), resolver.getAddresses());
        assertEquals(List.of("1.1.1.1:8082"), resolver.getAddresses());

        verifyNoInteractions(serviceDiscovery);
    }

    @Test
    void start_should_be_idempotent_and_shutdown_should_not_close_shared_discovery() {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(ServiceDiscoveryType.STATIC_ENV);
        when(serviceDiscovery.getHosts(endpoint)).thenReturn(List.of());
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);

        resolver.start();
        resolver.start();
        verify(serviceDiscovery, times(1)).getHosts(endpoint);

        resolver.shutdown();
        resolver.start();
        verify(serviceDiscovery, never()).shutdown();
    }

    @Test
    void should_return_empty_before_start() {
        OptimizerAddressResolver resolver = new OptimizerAddressResolver(
                mock(ServiceDiscovery.class),
                endpoint(ServiceDiscoveryType.VIPSERVER),
                PORT,
                POLL_INTERVAL_MS);

        assertTrue(resolver.getAddresses().isEmpty());
        resolver.shutdown();
    }

    private static void assertDynamicDiscoveryRefreshes(ServiceDiscoveryType type) throws Exception {
        ServiceDiscovery serviceDiscovery = mock(ServiceDiscovery.class);
        Endpoint endpoint = endpoint(type);
        when(serviceDiscovery.getHosts(endpoint))
                .thenReturn(List.of(WorkerHost.of("1.1.1.1", 8000)))
                .thenReturn(List.of(WorkerHost.of("2.2.2.2", 9000)));
        OptimizerAddressResolver resolver =
                new OptimizerAddressResolver(
                        serviceDiscovery, endpoint, PORT, POLL_INTERVAL_MS);

        resolver.start();
        verify(serviceDiscovery, timeout(1000).atLeast(2)).getHosts(endpoint);
        awaitAddresses(resolver, List.of("2.2.2.2:8082"));
        verify(serviceDiscovery, never()).listen(any(), any());
        resolver.shutdown();
    }

    private static void awaitAddresses(
            OptimizerAddressResolver resolver,
            List<String> expected) throws Exception {
        long deadline = System.nanoTime() + 1_000_000_000L;
        while (!resolver.getAddresses().equals(expected) && System.nanoTime() < deadline) {
            Thread.sleep(5);
        }
        assertEquals(expected, resolver.getAddresses());
    }

    private static Endpoint endpoint(ServiceDiscoveryType type) {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(type);
        if (type == ServiceDiscoveryType.STATIC_ENV) {
            discovery.setHosts(List.of("127.0.0.1:8080"));
        }
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(DOMAIN);
        endpoint.setProtocol("http");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }
}
