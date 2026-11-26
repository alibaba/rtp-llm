package org.flexlb;

import org.flexlb.config.ServiceDiscoveryConfiguration;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryProvider;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.discovery.ServiceHostListener;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks Mock DP configuration, VIP updates and integration with the discovery router.
 */
class MockDpServiceDiscoveryProviderTest {

    @Test
    void mockStartupWrapsTheVipProviderBeforeTheProductionRouterUsesIt() {
        try (var context = new AnnotationConfigApplicationContext()) {
            WhaleMockMasterMain.installDiscovery(
                    context.getBeanFactory(), Map.of("decode.test", 2));
            context.registerBean("vipServerDiscovery", RecordingVip.class, RecordingVip::new);
            context.register(ServiceDiscoveryConfiguration.class);
            context.refresh();

            var endpoint = new Endpoint();
            endpoint.setAddress("decode.test");
            endpoint.setProtocol("http");
            var discovery = new DiscoveryConfig();
            discovery.setType(ServiceDiscoveryType.VIPSERVER);
            endpoint.setDiscovery(discovery);
            var router = context.getBean(RoutingServiceDiscovery.class);
            router.validate(endpoint);
            var hosts = router.getHosts(endpoint);
            assertEquals(List.of(8089, 8099), hosts.stream().map(WorkerHost::getGrpcPort).toList());
            assertEquals(
                    List.of(8089, 8099),
                    hosts.stream().map(WorkerHost::getWorkerStatusPort).toList());
        }
    }

    @Test
    void readsPerRoleDpCountsAndDefaultsToOne() throws Exception {
        assertEquals(
                Map.of("prefill.test", 2, "decode.test", 8),
                MockDpServiceDiscoveryProvider.readLayout(
                        """
                {"role_endpoints":[
                  {"prefill_endpoint":{"address":"prefill.test","protocol":"http","discovery":{"type":"vipserver"},
                                       "mock_dp_per_pod":2},
                   "decode_endpoint":{"address":"decode.test","protocol":"http","discovery":{"type":"vipserver"},
                                      "mock_dp_per_pod":8}},
                  {"decode_endpoint":{"address":"single.test","discovery":{"type":"vipserver"},
                                      "mock_dp_per_pod":1}}
                ]}
                """));
    }

    @Test
    void rejectsMixedDpLayoutOnOneVip() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        MockDpServiceDiscoveryProvider.readLayout(
                                """
                {"role_endpoints":[
                  {"decode_endpoint":{"address":"shared.test","protocol":"http","discovery":{"type":"vipserver"},
                                      "mock_dp_per_pod":8}},
                  {"decode_endpoint":{"address":"shared.test","protocol":"http","discovery":{"type":"vipserver"}}}
                ]}
                """));
    }

    @Test
    void expandsSnapshotsAndRemovesEveryDpWhenTheVipMemberGoesAway() {
        var vip = new RecordingVip();
        var provider = new MockDpServiceDiscoveryProvider(vip, Map.of("decode.test", 8));
        var endpoint = new Endpoint();
        endpoint.setAddress("decode.test");
        var observed = new AtomicReference<List<WorkerHost>>();
        provider.listen(endpoint, observed::set);

        var hosts = provider.getHosts(endpoint);
        assertEquals(
                List.of(8088, 8098, 8108, 8118, 8128, 8138, 8148, 8158),
                hosts.stream().map(WorkerHost::getPort).toList());
        assertEquals("site", hosts.getLast().getSite());
        assertEquals("deployment", hosts.getLast().getDeploymentName());

        vip.listener.onHostsChanged(List.of(WorkerHost.of("192.0.2.2", 8088)));
        assertEquals(8, observed.get().size());
        assertEquals(
                List.of("192.0.2.2"),
                observed.get().stream().map(WorkerHost::getIp).distinct().toList());
        vip.listener.onHostsChanged(List.of());
        assertEquals(List.of(), observed.get());

        endpoint.setAddress("normal.test");
        assertSame(vip.hosts, provider.getHosts(endpoint));
    }

    /**
     * Provides repeatable VIP members and captures the host-change listener.
     */
    private static final class RecordingVip implements ServiceDiscoveryProvider {
        private final List<WorkerHost> hosts =
                List.of(WorkerHost.of("192.0.2.1", 8088, "site", "deployment"));
        private ServiceHostListener listener;

        @Override
        public ServiceDiscoveryType getType() {
            return ServiceDiscoveryType.VIPSERVER;
        }

        @Override
        public void validate(Endpoint endpoint) {}

        @Override
        public List<WorkerHost> getHosts(Endpoint endpoint) {
            return hosts;
        }

        @Override
        public void listen(Endpoint endpoint, ServiceHostListener listener) {
            this.listener = listener;
        }

        @Override
        public void shutdown() {}
    }
}
