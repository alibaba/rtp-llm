package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.enums.LogLevel;
import org.flexlb.service.address.FlexlbInstanceAddressService;
import org.flexlb.service.monitor.FlexlbLogManager;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class HttpLoadBalanceServerTest {

    @Test
    void masterInfoUsesCanonicalSchedulerQueueDepth() {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        ConfigService configService = mock(ConfigService.class);
        PriorityScheduler scheduler = mock(PriorityScheduler.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        MasterEngineSynchronizer synchronizer = mock(MasterEngineSynchronizer.class);
        FlexlbInstanceAddressService instanceAddressService = mock(FlexlbInstanceAddressService.class);
        FlexlbLogManager flexlbLogManager = mock(FlexlbLogManager.class);
        when(consistency.getMasterHostIpPort()).thenReturn("127.0.0.1:7001");
        when(scheduler.getQueuedRequestCount()).thenReturn(7);
        when(synchronizer.isReady()).thenReturn(true);
        when(instanceAddressService.getPodIp()).thenReturn("10.0.0.8");
        when(instanceAddressService.getInstanceIp()).thenReturn("192.168.0.8");

        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                consistency,
                configService,
                scheduler,
                endpointRegistry,
                synchronizer,
                new ServerScheduleLatencyRecorder(),
                instanceAddressService,
                flexlbLogManager);
        WebTestClient client = WebTestClient
                .bindToRouterFunction(server.loadBalancePrefill())
                .build();

        client.post()
                .uri("/rtp_llm/master/info")
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.APPLICATION_JSON)
                .bodyValue("{}")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.queue_length").isEqualTo(7)
                .jsonPath("$.real_master_host").isEqualTo("127.0.0.1:7001")
                .jsonPath("$.pod_ip").isEqualTo("10.0.0.8")
                .jsonPath("$.instance_ip").isEqualTo("192.168.0.8")
                .jsonPath("$.ready").isEqualTo(true);

        verify(scheduler).getQueuedRequestCount();
    }

    @Test
    void updatesFlexlbLogGroupThroughLegacyEndpoint() {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        ConfigService configService = mock(ConfigService.class);
        PriorityScheduler scheduler = mock(PriorityScheduler.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        FlexlbInstanceAddressService instanceAddressService = mock(FlexlbInstanceAddressService.class);
        FlexlbLogManager flexlbLogManager = mock(FlexlbLogManager.class);
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                consistency,
                configService,
                scheduler,
                endpointRegistry,
                null,
                new ServerScheduleLatencyRecorder(),
                instanceAddressService,
                flexlbLogManager);
        WebTestClient client = WebTestClient
                .bindToRouterFunction(server.loadBalancePrefill())
                .build();

        client.post()
                .uri("/rtp_llm/update_log_level")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"log_level\":\"warn\"}")
                .exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("Success! logLevel=WARN");

        verify(flexlbLogManager).setLogLevel(LogLevel.WARN);
    }

    @Test
    void removedTrafficPolicyEndpointReturnsNotFound() {
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                mock(LBStatusConsistencyService.class),
                mock(ConfigService.class),
                mock(PriorityScheduler.class),
                mock(EndpointRegistry.class),
                null,
                new ServerScheduleLatencyRecorder(),
                mock(FlexlbInstanceAddressService.class),
                mock(FlexlbLogManager.class));
        WebTestClient client = WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();

        client.post()
                .uri("/rtp_llm/update_traffic_policy")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{}")
                .exchange()
                .expectStatus().isNotFound();
    }
}
