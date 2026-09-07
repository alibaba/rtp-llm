package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.service.address.FlexlbInstanceAddressService;
import org.flexlb.service.monitor.FlexlbLogManager;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;

import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class HttpLoadBalanceServerTest {

    @Test
    void inflightStatusExposesExactBlockedRequestCount() {
        RequestScheduler scheduler = mock(RequestScheduler.class);
        when(scheduler.getInflightSize()).thenReturn(9);
        when(scheduler.getBlockedRequestCount()).thenReturn(2);
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                mock(LBStatusConsistencyService.class),
                mock(ConfigService.class, RETURNS_DEEP_STUBS),
                scheduler,
                mock(EndpointRegistry.class),
                mock(WorkerDirectory.class),
                mock(MasterEngineSynchronizer.class),
                new ServerScheduleLatencyRecorder(),
                mock(FlexlbInstanceAddressService.class),
                mock(FlexlbLogManager.class));
        WebTestClient client = WebTestClient
                .bindToRouterFunction(server.loadBalancePrefill())
                .build();

        client.get()
                .uri("/rtp_llm/inflight_status")
                .accept(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.scheduler_inflight").isEqualTo(9)
                .jsonPath("$.scheduler_blocked").isEqualTo(2);

        verify(scheduler).getBlockedRequestCount();
    }

    @Test
    void masterInfoUsesCanonicalSchedulerQueueDepth() {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        ConfigService configService = mock(ConfigService.class);
        RequestScheduler scheduler = mock(RequestScheduler.class);
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
                mock(WorkerDirectory.class),
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
    void notifyMasterSerializesTheResponseContract() {
        LBStatusConsistencyService consistency =
                mock(LBStatusConsistencyService.class);
        MasterChangeNotifyResp response = new MasterChangeNotifyResp();
        response.setSuccess(true);
        response.setMsg("refreshed");
        when(consistency.handleMasterChange(org.mockito.ArgumentMatchers.any()))
                .thenReturn(response);
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                consistency,
                mock(ConfigService.class),
                mock(RequestScheduler.class),
                mock(EndpointRegistry.class),
                mock(WorkerDirectory.class),
                mock(MasterEngineSynchronizer.class),
                new ServerScheduleLatencyRecorder(),
                mock(FlexlbInstanceAddressService.class),
                mock(FlexlbLogManager.class));
        WebTestClient client = WebTestClient
                .bindToRouterFunction(server.loadBalancePrefill())
                .build();

        client.post()
                .uri("/rtp_llm/notify_master")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"roleId\":\"role-a\"}")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.msg").isEqualTo("refreshed");
    }
}
