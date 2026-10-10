package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.MasterStatusService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.sync.synchronizer.WorkerStatusSynchronizer;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class HttpLoadBalanceServerTest {

    @Test
    void masterInfoUsesCanonicalSchedulerQueueDepth() {
        MasterStatusService consistency = mock(MasterStatusService.class);
        ConfigService configService = mock(ConfigService.class);
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        WorkerStatusSynchronizer synchronizer = mock(WorkerStatusSynchronizer.class);
        when(consistency.getMasterHostIpPort()).thenReturn("127.0.0.1:7001");
        when(requests.pendingDeliveryRequestCount()).thenReturn(7);
        when(synchronizer.isReady()).thenReturn(true);

        HttpLoadBalanceServer server = new HttpLoadBalanceServer(consistency, configService, requests, endpointRegistry, synchronizer, new ServerScheduleLatencyRecorder());
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
                .jsonPath("$.ready").isEqualTo(true);

        verify(requests).pendingDeliveryRequestCount();
    }

    @Test
    void inflightStatusExposesObservedAndIndexedCacheVersions() {
        ConfigService configService = mock(ConfigService.class);
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ownershipStats()).thenReturn(new PrefillState.Stats(7, 3, 2, 42L));
        when(status.cacheIndexSnapshot()).thenReturn(
                new WorkerStatus.CacheIndexSnapshot(
                        CacheStatus.builder()
                                .version(9L)
                                .cacheKeySize(4L)
                                .build(),
                        true,
                        8L));
        when(endpointRegistry.snapshotPrefillEndpoints())
                .thenReturn(Map.of("127.0.0.1:8080", endpoint));
        when(endpointRegistry.snapshotDecodeEndpoints()).thenReturn(Map.of());

        HttpLoadBalanceServer server = new HttpLoadBalanceServer(mock(MasterStatusService.class), configService, requests, endpointRegistry, mock(WorkerStatusSynchronizer.class), new ServerScheduleLatencyRecorder());
        WebTestClient client = WebTestClient
                .bindToRouterFunction(server.loadBalancePrefill())
                .build();

        client.get()
                .uri("/rtp_llm/inflight_status")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.prefill_endpoints[0].ip_port")
                .isEqualTo("127.0.0.1:8080")
                .jsonPath("$.prefill_endpoints[0].inflight_batches").isEqualTo(2)
                .jsonPath("$.prefill_endpoints[0].inflight_requests").isEqualTo(7)
                .jsonPath("$.prefill_endpoints[0].inflight_route_requests").isEqualTo(3)
                .jsonPath("$.prefill_endpoints[0].cache_version").isEqualTo(9)
                .jsonPath("$.prefill_endpoints[0].cache_indexed").isEqualTo(true)
                .jsonPath("$.prefill_endpoints[0].cache_indexed_version").isEqualTo(8)
                .jsonPath("$.prefill_endpoints[0].cache_key_size").isEqualTo(4);
        verify(endpoint).ownershipStats();
    }

    @Test
    void notifyMasterSerializesTheResponseContract() {
        MasterStatusService consistency =
                mock(MasterStatusService.class);
        MasterChangeNotifyResp response = new MasterChangeNotifyResp();
        response.setSuccess(true);
        response.setMsg("refreshed");
        when(consistency.handleMasterChange(org.mockito.ArgumentMatchers.any()))
                .thenReturn(response);
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(consistency, mock(ConfigService.class), mock(RequestRepository.class), mock(EndpointRegistry.class), mock(WorkerStatusSynchronizer.class), new ServerScheduleLatencyRecorder());
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
