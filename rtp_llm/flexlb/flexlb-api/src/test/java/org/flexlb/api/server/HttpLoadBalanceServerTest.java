package org.flexlb.api.server;

import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.master.MasterChangeNotifyResp;
import org.flexlb.master.MasterStatusService;
import org.flexlb.worker.cache.CacheStatus;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusSynchronizer;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;

import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that HTTP status endpoints expose the scheduler and worker state supplied by their owners.
 */
class HttpLoadBalanceServerTest {

    @Test
    void inflightStatusExposesExactBlockedRequestCount() {
        RequestRepository scheduler = mock(RequestRepository.class);
        when(scheduler.liveRequestCount()).thenReturn(9);
        var scheduling = mock(org.flexlb.balance.schedule.RequestScheduler.class);
        when(scheduling.blockedRequestCount()).thenReturn(2);
        HttpLoadBalanceServer server =
                new HttpLoadBalanceServer(
                        mock(MasterStatusService.class),
                        mock(ConfigService.class, RETURNS_DEEP_STUBS),
                        scheduler,
                        mock(EndpointRegistry.class),
                        mock(WorkerStatusSynchronizer.class),
                        new ServerScheduleLatencyRecorder());
        server.configureScheduling(scheduling);
        WebTestClient client =
                WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();

        client.get()
                .uri("/rtp_llm/inflight_status")
                .accept(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus()
                .isOk()
                .expectBody()
                .jsonPath("$.scheduler_inflight")
                .isEqualTo(9)
                .jsonPath("$.scheduler_tracked")
                .isEqualTo(9)
                .jsonPath("$.scheduler_blocked")
                .isEqualTo(2);

        verify(scheduling).blockedRequestCount();
    }

    @Test
    void masterInfoUsesCanonicalSchedulerQueueDepth() {
        MasterStatusService consistency = mock(MasterStatusService.class);
        ConfigService configService = mock(ConfigService.class);
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        WorkerStatusSynchronizer synchronizer = mock(WorkerStatusSynchronizer.class);
        var config = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(consistency.getMasterHostIpPort()).thenReturn("127.0.0.1:7001");
        var scheduler = mock(org.flexlb.balance.schedule.RequestScheduler.class);
        when(scheduler.queuedRequestCount()).thenReturn(7);
        when(synchronizer.isReady()).thenReturn(true);

        HttpLoadBalanceServer server =
                new HttpLoadBalanceServer(
                        consistency,
                        configService,
                        requests,
                        endpointRegistry,
                        synchronizer,
                        new ServerScheduleLatencyRecorder());
        server.configureScheduling(scheduler);
        WebTestClient client =
                WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();

        client.post()
                .uri("/rtp_llm/master/info")
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.APPLICATION_JSON)
                .bodyValue("{}")
                .exchange()
                .expectStatus()
                .isOk()
                .expectBody()
                .jsonPath("$.queue_length")
                .isEqualTo(7)
                .jsonPath("$.requires_generate_input")
                .isEqualTo(false)
                .jsonPath("$.real_master_host")
                .isEqualTo("127.0.0.1:7001")
                .jsonPath("$.ready")
                .isEqualTo(true);

        verify(scheduler).queuedRequestCount();
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
        when(endpoint.ipPort()).thenReturn("127.0.0.1:8080");
        when(endpoint.stats())
                .thenReturn(new org.flexlb.worker.endpoint.EndpointStats.PrefillCounts(7, 42L));
        when(status.cacheIndexSnapshot())
                .thenReturn(
                        new WorkerStatus.CacheIndexSnapshot(
                                CacheStatus.builder().version(9L).cacheKeySize(4L).build(),
                                true,
                                8L));
        when(endpointRegistry.endpoints(
                        org.mockito.ArgumentMatchers.eq(org.flexlb.config.service.RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.any()))
                .thenReturn(java.util.List.of(endpoint));
        when(endpointRegistry.endpoints(
                        org.mockito.ArgumentMatchers.eq(org.flexlb.config.service.RoleType.DECODE),
                        org.mockito.ArgumentMatchers.any()))
                .thenReturn(java.util.List.of());

        HttpLoadBalanceServer server =
                new HttpLoadBalanceServer(
                        mock(MasterStatusService.class),
                        configService,
                        requests,
                        endpointRegistry,
                        mock(WorkerStatusSynchronizer.class),
                        new ServerScheduleLatencyRecorder());
        WebTestClient client =
                WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();

        client.get()
                .uri("/rtp_llm/inflight_status")
                .exchange()
                .expectStatus()
                .isOk()
                .expectBody()
                .jsonPath("$.prefill_endpoints[0].ip_port")
                .isEqualTo("127.0.0.1:8080")
                .jsonPath("$.prefill_endpoints[0].inflight_requests")
                .isEqualTo(7)
                .jsonPath("$.prefill_endpoints[0].cache_version")
                .isEqualTo(9)
                .jsonPath("$.prefill_endpoints[0].cache_indexed")
                .isEqualTo(true)
                .jsonPath("$.prefill_endpoints[0].cache_indexed_version")
                .isEqualTo(8)
                .jsonPath("$.prefill_endpoints[0].cache_key_size")
                .isEqualTo(4);
        verify(endpoint).stats();
    }

    @Test
    void notifyMasterSerializesTheResponseContract() {
        MasterStatusService consistency = mock(MasterStatusService.class);
        MasterChangeNotifyResp response = new MasterChangeNotifyResp();
        response.setSuccess(true);
        response.setMsg("refreshed");
        when(consistency.handleMasterChange(org.mockito.ArgumentMatchers.any()))
                .thenReturn(response);
        HttpLoadBalanceServer server =
                new HttpLoadBalanceServer(
                        consistency,
                        mock(ConfigService.class),
                        mock(RequestRepository.class),
                        mock(EndpointRegistry.class),
                        mock(WorkerStatusSynchronizer.class),
                        new ServerScheduleLatencyRecorder());
        WebTestClient client =
                WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();

        client.post()
                .uri("/rtp_llm/notify_master")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"roleId\":\"role-a\"}")
                .exchange()
                .expectStatus()
                .isOk()
                .expectBody()
                .jsonPath("$.success")
                .isEqualTo(true)
                .jsonPath("$.msg")
                .isEqualTo("refreshed");
    }
}
