package org.flexlb.httpserver;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.service.grace.ApplicationLifecycle;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.http.MediaType;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.time.Duration;
import java.util.Map;

class HttpLoadBalanceServerTest {

    @Test
    void inflightStatusExposesExactBlockedRequestCount() {
        RequestScheduler scheduler = mock(RequestScheduler.class);
        when(scheduler.getTrackedRequestCount()).thenReturn(9);
        when(scheduler.getBlockedRequestCount()).thenReturn(2);
        HttpLoadBalanceServer server =
                new HttpLoadBalanceServer(
                        mock(LBStatusConsistencyService.class),
                        mock(ConfigService.class, RETURNS_DEEP_STUBS),
                        scheduler,
                        mock(EndpointRegistry.class),
                        mock(WorkerDirectory.class),
                        mock(MasterEngineSynchronizer.class),
                        new ServerScheduleLatencyRecorder());
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

        verify(scheduler).getBlockedRequestCount();
    }

    @ParameterizedTest
    @EnumSource(DispatcherConfig.Type.class)
    void masterInfoAdvertisesPayloadRequirementAndCanonicalQueueDepth(
            DispatcherConfig.Type dispatcher) {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        config.getDispatcher().setType(dispatcher);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestScheduler scheduler = mock(RequestScheduler.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        ApplicationLifecycle lifecycle = mock(ApplicationLifecycle.class);
        when(consistency.getMasterHostIpPort()).thenReturn("127.0.0.1:7001");
        when(scheduler.getQueuedRequestCount()).thenReturn(7);
        when(lifecycle.isReady()).thenReturn(true);

        HttpLoadBalanceServer server =
                new HttpLoadBalanceServer(
                        consistency,
                        configService,
                        scheduler,
                        endpointRegistry,
                        mock(WorkerDirectory.class),
                        lifecycle,
                        new ServerScheduleLatencyRecorder());
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
                .jsonPath("$.real_master_host")
                .isEqualTo("127.0.0.1:7001")
                .jsonPath("$.requires_generate_input")
                .isEqualTo(dispatcher == DispatcherConfig.Type.BATCH)
                .jsonPath("$.requiresGenerateInput")
                .doesNotExist()
                .jsonPath("$.ready")
                .isEqualTo(true);

        verify(scheduler).getQueuedRequestCount();
    }

    @ParameterizedTest
    @CsvSource({
        "false, true, false, true, true, false, false",
        "true, false, false, true, true, false, false",
        "true, true, true, true, true, false, false",
        "true, true, false, false, false, false, true",
        "true, true, false, false, false, true, true",
        "true, true, false, true, true, true, true",
        "true, true, false, true, false, true, true"
    })
    void masterInfoReadinessDependsOnlyOnTheMasterProcess(
            boolean processReady,
            boolean grpcServing,
            boolean shuttingDown,
            boolean consistencyEnabled,
            boolean master,
            boolean unhealthyWorker,
            boolean expectedReady) {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        when(consistency.isNeedConsistency()).thenReturn(consistencyEnabled);
        when(consistency.isMaster()).thenReturn(master);
        when(consistency.getMasterHostIpPort()).thenReturn("127.0.0.1:7001");
        FlexlbGrpcServer grpc = mock(FlexlbGrpcServer.class);
        when(grpc.isServing()).thenReturn(grpcServing);
        try (var context = new AnnotationConfigApplicationContext()) {
            ApplicationLifecycle lifecycle =
                    new ApplicationLifecycle(
                            consistency,
                            grpc,
                            mock(GracefulLifecycleReporter.class),
                            new MockEnvironment());
            context.registerBean(ApplicationLifecycle.class, () -> lifecycle);
            context.refresh();
            if (processReady) {
                context.publishEvent(
                        new ApplicationReadyEvent(
                                new SpringApplication(), new String[0], context, Duration.ZERO));
            }
            assertFalse(
                    lifecycle.isHealthy(),
                    "master readiness must not require the platform online hook");
            if (shuttingDown) {
                lifecycle.offline();
            }
            WorkerDirectory directory = mock(WorkerDirectory.class);
            if (unhealthyWorker) {
                WorkerStatus worker = mock(WorkerStatus.class);
                when(worker.pollHealth())
                        .thenReturn(new WorkerStatus.PollHealth(0, 0, 0, false, false));
                when(directory.statusSnapshot(RoleType.DECODE))
                        .thenReturn(Map.of("worker:8080", worker));
            }
            ConfigService config = mock(ConfigService.class);
            when(config.loadBalanceConfig()).thenReturn(new FlexlbConfig());
            HttpLoadBalanceServer server =
                    new HttpLoadBalanceServer(
                            consistency,
                            config,
                            mock(RequestScheduler.class),
                            mock(EndpointRegistry.class),
                            directory,
                            lifecycle,
                            new ServerScheduleLatencyRecorder());
            WebTestClient client =
                    WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();
            var body =
                    client.post()
                            .uri("/rtp_llm/master/info")
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue("{}")
                            .exchange()
                            .expectStatus()
                            .isOk()
                            .expectBody()
                            .jsonPath("$.success")
                            .isEqualTo(true)
                            .jsonPath("$.code")
                            .isEqualTo(200)
                            .jsonPath("$.real_master_host")
                            .isEqualTo("127.0.0.1:7001")
                            .jsonPath("$.ready")
                            .isEqualTo(expectedReady);
            if (unhealthyWorker) {
                body.jsonPath("$.worker_summary.DECODE.discovered")
                        .isEqualTo(1)
                        .jsonPath("$.worker_summary.DECODE.alive")
                        .isEqualTo(0);
            }
        }
    }

    @Test
    void inflightStatusExposesObservedAndIndexedCacheVersions() {
        ConfigService configService = mock(ConfigService.class);
        RequestScheduler scheduler = mock(RequestScheduler.class);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        when(endpoint.getStatus()).thenReturn(status);
        when(status.cacheIndexSnapshot())
                .thenReturn(
                        new WorkerStatus.CacheIndexSnapshot(
                                CacheStatus.builder().version(9L).cacheKeySize(4L).build(),
                                true,
                                8L));
        when(endpointRegistry.snapshotPrefillEndpoints())
                .thenReturn(Map.of("127.0.0.1:8080", endpoint));
        when(endpointRegistry.snapshotDecodeEndpoints()).thenReturn(Map.of());

        HttpLoadBalanceServer server =
                new HttpLoadBalanceServer(
                        mock(LBStatusConsistencyService.class),
                        configService,
                        scheduler,
                        endpointRegistry,
                        mock(WorkerDirectory.class),
                        mock(ApplicationLifecycle.class),
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
                .jsonPath("$.prefill_endpoints[0].cache_version")
                .isEqualTo(9)
                .jsonPath("$.prefill_endpoints[0].cache_indexed")
                .isEqualTo(true)
                .jsonPath("$.prefill_endpoints[0].cache_indexed_version")
                .isEqualTo(8)
                .jsonPath("$.prefill_endpoints[0].cache_key_size")
                .isEqualTo(4);
    }

    @Test
    void notifyMasterSerializesTheResponseContract() {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        MasterChangeNotifyResp response = new MasterChangeNotifyResp();
        response.setSuccess(true);
        response.setMsg("refreshed");
        when(consistency.handleMasterChange(org.mockito.ArgumentMatchers.any()))
                .thenReturn(response);
        HttpLoadBalanceServer server =
                new HttpLoadBalanceServer(
                        consistency,
                        mock(ConfigService.class),
                        mock(RequestScheduler.class),
                        mock(EndpointRegistry.class),
                        mock(WorkerDirectory.class),
                        mock(ApplicationLifecycle.class),
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
