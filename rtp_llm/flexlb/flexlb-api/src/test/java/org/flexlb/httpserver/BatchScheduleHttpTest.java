package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest.AllocationType;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class BatchScheduleHttpTest {
    private final BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
    private final EngineHealthReporter reporter = mock(EngineHealthReporter.class);
    private WebTestClient client;

    @BeforeEach
    void setUp() {
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                mock(LBStatusConsistencyService.class), mock(ConfigService.class),
                mock(RequestScheduler.class), mock(EndpointRegistry.class), mock(WorkerDirectory.class),
                mock(MasterEngineSynchronizer.class), mock(ServerScheduleLatencyRecorder.class),
                coordinator, reporter);
        client = WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();
    }

    @ParameterizedTest
    @CsvSource({"FE,false,true", "BE,true,false", "FE_AND_BE,true,true", ",true,false"})
    void completedAllocationIsSerializedWithoutRestamping(AllocationType type, boolean be, boolean fe) {
        BatchScheduleTarget target = BatchScheduleTarget.of(new WorkerHost("10.0.0.9", 8000), RoleType.PDFUSION, EngineType.LLM);
        BatchScheduleResponse allocation = BatchScheduleResponse.success(be ? List.of(target) : List.of());
        if (fe) {
            allocation.setFrontendUrls(List.of("http://master-selected-fe"));
        }
        when(coordinator.schedule(any())).thenReturn(Mono.just(allocation));
        var response = client.post().uri("/rtp_llm/batch_schedule").contentType(MediaType.APPLICATION_JSON)
                .bodyValue(type == null ? "{\"batch_count\":1}" : String.format(
                        "{\"batch_count\":1,\"allocation_type\":\"%s\"}", type))
                .exchange().expectStatus().isOk().expectBody();
        if (be) {
            response.jsonPath("$.server_status[0].server_ip").isEqualTo("10.0.0.9")
                    .jsonPath("$.server_status[0].http_port").isEqualTo(8000)
                    .jsonPath("$.server_status[0].grpc_port").isEqualTo(8001)
                    .jsonPath("$.server_status[0].fe_url").doesNotExist();
        } else {
            response.jsonPath("$.server_status").isEmpty();
        }
        if (fe) {
            response.jsonPath("$.frontend_urls[0]").isEqualTo("http://master-selected-fe");
        } else {
            response.jsonPath("$.frontend_urls").doesNotExist();
        }
        verify(coordinator).schedule(argThat(r -> r.getBatchCount() == 1
                && r.getAllocationType() == (type == null ? AllocationType.BE : type)));
    }

    @ParameterizedTest
    @CsvSource({"400,INVALID_REQUEST,rejected", "500,NO_AVAILABLE_WORKER,rejected", "500,,batch scheduling failed"})
    void failuresRemainStructuredAndTransportDetailsStayPrivate(int status, StrategyErrorType type, String message) {
        when(coordinator.schedule(any())).thenReturn(type == null
                ? Mono.error(new BatchScheduleTransportException("private-master:7001", "CONNECT_FAILED"))
                : Mono.just(BatchScheduleResponse.error(type, message)));
        client.post().uri("/rtp_llm/batch_schedule").contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"batch_count\":1}").exchange().expectStatus().isEqualTo(status)
                .expectBody().jsonPath("$.success").isEqualTo(false)
                .jsonPath("$.error_message").isEqualTo(message);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"", "{bad-json", "{\"batch_count\":\"bad\"}",
            "{\"batch_count\":1,\"allocation_type\":\"UNKNOWN\"}"})
    void malformedBodiesFailBeforeAllocation(String body) {
        client.post().uri("/rtp_llm/batch_schedule").contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body).exchange().expectStatus().isBadRequest();
        verifyNoInteractions(coordinator);
    }
}
