package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BatchScheduleContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
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

    @Test
    void completedAllocationIsSerializedWithoutRestamping() {
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setFeUrl("http://master-selected-fe");
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(List.of(target))));
        client.post().uri("/rtp_llm/batch_schedule").contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"batch_count\":1,\"assign_be\":false,\"assign_fe\":true}")
                .exchange().expectStatus().isOk().expectBody()
                .jsonPath("$.server_status[0].fe_url").isEqualTo("http://master-selected-fe");
        ArgumentCaptor<BatchScheduleRequest> request = ArgumentCaptor.forClass(BatchScheduleRequest.class);
        verify(coordinator).schedule(request.capture());
        assertEquals(1, request.getValue().getBatchCount());
        ArgumentCaptor<BatchScheduleContext> context = ArgumentCaptor.forClass(BatchScheduleContext.class);
        verify(reporter).reportBatchSchedule(context.capture());
        assertEquals(target.getFeUrl(), context.getValue().getBatchResponse().getServerStatus().getFirst().getFeUrl());
    }

    @ParameterizedTest
    @CsvSource({"INVALID_REQUEST,400", "NO_AVAILABLE_WORKER,500"})
    void businessFailurePreservesHttpStatus(StrategyErrorType type, int status) {
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.error(type, "rejected")));
        client.post().uri("/rtp_llm/batch_schedule").contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"batch_count\":1}").exchange().expectStatus().isEqualTo(status)
                .expectBody().jsonPath("$.success").isEqualTo(false)
                .jsonPath("$.error_message").isEqualTo("rejected");
    }

    @Test
    void transportFailureDoesNotExposeInternalAddresses() {
        when(coordinator.schedule(any())).thenReturn(Mono.error(
                new BatchScheduleTransportException("failed to dial private-master:7001", "CONNECT_FAILED")));
        client.post().uri("/rtp_llm/batch_schedule").contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"batch_count\":1}").exchange().expectStatus().is5xxServerError()
                .expectBody().jsonPath("$.error_message").isEqualTo("batch scheduling failed");
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"", "{bad-json", "{\"batch_count\":\"bad\"}"})
    void malformedBodiesFailBeforeAllocation(String body) {
        client.post().uri("/rtp_llm/batch_schedule").contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body).exchange().expectStatus().isBadRequest();
        verifyNoInteractions(coordinator);
    }
}
