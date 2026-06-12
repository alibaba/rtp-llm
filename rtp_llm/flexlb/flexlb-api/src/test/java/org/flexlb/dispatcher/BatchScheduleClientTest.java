package org.flexlb.dispatcher;

import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.BatchScheduleCoordinator;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class BatchScheduleClientTest {

    @Test
    void successPathReturnsTargets() {
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        List<BatchScheduleTarget> targets = List.of(
                new BatchScheduleTarget("10.0.0.1", 23840, 23841),
                new BatchScheduleTarget("10.0.0.2", 23840, 23841));
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(targets)));

        BatchScheduleClient client = new BatchScheduleClient(coordinator);

        StepVerifier.create(client.requestTargets(2))
                .assertNext(returned -> {
                    assertEquals(2, returned.size());
                    assertEquals("10.0.0.1", returned.get(0).getServerIp());
                    assertEquals(23841, returned.get(1).getGrpcPort());
                })
                .verifyComplete();

        ArgumentCaptor<BatchScheduleRequest> captor = ArgumentCaptor.forClass(BatchScheduleRequest.class);
        verify(coordinator).schedule(captor.capture());
        assertEquals(2, captor.getValue().getBatchCount(),
                "client must forward chunkCount as batchCount on the BatchScheduleRequest");
    }

    @Test
    void businessFailureCollapsesToEmptyList() {
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        when(coordinator.schedule(any())).thenReturn(Mono.just(
                BatchScheduleResponse.error(StrategyErrorType.NO_AVAILABLE_WORKER, "no BE")));

        BatchScheduleClient client = new BatchScheduleClient(coordinator);

        StepVerifier.create(client.requestTargets(3))
                .assertNext(returned -> assertEquals(0, returned.size(),
                        "business-level failure (success=false) must degrade silently to empty list"))
                .verifyComplete();
    }

    @Test
    void transportErrorCollapsesToEmptyList() {
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        when(coordinator.schedule(any())).thenReturn(Mono.error(new RuntimeException("boom")));

        BatchScheduleClient client = new BatchScheduleClient(coordinator);

        StepVerifier.create(client.requestTargets(5))
                .assertNext(returned -> assertEquals(0, returned.size(),
                        "transport error must degrade silently — never propagate up to the dispatcher"))
                .verifyComplete();
    }

    @Test
    void nullServerStatusTreatedAsEmpty() {
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        BatchScheduleResponse resp = new BatchScheduleResponse();
        resp.setSuccess(true);
        // serverStatus left null — coordinator returned success but no targets
        when(coordinator.schedule(any())).thenReturn(Mono.just(resp));

        BatchScheduleClient client = new BatchScheduleClient(coordinator);

        StepVerifier.create(client.requestTargets(1))
                .assertNext(returned -> assertEquals(0, returned.size()))
                .verifyComplete();
    }
}
