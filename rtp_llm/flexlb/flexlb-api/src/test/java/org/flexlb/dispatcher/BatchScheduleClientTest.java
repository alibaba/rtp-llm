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
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class BatchScheduleClientTest {

    @Test
    void successPathReturnsTargets() {
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        List<BatchScheduleTarget> targets = List.of(
                new BatchScheduleTarget("10.0.0.1", 23840, 23841),
                new BatchScheduleTarget("10.0.0.2", 23840, 23841));
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(targets)));

        BatchScheduleClient client = new BatchScheduleClient(coordinator, DispatcherTestSupport.noopFeAssigner());

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

        BatchScheduleClient client = new BatchScheduleClient(coordinator, DispatcherTestSupport.noopFeAssigner());

        StepVerifier.create(client.requestTargets(3))
                .assertNext(returned -> assertEquals(0, returned.size(),
                        "business-level failure (success=false) must degrade silently to empty list"))
                .verifyComplete();
    }

    @Test
    void transportErrorCollapsesToEmptyList() {
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        when(coordinator.schedule(any())).thenReturn(Mono.error(new RuntimeException("boom")));

        BatchScheduleClient client = new BatchScheduleClient(coordinator, DispatcherTestSupport.noopFeAssigner());

        StepVerifier.create(client.requestTargets(5))
                .assertNext(returned -> assertEquals(0, returned.size(),
                        "transport error must degrade silently — never propagate up to the dispatcher"))
                .verifyComplete();
    }

    @Test
    void hungCoordinatorTimesOutAndCollapsesToEmptyList() {
        // A hung transport (slave forwarding to a wedged master) must not pin the /dispatcher
        // request: the whole-call timeout fires and degrades to the same empty-list path as any
        // other failure. Virtual time keeps the 3s bound from slowing the suite.
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        when(coordinator.schedule(any())).thenReturn(Mono.never());

        StepVerifier.withVirtualTime(() -> new BatchScheduleClient(coordinator, DispatcherTestSupport.noopFeAssigner()).requestTargets(4))
                .thenAwait(BatchScheduleClient.REQUEST_TIMEOUT.plusSeconds(1))
                .assertNext(returned -> assertEquals(0, returned.size(),
                        "a timed-out pre-assign call must degrade silently to empty list"))
                .verifyComplete();
    }

    @Test
    void nullServerStatusTreatedAsEmpty() {
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        BatchScheduleResponse resp = new BatchScheduleResponse();
        resp.setSuccess(true);
        // serverStatus left null — coordinator returned success but no targets
        when(coordinator.schedule(any())).thenReturn(Mono.just(resp));

        BatchScheduleClient client = new BatchScheduleClient(coordinator, DispatcherTestSupport.noopFeAssigner());

        StepVerifier.create(client.requestTargets(1))
                .assertNext(returned -> assertEquals(0, returned.size()))
                .verifyComplete();
    }

    @Test
    void masterLocalResolutionStampsFeUrlFromMasterPool() {
        // The dispatcher resolves in-process — on the elected master (or with consistency off) that
        // takes the LOCAL coordinator branch, NOT the HTTP handler — so BatchScheduleClient must
        // apply the master FePool stamp itself, or the master's own batch chunks all fail with
        // CHUNK_NO_FE. Mutation guard: drop the masterFeAssigner.assign() call and the returned
        // targets come back with null fe_url.
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        List<BatchScheduleTarget> targets = List.of(
                new BatchScheduleTarget("10.0.0.1", 23840, 23841),
                new BatchScheduleTarget("10.0.0.2", 23840, 23841));
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(targets)));
        FePool pool = mock(FePool.class);
        when(pool.next()).thenReturn("http://fe-1", "http://fe-2");
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool, true, true);

        BatchScheduleClient client = new BatchScheduleClient(coordinator, assigner);

        StepVerifier.create(client.requestTargets(2))
                .assertNext(returned -> {
                    assertEquals("http://fe-1", returned.get(0).getFeUrl(),
                            "master-local resolution must stamp fe_url from the master cursor");
                    assertEquals("http://fe-2", returned.get(1).getFeUrl());
                })
                .verifyComplete();
    }

    @Test
    void slaveForwardResolutionDoesNotRestampMasterFeUrl() {
        // A slave forwarded to the master; the coordinator's response already carries the master's
        // fe_url. The client must NOT overwrite it with a second (local) cursor — that reintroduces
        // the collision the feature removes. Guard: isMaster()==false → assign() is a no-op, the
        // pool is never touched. Mutation guard: drop the resolvedLocally guard and this restamps.
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        BatchScheduleTarget fromMaster = new BatchScheduleTarget("10.0.0.1", 23840, 23841);
        fromMaster.setFeUrl("http://master-picked-fe");
        when(coordinator.schedule(any()))
                .thenReturn(Mono.just(BatchScheduleResponse.success(List.of(fromMaster))));
        FePool pool = mock(FePool.class);
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool, true, false);

        BatchScheduleClient client = new BatchScheduleClient(coordinator, assigner);

        StepVerifier.create(client.requestTargets(1))
                .assertNext(returned -> assertEquals("http://master-picked-fe", returned.get(0).getFeUrl(),
                        "a forwarding slave must preserve the master's fe_url, not restamp it"))
                .verifyComplete();
        verifyNoInteractions(pool);
    }

    @Test
    void consistencyOffSingleNodeStampsFeUrlLocally() {
        // With consistency disabled there is no master/slave split — every node resolves locally, so
        // the client must stamp via the !isNeedConsistency() operand of resolvedLocally (true even
        // though isMaster() is false). The commit calls this path out explicitly ("any single-node
        // consistency-off deploy"); it is the one operand the isMaster() case does not exercise.
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        List<BatchScheduleTarget> targets = List.of(new BatchScheduleTarget("10.0.0.1", 23840, 23841));
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(targets)));
        FePool pool = mock(FePool.class);
        when(pool.next()).thenReturn("http://fe-solo");
        MasterFeAssigner assigner = DispatcherTestSupport.masterFeAssigner(pool, false, false);

        BatchScheduleClient client = new BatchScheduleClient(coordinator, assigner);

        StepVerifier.create(client.requestTargets(1))
                .assertNext(returned -> assertEquals("http://fe-solo", returned.get(0).getFeUrl(),
                        "a consistency-off node resolves locally and must stamp fe_url itself"))
                .verifyComplete();
    }
}
