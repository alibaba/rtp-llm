package org.flexlb.sync.runner;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class GrpcWorkerHealthIsolationTest {
    private GrpcWorkerStatusRunner runner(WorkerStatus worker, EngineGrpcService rpc, Executor application) {
        return new GrpcWorkerStatusRunner("model", "127.0.0.1:8000", "site", RoleType.PREFILL, "group",
                worker, Map.of("127.0.0.1:8000", worker), mock(EngineHealthReporter.class), rpc,
                5000, null, null, application);
    }

    @Test void conversionFailureIsLocalAndDoesNotIncrementTransportFailureCount() {
        var worker=new WorkerStatus();
        worker.setAlive(true);
        var rpc=mock(EngineGrpcService.class);
        var invalid=EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true)
                .setStatusVersion(1).setRole("invalid-role").build();
        when(rpc.getWorkerStatusAsync(anyString(),anyInt(),anyLong(),anyLong(),any()))
                .thenReturn(CompletableFuture.completedFuture(invalid));
        runner(worker,rpc,Runnable::run).run();
        assertTrue(worker.getPollState().lastRpcSuccessUs()>0);
        assertEquals(0,worker.getPollState().lastAppliedUs());
        assertEquals(0,worker.getConsecutiveFailures().get());
        assertFalse(worker.getPollState().isRetired());
        assertFalse(worker.getStatusCheckInProgress().get());
    }

    @Test void freshRpcReleasesPollingWhileOneLatestApplicationRemainsQueued() {
        var worker=new WorkerStatus();worker.setAlive(true);worker.getPollState().configure(10000);
        var rpc=mock(EngineGrpcService.class);
        var first=EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).setStatusVersion(1)
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL).build();
        var second=first.toBuilder().setStatusVersion(2).build();
        when(rpc.getWorkerStatusAsync(anyString(),anyInt(),anyLong(),anyLong(),any()))
                .thenReturn(CompletableFuture.completedFuture(first),CompletableFuture.completedFuture(second));
        var queued=new ArrayList<Runnable>();
        runner(worker,rpc,queued::add).run();
        long received=worker.getPollState().lastRpcSuccessUs();
        assertFalse(worker.getStatusCheckInProgress().get());
        runner(worker,rpc,queued::add).run();
        assertTrue(worker.getPollState().lastRpcSuccessUs()>=received);
        assertEquals(1,queued.size());
        assertFalse(worker.isSchedulingStatusFresh());
        queued.remove(0).run();
        assertEquals(2,worker.getStatusVersion().get());
        assertTrue(worker.isSchedulingStatusFresh());
    }
}
