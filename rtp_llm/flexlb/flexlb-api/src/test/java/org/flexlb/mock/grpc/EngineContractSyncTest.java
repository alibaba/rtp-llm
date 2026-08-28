package org.flexlb.mock.grpc;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stage-0 engine-contract acceptance, end to end through the real gRPC client
 * and the production status transaction:
 *
 * <ul>
 *   <li>kv_tokens read-back non-zero: a contract-shaped engine reports
 *       per-request KV usage and the master observes it (observation only)</li>
 *   <li>lost-increment convergence: when the master drops one status
 *       response (transport success but no publish), its finished cursor stays
 *       behind, the engine's finished window replays the terminal on the next
 *       pull, and one successful publish converges the cursor — terminal
 *       delivery is state-based, not one-shot</li>
 * </ul>
 */
class EngineContractSyncTest extends FlexLBMockTestBase {

    private static final long SYNC_TIMEOUT_MS = 5000;

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .availableConcurrency(10)
                .build();
    }

    @Test
    @Timeout(20)
    void kvTokensReadBackNonZeroThroughStatusSync() throws Exception {
        // Baseline round-trip so the endpoint has a committed status.
        CompletableFuture<org.flexlb.dao.loadbalance.Response> future = submitRequest(31001);
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());

        EngineRpcService.TaskInfoPB running = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(42L)
                .setInputLength(1024L)
                .setBatchId(7L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .setKvTokens(2048L)
                .build();
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder()
                .availableConcurrency(10)
                .runningTasks(List.of(running))
                .runningDetailTruncated(true)
                .build());

        EngineRpcService.WorkerStatusPB pb = fetchWorkerStatus();
        assertEquals(2048L, pb.getRunningTaskInfo(0).getKvTokens(),
                "kv_tokens must survive the gRPC round trip");
        assertTrue(pb.getRunningDetailTruncated(),
                "running_detail_truncated must survive the gRPC round trip");

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(pb);
        response.setRole(RoleType.PREFILL);
        applyWorkerStatusResponse(getPrefillEndpoint().getStatus(), response);

        WorkerStatus ws = getPrefillEndpoint().getStatus();
        WorkerStatus.TaskObservation observed =
                ws.committedEngineObservation().runningTaskList().get("42");
        assertNotNull(observed, "running task 42 must be committed");
        assertEquals(2048L, observed.kvTokens(),
                "committed observation must carry the reported kv_tokens");
    }

    @Test
    @Timeout(20)
    void lostIncrementReplayedWithinWindowAndConverges() throws Exception {
        CompletableFuture<org.flexlb.dao.loadbalance.Response> future = submitRequest(31002);
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());

        EngineRpcService.TaskInfoPB terminal = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(43L)
                .setInputLength(512L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .setKvTokens(768L)
                .build();
        // The mock mirrors the engine's finished-window semantics: the terminal
        // is re-sent while the requester's cursor (latest_finished_version) is
        // behind the engine's latest, and stops once the cursor catches up.
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder()
                .availableConcurrency(10)
                .finishedTasks(List.of(terminal))
                .latestFinishedVersion(5L)
                .build());

        // Pull 1: master receives the terminal but drops it (no publish —
        // e.g. the callback was lost before the transaction committed).
        EngineRpcService.WorkerStatusPB pb1 = fetchWorkerStatus();
        assertEquals(1, pb1.getFinishedTaskListCount(),
                "first pull must deliver the terminal");
        assertEquals(43L, pb1.getFinishedTaskList(0).getRequestId());
        assertEquals(5L, pb1.getLatestFinishedVersion());

        // The cursor stayed behind, so the window replays the same terminal.
        EngineRpcService.WorkerStatusPB pb2 = fetchWorkerStatus();
        assertEquals(1, pb2.getFinishedTaskListCount(),
                "window must replay the terminal after the lost increment");
        assertEquals(43L, pb2.getFinishedTaskList(0).getRequestId());

        // This time the transaction succeeds; the cursor advances only now
        // (publish-after-reduce), exactly as GrpcWorkerStatusRunner does.
        WorkerStatus ws = getPrefillEndpoint().getStatus();
        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(pb2);
        response.setRole(RoleType.PREFILL);
        publishThroughStatusTransaction(ws, response);
        assertEquals(5L, ws.appliedStatusCursor().latestFinishedTaskVersion(),
                "finished cursor must advance only after the successful publish");

        // Pull 3: converged — the window stops replaying the terminal.
        EngineRpcService.WorkerStatusPB pb3 = fetchWorkerStatus();
        assertEquals(0, pb3.getFinishedTaskListCount(),
                "no replay once the cursor converged");
    }

    private EngineRpcService.WorkerStatusPB fetchWorkerStatus() throws Exception {
        WorkerStatus.AppliedStatusCursor cursor =
                getPrefillEndpoint().getStatus().appliedStatusCursor();
        EngineRpcService.StatusVersionPB request = EngineRpcService.StatusVersionPB.newBuilder()
                .setLatestCacheVersion(cursor.statusVersion())
                .setLatestFinishedVersion(cursor.latestFinishedTaskVersion())
                .build();
        return grpcClient.getWorkerStatusAsync(prefillIp, prefillGrpcPort, request, SYNC_TIMEOUT_MS)
                .get(SYNC_TIMEOUT_MS, TimeUnit.MILLISECONDS);
    }

    /**
     * Commit one strictly newer response through the production two-phase
     * transaction (freeze -> prepare -> publish), the same linearization the
     * runner performs after its reducers succeed.
     */
    private static void publishThroughStatusTransaction(
            WorkerStatus ws, WorkerStatusResponse response) {
        ws.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    ws.prepareNewStatus(ws.freezeStatusResponse(response));
            ws.publishPreparedStatus(prepared);
        } finally {
            ws.lock.unlock();
        }
    }
}
