package org.flexlb.service.grpc;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stage-0 engine-contract passthrough: kv_tokens and running_detail_truncated
 * flow PB -> converter -> POJO -> frozen observation, and a legacy-shaped
 * payload (contract fields absent/default) produces the exact default values
 * with every existing field untouched — the mixed-fleet guarantee.
 */
class EngineStatusConverterContractTest {

    private static WorkerStatus newWorkerStatus() {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, "test-group", "127.0.0.1", 1234, 5678, null);
    }

    @Test
    void contractFieldsFlowThroughConverterAndFreeze() {
        EngineRpcService.TaskInfoPB running = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(42L)
                .setInputLength(1024L)
                .setBatchId(7L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .setKvTokens(2048L)
                .build();
        EngineRpcService.TaskInfoPB finished = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(43L)
                .setInputLength(512L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .setKvTokens(768L)
                .build();
        EngineRpcService.WorkerStatusPB pb = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("RoleType.PREFILL")
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(10L)
                .setLatestFinishedVersion(20L)
                .setRunningDetailTruncated(true)
                .addRunningTaskInfo(running)
                .addFinishedTaskList(finished)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(pb);
        assertEquals(2048L, response.getRunningTaskInfo().get("42").getKvTokens());
        assertEquals(768L, response.getFinishedTaskInfo().get("43").getKvTokens());
        assertTrue(response.isRunningDetailTruncated());

        WorkerStatus status = newWorkerStatus();
        WorkerStatus.StatusObservation observation =
                status.freezeStatusResponse(response);
        assertEquals(2048L, observation.runningTasks().get("42").kvTokens());
        assertEquals(768L, observation.finishedTasks().get("43").kvTokens());
        assertTrue(observation.runningDetailTruncated());
        assertEquals(10L, observation.statusVersion());
        assertEquals(20L, observation.latestFinishedVersion());
    }

    @Test
    void legacyShapedPayloadYieldsDefaultValuesAndUnchangedExistingFields() {
        // Exactly the payload a pre-contract engine emits: no kv_tokens and no
        // running_detail_truncated on the wire, phase/batch ids as before.
        EngineRpcService.TaskInfoPB running = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(42L)
                .setInputLength(1024L)
                .setIterateCount(3L)
                .setBatchId(7L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED)
                .build();
        EngineRpcService.WorkerStatusPB pb = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("RoleType.PREFILL")
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(10L)
                .setLatestFinishedVersion(20L)
                .setAvailableKvCache(1000L)
                .setTotalKvCache(2000L)
                .addRunningTaskInfo(running)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(pb);

        // Default-value path: zero usage, complete detail, existing behavior
        // is indistinguishable from before the contract existed.
        TaskInfo task = response.getRunningTaskInfo().get("42");
        assertEquals(0L, task.getKvTokens());
        assertFalse(response.isRunningDetailTruncated());
        assertEquals(42L, task.getRequestId());
        assertEquals(1024L, task.getInputLength());
        assertEquals(3L, task.getIterateCount());
        assertEquals(7L, task.getBatchId());
        assertEquals(TaskPhase.KV_ALLOCATED, task.getPhase());
        assertEquals(1000L, response.getAvailableKvCacheTokens());
        assertEquals(2000L, response.getTotalKvCacheTokens());

        WorkerStatus status = newWorkerStatus();
        WorkerStatus.StatusObservation observation =
                status.freezeStatusResponse(response);
        assertEquals(0L, observation.runningTasks().get("42").kvTokens());
        assertFalse(observation.runningDetailTruncated());
    }
}
