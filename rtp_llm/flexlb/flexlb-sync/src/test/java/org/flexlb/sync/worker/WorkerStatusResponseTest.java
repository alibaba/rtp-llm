package org.flexlb.sync.worker;

import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class WorkerStatusResponseTest {

    @Test
    void testConfigLoader() throws Exception {
        String TEST_JSON = "{\"role\":\"PREFILL\",\"available_concurrency\":1637,\"running_task_info\":{},\"finished_task_info\":{},\"step_latency_ms\":36.636,\"iterate_count\":1,\"dp_size\":1,\"tp_size\":1,\"alive\":true,\"version\":1,\"status_version\":1752025357566,\"cache_status\":{\"available_kv_cache\":82944,\"total_kv_cache\":82944,\"block_size\":256,\"version\":-1},\"waiting_query_len\":0,\"running_query_len\":0,\"max_seq_len\":131072,\"max_batch_tokens_size\":262144}";
        WorkerStatusResponse workerStatusResponse = JsonUtils.toObject(TEST_JSON, new com.fasterxml.jackson.core.type.TypeReference<WorkerStatusResponse>() {
        });
        Assertions.assertEquals(RoleType.PREFILL, workerStatusResponse.getRole());
        Assertions.assertTrue(workerStatusResponse.isAlive());
        Assertions.assertEquals(1637, workerStatusResponse.getAvailableConcurrency());
        Assertions.assertEquals(131072, workerStatusResponse.getMaxSeqLen());
        Assertions.assertEquals(262144, workerStatusResponse.getMaxBatchTokensSize());
    }

    @Test
    void converterCopiesEngineBatchLimits() {
        EngineRpcService.WorkerStatusPB proto = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setMaxSeqLen(131072L)
                .setMaxBatchTokensSize(262144L)
                .build();

        WorkerStatusResponse response = EngineStatusConverter.convertToWorkerStatusResponse(proto);

        Assertions.assertEquals(RoleType.PREFILL, response.getRole());
        Assertions.assertEquals(131072L, response.getMaxSeqLen());
        Assertions.assertEquals(262144L, response.getMaxBatchTokensSize());
    }

    @Test
    void converterReadsLegacyWorkerRoleAndTaskState() {
        EngineRpcService.TaskInfoPB oldWaiting = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(1L)
                .setIsWaiting(true)
                .build();
        // An old proto3 writer omits is_waiting=false from the wire. The new
        // reader must use the running_task_info container as the fallback.
        EngineRpcService.TaskInfoPB oldRunning = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(2L)
                .build();
        EngineRpcService.WorkerStatusPB proto = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("RoleType.PREFILL")
                .addRunningTaskInfo(oldWaiting)
                .addRunningTaskInfo(oldRunning)
                .build();

        WorkerStatusResponse response = EngineStatusConverter.convertToWorkerStatusResponse(proto);

        assertEquals(RoleType.PREFILL, response.getRole());
        assertEquals(org.flexlb.enums.TaskPhase.PENDING,
                response.getRunningTaskInfo().get("1").getPhase());
        assertEquals(org.flexlb.enums.TaskPhase.RUNNING,
                response.getRunningTaskInfo().get("2").getPhase());
    }

    @Test
    void converterReadsAndValidatesDualWorkerStatus() {
        EngineRpcService.TaskInfoPB task = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(3L)
                .setIsWaiting(true)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED)
                .build();
        EngineRpcService.WorkerStatusPB proto = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("DECODE")
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .addRunningTaskInfo(task)
                .build();

        WorkerStatusResponse response = EngineStatusConverter.convertToWorkerStatusResponse(proto);

        assertEquals(RoleType.DECODE, response.getRole());
        assertEquals(org.flexlb.enums.TaskPhase.KV_ALLOCATED,
                response.getRunningTaskInfo().get("3").getPhase());
    }

    @Test
    void converterRejectsConflictingDualWorkerStatus() {
        EngineRpcService.WorkerStatusPB roleConflict = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .build();
        assertThrows(IllegalArgumentException.class,
                () -> EngineStatusConverter.convertToWorkerStatusResponse(roleConflict));

        EngineRpcService.TaskInfoPB taskConflict = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(4L)
                .setIsWaiting(false)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED)
                .build();
        EngineRpcService.WorkerStatusPB status = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .addRunningTaskInfo(taskConflict)
                .build();
        assertThrows(IllegalArgumentException.class,
                () -> EngineStatusConverter.convertToWorkerStatusResponse(status));
    }

    @Test
    void converterPreservesAuthoritativePriorityCanceledTerminal() {
        EngineRpcService.TaskInfoPB canceled = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(8429001L)
                .setPriorityPreemptionProgress(EngineRpcService.PriorityPreemptionProgressPB
                        .PRIORITY_PREEMPTION_CANCELED)
                .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                        .setErrorCode(8429L)
                        .setErrorMessage("preempted by a higher-priority request"))
                .build();
        EngineRpcService.WorkerStatusPB proto = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .addFinishedTaskList(canceled)
                .build();

        WorkerStatusResponse response = EngineStatusConverter.convertToWorkerStatusResponse(proto);
        var converted = response.getFinishedTaskInfo().get("8429001");

        Assertions.assertEquals(RoleType.PREFILL, response.getRole());
        Assertions.assertNotNull(converted);
        Assertions.assertEquals(PriorityPreemptionProgress.CANCELED,
                converted.getPriorityPreemptionProgress());
        Assertions.assertEquals(8429L, converted.getErrorCode());
    }
}
