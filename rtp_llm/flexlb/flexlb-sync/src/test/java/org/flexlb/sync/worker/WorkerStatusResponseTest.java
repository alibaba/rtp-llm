package org.flexlb.sync.worker;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatus.StatusObservation;
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

        StatusObservation response = convert(proto);

        Assertions.assertEquals(RoleType.PREFILL, response.role());
        Assertions.assertEquals(131072L, response.engine().maxSeqLen());
        Assertions.assertEquals(262144L, response.engine().maxBatchTokensSize());
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

        StatusObservation response = convert(proto);

        assertEquals(RoleType.PREFILL, response.role());
        assertEquals(org.flexlb.enums.TaskPhase.PENDING,
                response.runningTasks().get("1").phase());
        assertEquals(org.flexlb.enums.TaskPhase.RUNNING,
                response.runningTasks().get("2").phase());
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

        StatusObservation response = convert(proto);

        assertEquals(RoleType.DECODE, response.role());
        assertEquals(org.flexlb.enums.TaskPhase.KV_ALLOCATED,
                response.runningTasks().get("3").phase());
    }

    @Test
    void converterRejectsConflictingDualWorkerRole() {
        EngineRpcService.WorkerStatusPB roleConflict = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .build();
        assertThrows(IllegalArgumentException.class,
                () -> convert(roleConflict));
    }

    @Test
    void converterTrustsExplicitTaskPhaseWhenLegacyFlagIsAbsent() throws Exception {
        // e0 reserved field 9 and therefore sent no is_waiting value. A new
        // reader observes the proto3 default false, which must not invalidate
        // the explicit phase carried in field 12.
        EngineRpcService.TaskInfoPB receivedFromE0 = EngineRpcService.TaskInfoPB.parseFrom(
                EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(4L)
                        .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED)
                        .build()
                        .toByteArray());
        EngineRpcService.TaskInfoPB kvAllocatedFromE0 = EngineRpcService.TaskInfoPB.parseFrom(
                EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(5L)
                        .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED)
                        .build()
                        .toByteArray());
        EngineRpcService.WorkerStatusPB status = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .addRunningTaskInfo(receivedFromE0)
                .addRunningTaskInfo(kvAllocatedFromE0)
                .build();

        StatusObservation response = convert(status);

        assertEquals(org.flexlb.enums.TaskPhase.RECEIVED,
                response.runningTasks().get("4").phase());
        assertEquals(org.flexlb.enums.TaskPhase.KV_ALLOCATED,
                response.runningTasks().get("5").phase());
    }

    @Test
    void converterKeepsExplicitPhaseWhenLegacyFlagDisagrees() {
        EngineRpcService.TaskInfoPB runningButWaiting = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(6L)
                .setIsWaiting(true)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .build();
        EngineRpcService.WorkerStatusPB status = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .addRunningTaskInfo(runningButWaiting)
                .build();

        StatusObservation response = convert(status);

        assertEquals(org.flexlb.enums.TaskPhase.RUNNING,
                response.runningTasks().get("6").phase());
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

        StatusObservation response = convert(proto);
        var converted = response.finishedTasks().get("8429001");

        Assertions.assertEquals(RoleType.PREFILL, response.role());
        Assertions.assertNotNull(converted);
        Assertions.assertEquals(PriorityPreemptionProgress.CANCELED,
                converted.priorityPreemptionProgress());
        Assertions.assertEquals(8429L, converted.errorCode());
    }

    private static StatusObservation convert(
            EngineRpcService.WorkerStatusPB response) {
        WorkerStatus owner = WorkerStatus.createDiscovered(
                RoleType.PREFILL,
                "test-group",
                "127.0.0.1",
                8080,
                8081,
                "test-site");
        return EngineStatusConverter.convertToStatusObservation(
                owner, response);
    }
}
