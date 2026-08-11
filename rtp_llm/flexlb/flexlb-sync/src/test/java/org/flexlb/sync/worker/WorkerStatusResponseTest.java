package org.flexlb.sync.worker;

import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

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
