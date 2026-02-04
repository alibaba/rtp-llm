package org.flexlb.sync.worker;

import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class WorkerStatusResponseTest {

    @Test
    void testConfigLoader() throws Exception {
        String TEST_JSON = "{\"role\":\"RoleType.PREFILL\",\"available_concurrency\":1637,\"running_task_info\":{},\"finished_task_info\":{},\"step_latency_ms\":36.636,\"iterate_count\":1,\"dp_size\":1,\"tp_size\":1,\"alive\":true,\"version\":1,\"status_version\":1752025357566,\"cache_status\":{\"available_kv_cache\":82944,\"total_kv_cache\":82944,\"block_size\":256,\"version\":-1},\"waiting_query_len\":0,\"running_query_len\":0}";
        WorkerStatusResponse workerStatusResponse = JsonUtils.toObject(TEST_JSON, new com.fasterxml.jackson.core.type.TypeReference<WorkerStatusResponse>() {
        });
        Assertions.assertEquals("RoleType.PREFILL", workerStatusResponse.getRole());
        Assertions.assertTrue(workerStatusResponse.isAlive());
        Assertions.assertEquals(1637, workerStatusResponse.getAvailableConcurrency());
    }
}