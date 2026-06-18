package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class RequestTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void should_deserialize_frontend_schedule_payload() throws Exception {
        Request request = objectMapper.readValue("""
                {
                  "model": "engine_service",
                  "block_cache_keys": [1, 2, 3],
                  "cache_key_block_size": 1024,
                  "seq_len": 8192,
                  "debug": false,
                  "request_priority": 100,
                  "generate_timeout": 5000,
                  "request_id": 12345,
                  "request_time_ms": 1710000000000
                }
                """, Request.class);

        assertEquals(12345L, request.getRequestId());
        assertEquals(8192L, request.getSeqLen());
        assertEquals(1024L, request.getCacheKeyBlockSize());
        assertEquals(3, request.getBlockCacheKeys().size());
        assertEquals(5000L, request.getGenerateTimeout());
    }

    @Test
    void should_not_include_api_key_in_to_string() {
        Request request = new Request();
        request.setRequestId(12345L);
        request.setApiKey("secret-api-key");

        assertFalse(request.toString().contains("secret-api-key"));
    }

    @Test
    void should_match_excluded_worker_by_role_and_ip_when_port_is_wildcard() {
        Request request = new Request();
        ExcludedWorker excludedWorker = new ExcludedWorker();
        excludedWorker.setRole(RoleType.DECODE);
        excludedWorker.setServerIp("10.0.0.1");
        excludedWorker.setHttpPort(0);
        request.setExcludedWorkers(List.of(excludedWorker));

        Assertions.assertTrue(request.isWorkerExcluded(RoleType.DECODE, "10.0.0.1", 26650));
        Assertions.assertTrue(request.isWorkerExcluded(RoleType.DECODE, "10.0.0.1", 26660));
        Assertions.assertFalse(request.isWorkerExcluded(RoleType.PREFILL, "10.0.0.1", 26650));
        Assertions.assertFalse(request.isWorkerExcluded(RoleType.DECODE, "10.0.0.2", 26650));
    }

    @Test
    void should_match_excluded_worker_by_exact_port_when_port_is_set() {
        Request request = new Request();
        ExcludedWorker excludedWorker = new ExcludedWorker();
        excludedWorker.setRole(RoleType.PREFILL);
        excludedWorker.setServerIp("10.0.0.2");
        excludedWorker.setHttpPort(25850);
        request.setExcludedWorkers(List.of(excludedWorker));

        Assertions.assertTrue(request.isWorkerExcluded(RoleType.PREFILL, "10.0.0.2", 25850));
        Assertions.assertFalse(request.isWorkerExcluded(RoleType.PREFILL, "10.0.0.2", 25860));
    }
}
