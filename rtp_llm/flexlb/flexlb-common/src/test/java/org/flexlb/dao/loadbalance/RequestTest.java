package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class RequestTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void readsNewAndLegacyCacheAffinityRequests() throws Exception {
        for (String payload : new String[] {
                "{\"cache_affinity_keys\":[\"image\"],\"vit_route_only\":true}",
                "{\"media_keys\":[\"image\"],\"vit_route_only\":true}"}) {
            Request request = objectMapper.readValue(payload, Request.class);
            assertEquals(java.util.List.of("image"), request.getCacheAffinityKeys());
            org.junit.jupiter.api.Assertions.assertTrue(request.isVitRouteOnly());
            String output = objectMapper.writeValueAsString(request);
            assertFalse(output.contains("media_keys"));
        }
    }

    @Test
    void should_deserialize_frontend_schedule_payload() throws Exception {
        Request request = objectMapper.readValue("""
                {
                  "model": "engine_service",
                  "block_cache_keys": [1, 2, 3],
                  "cache_key_block_size": 1024,
                  "seq_len": 8192,
                  "max_new_tokens": 64,
                  "num_beams": 1,
                  "force_disable_sp_run": false,
                  "debug": false,
                  "request_priority": 100,
                  "generate_timeout": 5000,
                  "request_id": 12345,
                  "request_time_ms": 1710000000000
                }
                """, Request.class);

        assertEquals("12345", request.getRequestId());
        assertEquals(8192L, request.getSeqLen());
        assertEquals(1024L, request.getCacheKeyBlockSize());
        assertEquals(3, request.getBlockCacheKeys().size());
        assertEquals(5000L, request.getGenerateTimeout());
        assertEquals(64, request.getMaxNewTokens());
        assertEquals(1, request.getNumBeams());
        assertEquals("engine_service", request.getModel());
    }

    @Test
    void should_not_include_api_key_in_to_string() {
        Request request = new Request();
        request.setRequestId("12345");
        request.setApiKey("secret-api-key");

        assertFalse(request.toString().contains("secret-api-key"));
    }
}
