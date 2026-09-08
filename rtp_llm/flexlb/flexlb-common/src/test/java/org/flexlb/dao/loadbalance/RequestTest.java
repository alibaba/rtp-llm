package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

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

    @Test
    void should_use_defaults_when_payload_omits_optional_fields() throws Exception {
        Request request = objectMapper.readValue("{\"request_id\":123}", Request.class);

        assertEquals(Request.DEFAULT_GENERATE_TIMEOUT_MS, request.getGenerateTimeout());
        assertEquals(0, request.getMaxNewTokens());
    }

    @Test
    void should_accept_input_ids_without_serializing_them() throws Exception {
        Request request = objectMapper.readValue(
                "{\"request_id\":\"input-json\",\"input_ids\":[11,22,33]}",
                Request.class);

        assertEquals(3, request.getInputIds().size());
        assertEquals(22, request.getInputIds().getInt(1));
        var json = objectMapper.readTree(objectMapper.writeValueAsString(request));
        assertFalse(json.has("input_ids"));
        assertFalse(json.has("inputIds"));
        assertEquals("input-json", json.get("request_id").asText());
    }

    @Test
    void should_not_read_tokens_when_serializing_or_printing_request() throws Exception {
        Request request = new Request();
        request.setRequestId("large-input");
        request.setInputIds(TokenIds.wrap(1_000_000, index -> {
            throw new AssertionError("Request output must not read input tokens");
        }));

        var json = objectMapper.readTree(objectMapper.writeValueAsString(request));
        assertFalse(json.has("input_ids"));
        assertFalse(json.has("inputIds"));
        assertEquals("large-input", json.get("request_id").asText());
        assertFalse(request.toString().contains("inputIds"));
    }

    @Test
    void should_accept_empty_and_null_input_ids() throws Exception {
        Request empty = objectMapper.readValue("{\"input_ids\":[]}", Request.class);
        Request absent = objectMapper.readValue("{\"input_ids\":null}", Request.class);

        assertEquals(0, empty.getInputIds().size());
        assertNull(absent.getInputIds());
    }
}
