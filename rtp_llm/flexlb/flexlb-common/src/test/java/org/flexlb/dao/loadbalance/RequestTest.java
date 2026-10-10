package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

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

        assertEquals(12345L, request.getRequestId());
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
        Request request = Request.builder()
                .requestId(12345L)
                .apiKey("secret-api-key").build();

        assertFalse(request.toString().contains("secret-api-key"));
    }

    @Test
    void should_use_default_timeout_when_payload_omits_generate_timeout() throws Exception {
        Request request = objectMapper.readValue("{\"request_id\":123}", Request.class);

        assertEquals(Request.DEFAULT_GENERATE_TIMEOUT_MS, request.getGenerateTimeout());
    }

    @Test
    void constructionCopiesCacheKeysAndBuilderReuseDoesNotChangeTheRequest() {
        var keys = new ArrayList<>(List.of(11L, 22L));
        var builder = Request.builder().requestId(7L).blockCacheKeys(keys).seqLen(64L).maxNewTokens(32);
        var request = builder.build();
        keys.clear();
        builder.seqLen(1L).maxNewTokens(2);
        assertEquals(List.of(11L, 22L), request.getBlockCacheKeys());
        assertThrows(UnsupportedOperationException.class, () -> request.getBlockCacheKeys().clear());
        assertEquals(64L, request.getHardKvTokens());
        assertEquals(96L, request.getExpectedKvTokens());
    }

    @Test
    void immutableJsonInputPreservesAliasesDefaultsAndWirePriority() throws Exception {
        for (String alias : List.of("api_key", "apikey", "apiKey")) {
            var request = objectMapper.readValue("{\"" + alias + "\":\"secret\",\"priority\":73}", Request.class);
            assertEquals("secret", request.getApiKey());
            assertEquals(73, request.getRawPriority());
            assertEquals(1, request.getMaxNewTokens());
            assertEquals(1, request.getNumBeams());
            assertEquals(Request.DEFAULT_GENERATE_TIMEOUT_MS, request.getGenerateTimeout());
            assertEquals(List.of(), request.getBlockCacheKeys());
            var json = objectMapper.valueToTree(request);
            assertEquals(73, json.get("priority").asInt());
            assertFalse(json.has("rawPriority"));
            assertFalse(json.has("hardKvTokens"));
            assertFalse(json.has("expectedKvTokens"));
            assertEquals(request.getRawPriority(), objectMapper.treeToValue(json, Request.class).getRawPriority());
        }
    }
}
