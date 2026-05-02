package org.flexlb.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * V1-α: verify the new DP fields on FlexlbConfig — backward compatible with legacy
 * payloads, round-trip correct for new payloads.
 */
class FlexlbConfigDpTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void default_dp_disabled_safe_defaults() {
        FlexlbConfig cfg = new FlexlbConfig();
        assertFalse(cfg.isDpBalanceEnabled(), "DP must be off by default to not affect existing deployments");
        assertEquals(0, cfg.getDpBatchSizeMax(), "0 means auto-derive from worker.dpSize");
        assertEquals(30, cfg.getDpBatchWindowMs());
        assertEquals(100, cfg.getDpBatchTimeoutMs());
        assertEquals("RR", cfg.getDpAssignStrategy());
    }

    @Test
    void legacy_payload_without_dp_fields_still_loads() throws Exception {
        // Simulate the existing production FLEXLB_CONFIG (no dp* fields).
        String legacy = "{\"loadBalanceStrategy\":\"SHORTEST_TTFT\","
                + "\"weightedCacheDecayFactor\":0.005,"
                + "\"enableQueueing\":true,"
                + "\"maxQueueSize\":500}";
        FlexlbConfig cfg = mapper.readValue(legacy, FlexlbConfig.class);
        assertTrue(cfg.isEnableQueueing());
        assertEquals(500, cfg.getMaxQueueSize());
        assertFalse(cfg.isDpBalanceEnabled(), "legacy config must silently fall back to the old path");
    }

    @Test
    void new_dp_payload_round_trip() throws Exception {
        String json = "{\"loadBalanceStrategy\":\"SHORTEST_TTFT\","
                + "\"dpBalanceEnabled\":true,"
                + "\"dpBatchSizeMax\":4,"
                + "\"dpBatchWindowMs\":50,"
                + "\"dpBatchTimeoutMs\":200,"
                + "\"dpAssignStrategy\":\"RR\"}";
        FlexlbConfig cfg = mapper.readValue(json, FlexlbConfig.class);
        assertTrue(cfg.isDpBalanceEnabled());
        assertEquals(4, cfg.getDpBatchSizeMax());
        assertEquals(50, cfg.getDpBatchWindowMs());
        assertEquals(200, cfg.getDpBatchTimeoutMs());
        assertEquals("RR", cfg.getDpAssignStrategy());
    }
}
