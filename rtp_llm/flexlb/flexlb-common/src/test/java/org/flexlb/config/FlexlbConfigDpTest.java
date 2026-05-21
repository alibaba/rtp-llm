package org.flexlb.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verify the DP fields on FlexlbConfig — V1 is the default path; round-trip
 * correct for new payloads; payloads can still opt out by setting
 * dpBalanceEnabled=false explicitly.
 */
class FlexlbConfigDpTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void default_dp_enabled_v1_path() {
        FlexlbConfig cfg = new FlexlbConfig();
        assertTrue(cfg.isDpBalanceEnabled(), "V1 DP-batching is the default path");
        assertEquals(0, cfg.getDpBatchSizeMax(), "0 means auto-derive from worker.dpSize");
        assertEquals(30, cfg.getDpBatchWindowMs());
        assertEquals(100, cfg.getDpBatchTimeoutMs());
        assertEquals("RR", cfg.getDpAssignStrategy());
    }

    @Test
    void payload_without_dp_fields_inherits_v1_default() throws Exception {
        String payload = "{\"loadBalanceStrategy\":\"SHORTEST_TTFT\","
                + "\"weightedCacheDecayFactor\":0.005,"
                + "\"enableQueueing\":true,"
                + "\"maxQueueSize\":500}";
        FlexlbConfig cfg = mapper.readValue(payload, FlexlbConfig.class);
        assertTrue(cfg.isEnableQueueing());
        assertEquals(500, cfg.getMaxQueueSize());
        assertTrue(cfg.isDpBalanceEnabled(),
                "absent dpBalanceEnabled inherits the V1 default; set false explicitly to opt out");
    }

    @Test
    void explicit_opt_out_disables_v1() throws Exception {
        String payload = "{\"dpBalanceEnabled\":false}";
        FlexlbConfig cfg = mapper.readValue(payload, FlexlbConfig.class);
        assertFalse(cfg.isDpBalanceEnabled());
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
