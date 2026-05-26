package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * V1-α: verify the new DP-related fields on Request / Response / ServerStatus:
 *
 *  - JSON round-trip preserves all fields (snake_case naming matches Python frontend)
 *  - Legacy payloads without the new fields fall back to safe defaults (DP-off path),
 *    so existing callers are not broken.
 */
class RequestResponseDpTest {

    private final ObjectMapper mapper = new ObjectMapper();

    // ============== Request ==============

    @Test
    void request_legacy_payload_uses_safe_defaults() throws JsonProcessingException {
        String legacy = "{\"block_cache_keys\":[1,2,3],\"seq_len\":100,\"request_id\":42}";
        Request req = mapper.readValue(legacy, Request.class);
        assertEquals(3, req.getBlockCacheKeys().size());
        assertEquals(100, req.getSeqLen());
        assertEquals(42, req.getRequestId());
        // Defaults ensure RouteService.shouldUseDpBatch falls back to the SP/beam path.
        assertEquals(1, req.getMaxNewTokens());
        assertEquals(1, req.getNumBeams());
        assertFalse(req.isForceDisableSpRun());
        assertEquals("", req.getModel());
    }

    @Test
    void request_new_payload_round_trip() throws JsonProcessingException {
        Request req = new Request();
        req.setRequestId(7);
        req.setSeqLen(512);
        req.setMaxNewTokens(128);
        req.setNumBeams(1);
        req.setForceDisableSpRun(true);
        req.setModel("qwen3");

        String json = mapper.writeValueAsString(req);
        // snake_case field names (Python frontend convention)
        assertTrue(json.contains("\"max_new_tokens\":128"), json);
        assertTrue(json.contains("\"num_beams\":1"), json);
        assertTrue(json.contains("\"force_disable_sp_run\":true"), json);
        assertTrue(json.contains("\"model\":\"qwen3\""), json);

        Request decoded = mapper.readValue(json, Request.class);
        assertEquals(req.getMaxNewTokens(), decoded.getMaxNewTokens());
        assertEquals(req.getNumBeams(), decoded.getNumBeams());
        assertEquals(req.isForceDisableSpRun(), decoded.isForceDisableSpRun());
        assertEquals(req.getModel(), decoded.getModel());
    }

    // ============== ServerStatus.dpRank ==============

    @Test
    void serverStatus_default_dpRank_is_negative_one() {
        ServerStatus ss = new ServerStatus();
        assertEquals(-1, ss.getDpRank(),
                "default dpRank=-1 means unassigned; the DP-off path must not be misread as rank 0");
    }

    @Test
    void serverStatus_dpRank_round_trip() throws JsonProcessingException {
        ServerStatus ss = new ServerStatus();
        ss.setRole(RoleType.PREFILL);
        ss.setServerIp("10.0.0.1");
        ss.setHttpPort(8080);
        ss.setSuccess(true);
        ss.setDpRank(2);

        String json = mapper.writeValueAsString(ss);
        assertTrue(json.contains("\"dp_rank\":2"), json);

        ServerStatus decoded = mapper.readValue(json, ServerStatus.class);
        assertEquals(2, decoded.getDpRank());
    }

    @Test
    void serverStatus_legacy_payload_dpRank_defaults_to_minus_one() throws JsonProcessingException {
        String legacy = "{\"role\":\"PREFILL\",\"server_ip\":\"10.0.0.1\",\"http_port\":8080,\"success\":true}";
        ServerStatus decoded = mapper.readValue(legacy, ServerStatus.class);
        assertEquals(-1, decoded.getDpRank());
    }

    // ============== Response.enqueuedByMaster ==============

    @Test
    void response_default_enqueuedByMaster_is_false() {
        Response r = new Response();
        assertFalse(r.isEnqueuedByMaster(),
                "default enqueuedByMaster=false keeps legacy frontends transparent — they won't accidentally switch to Decode.FetchResponse");
    }

    @Test
    void response_enqueuedByMaster_round_trip() throws JsonProcessingException {
        Response r = new Response();
        r.setSuccess(true);
        r.setEnqueuedByMaster(true);

        String json = mapper.writeValueAsString(r);
        assertTrue(json.contains("\"enqueued_by_master\":true"), json);

        Response decoded = mapper.readValue(json, Response.class);
        assertTrue(decoded.isEnqueuedByMaster());
    }

    @Test
    void response_legacy_payload_enqueuedByMaster_defaults_to_false() throws JsonProcessingException {
        String legacy = "{\"success\":true,\"code\":200,\"server_status\":[]}";
        Response decoded = mapper.readValue(legacy, Response.class);
        assertFalse(decoded.isEnqueuedByMaster());
    }
}
