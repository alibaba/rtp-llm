package org.flexlb.dao.loadbalance;

import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Wire compatibility for the {@code fe_url} field the master stamps into a batch_schedule target.
 * The field travels master → slave over HTTP (a forwarding slave rebuilds the response from the
 * body alone), so it must (de)serialize under the same {@code fe_url} key, and a slave running an
 * older build that never sends it must still parse — the field is additive, not required.
 */
class BatchScheduleTargetJsonTest {

    @Test
    void feUrlSurvivesRoundTripUnderItsWireKey() {
        BatchScheduleTarget target = new BatchScheduleTarget("10.0.0.1", 8088, 50051);
        target.setFeUrl("http://10.0.0.9:26002");

        String json = JsonUtils.toString(target);
        assertTrue(json.contains("\"fe_url\":\"http://10.0.0.9:26002\""),
                "fe_url must serialize under its snake_case wire key, got: " + json);

        BatchScheduleTarget back = JsonUtils.toObject(json, BatchScheduleTarget.class);
        assertEquals("http://10.0.0.9:26002", back.getFeUrl());
        assertEquals("10.0.0.1", back.getServerIp());
        assertEquals(50051, back.getGrpcPort());
    }

    @Test
    void nullFeUrlIsOmittedFromJson() {
        // @JsonInclude(NON_NULL): a target with no master FE assignment must not emit fe_url at all,
        // so an unfilled target is byte-identical on the wire to the pre-feature schema.
        BatchScheduleTarget target = new BatchScheduleTarget("10.0.0.1", 8088, 50051);

        String json = JsonUtils.toString(target);
        assertTrue(!json.contains("fe_url"), "null fe_url must be omitted, got: " + json);
    }

    @Test
    void legacyJsonWithoutFeUrlStillDeserializes() {
        // An older master build never sends fe_url; the newer peer must parse the body and leave
        // feUrl null (those chunks then fail with CHUNK_NO_FE, no fallback) rather than fail to parse.
        String legacy = "{\"server_ip\":\"10.0.0.1\",\"http_port\":8088,\"grpc_port\":50051,\"role\":\"PREFILL\"}";

        BatchScheduleTarget back = JsonUtils.toObject(legacy, BatchScheduleTarget.class);
        assertNull(back.getFeUrl(), "a body without fe_url must leave it null, not fail to parse");
        assertEquals("10.0.0.1", back.getServerIp());
    }
}
