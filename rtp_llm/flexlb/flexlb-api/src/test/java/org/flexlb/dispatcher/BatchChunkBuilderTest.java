package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pure unit tests for the three transforms shared by {@link GenericBatchHandler} and
 * {@link DispatcherDryRunHandler}. No Spring, no reactor — just static method calls against
 * fixture JSON. Owns every assertion about chunk-body wire shape (force_batch injection,
 * role_addrs stamping, split distribution) so the handler tests can stay focused on
 * orchestration concerns.
 */
class BatchChunkBuilderTest {

    private final ObjectMapper mapper = new ObjectMapper();

    // ─────────────────────── buildChunkBodies ───────────────────────

    @Test
    void buildChunkBodiesPreservesTopLevelEnvelopeFields() {
        ObjectNode envelope = mapper.createObjectNode();
        envelope.put("model", "qwen-7b");
        envelope.put("custom_top_level", 42);
        envelope.putArray("prompt_batch").add("a").add("b").add("c");

        List<ArrayNode> chunks = BatchSplitter.split(
                (ArrayNode) envelope.get("prompt_batch"), SubBatchSpec.parse("size:2"), mapper);
        List<ObjectNode> bodies = BatchChunkBuilder.buildChunkBodies(envelope, chunks, "prompt_batch");

        for (ObjectNode body : bodies) {
            assertEquals("qwen-7b", body.get("model").asText(),
                    "every chunk inherits envelope's top-level model");
            assertEquals(42, body.get("custom_top_level").asInt(),
                    "every chunk inherits envelope's arbitrary top-level fields");
        }
    }

    @Test
    void buildChunkBodiesReplacesRequestArrayWithChunkSlice() {
        ObjectNode envelope = mapper.createObjectNode();
        envelope.putArray("prompt_batch").add("a").add("b").add("c");

        List<ArrayNode> chunks = BatchSplitter.split(
                (ArrayNode) envelope.get("prompt_batch"), SubBatchSpec.parse("size:2"), mapper);
        List<ObjectNode> bodies = BatchChunkBuilder.buildChunkBodies(envelope, chunks, "prompt_batch");

        assertEquals(2, bodies.get(0).get("prompt_batch").size());
        assertEquals("a", bodies.get(0).get("prompt_batch").get(0).asText());
        assertEquals(1, bodies.get(1).get("prompt_batch").size());
        assertEquals("c", bodies.get(1).get("prompt_batch").get(0).asText());
    }

    @Test
    void buildChunkBodiesInjectsForceBatchIntoEveryChunk() {
        ObjectNode envelope = mapper.createObjectNode();
        envelope.putArray("prompt_batch").add("a").add("b").add("c").add("d").add("e");

        List<ArrayNode> chunks = BatchSplitter.split(
                (ArrayNode) envelope.get("prompt_batch"), SubBatchSpec.parse("size:2"), mapper);
        List<ObjectNode> bodies = BatchChunkBuilder.buildChunkBodies(envelope, chunks, "prompt_batch");

        assertEquals(3, bodies.size());
        for (ObjectNode body : bodies) {
            assertTrue(body.get("generate_config").get("force_batch").asBoolean(),
                    "every chunk must carry force_batch=true so the per-chunk FE schedules its prompts together");
        }
    }

    // ─────────────────────── injectForceBatch ───────────────────────

    @Test
    void injectForceBatchCreatesGenerateConfigWhenAbsent() {
        ObjectNode body = mapper.createObjectNode();
        BatchChunkBuilder.injectForceBatch(body);

        JsonNode gc = body.get("generate_config");
        assertNotNull(gc, "generate_config must be created when absent");
        assertTrue(gc.isObject());
        assertTrue(gc.get("force_batch").asBoolean());
    }

    @Test
    void injectForceBatchPreservesUserGenerateConfigFields() {
        ObjectNode body = mapper.createObjectNode();
        ObjectNode gc = body.putObject("generate_config");
        gc.put("temperature", 0.7);
        gc.put("max_new_tokens", 100);

        BatchChunkBuilder.injectForceBatch(body);

        assertEquals(0.7, body.get("generate_config").get("temperature").asDouble(), 1e-9);
        assertEquals(100, body.get("generate_config").get("max_new_tokens").asInt());
        assertTrue(body.get("generate_config").get("force_batch").asBoolean());
    }

    @Test
    void injectForceBatchHonorsUserSetTrue() {
        ObjectNode body = mapper.createObjectNode();
        body.putObject("generate_config").put("force_batch", true);
        BatchChunkBuilder.injectForceBatch(body);
        assertTrue(body.get("generate_config").get("force_batch").asBoolean());
    }

    @Test
    void injectForceBatchHonorsUserSetFalse() {
        ObjectNode body = mapper.createObjectNode();
        body.putObject("generate_config").put("force_batch", false);
        BatchChunkBuilder.injectForceBatch(body);
        assertFalse(body.get("generate_config").get("force_batch").asBoolean(),
                "user's explicit force_batch=false opt-out must be honored, not overwritten");
    }

    // ─────────────────────── stampPreAssignedBe ───────────────────────

    @Test
    void stampPreAssignedBeStampsInChunkOrder() {
        List<ObjectNode> chunks = chunkBodiesWithCount(3);
        List<BatchScheduleTarget> targets = List.of(
                target("10.0.0.1"), target("10.0.0.2"), target("10.0.0.3"));

        BatchChunkBuilder.stampPreAssignedBe(chunks, targets);

        for (int i = 0; i < 3; i++) {
            JsonNode roleAddrs = chunks.get(i).get("generate_config").get("role_addrs");
            assertEquals(1, roleAddrs.size(), "exactly one role_addr per chunk for single-role batch_schedule");
            JsonNode addr = roleAddrs.get(0);
            assertEquals("PDFUSION", addr.get("role").asText(),
                    "role serializes as enum.name() to match Python RoleType enum");
            assertEquals("10.0.0." + (i + 1), addr.get("ip").asText(),
                    "chunk " + i + " must carry the i-th target");
            assertEquals(23840, addr.get("http_port").asInt());
            assertEquals(23841, addr.get("grpc_port").asInt());
        }
    }

    @Test
    void stampPreAssignedBeUsesIpKeyNotServerIp() {
        // Defensive: BatchScheduleTarget exposes server_ip on its wire shape, but FE pydantic
        // expects ip. The rename in stampPreAssignedBe is load-bearing — if it ever drifted
        // back to server_ip, FE would silently accept the field and route to no one.
        List<ObjectNode> chunks = chunkBodiesWithCount(1);
        BatchChunkBuilder.stampPreAssignedBe(chunks, List.of(target("10.0.0.1")));
        JsonNode addr = chunks.get(0).get("generate_config").get("role_addrs").get(0);
        assertNotNull(addr.get("ip"));
        assertNull(addr.get("server_ip"));
    }

    @Test
    void stampPreAssignedBePreservesUserSuppliedRoleAddrs() {
        // If a user already set generate_config.role_addrs on the request body (rare but legitimate,
        // e.g. integration-testing a specific BE), the dispatcher appends its target rather than
        // wiping the user's entry. FE will see both addrs and pick whichever role matches its
        // routing pipeline.
        ObjectNode chunk = mapper.createObjectNode();
        ObjectNode gc = chunk.putObject("generate_config");
        ArrayNode userAddrs = gc.putArray("role_addrs");
        ObjectNode userAddr = userAddrs.addObject();
        userAddr.put("role", "DECODE");
        userAddr.put("ip", "192.168.1.99");
        userAddr.put("http_port", 9000);
        userAddr.put("grpc_port", 9001);

        BatchChunkBuilder.stampPreAssignedBe(List.of(chunk), List.of(target("10.0.0.42")));

        JsonNode roleAddrs = chunk.get("generate_config").get("role_addrs");
        assertEquals(2, roleAddrs.size(),
                "user-supplied role_addr must be preserved, dispatcher's appended");
        assertEquals("DECODE", roleAddrs.get(0).get("role").asText(),
                "user entry preserved at index 0 (insertion order)");
        assertEquals("PDFUSION", roleAddrs.get(1).get("role").asText());
        assertEquals("10.0.0.42", roleAddrs.get(1).get("ip").asText());
    }

    @Test
    void stampPreAssignedBeEmptyTargetListIsNoOp() {
        List<ObjectNode> chunks = chunkBodiesWithCount(3);
        BatchChunkBuilder.stampPreAssignedBe(chunks, List.of());
        for (ObjectNode chunk : chunks) {
            JsonNode gc = chunk.get("generate_config");
            assertNull(gc == null ? null : gc.get("role_addrs"),
                    "empty target list must leave generate_config.role_addrs unset — never half-stamp");
        }
    }

    @Test
    void stampPreAssignedBeShorterTargetListPartiallyStamps() {
        // Tolerates a short target list — only the first N chunks get stamped. This is the
        // degradation path when BatchScheduleClient returns fewer targets than requested.
        List<ObjectNode> chunks = chunkBodiesWithCount(3);
        BatchChunkBuilder.stampPreAssignedBe(chunks, List.of(target("10.0.0.1")));

        assertNotNull(chunks.get(0).get("generate_config").get("role_addrs"),
                "chunk 0 must be stamped");
        assertEquals(1, chunks.get(0).get("generate_config").get("role_addrs").size());

        // Unstamped chunks may have no generate_config at all (the helper builds bare chunks);
        // either generate_config absent OR role_addrs absent both encode "not stamped".
        for (int i : new int[]{1, 2}) {
            JsonNode gc = chunks.get(i).get("generate_config");
            assertTrue(gc == null || gc.get("role_addrs") == null,
                    "chunk " + i + " unstamped when target list shorter than chunk list");
        }
    }

    // ─────────────────────── helpers ───────────────────────

    private List<ObjectNode> chunkBodiesWithCount(int n) {
        List<ObjectNode> chunks = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ObjectNode body = mapper.createObjectNode();
            body.putArray("prompt_batch").add("p" + i);
            chunks.add(body);
        }
        return chunks;
    }

    private static BatchScheduleTarget target(String ip) {
        return new BatchScheduleTarget(ip, 23840, 23841, RoleType.PDFUSION);
    }
}
