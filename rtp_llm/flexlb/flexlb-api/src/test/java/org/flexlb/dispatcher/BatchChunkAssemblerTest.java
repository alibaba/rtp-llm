package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchChunkAssemblerTest {

    @Test
    void splitArrayDividesEvenly() {
        JSONArray arr = JSONArray.of("a", "b", "c", "d");
        List<JSONArray> chunks = BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.SIZE, 2));
        assertEquals(2, chunks.size());
        assertEquals(JSONArray.of("a", "b"), chunks.get(0));
        assertEquals(JSONArray.of("c", "d"), chunks.get(1));
    }

    @Test
    void splitArrayLastChunkShorter() {
        JSONArray arr = JSONArray.of("a", "b", "c", "d", "e");
        List<JSONArray> chunks = BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.SIZE, 2));
        assertEquals(3, chunks.size());
        assertEquals(1, chunks.get(2).size());
    }

    @Test
    void splitArrayEmptyReturnsEmptyList() {
        assertTrue(BatchChunkAssembler.split(new JSONArray(), new SubBatchSpec(SubBatchSpec.Mode.SIZE, 5)).isEmpty());
    }

    @Test
    void splitByCountFrontLoadsRemainder() {
        JSONArray arr = JSONArray.of(1, 2, 3, 4, 5, 6, 7);
        List<JSONArray> chunks = BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.COUNT, 3));
        assertEquals(3, chunks.size());
        assertEquals(3, chunks.get(0).size());
        assertEquals(2, chunks.get(1).size());
        assertEquals(2, chunks.get(2).size());
    }

    @Test
    void splitByCountClampsToTotalWhenRequestedExceeds() {
        JSONArray arr = JSONArray.of("a", "b");
        List<JSONArray> chunks = BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.COUNT, 5));
        assertEquals(2, chunks.size());
    }

    @Test
    void specAwareSplitRoutesBySizeAndCount() {
        JSONArray arr = JSONArray.of(1, 2, 3, 4);
        assertEquals(2, BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.SIZE, 2)).size());
        assertEquals(3, BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.COUNT, 3)).size());
    }

    @Test
    void buildChunkBodiesDeepClonesAndReplacesArray() {
        JSONObject envelope = new JSONObject();
        envelope.put("model", "m");
        envelope.put("prompt_batch", JSONArray.of("a", "b", "c"));
        JSONObject gc = new JSONObject();
        gc.put("temperature", 0.5);
        envelope.put("generate_config", gc);

        List<JSONArray> chunks = List.of(JSONArray.of("a"), JSONArray.of("b", "c"));
        List<JSONObject> bodies = BatchChunkAssembler.buildChunkBodies(envelope, chunks, BatchEndpointSpec.BATCH_INFER, true);

        assertEquals(2, bodies.size());
        assertEquals(JSONArray.of("a"), bodies.get(0).getJSONArray("prompt_batch"));
        assertEquals(JSONArray.of("b", "c"), bodies.get(1).getJSONArray("prompt_batch"));
        assertEquals("m", bodies.get(0).getString("model"));
        // Each chunk has its own generate_config so per-chunk mutations don't leak.
        assertNotSame(bodies.get(0).getJSONObject("generate_config"),
                bodies.get(1).getJSONObject("generate_config"));
        // Original envelope is untouched.
        assertEquals(3, envelope.getJSONArray("prompt_batch").size());
        assertFalse(envelope.getJSONObject("generate_config").containsKey("force_batch"));
    }

    @Test
    void buildChunkBodiesStripsReservedCallerRoleAddrs() {
        JSONObject envelope = new JSONObject();
        envelope.put("model", "m");
        envelope.put("prompt_batch", JSONArray.of("a", "b"));
        JSONObject gc = new JSONObject();
        JSONArray roleAddrs = new JSONArray();
        roleAddrs.add("PREFILL@10.0.0.1:8000");
        gc.put("role_addrs", roleAddrs);
        envelope.put("generate_config", gc);

        List<JSONArray> chunks = List.of(JSONArray.of("a"), JSONArray.of("b"));
        List<JSONObject> bodies = BatchChunkAssembler.buildChunkBodies(envelope, chunks, BatchEndpointSpec.BATCH_INFER, true);

        assertFalse(bodies.get(0).getJSONObject("generate_config").containsKey("role_addrs"));
        assertFalse(bodies.get(1).getJSONObject("generate_config").containsKey("role_addrs"));
        assertEquals(1, envelope.getJSONObject("generate_config").getJSONArray("role_addrs").size(),
                "defense-in-depth stripping must not mutate the source envelope");
    }

    @Test
    void buildChunkBodiesNormalizesLegacyGenerationConfigWithoutLosingSettings() {
        JSONObject legacy = JSONObject.of(
                "max_new_tokens", 37,
                "temperature", 0.25,
                "adapter_name", "qa-lora");
        JSONObject envelope = JSONObject.of(
                "prompt_batch", JSONArray.of("a", "b"),
                "generation_config", legacy);

        List<JSONObject> bodies = BatchChunkAssembler.buildChunkBodies(
                envelope, List.of(JSONArray.of("a"), JSONArray.of("b")), BatchEndpointSpec.BATCH_INFER, true);

        for (JSONObject body : bodies) {
            assertFalse(body.containsKey("generation_config"));
            JSONObject gc = body.getJSONObject("generate_config");
            assertEquals(37, gc.getIntValue("max_new_tokens"));
            assertEquals(0.25, gc.getDoubleValue("temperature"));
            assertEquals("qa-lora", gc.getString("adapter_name"));
            assertTrue(gc.getBooleanValue("force_batch"));
        }
        assertFalse(legacy.containsKey("force_batch"),
                "normalization must not mutate the caller's config object");
    }

    @Test
    void buildChunkBodiesStripsTopLevelRoleAddrsAsDefenseInDepth() {
        JSONObject envelope = JSONObject.of(
                "prompt_batch", JSONArray.of("a"),
                "role_addrs", JSONArray.of(JSONObject.of("ip", "1.2.3.4")));

        JSONObject body = BatchChunkAssembler.buildChunkBodies(
                envelope, List.of(JSONArray.of("a")), BatchEndpointSpec.BATCH_INFER, true).getFirst();

        assertFalse(body.containsKey("role_addrs"));
        assertTrue(envelope.containsKey("role_addrs"));
    }

    @Test
    void validateGenerateConfigRejectsEveryCallerRoutingSpelling() {
        assertTrue(BatchChunkAssembler.validateGenerateConfig(
                JSONObject.of("role_addrs", new JSONArray())).contains("role_addrs"));
        assertTrue(BatchChunkAssembler.validateGenerateConfig(JSONObject.of(
                "generation_config", JSONObject.of("role_addrs", new JSONArray())))
                .contains("role_addrs"));
        assertEquals("generation_config must be a JSON object",
                BatchChunkAssembler.validateGenerateConfig(
                        JSONObject.of("generation_config", "invalid")));
    }

    @Test
    void buildChunkBodiesStampsForceBatchOnlyForPromptBatchEndpoints() {
        JSONObject envelope = new JSONObject();
        envelope.put("input", JSONArray.of("a", "b"));
        List<JSONArray> chunks = List.of(JSONArray.of("a"), JSONArray.of("b"));

        // prompt_batch generation endpoint: force_batch is stamped on every chunk.
        List<JSONObject> promptBodies = BatchChunkAssembler.buildChunkBodies(
                envelope, chunks, BatchEndpointSpec.BATCH_INFER, true);
        assertTrue(promptBodies.get(0).getJSONObject("generate_config").getBoolean("force_batch"));

        // Non-prompt_batch endpoints (embedding "input", openai "requests"): force_batch is a
        // generation generate_config flag with no meaning here, so no generate_config is fabricated.
        List<JSONObject> embeddingBodies = BatchChunkAssembler.buildChunkBodies(
                envelope, chunks, BatchEndpointSpec.EMBEDDING, true);
        assertFalse(embeddingBodies.get(0).containsKey("generate_config"));
        List<JSONObject> openaiBodies = BatchChunkAssembler.buildChunkBodies(
                envelope, chunks, BatchEndpointSpec.CHAT, true);
        assertFalse(openaiBodies.get(0).containsKey("generate_config"));
    }

    @Test
    void injectForceBatchAddsWhenAbsent() {
        JSONObject body = new JSONObject();
        BatchChunkAssembler.injectForceBatch(body);
        assertEquals(true, body.getJSONObject("generate_config").getBoolean("force_batch"));
    }

    @Test
    void injectForceBatchPreservesUserFalse() {
        JSONObject body = new JSONObject();
        JSONObject gc = new JSONObject();
        gc.put("force_batch", false);
        body.put("generate_config", gc);
        BatchChunkAssembler.injectForceBatch(body);
        assertEquals(false, body.getJSONObject("generate_config").getBoolean("force_batch"));
    }

    @Test
    void policyFallbackOverridesAndRemovesTopLevelForceBatch() {
        JSONObject envelope = JSONObject.of(
                "prompt_batch", JSONArray.of("a", "b"),
                "force_batch", true,
                "generate_config", JSONObject.of("temperature", 0.5));

        JSONObject body = BatchChunkAssembler.buildChunkBodies(
                envelope, List.of(JSONArray.of("a")), BatchEndpointSpec.BATCH_INFER, false).getFirst();

        assertFalse(body.containsKey("force_batch"),
                "top-level GenerateConfig fields override nested fields in RequestExtractor");
        assertFalse(body.getJSONObject("generate_config").getBooleanValue("force_batch"));
        assertTrue(envelope.getBooleanValue("force_batch"),
                "policy fallback must not mutate the caller envelope");
    }

    @Test
    void stampPreAssignedBeAppendsRoleAddrs() {
        JSONObject body = new JSONObject();
        List<JSONObject> bodies = List.of(body);
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setRole(RoleType.PDFUSION);
        target.setServerIp("10.0.0.1");
        target.setHttpPort(8088);
        target.setGrpcPort(50051);

        BatchChunkAssembler.stampPreAssignedBe(bodies, List.of(target));

        JSONArray addrs = body.getJSONObject("generate_config").getJSONArray("role_addrs");
        assertEquals(1, addrs.size());
        JSONObject addr = addrs.getJSONObject(0);
        assertEquals("PDFUSION", addr.getString("role"));
        assertEquals("10.0.0.1", addr.getString("ip"));
        assertEquals(8088, addr.getIntValue("http_port"));
        assertEquals(50051, addr.getIntValue("grpc_port"));
    }

    @Test
    void stampPreAssignedBeSkipsTargetWithoutRole() {
        // Pre-assignment must never be able to fail a request: a target missing its role
        // (heterogeneous master response) is skipped like a missing grpc_port, not an NPE
        // that turns the whole batch into a 500.
        JSONObject body = new JSONObject();
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setServerIp("10.0.0.1");
        target.setHttpPort(8088);
        target.setGrpcPort(50051);
        // role left null

        BatchChunkAssembler.stampPreAssignedBe(List.of(body), List.of(target));

        assertNull(body.getJSONObject("generate_config"),
                "role-less target must be skipped without stamping anything");
    }

    @Test
    void stampPreAssignedBeReplacesAnyExistingRoleAddrs() {
        JSONObject body = new JSONObject();
        JSONObject gc = new JSONObject();
        JSONArray userAddrs = new JSONArray();
        userAddrs.add(JSONObject.of("role", "PREFILL", "ip", "1.1.1.1", "http_port", 80, "grpc_port", 50));
        gc.put("role_addrs", userAddrs);
        body.put("generate_config", gc);

        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setRole(RoleType.PDFUSION);
        target.setServerIp("10.0.0.1");
        target.setHttpPort(8088);
        target.setGrpcPort(50051);

        BatchChunkAssembler.stampPreAssignedBe(List.of(body), List.of(target));

        JSONArray addrs = body.getJSONObject("generate_config").getJSONArray("role_addrs");
        assertEquals(1, addrs.size());
        assertEquals("PDFUSION", addrs.getJSONObject(0).getString("role"));
    }

    @Test
    void stampPreAssignedBeNoOpOnEmptyTargets() {
        JSONObject body = new JSONObject();
        BatchChunkAssembler.stampPreAssignedBe(List.of(body), List.of());
        assertTrue(body.isEmpty());
    }

    @Test
    void stampPreAssignedBeToleratesShortTargetList() {
        JSONObject body0 = new JSONObject();
        JSONObject body1 = new JSONObject();
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.setRole(RoleType.PDFUSION);
        target.setServerIp("10.0.0.1");
        target.setHttpPort(8088);
        target.setGrpcPort(50051);

        BatchChunkAssembler.stampPreAssignedBe(List.of(body0, body1), List.of(target));

        assertFalse(body0.isEmpty());
        assertTrue(body1.isEmpty());
    }

    @Test
    void nonPositiveChunkSizeFailsExplicitlyRatherThanRelyingOnAssertions() {
        // These are public pure functions and assertions are off by default in production, so a
        // zero must raise here instead of reaching the chunk-count division as an ArithmeticException.
        JSONArray arr = new JSONArray();
        arr.add("a");

        assertThrows(IllegalArgumentException.class, () -> BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.SIZE, 0)));
        assertThrows(IllegalArgumentException.class, () -> BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.COUNT, 0)));
        assertThrows(IllegalArgumentException.class, () -> BatchChunkAssembler.split(arr, new SubBatchSpec(SubBatchSpec.Mode.SIZE, -1)));
    }

    @Test
    void chunkCountIsOverflowSafeAtIntegerMaxValue() {
        assertEquals(1_073_741_824, BatchChunkAssembler.chunkCount(
                Integer.MAX_VALUE, new SubBatchSpec(SubBatchSpec.Mode.SIZE, 2)));
        assertEquals(7, BatchChunkAssembler.chunkCount(
                Integer.MAX_VALUE, new SubBatchSpec(SubBatchSpec.Mode.COUNT, 7)));
        assertEquals(0, BatchChunkAssembler.chunkCount(
                0, new SubBatchSpec(SubBatchSpec.Mode.SIZE, 1)));
    }

    @Test
    void projectedOutboundBytesAccountsForRepeatedEnvelope() {
        JSONObject body = new JSONObject();
        body.put("model", "x".repeat(4096));
        JSONArray inputs = JSONArray.of("a", "b", "c");
        body.put("input", inputs);
        BatchEndpointSpec embeddings = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");

        long oneChunk = BatchChunkAssembler.projectedChunkBytes(
                body, inputs, 1, embeddings, true, List.of());
        long threeChunks = BatchChunkAssembler.projectedChunkBytes(
                body, inputs, 3, embeddings, true, List.of());

        assertTrue(threeChunks > oneChunk + 8_000,
                "the shared 4KiB envelope must be charged once per chunk");
    }

    @Test
    void projectionMatchesSerializedChunksAcrossEndpointRewritesAndRouting() {
        BatchScheduleTarget target = new BatchScheduleTarget("backend-中\"\\", 8088, 50051);
        target.setRole(RoleType.PDFUSION);
        for (BatchEndpointSpec spec : BatchEndpointSpec.SPECS) {
            for (String splitMode : List.of("size:2", "count:3", "count:20")) {
                for (boolean atomicAllowed : List.of(true, false)) {
                    for (int size : List.of(0, 1, 7)) {
                        JSONArray items = new JSONArray();
                        for (int i = 0; i < size; i++) {
                            items.add("中\"\\\n" + i);
                        }
                        JSONObject body = JSONObject.of(spec.getRequestArrayField(), items,
                                "model", "中-model", "query", "q", "sorted", true, "top_k", 2);
                        body.put("tools", null);
                        body.put("generation_config", JSONObject.of("temperature", 0.7, "seed", null));
                        SubBatchSpec split = SubBatchSpec.parse(splitMode);
                        List<JSONArray> chunks = BatchChunkAssembler.split(items, split);
                        List<BatchScheduleTarget> targets = spec.isPreAssignable() ? List.of(target) : List.of();
                        long projected = BatchChunkAssembler.projectedChunkBytes(
                                body, items, chunks.size(), spec, atomicAllowed, targets);
                        List<JSONObject> bodies = BatchChunkAssembler.buildChunkBodies(
                                body, chunks, spec, atomicAllowed);
                        BatchChunkAssembler.stampPreAssignedBe(bodies, targets);
                        long actual = bodies.stream().mapToLong(chunk -> BatchBodyParser.serialize(chunk).length).sum();
                        assertEquals(actual, projected, spec + "/" + splitMode + "/" + atomicAllowed + "/" + size);
                    }
                }
            }
        }
    }
}
