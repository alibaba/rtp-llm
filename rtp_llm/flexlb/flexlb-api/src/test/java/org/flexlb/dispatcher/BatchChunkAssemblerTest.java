package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchChunkAssemblerTest {
    @ParameterizedTest
    @CsvSource(delimiter = '|', textBlock = """
            size:2 | [1,2,3,4] | [[1,2],[3,4]]
            size:2 | [1,2,3,4,5] | [[1,2],[3,4],[5]]
            size:5 | [] | []
            count:3 | [1,2,3,4,5,6,7] | [[1,2,3],[4,5],[6,7]]
            count:5 | [1,2] | [[1],[2]]
            count:3 | [1,2,3,4] | [[1,2],[3],[4]]
            """)
    void splittingPreservesOrderAndDistribution(String mode, String input, String expected) {
        BatchChunkAssembler batch = batch(JSONObject.of("prompt_batch", JSON.parseArray(input)), mode, true);
        List<JSONArray> actual = batch.chunks(List.of()).stream().map(b -> b.getJSONArray("prompt_batch")).toList();
        assertEquals(JSON.parseArray(expected), actual);
        assertEquals(actual.size(), batch.chunkCount());
        for (int i = 0; i < actual.size(); i++) {
            assertEquals(actual.get(i).size(), batch.chunkSize(i));
        }
    }

    @Test
    void chunksIsolateMutableFieldsAndShareLargeReadOnlyFields() {
        JSONObject tools = JSONObject.of("schema", "x".repeat(4096));
        JSONObject source = JSONObject.of("model", "m", "tools", tools,
                "prompt_batch", JSONArray.of("a", "b", "c"),
                "generate_config", JSONObject.of("temperature", 0.5));
        List<JSONObject> chunks = batch(source, "size:2", true).chunks(List.of());
        assertEquals("m", chunks.getFirst().getString("model"));
        assertSame(tools, chunks.getFirst().get("tools"));
        assertNotSame(source.get("generate_config"), chunks.getFirst().get("generate_config"));
        assertNotSame(chunks.getFirst().get("generate_config"), chunks.getLast().get("generate_config"));
        chunks.getFirst().getJSONObject("generate_config").put("temperature", 9);
        assertEquals(0.5, chunks.getLast().getJSONObject("generate_config").getDouble("temperature"));
        assertEquals(3, source.getJSONArray("prompt_batch").size());
        assertFalse(source.getJSONObject("generate_config").containsKey("force_batch"));
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            {"generation_config":{"max_new_tokens":37,"temperature":0.25,"adapter_name":"qa-lora"}} | true | {"max_new_tokens":37,"temperature":0.25,"adapter_name":"qa-lora","force_batch":true}
            {"generate_config":{"temperature":0.5},"generation_config":{"temperature":9}} | true | {"temperature":0.5,"force_batch":true}
            {} | true | {"force_batch":true}
            {"generate_config":{"force_batch":false}} | true | {"force_batch":false}
            {"generate_config":{"force_batch":null}} | true | {"force_batch":null}
            {"force_batch":true,"generate_config":{"temperature":0.5}} | false | {"temperature":0.5,"force_batch":false}
            {"role_addrs":[{}],"generate_config":{"role_addrs":[{}]}} | true | {"force_batch":true}
            """)
    void configNormalizationPreservesCallerSettings(String json, boolean allowed, String expected) {
        JSONObject source = JSON.parseObject(json);
        source.put("prompt_batch", JSONArray.of("a", "b"));
        byte[] original = BatchBodyParser.serialize(source);
        for (JSONObject chunk : batch(source, "size:1", allowed).chunks(List.of())) {
            assertEquals(JSON.parseObject(expected), chunk.getJSONObject("generate_config"));
            assertFalse(chunk.containsKey("generation_config"));
            assertFalse(chunk.containsKey("role_addrs"));
            if (!allowed) {
                assertFalse(chunk.containsKey("force_batch"));
            }
        }
        assertArrayEquals(original, BatchBodyParser.serialize(source));
    }

    @Test
    void forceBatchBelongsOnlyToPromptEndpoints() {
        for (BatchEndpointSpec spec : BatchEndpointSpec.SPECS) {
            JSONObject body = JSONObject.of(spec.getRequestArrayField(), JSONArray.of("a"));
            JSONObject chunk = new BatchChunkAssembler(body, spec, SubBatchSpec.parse("size:1"), true)
                    .chunks(List.of()).getFirst();
            assertEquals(spec.isPreAssignable(), chunk.containsKey("generate_config"));
        }
    }

    @Test
    void stampsOnlyAvailableGrpcTargetsAndNeverAppendsCallerAddresses() {
        JSONObject source = JSONObject.of("prompt_batch", JSONArray.of("a", "b", "c", "d"),
                "generate_config", JSONObject.of("role_addrs", JSONArray.of(JSONObject.of("ip", "caller"))));
        BatchScheduleTarget valid = new BatchScheduleTarget("10.0.0.1", 8088, 50051, RoleType.PDFUSION);
        BatchScheduleTarget noRole = new BatchScheduleTarget("10.0.0.2", 8088, 50051);
        BatchScheduleTarget noGrpc = new BatchScheduleTarget();
        noGrpc.setRole(RoleType.PDFUSION);
        BatchChunkAssembler batch = batch(source, "size:1", true);
        List<JSONObject> chunks = batch.chunks(List.of(valid, noRole, noGrpc));
        assertEquals(JSON.parseArray("""
                [{"role":"PDFUSION","ip":"10.0.0.1","http_port":8088,"grpc_port":50051}]
                """), chunks.getFirst().getJSONObject("generate_config").getJSONArray("role_addrs"));
        for (int i = 1; i < chunks.size(); i++) {
            assertFalse(chunks.get(i).getJSONObject("generate_config").containsKey("role_addrs"));
        }
        assertTrue(batch.chunks(List.of()).stream().noneMatch(c -> c.getJSONObject("generate_config").containsKey("role_addrs")));
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            {"role_addrs":[]} | role_addrs
            {"generation_config":{"role_addrs":[]}} | role_addrs
            {"generation_config":"invalid"} | generation_config must be a JSON object
            """)
    void boundaryRejectsReservedRoutingFields(String json, String reason) {
        assertTrue(BatchEndpointSpec.ROOT.validateRequest(JSON.parseObject(json)).contains(reason));
    }

    @Test
    void countsAreOverflowSafeAndRejectInvalidSizes() {
        assertEquals(1_073_741_824, BatchChunkAssembler.chunkCount(Integer.MAX_VALUE, SubBatchSpec.parse("size:2")));
        assertEquals(7, BatchChunkAssembler.chunkCount(Integer.MAX_VALUE, SubBatchSpec.parse("count:7")));
        assertEquals(0, BatchChunkAssembler.chunkCount(0, SubBatchSpec.parse("size:1")));
        for (SubBatchSpec.Mode mode : SubBatchSpec.Mode.values()) {
            for (int size : List.of(0, -1)) {
                assertThrows(IllegalArgumentException.class,
                        () -> BatchChunkAssembler.chunkCount(1, new SubBatchSpec(mode, size)));
            }
        }
    }

    @Test
    void repeatedEnvelopesAreCounted() {
        JSONObject body = JSONObject.of("model", "x".repeat(4096), "prompt_batch", JSONArray.of("a", "b", "c"));
        long one = batch(body, "count:1", true).projectedBytes(List.of());
        assertTrue(batch(body, "count:3", true).projectedBytes(List.of()) > one + 8000);
    }

    @Test
    void projectionMatchesActualWireBytesAcrossEndpointRewritesAndRouting() {
        BatchScheduleTarget target = new BatchScheduleTarget("backend-中\"\\", 8088, 50051, RoleType.PDFUSION);
        for (BatchEndpointSpec spec : BatchEndpointSpec.SPECS) {
            for (String mode : List.of("size:2", "count:3", "count:20")) {
                for (boolean allowed : List.of(true, false)) {
                    for (int size : List.of(0, 1, 7)) {
                        JSONArray items = new JSONArray();
                        for (int i = 0; i < size; i++) {
                            items.add("中\"\\\n" + i);
                        }
                        JSONObject body = JSONObject.of(spec.getRequestArrayField(), items,
                                "model", "中-model", "query", "q", "sorted", true, "top_k", 2);
                        body.put("tools", null);
                        body.put("generation_config", JSONObject.of("temperature", 0.7, "seed", null));
                        BatchChunkAssembler batch = new BatchChunkAssembler(body, spec, SubBatchSpec.parse(mode), allowed);
                        List<BatchScheduleTarget> targets = spec.isPreAssignable() ? List.of(target) : List.of();
                        long actual = batch.chunks(targets).stream().mapToLong(c -> BatchBodyParser.serialize(c).length).sum();
                        assertEquals(actual, batch.projectedBytes(targets), spec + "/" + mode + "/" + allowed + "/" + size);
                    }
                }
            }
        }
    }

    private static BatchChunkAssembler batch(JSONObject body, String mode, boolean allowed) {
        return new BatchChunkAssembler(body, BatchEndpointSpec.BATCH_INFER, SubBatchSpec.parse(mode), allowed);
    }
}
