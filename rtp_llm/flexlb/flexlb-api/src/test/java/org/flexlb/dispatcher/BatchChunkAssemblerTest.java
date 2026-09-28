package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

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
        BatchChunkAssembler batch = batch(JSONObject.of("prompt_batch", JSON.parseArray(input)), mode);
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
        List<JSONObject> chunks = batch(source, "size:2").chunks(List.of());
        assertEquals("m", chunks.getFirst().getString("model"));
        assertSame(tools, chunks.getFirst().get("tools"));
        assertNotSame(source.get("generate_config"), chunks.getFirst().get("generate_config"));
        assertNotSame(chunks.getFirst().get("generate_config"), chunks.getLast().get("generate_config"));
        chunks.getFirst().getJSONObject("generate_config").put("temperature", 9);
        assertEquals(0.5, chunks.getLast().getJSONObject("generate_config").getDouble("temperature"));
        assertEquals(3, source.getJSONArray("prompt_batch").size());
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            {"generation_config":{"max_new_tokens":37,"temperature":0.25,"adapter_name":"qa-lora"}}
            {"generate_config":{"temperature":0.5},"generation_config":{"temperature":9}}
            {}
            """)
    void chunksPreserveCallerConfigWithoutNormalization(String json) {
        JSONObject source = JSON.parseObject(json);
        source.put("prompt_batch", JSONArray.of("a", "b"));
        byte[] original = BatchBodyParser.serialize(source);
        for (JSONObject chunk : batch(source, "size:1").chunks(List.of())) {
            chunk.remove("prompt_batch");
            assertEquals(JSON.parseObject(json), chunk);
        }
        assertArrayEquals(original, BatchBodyParser.serialize(source));
    }

    @Test
    void chunkCountsAreOverflowSafe() {
        assertEquals(1_073_741_824, BatchChunkAssembler.chunkCount(Integer.MAX_VALUE, SubBatchSpec.parse("size:2")));
        assertEquals(7, BatchChunkAssembler.chunkCount(Integer.MAX_VALUE, SubBatchSpec.parse("count:7")));
        assertEquals(0, BatchChunkAssembler.chunkCount(0, SubBatchSpec.parse("size:1")));
    }

    @Test
    void projectionMatchesActualWireBytesAcrossEndpointRewrites() {
        for (BatchEndpointSpec spec : BatchEndpointSpec.SPECS) {
            for (String mode : List.of("size:2", "count:3", "count:20")) {
                for (int size : List.of(0, 1, 7)) {
                    JSONArray items = new JSONArray();
                    for (int i = 0; i < size; i++) {
                        items.add("中\"\\\n" + i);
                    }
                    JSONObject body = JSONObject.of(spec.getRequestArrayField(), items,
                            "model", "中-model", "query", "q", "sorted", true, "top_k", 2);
                    body.put("tools", null);
                    body.put("generation_config", JSONObject.of("temperature", 0.7, "seed", null));
                    BatchChunkAssembler batch = new BatchChunkAssembler(body, spec, SubBatchSpec.parse(mode));
                    long actual = batch.chunks(List.of()).stream().mapToLong(c -> BatchBodyParser.serialize(c).length).sum();
                    assertEquals(actual, batch.projectedBytes(), spec + "/" + mode + "/" + size);
                }
            }
        }
    }

    private static BatchChunkAssembler batch(JSONObject body, String mode) {
        return new BatchChunkAssembler(body, BatchEndpointSpec.BATCH_INFER, SubBatchSpec.parse(mode));
    }
}
