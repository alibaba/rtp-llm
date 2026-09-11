package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchEndpointSpecTest {

    @Test
    void specsTableIsIndexedByPath() {
        assertNotNull(BatchEndpointSpec.BY_PATH.get("/batch_infer"));
        assertNotNull(BatchEndpointSpec.BY_PATH.get("/v1/embeddings"));
        assertNotNull(BatchEndpointSpec.BY_PATH.get("/v1/reranker"));
        assertNull(BatchEndpointSpec.BY_PATH.get("/no-such-path"));
    }

    @Test
    void batchInferSpecShape() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        assertEquals("prompt_batch", spec.getRequestArrayField());
        assertEquals("response_batch", spec.getResponseArrayField());
        assertNull(spec.failedItem(0, "failed"));
    }

    @Test
    void registeredEndpointsHaveExpectedRoutingSemantics() {
        assertEquals(5, BatchEndpointSpec.BY_PATH.size());
        for (BatchEndpointSpec spec : BatchEndpointSpec.SPECS) {
            boolean generation = spec.getPath().equals("/") || spec.getPath().equals("/batch_infer");
            assertEquals(generation, spec.isPreAssignable());
            assertEquals(spec.getPath().equals("/v1/reranker"), spec.isFailOnPartialFailure());
        }
    }

    @Test
    void embeddingsSpecShape() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");
        assertEquals("input", spec.getRequestArrayField());
        assertEquals("data", spec.getResponseArrayField());
        assertTrue(spec.failedItem(0, "failed") instanceof JSONObject);
    }

    @Test
    void nullFactoryReturnsNullPlaceholder() {
        assertNull(BatchEndpointSpec.BATCH_INFER.failedItem(3, "boom"));
    }

    @Test
    void openaiErrorFactoryShape() {
        Object placeholder = BatchEndpointSpec.CHAT.failedItem(7, "timeout");
        assertTrue(placeholder instanceof JSONObject);
        JSONObject item = (JSONObject) placeholder;
        assertEquals(7, item.getIntValue("index"));
        JSONObject err = item.getJSONObject("error");
        assertEquals("dispatcher_sub_batch_failed", err.getString("code"));
        assertEquals("timeout", err.getString("message"));
    }

    @Test
    void promptBatchWithTopLevelImagesRequiresWholeBody() {
        // FE root `/` aligns top-level `images`/`urls` (list[list]) to the prompt count; a split
        // chunk would carry the full-length companion against a shorter prompt slice and FE would
        // reject every chunk. Such bodies must be forwarded whole instead of split.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/");
        JSONObject withImages = JSONObject.of("prompt_batch", new String[]{"a", "b"},
                "images", new Object[]{new String[]{"u0"}, new String[]{"u1"}});
        assertTrue(spec.requiresWholeBody(withImages));

        JSONObject withUrls = JSONObject.of("prompt_batch", new String[]{"a", "b"},
                "urls", new Object[]{new String[]{"u0"}, new String[]{"u1"}});
        assertTrue(spec.requiresWholeBody(withUrls));
    }

    @Test
    void promptBatchWithListAdapterNameRequiresWholeBody() {
        // FE `_get_adapter` rejects when a list-form adapter_name length != prompt count.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        JSONObject gc = JSONObject.of("adapter_name", new String[]{"lora0", "lora1"});
        JSONObject body = JSONObject.of("prompt_batch", new String[]{"a", "b"},
                "generate_config", gc);
        assertTrue(spec.requiresWholeBody(body));
    }

    @Test
    void promptBatchWithLegacyListAdapterNameRequiresWholeBody() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/");
        JSONObject body = JSONObject.of("prompt_batch", new String[]{"a", "b"},
                "generation_config", JSONObject.of(
                        "adapter_name", new String[]{"lora0", "lora1"}));

        assertTrue(spec.requiresWholeBody(body));
    }

    @Test
    void promptBatchUsesTopLevelAdapterNamePrecedence() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/");
        JSONObject nestedScalar = JSONObject.of("adapter_name", "nested-lora");
        JSONObject topLevelList = JSONObject.of("prompt_batch", new String[]{"a", "b"},
                "generate_config", nestedScalar,
                "adapter_name", new String[]{"lora0", "lora1"});
        assertTrue(spec.requiresWholeBody(topLevelList));

        // RequestExtractor applies top-level config after nested config. An explicit null therefore
        // disables the nested adapter instead of exposing its list to positional validation.
        JSONObject topLevelNull = JSONObject.of("prompt_batch", new String[]{"a", "b"},
                "generate_config", JSONObject.of(
                        "adapter_name", new String[]{"lora0", "lora1"}));
        topLevelNull.put("adapter_name", null);
        assertFalse(spec.requiresWholeBody(topLevelNull));
    }

    @Test
    void streamingPromptBatchFormsRequireWholeBody() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/");

        assertTrue(spec.requiresWholeBody(JSONObject.of(
                "prompt_batch", new String[]{"a"}, "stream", true)));
        assertTrue(spec.requiresWholeBody(JSONObject.of(
                "prompt_batch", new String[]{"a"}, "yield_generator", true)));
        assertTrue(spec.requiresWholeBody(JSONObject.of(
                "prompt_batch", new String[]{"a"}, "generation_config",
                JSONObject.of("yield_generator", true))));
        assertTrue(spec.requiresWholeBody(JSONObject.of(
                "prompt_batch", new String[]{"a"}, "generate_config",
                JSONObject.of("is_streaming", true))));

        assertFalse(spec.requiresWholeBody(JSONObject.of(
                "prompt_batch", new String[]{"a"}, "stream", false,
                "generate_config", JSONObject.of("yield_generator", false))));
    }

    @Test
    void promptBatchWithoutAlignedCompanionsSplits() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        JSONObject plain = JSONObject.of("prompt_batch", new String[]{"a", "b"});
        assertFalse(spec.requiresWholeBody(plain));
    }

    @Test
    void promptBatchWithScalarAdapterNameRequiresWholeBody() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        // FE only accepts a scalar adapter for one input. Splitting could otherwise turn an
        // invalid multi-prompt request into independently valid requests.
        JSONObject scalarAdapter = JSONObject.of("prompt_batch", new String[]{"a", "b"},
                "generate_config", JSONObject.of("adapter_name", "lora"));
        assertTrue(spec.requiresWholeBody(scalarAdapter));
    }

    @Test
    void nonPromptBatchEndpointNeverRequiresWholeBodyForCompanions() {
        // Only the prompt_batch endpoints carry these top-level companion arrays; `requests`
        // items on /v1/batch/chat/completions are self-contained.
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/batch/chat/completions");
        JSONObject body = JSONObject.of("requests", new Object[]{JSONObject.of("messages", "x")},
                "images", new Object[]{new String[]{"u0"}});
        assertFalse(spec.requiresWholeBody(body));
    }

    @Test
    void embeddingNullFactoryShape() {
        Object placeholder = BatchEndpointSpec.EMBEDDING.failedItem(2, "no_route");
        assertTrue(placeholder instanceof JSONObject);
        JSONObject item = (JSONObject) placeholder;
        assertEquals(2, item.getIntValue("index"));
        assertNull(item.get("embedding"));
        assertEquals("no_route", item.getString("error"));
    }
}
