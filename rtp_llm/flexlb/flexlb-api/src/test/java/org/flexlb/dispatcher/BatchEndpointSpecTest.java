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
        assertNull(BatchEndpointSpec.BY_PATH.get("/no-such-path"));
    }

    @Test
    void batchInferSpecShape() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        assertEquals("prompt_batch", spec.getRequestArrayField());
        assertEquals("response_batch", spec.getResponseArrayField());
        assertEquals(BatchEndpointSpec.FailedItemFactory.NULL, spec.getFailedItemFactory());
    }

    @Test
    void embeddingsSpecShape() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");
        assertEquals("input", spec.getRequestArrayField());
        assertEquals("data", spec.getResponseArrayField());
        assertEquals(BatchEndpointSpec.FailedItemFactory.EMBEDDING_NULL,
                spec.getFailedItemFactory());
    }

    @Test
    void nullFactoryReturnsNullPlaceholder() {
        assertNull(BatchEndpointSpec.FailedItemFactory.NULL.build(3, "boom"));
    }

    @Test
    void openaiErrorFactoryShape() {
        Object placeholder = BatchEndpointSpec.FailedItemFactory.OPENAI_ERROR.build(7, "timeout");
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
    void promptBatchWithoutAlignedCompanionsSplits() {
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get("/batch_infer");
        JSONObject plain = JSONObject.of("prompt_batch", new String[]{"a", "b"});
        assertFalse(spec.requiresWholeBody(plain));

        // A scalar adapter_name applies to the whole batch — no alignment, safe to split.
        JSONObject scalarAdapter = JSONObject.of("prompt_batch", new String[]{"a", "b"},
                "generate_config", JSONObject.of("adapter_name", "lora"));
        assertFalse(spec.requiresWholeBody(scalarAdapter));
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
        Object placeholder = BatchEndpointSpec.FailedItemFactory.EMBEDDING_NULL.build(2, "no_route");
        assertTrue(placeholder instanceof JSONObject);
        JSONObject item = (JSONObject) placeholder;
        assertEquals(2, item.getIntValue("index"));
        assertNull(item.get("embedding"));
        assertEquals("no_route", item.getString("error"));
    }
}
