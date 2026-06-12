package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
    void embeddingNullFactoryShape() {
        Object placeholder = BatchEndpointSpec.FailedItemFactory.EMBEDDING_NULL.build(2, "no_route");
        assertTrue(placeholder instanceof JSONObject);
        JSONObject item = (JSONObject) placeholder;
        assertEquals(2, item.getIntValue("index"));
        assertNull(item.get("embedding"));
        assertEquals("no_route", item.getString("error"));
    }
}
