package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BatchEndpointRegistryTest {

    @Test
    void registryContainsAllThreeSpecsKeyedByPath() {
        BatchEndpointRegistry registry = new BatchEndpointRegistry();
        List<BatchEndpointSpec> specs = registry.batchSpecs();
        Map<String, BatchEndpointSpec> byPath = registry.batchSpecsByPath(specs);

        assertEquals(3, specs.size());
        assertEquals(3, byPath.size());
        assertNull(byPath.get("/"));

        BatchEndpointSpec batchInfer = byPath.get("/batch_infer");
        assertEquals("prompt_batch", batchInfer.getRequestArrayField());
        assertEquals("response_batch", batchInfer.getResponseArrayField());
        assertSame(FailedItemFactory.NULL, batchInfer.getFailedItemFactory());
        assertNull(batchInfer.getPostMerger());

        BatchEndpointSpec openAi = byPath.get("/v1/batch/chat/completions");
        assertEquals("requests", openAi.getRequestArrayField());
        assertEquals("responses", openAi.getResponseArrayField());
        assertSame(FailedItemFactory.OPENAI_ERROR, openAi.getFailedItemFactory());
        assertNull(openAi.getPostMerger());

        BatchEndpointSpec embeddings = byPath.get("/v1/embeddings");
        assertEquals("input", embeddings.getRequestArrayField());
        assertEquals("data", embeddings.getResponseArrayField());
        assertSame(FailedItemFactory.EMBEDDING_NULL, embeddings.getFailedItemFactory());
        assertSame(EmbeddingPostMerger.INSTANCE, embeddings.getPostMerger());
    }

    @Test
    void duplicatePathsFailAtMapBuildTime() {
        BatchEndpointRegistry registry = new BatchEndpointRegistry();
        BatchEndpointSpec a = new BatchEndpointSpec(
                "/dup", "in", "out", FailedItemFactory.NULL, null);
        BatchEndpointSpec b = new BatchEndpointSpec(
                "/dup", "in2", "out2", FailedItemFactory.NULL, null);

        assertThrows(IllegalStateException.class,
                () -> registry.batchSpecsByPath(List.of(a, b)));
    }
}
