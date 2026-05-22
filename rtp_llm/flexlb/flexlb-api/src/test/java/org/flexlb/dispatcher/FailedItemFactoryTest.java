package org.flexlb.dispatcher;

import static org.flexlb.dispatcher.BatchEndpointSpec.FailedItemFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FailedItemFactoryTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void nullFactoryReturnsJsonNull() {
        JsonNode n = FailedItemFactory.NULL.build(7, "fe_timeout", mapper);
        assertTrue(n.isNull());
    }

    @Test
    void openAiErrorFactoryBuildsErrorObject() {
        JsonNode n = FailedItemFactory.OPENAI_ERROR.build(7, "fe_timeout", mapper);
        assertEquals(7, n.get("index").asInt());
        assertEquals("fe_timeout", n.get("error").get("message").asText());
        assertEquals("dispatcher_sub_batch_failed", n.get("error").get("code").asText());
    }

    @Test
    void embeddingNullFactoryBuildsDataItemWithNullEmbedding() {
        JsonNode n = FailedItemFactory.EMBEDDING_NULL.build(7, "fe_timeout", mapper);
        assertEquals(7, n.get("index").asInt());
        assertTrue(n.get("embedding").isNull());
        assertEquals("fe_timeout", n.get("error").asText());
    }
}
