package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RequestTest {
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void shouldDeserializeChatIdFromSchedulePayload() throws Exception {
        Request request =
                objectMapper.readValue(
                        """
                        {
                          "block_cache_keys": [11, 22],
                          "seq_len": 128,
                          "request_id": 12345,
                          "generate_timeout": 60000,
                          "request_time_ms": 98765,
                          "chat_id": "chat-a"
                        }
                        """,
                        Request.class);

        assertEquals("chat-a", request.getChatId());
    }

    @Test
    void shouldSerializeChatIdToSchedulePayload() throws Exception {
        Request request = new Request();
        request.setChatId("chat-a");

        String json = objectMapper.writeValueAsString(request);

        assertEquals("chat-a", objectMapper.readTree(json).get("chat_id").asText());
    }
}
