package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class BatchBodyParserTest {

    @Test
    void parseObjectReturnsObjectForTopLevelObject() {
        byte[] body = "{\"prompt_batch\":[\"a\",\"b\"],\"model\":\"m\"}".getBytes(StandardCharsets.UTF_8);
        JSONObject obj = BatchBodyParser.parseObject(body);
        assertNotNull(obj);
        assertEquals("m", obj.getString("model"));
    }

    @Test
    void parseObjectReturnsNullForTopLevelArray() {
        byte[] body = "[1,2,3]".getBytes(StandardCharsets.UTF_8);
        assertNull(BatchBodyParser.parseObject(body));
    }

    @Test
    void parseObjectReturnsNullForTopLevelScalar() {
        assertNull(BatchBodyParser.parseObject("42".getBytes(StandardCharsets.UTF_8)));
        assertNull(BatchBodyParser.parseObject("\"hi\"".getBytes(StandardCharsets.UTF_8)));
        assertNull(BatchBodyParser.parseObject("null".getBytes(StandardCharsets.UTF_8)));
    }

    @Test
    void parseObjectReturnsNullForEmptyBytes() {
        // A body-less POST reaches the handler as zero bytes via defaultIfEmpty.
        assertNull(BatchBodyParser.parseObject(new byte[0]));
    }

    @Test
    void parseObjectReturnsNullOnMalformedJson() {
        // Malformed input and "valid JSON but not an object" both map to 400 at the handler with
        // the same envelope, so parser collapses both into null rather than throw a JSONException
        // that the handler would have to catch separately.
        byte[] body = "{not json".getBytes(StandardCharsets.UTF_8);
        assertNull(BatchBodyParser.parseObject(body));
    }

    @Test
    void findArrayFieldReturnsTheArray() {
        JSONObject obj = JSONObject.of("prompt_batch", JSONArray.of("a", "b"));
        JSONArray arr = BatchBodyParser.findArrayField(obj, "prompt_batch");
        assertNotNull(arr);
        assertEquals(2, arr.size());
    }

    @Test
    void findArrayFieldReturnsNullWhenMissing() {
        JSONObject obj = JSONObject.of("model", "m");
        assertNull(BatchBodyParser.findArrayField(obj, "prompt_batch"));
    }

    @Test
    void findArrayFieldReturnsNullWhenNotArray() {
        JSONObject obj = JSONObject.of("prompt_batch", "not-an-array");
        assertNull(BatchBodyParser.findArrayField(obj, "prompt_batch"));
    }
}
