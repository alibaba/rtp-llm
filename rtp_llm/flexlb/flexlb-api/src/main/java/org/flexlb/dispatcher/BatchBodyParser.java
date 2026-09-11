package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONException;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;

import java.util.Objects;

/** UTF-8 JSON parsing and serialization shared by dispatcher request and response paths. */
public final class BatchBodyParser {

    private BatchBodyParser() {}

    /**
     * Parses a UTF-8 encoded JSON object body. Returns {@code null} when the body is not a JSON
     * object — either because it's a top-level array / scalar / {@code null}, or because the
     * bytes don't parse as valid JSON at all. The handler maps both cases to 400 (invalid
     * batch request) with the same envelope, so the loss of distinction is cosmetic.
     *
     * <p>The typed byte-array parser avoids an intermediate String allocation.
     */
    public static JSONObject parseObject(byte[] body) {
        Objects.requireNonNull(body, "body");
        if (body.length == 0) {
            return null;
        }
        try {
            return JSON.parseObject(body);
        } catch (JSONException e) {
            return null;
        }
    }

    /**
     * Returns the value of a top-level field as a {@link JSONArray}, or {@code null} when the
     * field is missing or not an array.
     */
    public static JSONArray findArrayField(JSONObject body, String fieldName) {
        Objects.requireNonNull(body, "body");
        Objects.requireNonNull(fieldName, "fieldName");
        Object value = body.get(fieldName);
        return value instanceof JSONArray arr ? arr : null;
    }

    /**
     * WriteNulls preserves explicit nulls on the wire (e.g. {@code embedding: null} from
     * {@link BatchEndpointSpec#failedItem}); fastjson2 strips null
     * entries by default.
     */
    static byte[] serialize(Object value) {
        return JSON.toJSONBytes(value, JSONWriter.Feature.WriteNulls);
    }
}
