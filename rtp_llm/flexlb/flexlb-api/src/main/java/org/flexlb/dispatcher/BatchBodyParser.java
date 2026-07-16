package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONException;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;

import java.util.Objects;

/**
 * Thin entry point for parsing dispatcher batch request bodies with fastjson2.
 * {@link JSON#parseObject(byte[])} consumes UTF-8 input directly without an intermediate
 * {@code String} allocation.
 *
 * <p>The dispatcher is array-only by design — every batch endpoint declares a
 * {@code requestArrayField} that must be present and shaped as a JSON array. {@link
 * #findArrayField(JSONObject, String)} returns {@code null} when the field is absent or not an
 * array so the handler can reject with a 400 instead of relaying a non-batch request to FE.
 */
public final class BatchBodyParser {

    private BatchBodyParser() {}

    /**
     * Parses a UTF-8 encoded JSON object body. Returns {@code null} when the body is not a JSON
     * object — either because it's a top-level array / scalar / {@code null}, or because the
     * bytes don't parse as valid JSON at all. The handler maps both cases to 400 (invalid
     * batch request) with the same envelope, so the loss of distinction is cosmetic.
     *
     * <p>Uses {@link JSON#parseObject(byte[])} (the typed entry point) rather than
     * {@link JSON#parse(byte[])} + cast — the typed entry hits fastjson2's JSONObject-specific
     * ObjectReader path that skips the runtime type-dispatch the untyped {@code parse} does on
     * every nested value. Measured ~21% faster on a 752KB envelope (500 CJK prompts × 500 chars).
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

    static JSONObject deepCopy(JSONObject source) {
        return JSON.parseObject(JSON.toJSONBytes(source));
    }

    /**
     * WriteNulls preserves explicit nulls on the wire (e.g. {@code embedding: null} from
     * {@link BatchEndpointSpec.FailedItemFactory#EMBEDDING_NULL}); fastjson2 strips null
     * entries by default.
     */
    static byte[] serialize(JSONObject body) {
        return JSON.toJSONBytes(body, JSONWriter.Feature.WriteNulls);
    }
}
