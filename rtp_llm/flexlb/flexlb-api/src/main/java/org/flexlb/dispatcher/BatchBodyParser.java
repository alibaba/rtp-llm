package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONException;
import com.alibaba.fastjson2.JSONObject;

import java.util.Objects;

/**
 * Thin entry point for parsing dispatcher batch request bodies with fastjson2. Replaces the
 * hand-rolled byte-level scanner; fastjson2's {@link JSON#parseObject(byte[])} consumes UTF-8
 * input directly without an intermediate {@code String} allocation.
 *
 * <p>The dispatcher is array-only by design — every batch endpoint declares a
 * {@code requestArrayField} that must be present and shaped as a JSON array. {@link
 * #findArrayField(JSONObject, String)} returns {@code null} when the field is absent or not an
 * array so the handler can reject with a 400 instead of relaying a non-batch request to FE.
 */
public final class BatchBodyParser {

    private BatchBodyParser() {}

    /**
     * Parses a UTF-8 encoded JSON object body. Returns {@code null} when the top-level value is
     * not a JSON object (e.g. an array, scalar, or {@code null}); the handler maps that to a 400.
     *
     * @throws JSONException on malformed JSON, propagated to the handler's {@code onErrorResume}.
     */
    public static JSONObject parseObject(byte[] body) {
        Objects.requireNonNull(body, "body");
        Object root = JSON.parse(body);
        return root instanceof JSONObject obj ? obj : null;
    }

    /**
     * Returns the value of a top-level field as a {@link JSONArray}, or {@code null} when the
     * field is missing or not an array. Mirrors the validation
     * {@code GenericBatchHandler} performs on the Jackson side.
     */
    public static JSONArray findArrayField(JSONObject body, String fieldName) {
        Objects.requireNonNull(body, "body");
        Objects.requireNonNull(fieldName, "fieldName");
        Object value = body.get(fieldName);
        return value instanceof JSONArray arr ? arr : null;
    }
}
