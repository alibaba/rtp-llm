package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONException;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;

/** UTF-8 JSON parsing and serialization for batch requests and FE responses. */
public final class BatchBodyParser {

    private BatchBodyParser() {}

    /** Parses a UTF-8 encoded JSON object body. */
    public static JSONObject parseObject(byte[] body) {
        if (body.length == 0) {
            return null;
        }
        try {
            return JSON.parseObject(body);
        } catch (JSONException e) {
            return null;
        }
    }

    /** Preserve explicit nulls, including top-level overrides of nested configuration. */
    static byte[] serialize(Object value) {
        return JSON.toJSONBytes(value, JSONWriter.Feature.WriteNulls);
    }
}
