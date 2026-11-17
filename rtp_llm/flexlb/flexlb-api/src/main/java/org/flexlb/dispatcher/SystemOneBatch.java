package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;

/** Batch items are complete SystemOne requests; questions within one item stay together. */
final class SystemOneBatch {

    private SystemOneBatch() {}

    static String validate(JSONObject body) {
        for (String key : body.keySet()) {
            if (!key.equals("requests") && !key.equals("model")) {
                return "SystemOne batch envelope accepts only requests and model";
            }
        }
        if (body.containsKey("model")
                && (!(body.get("model") instanceof String model) || model.isBlank())) {
            return "model must be a nonempty string";
        }
        for (Object value : body.getJSONArray("requests")) {
            if (!(value instanceof JSONObject item)
                    || item.containsKey("requests")
                    || !item.containsKey("state")
                    || !(item.get("questions") instanceof JSONObject questions)
                    || questions.isEmpty() || questions.size() > 64) {
                return "requests must contain single SystemOne objects with state and 1..64 questions";
            }
        }
        return null;
    }
}
