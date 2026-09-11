package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;

import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/** A batch's normalized envelope, shared by size projection and chunk materialization. */
public final class BatchChunkAssembler {
    private final JSONObject template;
    private final JSONArray items;
    private final BatchEndpointSpec endpoint;
    private final SubBatchSpec split;
    private final int count;
    private final String configKey;

    public BatchChunkAssembler(JSONObject body, BatchEndpointSpec endpoint,
                                SubBatchSpec split, boolean atomicBatchAllowed) {
        this.items = body.getJSONArray(endpoint.getRequestArrayField());
        this.endpoint = endpoint;
        this.split = split;
        this.count = chunkCount(items.size(), split);
        template = new JSONObject(body);
        template.put(endpoint.getRequestArrayField(), new JSONArray());
        String key = body.containsKey("generate_config") ? "generate_config" : "generation_config";
        JSONObject originalConfig = body.getJSONObject(key);
        if (originalConfig != null || endpoint.isPreAssignable()) {
            JSONObject config = originalConfig == null ? new JSONObject() : new JSONObject(originalConfig);
            if (endpoint.isPreAssignable()) {
                template.remove("generation_config");
                if (!atomicBatchAllowed) {
                    template.remove("force_batch");
                    config.put("force_batch", false);
                } else if (!config.containsKey("force_batch")) {
                    config.put("force_batch", true);
                }
                key = "generate_config";
            }
            template.put(key, config);
        }
        configKey = template.get(key) instanceof JSONObject ? key : null;
        endpoint.prepareChunkBody(template);
    }

    public int chunkCount() {
        return count;
    }

    public int chunkSize(int index) {
        return split.mode() == SubBatchSpec.Mode.SIZE
                ? Math.min(split.value(), items.size() - index * split.value())
                : items.size() / count + (index < items.size() % count ? 1 : 0);
    }

    static int chunkCount(int total, SubBatchSpec split) {
        if (total == 0) {
            return 0;
        }
        return split.mode() == SubBatchSpec.Mode.SIZE
                ? 1 + (total - 1) / split.value() : Math.min(total, split.value());
    }

    /** Exact wire bytes without allocating repeated envelopes; JSON byte arrays have int lengths. */
    public long projectedBytes() {
        if (count == 0) {
            return 0;
        }
        long templateBytes = BatchBodyParser.serialize(template).length;
        long arrayBytes = JSON.writeTo(OutputStream.nullOutputStream(), items, JSONWriter.Feature.WriteNulls);
        // The product of two positive int lengths plus their framing fits in a signed long.
        return (templateBytes - 2) * count + arrayBytes + count - 1L;
    }

    /** Each chunk owns its array, envelope and mutable config; large read-only fields stay shared. */
    public List<JSONObject> chunks(List<BatchScheduleTarget> targets) {
        List<JSONObject> chunks = new ArrayList<>(count);
        int offset = 0;
        for (int i = 0; i < count; i++) {
            int size = chunkSize(i);
            JSONObject chunk = new JSONObject(template);
            chunk.put(endpoint.getRequestArrayField(), new JSONArray(items.subList(offset, offset + size)));
            if (configKey != null) {
                chunk.put(configKey, new JSONObject(template.getJSONObject(configKey)));
            }
            if (i < targets.size() && targets.get(i).getGrpcPort() != null) {
                JSONObject config = chunk.getJSONObject("generate_config");
                config.put("role_addrs", roleAddrs(targets.get(i)));
            }
            chunks.add(chunk);
            offset += size;
        }
        return chunks;
    }

    private static JSONArray roleAddrs(BatchScheduleTarget target) {
        return JSONArray.of(JSONObject.of("role", target.getRole().name(), "ip", target.getServerIp(),
                "http_port", target.getHttpPort(), "grpc_port", target.getGrpcPort()));
    }
}
