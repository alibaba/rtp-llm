package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;

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
        template.remove("role_addrs");
        String key = body.containsKey("generate_config") ? "generate_config" : "generation_config";
        JSONObject originalConfig = body.getJSONObject(key);
        if (originalConfig != null || endpoint.isPreAssignable()) {
            JSONObject config = originalConfig == null ? new JSONObject() : new JSONObject(originalConfig);
            config.remove("role_addrs");
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
        if (count == 0) {
            return 0;
        }
        return split.mode() == SubBatchSpec.Mode.SIZE
                ? Math.min(split.value(), items.size() - index * split.value())
                : items.size() / count + (index < items.size() % count ? 1 : 0);
    }

    static int chunkCount(int total, SubBatchSpec split) {
        if (total < 0 || split == null || split.value() < 1) {
            throw new IllegalArgumentException("total must be non-negative and subBatch value must be positive");
        }
        if (total == 0) {
            return 0;
        }
        return split.mode() == SubBatchSpec.Mode.SIZE
                ? 1 + (total - 1) / split.value() : Math.min(total, split.value());
    }

    /** Exact wire bytes without allocating repeated envelopes; JSON byte arrays have int lengths. */
    public long projectedBytes(List<BatchScheduleTarget> targets) {
        if (count == 0) {
            return 0;
        }
        long templateBytes = BatchBodyParser.serialize(template).length;
        long arrayBytes = BatchBodyParser.serialize(items).length;
        // The product of two positive int lengths plus their framing fits in a signed long.
        long bytes = (templateBytes - 2) * count + arrayBytes + count - 1L;
        for (int i = 0; i < Math.min(count, targets.size()); i++) {
            if (isPreAssignable(targets.get(i))) {
                long extra = 14L + BatchBodyParser.serialize(roleAddrs(targets.get(i))).length;
                bytes = bytes > Long.MAX_VALUE - extra ? Long.MAX_VALUE : bytes + extra;
            }
        }
        return bytes;
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
            if (i < targets.size() && isPreAssignable(targets.get(i))) {
                JSONObject config = chunk.getJSONObject("generate_config");
                if (config == null) {
                    config = new JSONObject();
                    chunk.put("generate_config", config);
                }
                config.put("role_addrs", roleAddrs(targets.get(i)));
            }
            chunks.add(chunk);
            offset += size;
        }
        return chunks;
    }

    static boolean isPreAssignable(BatchScheduleTarget target) {
        return target.getGrpcPort() != null && target.getRole() != null;
    }

    private static JSONArray roleAddrs(BatchScheduleTarget target) {
        return JSONArray.of(JSONObject.of("role", target.getRole().name(), "ip", target.getServerIp(),
                "http_port", target.getHttpPort(), "grpc_port", target.getGrpcPort()));
    }
}
