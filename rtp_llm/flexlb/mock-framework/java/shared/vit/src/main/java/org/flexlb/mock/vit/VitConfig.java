package org.flexlb.mock.vit;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.util.*;
import org.flexlb.mock.compute.ComputeModel;

public record VitConfig(String id, int patch, int merge, int hidden, int batchSize, int batchImages,
                           int waitMs, int queueSize, long deviceBytes, long hostBytes, long hashBytes,
                           ComputeModel compute, ObjectNode effective) {
    public static VitConfig load(Path path) throws Exception {
        var json = new ObjectMapper(); var root = json.readTree(path.toFile());
        if (!root.path("component_role").asText().equals("vit") || !root.path("id").isTextual())
            throw new IllegalArgumentException("V instance requires role vit and an id");
        if (root.path("component").has("engine"))
            throw new IllegalArgumentException("V uses a shared implementation; component.engine selects only the P/D backend");
        var config = root.path("component").path("config");
        if (!config.isObject()) throw new IllegalArgumentException("V config must be an object");
        var names = Set.of("patch_size", "spatial_merge_size", "hidden_size", "gpu_max_batch_size", "gpu_max_batch_images", "gpu_batch_wait_ms", "mm_max_queue_size",
            "mm_cache_gpu_max_bytes", "mm_cache_cpu_max_bytes", "mm_hash_key_cache_max_bytes", "compute");
        config.fieldNames().forEachRemaining(key -> { if (!names.contains(key)) throw new IllegalArgumentException("Unknown V config: " + key); });
        var computeConfig = config.path("compute");
        computeConfig.fieldNames().forEachRemaining(key -> { if (!key.equals("forward")) throw new IllegalArgumentException("V only has forward compute"); });
        var compute = ComputeModel.fromJson(computeConfig);
        var effective = ((ObjectNode) config).deepCopy(); effective.put("id", root.path("id").asText()).put("role", "vit");
        var result = new VitConfig(root.path("id").asText(), integer(config,"patch_size",14,1,1024), integer(config,"spatial_merge_size",2,1,32), integer(config,"hidden_size",4096,1,65536),
            integer(config,"gpu_max_batch_size",8,1,1024), integer(config,"gpu_max_batch_images",200,1,4096), integer(config,"gpu_batch_wait_ms",10,0,60000), integer(config,"mm_max_queue_size",1024,1,65536),
            number(config,"mm_cache_gpu_max_bytes",0), number(config,"mm_cache_cpu_max_bytes",0), number(config,"mm_hash_key_cache_max_bytes",16777216),compute,effective);
        effective.put("patch_size",result.patch).put("spatial_merge_size",result.merge).put("hidden_size",result.hidden)
            .put("gpu_max_batch_size",result.batchSize).put("gpu_max_batch_images",result.batchImages).put("gpu_batch_wait_ms",result.batchSize == 1 ? 0 : result.waitMs)
            .put("mm_max_queue_size",result.queueSize).put("mm_cache_gpu_max_bytes",result.deviceBytes).put("mm_cache_cpu_max_bytes",result.hostBytes).put("mm_hash_key_cache_max_bytes",result.hashBytes);
        effective.set("compute",compute.effectiveConfiguration());
        return result;
    }
    private static int integer(JsonNode node,String key,int fallback,int low,int high) {
        long value = number(node,key,fallback); if (value < low || value > high) throw new IllegalArgumentException("Invalid V " + key); return (int)value;
    }
    private static long number(JsonNode node,String key,long fallback) {
        var value = node.get(key); if (value == null) return fallback;
        if (!value.isIntegralNumber() || !value.canConvertToLong() || value.longValue() < 0) throw new IllegalArgumentException("Invalid V " + key);
        return value.longValue();
    }
}
