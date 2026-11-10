package org.flexlb.mock.engine.vit;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.util.*;
import org.flexlb.mock.engine.forward.ComputeModel;
import org.flexlb.mock.protocol.ConfigInput;

/**
 * Reads and validates the ViT model and Engine settings.
 */
public record VitConfig(String id, int patch, int merge, int hidden, int minPixels, int maxTokens, int batchSize,
                           int waitMs, int concurrency, int queueSize, boolean skipDownload, int timeoutMs, int maxFileSizeKb, Map<String,String> downloadHeaders, long deviceBytes, long hostBytes, long hashBytes,
                           ComputeModel compute, ObjectNode effective) {
    private static final Set<String> NATIVE = Set.of("patch_size", "downsample_ratio", "hidden_size", "min_pixels", "max_image_tokens", "mm_timeout_ms", "mm_image_max_file_size_kb", "download_headers", "gpu_max_batch_size", "gpu_batch_wait_ms", "vit_concurrency", "vit_max_queue_size",
        "mm_cache_gpu_max_bytes", "mm_cache_cpu_max_bytes", "mm_hash_key_cache_max_bytes");

    public static VitConfig fromEnvironment(String id, Map<String, String> env) throws Exception {
        var values = new HashMap<>(env);
        // Real ViT ENV names differ from its configuration attributes.
        for (String name : List.of("GPU_MAX_BATCH_SIZE", "GPU_BATCH_WAIT_MS")) {
            values.remove(name);
            if (env.containsKey("VIT_" + name)) values.put(name, env.get("VIT_" + name));
        }
        var root = new ObjectMapper().createObjectNode().put("id", id).put("component_role", "vit");
        root.putObject("component").set("config", ConfigInput.environment(values, NATIVE, mockNames()));
        return load(root);
    }

    public static VitConfig load(Path path) throws Exception {
        return load(ConfigInput.read(path));
    }

    public static VitConfig load(JsonNode root) throws Exception {
        if (!root.path("component_role").asText().equals("vit") || !root.path("id").isTextual())
            throw new IllegalArgumentException("V instance requires role vit and an id");
        if (root.path("component").has("engine"))
            throw new IllegalArgumentException("V uses a shared implementation; component.engine selects only the P/D backend");
        var config = root.path("component").path("config");
        if (!config.isObject()) throw new IllegalArgumentException("V config must be an object");
        config = ConfigInput.normalize(config, NATIVE, mockNames());
        var compute = ComputeModel.fromSettings(config, Map.of());
        var effective = new ObjectMapper().createObjectNode().put("id", root.path("id").asText()).put("role", "vit");
        int batchSize = integer(config,"gpu_max_batch_size",1,1,Integer.MAX_VALUE);
        int concurrency = Math.max(1, signedInteger(config,"vit_concurrency",64));
        int pending = Math.max(0, signedInteger(config,"vit_max_queue_size",64));
        var result = new VitConfig(root.path("id").asText(), integer(config,"patch_size",null,1,Integer.MAX_VALUE), integer(config,"downsample_ratio",null,1,Integer.MAX_VALUE), integer(config,"hidden_size",null,1,Integer.MAX_VALUE),
            integer(config,"min_pixels",null,0,Integer.MAX_VALUE), integer(config,"max_image_tokens",null,4,Integer.MAX_VALUE),
            batchSize, integer(config,"gpu_batch_wait_ms",10,0,Integer.MAX_VALUE), concurrency, Math.addExact(concurrency,pending),
            bool(config,"mock.vit.skip.download",false), signedInteger(config,"mm_timeout_ms",120000),
            signedInteger(config,"mm_image_max_file_size_kb",102400), headers(config),
            number(config,"mm_cache_gpu_max_bytes",2147483648L), number(config,"mm_cache_cpu_max_bytes",2147483648L), number(config,"mm_hash_key_cache_max_bytes",268435456L),compute,effective);
        effective.put("patch_size", result.patch).put("downsample_ratio", result.merge)
            .put("hidden_size", result.hidden).put("min_pixels", result.minPixels).put("max_image_tokens", result.maxTokens)
            .put("gpu_max_batch_size",result.batchSize).put("gpu_batch_wait_ms",result.batchSize == 1 ? 0 : result.waitMs)
            .put("vit_concurrency",concurrency).put("vit_max_queue_size",pending)
            .put("mock.vit.skip.download",result.skipDownload).put("mm_timeout_ms",result.timeoutMs)
            .put("mm_image_max_file_size_kb",result.maxFileSizeKb).put("download_headers",config.path("download_headers").asText(""))
            .put("mm_cache_gpu_max_bytes",result.deviceBytes).put("mm_cache_cpu_max_bytes",result.hostBytes).put("mm_hash_key_cache_max_bytes",result.hashBytes);
        compute.effectiveSettings().fields().forEachRemaining(entry -> {
            if (mockNames().contains(entry.getKey())) effective.set(entry.getKey(), entry.getValue());
        });
        return result;
    }

    private static Set<String> mockNames() {
        var names = new HashSet<>(ComputeModel.settingNames());
        names.removeIf(name -> !name.startsWith("mock.compute.forward."));
        names.add("mock.vit.skip.download");
        return Set.copyOf(names);
    }

    private static boolean bool(JsonNode config,String key,boolean fallback) {
        var value=config.get(key); if(value==null) return fallback;
        if(value.isBoolean()) return value.booleanValue();
        if(value.isTextual() && Set.of("true","false","1","0").contains(value.asText()))
            return value.asText().equals("true") || value.asText().equals("1");
        throw new IllegalArgumentException("Invalid V " + key);
    }

    private static Map<String,String> headers(JsonNode config) throws Exception {
        var value=config.get("download_headers");
        if(value==null || value.isTextual() && value.asText().isEmpty()) return Map.of();
        if(!value.isTextual()) throw new IllegalArgumentException("download_headers must be a JSON string");
        var object=new ObjectMapper().readTree(value.asText());
        if(object==null || !object.isObject()) throw new IllegalArgumentException("download_headers must contain an object");
        var headers=new HashMap<String,String>();
        object.fields().forEachRemaining(entry -> {
            if(!entry.getValue().isTextual()) throw new IllegalArgumentException("download_headers values must be strings");
            headers.put(entry.getKey(),entry.getValue().asText());
        });
        return Map.copyOf(headers);
    }

    private static int signedInteger(JsonNode config, String key, int fallback) {
        var value = config.get(key);
        if (value == null) return fallback;
        if (value.isIntegralNumber() || value.isTextual() && value.asText().matches("[+-]?\\d+")) {
            try { return Integer.parseInt(value.asText()); }
            catch (NumberFormatException ignored) { }
        }
        throw new IllegalArgumentException("Invalid V " + key);
    }

    private static int integer(JsonNode node,String key,Integer fallback,int low,int high) {
        long value = number(node,key,fallback == null ? null : fallback.longValue()); if (value < low || value > high) throw new IllegalArgumentException("Invalid V " + key); return (int)value;
    }

    private static long number(JsonNode node,String key,Long fallback) {
        var value = node.get(key); if (value == null) { if (fallback == null) throw new IllegalArgumentException("Missing model setting: " + key); return fallback; }
        if (value.isIntegralNumber() || value.isTextual() && value.textValue().matches("[+]?\\d+")) {
            try {
                long result = Long.parseLong(value.asText());
                if (result >= 0) return result;
            } catch (NumberFormatException ignored) { }
        }
        throw new IllegalArgumentException("Invalid V " + key);
    }
}
