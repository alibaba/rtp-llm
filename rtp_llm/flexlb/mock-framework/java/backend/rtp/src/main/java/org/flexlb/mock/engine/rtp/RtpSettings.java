package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.*;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.ComputeModel;
import org.flexlb.mock.runtime.EngineConfig;

/** Only implemented native settings enter the effective configuration. */
final class RtpSettings {
    private static final Set<String> COMMON = Set.of(
        "checkpoint_path", "test_block_num", "reserve_block_ratio", "device_eviction_policy", "seq_size_per_block", "reuse_cache",
        "enable_device_cache", "disable_decode_first_malloc_device_reuse", "max_seq_len", "max_batch_tokens_size",
        "kv_cache_event_publisher_type", "kv_cache_event_manager_endpoint", "kv_cache_event_instance_id",
        "kv_cache_event_instance_group", "kv_cache_event_host_ip_port", "kv_cache_event_snapshot_interval_ms",
        "kvcm_enable_vipserver", "kvcm_vipserver_domain", "kvcm_server_address",
        "enable_remote_cache", "model_name", "concurrency_limit", "tp_size", "dp_size", "cp_rotate_method", "enable_memory_cache", "memory_eviction_policy",
        "block_tree_memory_evict_low_watermark_ratio", "block_tree_memory_evict_high_watermark_ratio",
        "block_tree_device_evict_low_watermark_ratio", "block_tree_device_evict_high_watermark_ratio");

    static Set<String> nativeNames(Role role) {
        var names = new HashSet<>(COMMON);
        if (role == Role.prefill) names.addAll(Set.of("max_batch_tokens_without_cache", "max_inited_kv_cache_streams"));
        return Set.copyOf(names);
    }
    static Set<String> environmentNames(Role role) {
        var names = new HashSet<>(nativeNames(role));
        // These are real internal attributes, without public server ENV bindings.
        names.removeAll(Set.of("device_eviction_policy", "memory_eviction_policy", "disable_decode_first_malloc_device_reuse"));
        names.removeAll(Set.of("kvcm_enable_vipserver", "kvcm_vipserver_domain", "kvcm_server_address"));
        return Set.copyOf(names);
    }
    static ObjectNode environment(Map<String, String> env, Role role) {
        var result = EngineConfig.environment(env, environmentNames(role), mockNames());
        // Real ENV bindings differ from their native config attribute names.
        for (String suffix : List.of("enable_vipserver", "vipserver_domain", "server_address")) {
            String value = env.get("RECO_" + suffix.toUpperCase(Locale.ROOT));
            if (value != null) result.put("kvcm_" + suffix, value);
        }
        return result;
    }
    static Set<String> mockNames() {
        var names = new HashSet<>(ComputeModel.settingNames());
        names.add("mock.cache.memory.blocks");
        return Set.copyOf(names);
    }
    static ObjectNode normalize(JsonNode config, Role role) {
        return EngineConfig.normalize(config, nativeNames(role), mockNames());
    }
    private RtpSettings() {}
}
