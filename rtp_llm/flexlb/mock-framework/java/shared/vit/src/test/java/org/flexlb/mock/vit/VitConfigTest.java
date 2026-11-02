package org.flexlb.mock.vit;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class VitConfigTest {
    private ObjectNode root() {
        var root = new ObjectMapper().createObjectNode().put("id", "vit-0").put("component_role", "vit");
        root.putObject("component").putObject("config").put("mock.compute.forward.expression", "batchSize + 1");
        return root;
    }

    @Test void nativeDefaultsAndEnvironmentNamesMatchRealVit() throws Exception {
        var root = root();
        var defaults = VitConfig.load(root);
        assertEquals(1, defaults.batchSize());
        assertEquals(0, defaults.effective().path("gpu_batch_wait_ms").intValue());
        assertEquals(2L * 1024 * 1024 * 1024, defaults.deviceBytes());
        assertEquals(defaults.deviceBytes(), defaults.hostBytes());
        assertEquals(256L * 1024 * 1024, defaults.hashBytes());
        assertEquals(defaults.effective(), VitConfig.fromEnvironment("vit-0", Map.of(
            "MOCK_COMPUTE_FORWARD_EXPRESSION", "batchSize + 1", "GPU_MAX_BATCH_SIZE", "99")).effective());
        var config = (ObjectNode) root.path("component").path("config");
        config.put("gpu_max_batch_size", 4).put("vit_concurrency", 4).put("vit_max_queue_size", 3).put("gpu_batch_wait_ms", 7)
            .put("mm_cache_gpu_max_bytes", 0).put("mm_cache_cpu_max_bytes", 4096)
            .put("mock.vit.patch.size", 16).put("mock.vit.hidden.size", 8);
        var parsed = VitConfig.load(root);
        assertEquals(7, parsed.queueSize()); assertEquals(4, parsed.batchSize());
        assertFalse(parsed.effective().has("mm_max_queue_size"));
        assertEquals(parsed.effective(), VitConfig.fromEnvironment("vit-0", Map.of(
            "MOCK_COMPUTE_FORWARD_EXPRESSION", "batchSize + 1", "VIT_GPU_MAX_BATCH_SIZE", "4",
            "VIT_CONCURRENCY", "4", "VIT_MAX_QUEUE_SIZE", "3", "VIT_GPU_BATCH_WAIT_MS", "7", "MM_CACHE_GPU_MAX_BYTES", "0",
            "MM_CACHE_CPU_MAX_BYTES", "4096", "MOCK_VIT_PATCH_SIZE", "16", "MOCK_VIT_HIDDEN_SIZE", "8")).effective());
        ((ObjectNode) root.path("component")).set("config", parsed.effective());
        assertEquals(parsed.effective(), VitConfig.load(root).effective());
    }

    @Test void unsupportedRealInputsAreIgnoredAndMockModelsAreStrict() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        var before = VitConfig.load(root).effective();
        config.putObject("model_config").put("hidden_size", 100);
        config.put("unsupported_future_option", false);
        assertEquals(before, VitConfig.load(root).effective());
        config.put("mock.vit.hiden.size", 100);
        assertThrows(IllegalArgumentException.class, () -> VitConfig.load(root));
        config.remove("mock.vit.hiden.size"); config.put("gpu_max_batch_size", 1.5);
        assertThrows(IllegalArgumentException.class, () -> VitConfig.load(root));
        assertThrows(Exception.class, () -> VitConfig.fromEnvironment("vit-0", Map.of(
            "MOCK_COMPUTE_FORWARD_EXPRESSION", "1", "VIT_GPU_MAX_BATCH_SIZE", "1.5")));
        assertThrows(IllegalArgumentException.class, () -> VitConfig.fromEnvironment("vit-0", Map.of(
            "MOCK_COMPUTE_FORWARD_EXPRESSION", "1", "MOCK_VIT_HIDEN_SIZE", "8")));
    }
}
