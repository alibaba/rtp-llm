package org.flexlb.mock.engine.vit;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies how ViT settings are read and validated.
 */
class VitConfigTest {
    @Test void nativeDefaultsAndEnvironmentNamesMatchRealVit() throws Exception {
        var root = root();
        var defaults = VitConfig.load(root);
        assertEquals(1, defaults.batchSize()); assertFalse(defaults.skipDownload());
        assertEquals(0, defaults.effective().path("gpu_batch_wait_ms").intValue());
        assertEquals(2L * 1024 * 1024 * 1024, defaults.deviceBytes());
        assertEquals(defaults.deviceBytes(), defaults.hostBytes());
        assertEquals(256L * 1024 * 1024, defaults.hashBytes());
        assertEquals(defaults.effective(), VitConfig.fromEnvironment("vit-0", env(14, 4096, Map.of("GPU_MAX_BATCH_SIZE", "99"))).effective());
        var config = (ObjectNode) root.path("component").path("config");
        config.put("mock.vit.skip.download", true).put("gpu_max_batch_size", 4).put("vit_concurrency", 4).put("vit_max_queue_size", 3).put("gpu_batch_wait_ms", 7)
            .put("mm_cache_gpu_max_bytes", 0).put("mm_cache_cpu_max_bytes", 4096);
        config.put("patch_size", 16).put("hidden_size", 8);
        var parsed = VitConfig.load(root);
        assertTrue(parsed.skipDownload()); assertEquals(4, parsed.concurrency());
        assertEquals(16, parsed.patch()); assertEquals(8, parsed.hidden());
        assertEquals(7, parsed.queueSize()); assertEquals(4, parsed.batchSize());
        assertEquals(parsed.effective(), VitConfig.fromEnvironment("vit-0", env(16, 8, Map.of(
            "VIT_GPU_MAX_BATCH_SIZE", "4",
            "VIT_CONCURRENCY", "4", "VIT_MAX_QUEUE_SIZE", "3", "VIT_GPU_BATCH_WAIT_MS", "7", "MM_CACHE_GPU_MAX_BYTES", "0",
            "MM_CACHE_CPU_MAX_BYTES", "4096", "MOCK_VIT_SKIP_DOWNLOAD", "true"))).effective());
        ((ObjectNode) root.path("component")).set("config", parsed.effective());
        assertEquals(parsed.effective(), VitConfig.load(root).effective());
    }

    @Test void unsupportedRealInputsAreIgnoredAndMockModelsAreStrict() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        var before = VitConfig.load(root).effective();
        config.put("unsupported_future_option", false);
        assertEquals(before, VitConfig.load(root).effective());
        config.put("mock.vit.patch.size", 16);
        assertThrows(IllegalArgumentException.class, () -> VitConfig.load(root));
        config.remove("mock.vit.patch.size"); config.put("gpu_max_batch_size", 1.5);
        assertThrows(IllegalArgumentException.class, () -> VitConfig.load(root));
        assertThrows(Exception.class, () -> VitConfig.fromEnvironment("vit-0", env(14, 4096, Map.of("VIT_GPU_MAX_BATCH_SIZE", "1.5"))));
        assertThrows(IllegalArgumentException.class, () -> VitConfig.fromEnvironment("vit-0", env(14, 4096, Map.of("MOCK_VIT_PATCH_SIZE", "8"))));
    }

    @Test void normalizesAsyncAdmissionAndDoesNotInventATimeoutCeiling() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("vit_concurrency", 0).put("vit_max_queue_size", -3).put("mm_timeout_ms", 7200000);
        var parsed = VitConfig.load(root);
        assertEquals(1, parsed.concurrency()); assertEquals(1, parsed.queueSize());
        assertEquals(0, parsed.effective().path("vit_max_queue_size").intValue());
        assertEquals(7200000, parsed.timeoutMs());
    }

    @Test void nativeValuesHaveNoInventedUpperBounds() throws Exception {
        var root = root();
        var settings = (ObjectNode) root.path("component").path("config");
        settings.put("gpu_max_batch_size", 2048).put("gpu_batch_wait_ms", 60001)
            .put("mm_image_max_file_size_kb", -1).put("mm_timeout_ms", 0);
        settings.put("patch_size", 2048).put("downsample_ratio", 64)
            .put("max_image_tokens", 1048577).put("hidden_size", 65537);
        var parsed = VitConfig.load(root);
        assertEquals(2048, parsed.batchSize()); assertEquals(60001, parsed.waitMs());
        assertEquals(-1, parsed.maxFileSizeKb()); assertEquals(65537, parsed.hidden());
        assertEquals(0, parsed.timeoutMs());
        var image = VitTest.input("http://example/" + "x".repeat(5000)).getMultimodalInputs(0).toBuilder()
            .setMmPreprocessConfig(org.flexlb.engine.grpc.EngineRpcService.MMPreprocessConfigPB.newBuilder()
                .setWidth(40000).setHeight(40000)).build();
        var inputs = org.flexlb.engine.grpc.EngineRpcService.MultimodalInputsPB.newBuilder();
        for (int i = 0; i < 257; i++) inputs.addMultimodalInputs(image);
        assertDoesNotThrow(() -> VitGeometry.validate(inputs.build(), true));
        assertTrue(VitGeometry.rows(40000, 40000, 14, 3, 0, 1024) <= 1024);
    }

    private ObjectNode root() {
        var json = new ObjectMapper();
        var root = json.createObjectNode().put("id", "vit-0").put("component_role", "vit");
        root.putObject("component").putObject("config").put("patch_size", 14).put("downsample_ratio", 3)
            .put("min_pixels", 295936).put("max_image_tokens", 1024).put("hidden_size", 4096)
            .put("mock.compute.forward.expression", "batchSize + 1");
        return root;
    }

    private Map<String, String> env(int patch, int hidden, Map<String, String> overrides) {
        var values = new HashMap<>(Map.of("PATCH_SIZE", Integer.toString(patch), "HIDDEN_SIZE", Integer.toString(hidden),
            "DOWNSAMPLE_RATIO", "3", "MIN_PIXELS", "295936", "MAX_IMAGE_TOKENS", "1024",
            "MOCK_COMPUTE_FORWARD_EXPRESSION", "batchSize + 1"));
        values.putAll(overrides);
        return values;
    }

}
