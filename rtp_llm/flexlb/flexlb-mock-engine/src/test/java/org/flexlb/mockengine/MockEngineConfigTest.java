package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class MockEngineConfigTest {
    @TempDir Path directory;

    @Test void finalizedBehaviorOverridesAllSimulationDefaults() throws Exception {
        ObjectNode document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        ObjectNode role = (ObjectNode)document.path("roles").path("prefill");
        ((ObjectNode)role.path("scheduler")).put("max_generate_batch_size", 3).put("max_batch_tokens_size", 100);
        ((ObjectNode)role.path("kv_cache")).put("seq_size_per_block", 16).put("reuse_cache", false);
        ((ObjectNode)role.path("capacity")).put("seq_size_per_block", 16);
        var model = MockEngineConfig.parse(document).configure("prefill", MockEngineTestSupport.performanceModel(directory, "1"));
        assertEquals(16, model.blockSize());
        assertFalse(model.prefillReuseCache);
        assertEquals(3, model.prefillBatchPolicy().maxRequests());
        assertEquals(100, model.prefillBatchPolicy().maxTokens());
        assertFalse(model.prefillBatchPolicy().faultLimitsEnabled());
    }

    @Test void missingInvalidAndUnsupportedBehaviorNeverFallsBack() {
        for (String field : new String[]{"scheduler", "capacity", "kv_cache", "model"}) {
            ObjectNode document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
            ((ObjectNode)document.path("roles").path("prefill")).remove(field);
            assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document), field);
        }
        ObjectNode document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        ObjectNode role = (ObjectNode)document.path("roles").path("prefill");
        ((ObjectNode)role.path("capacity")).put("layout", "multi_pool");
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
        ((ObjectNode)role.path("capacity")).put("layout", "single_pool");
        ((ObjectNode)role.path("scheduler")).put("type", "ratio");
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
        ((ObjectNode)role.path("scheduler")).put("type", "fifo").put("max_batch_tokens_size", 0);
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
    }

    @Test void vitBudgetMustBeExplicitAndEveryDimensionValidated() {
        ObjectNode document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        var vit = (ObjectNode)document.path("roles").path("vit");
        vit.remove("work_budget");
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
        var budget = vit.putObject("work_budget");
        budget.put("input_patches", 8).put("output_tokens", 0).put("estimated_workspace_bytes", 0)
                .put("max_attention_segment", 0).put("attention_work", 0);
        assertEquals(new MockVitWorkEstimate(8, 0, 0, 0, 0), MockEngineConfig.vitWorkBudget(vit));
        assertDoesNotThrow(() -> MockEngineConfig.parse(document));
        budget.put("attention_work", -1);
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
        budget.put("attention_work", 0).remove("max_attention_segment");
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
    }

    @Test void evictionPoliciesCannotSilentlyFallBackToLru() {
        ObjectNode document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        var kv = (ObjectNode)document.path("roles").path("prefill").path("kv_cache");
        kv.put("device_eviction_policy", "fifo");
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
        kv.put("device_eviction_policy", "lru").put("memory_eviction_policy", "fifo");
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
    }

    @Test void serialVitRequiresRealResolvedLimitsAndImageSizeUsesKilobytes() {
        ObjectNode document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        var vit = (ObjectNode)document.path("roles").path("vit");
        vit.put("mm_image_max_file_size_kb", 123);
        assertEquals(123 * 1024, MockEngineConfig.vitImageBytes(vit));
        vit.put("gpu_max_batch_images", 200);
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
        vit.put("gpu_max_batch_images", Long.MAX_VALUE).put("gpu_batch_wait_ms", 10);
        assertThrows(IllegalArgumentException.class, () -> MockEngineConfig.parse(document));
    }

    @Test void calibrationCannotChangeAdmissionOrCapacity() throws Exception {
        Path file = directory.resolve("calibration.json");
        Files.writeString(file, "{\"prefill\":{\"fifo\":{}}}");
        assertThrows(IllegalArgumentException.class, () -> MockPerformanceModel.loadCalibration(file.toString()));
        Files.writeString(file, "{\"block_size\":64}");
        assertThrows(IllegalArgumentException.class, () -> MockPerformanceModel.loadCalibration(file.toString()));
    }

    @Test void snapshotsCannotMutateRunningConfiguration() {
        var config = MockEngineConfigFixture.load();
        ObjectNode copy = (ObjectNode)config.snapshot();
        ((ObjectNode)copy.path("roles").path("decode").path("capacity")).put("device_blocks", 1);
        assertEquals(27686, config.blocks("decode"));
    }

    @Test void realSerialVitImageLimitSurvivesImport() {
        ObjectNode document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        ((ObjectNode)document.path("roles").path("vit")).put("gpu_max_batch_size", 1)
                .put("gpu_max_batch_images", Long.MAX_VALUE).put("gpu_batch_wait_ms", 0);
        var config = MockEngineConfig.parse(document);
        assertEquals(Long.MAX_VALUE, config.role("vit").path("gpu_max_batch_images").longValue());
    }
}
