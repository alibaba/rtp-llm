package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RtpConfigTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    @TempDir Path directory;
    private RtpConfig load(ObjectNode value) throws Exception {
        var path = directory.resolve("component.json"); JSON.writeValue(path.toFile(), value); return RtpConfig.load(path);
    }
    private ObjectNode root() { return JSON.valueToTree(RtpNodeTest.configuration("prefill")); }

    @Test void normalizesRecognizedNativeEnvValuesAndKeepsAnImmutableRequestedConfiguration() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.remove("seq_size_per_block"); config.put("SEQ_SIZE_PER_BLOCK", "32").put("REUSE_CACHE", "on")
            .put("TP_SIZE", "2").put("DISABLE_DECODE_FIRST_MALLOC_DEVICE_REUSE", "true");
        var parsed = load(root);
        assertEquals(32, parsed.blockSize()); assertTrue(parsed.reuseCache()); assertEquals(2, parsed.tpSize());
        assertTrue(parsed.disableDecodeFirstReuse());
        assertEquals("2", parsed.business().path("tp_size").textValue());
        assertFalse(parsed.business().has("TP_SIZE"));
        ((ObjectNode) parsed.business()).removeAll(); assertFalse(parsed.business().isEmpty());
    }

    @Test void nativePhysicalBlockCountExcludesNullBlockAndReserveUsesFloor() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("test_block_num", 100).remove("reserve_block_ratio");
        assertEquals(99, load(root).deviceBlocks()); assertEquals(4, load(root).reserveBlocks());
        config.put("reserve_block_ratio", 0); assertEquals(0, load(root).reserveBlocks());
        config.remove("test_block_num"); config.put("TEST_BLOCK_NUM", "25").put("reserve_block_ratio", 5);
        assertEquals(24, load(root).deviceBlocks()); assertEquals(1, load(root).reserveBlocks());
        config.put("TEST_BLOCK_NUM", "1"); assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void rejectsAmbiguousAliasesAndWrongNativeTypes() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("SEQ_SIZE_PER_BLOCK", "16");
        assertThrows(IllegalArgumentException.class, () -> load(root));
        config.remove("SEQ_SIZE_PER_BLOCK"); config.put("seq_size_per_block", 0);
        assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("seq_size_per_block", 16).put("REUSE_CACHE", 1);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void rejectsUnsupportedDpAndRemovedMockCapacitySettings() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("dp_size", 2); assertThrows(IllegalArgumentException.class, () -> load(root));
        config.remove("dp_size"); config.putObject("capacity").put("device_blocks", 8);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void rejectsOldComputeAndInvalidOrMisplacedModels() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        ((ObjectNode) root.path("environment")).putObject("compute");
        assertThrows(IllegalArgumentException.class, () -> load(root)); ((ObjectNode) root.path("environment")).remove("compute");
        var compute = (ObjectNode) config.path("compute");
        compute.putObject("eos").put("probability", .1);
        assertThrows(Exception.class, () -> load(root));
        ((ObjectNode) compute.path("eos")).put("seed", 42);
        assertEquals(.1, load(root).compute().configuration().eos().probability());
        ((ObjectNode) compute.path("eos")).put("extra", 1); assertThrows(Exception.class, () -> load(root));
        compute.remove("eos"); compute.putObject("decode").put("tokens_per_step", 2.6);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void componentLoaderOwnsDuplicateFieldsAndEnvelopeValidation() throws Exception {
        var path = directory.resolve("duplicate.json");
        Files.writeString(path, JSON.writeValueAsString(root()).replace("\"compute\":{", "\"compute\":{},\"compute\":{"));
        assertThrows(java.io.IOException.class, () -> RtpConfig.load(path));
    }

    @Test void requiresExplicitGeometryAndMatchingRoleFormula() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.remove("seq_size_per_block"); assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("seq_size_per_block", 16); config.remove("compute");
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }
    @Test void fractionalDecodeModelLoadsAndItsEffectiveConfigurationCanBeReadBack() throws Exception {
        var root = root(); root.put("component_role", "decode");
        var config = (ObjectNode) root.path("component").path("config");
        ((ObjectNode) config.path("compute")).putObject("decode").put("tokens_per_step", 2.6);
        var model = load(root);
        assertEquals(2.6, model.compute().configuration().decode().tokens_per_step());
        config.set("compute", model.effective().get("compute"));
        assertEquals(2.6, load(root).compute().configuration().decode().tokens_per_step());
    }
    @Test void nativeEvictionOrderingUsesOnlyTreeAndRejectsUnimplementedTiers() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        assertEquals("lru", load(root).effective().path("device_eviction_policy").asText());
        for (String policy : java.util.List.of("lru", "lfu", "fifo")) {
            config.put("device_eviction_policy", policy);
            assertEquals(policy, load(root).evictionPolicy().name().toLowerCase(java.util.Locale.ROOT));
        }
        config.put("device_eviction_policy", "tree"); assertThrows(IllegalArgumentException.class, () -> load(root));
        config.remove("device_eviction_policy"); config.put("DEVICE_EVICTION_POLICY", "LFU");
        assertEquals("lfu", load(root).effective().path("device_eviction_policy").asText());
        for (String tier : java.util.List.of("enable_memory_cache", "enable_disk_cache", "enable_remote_cache")) {
            config.put(tier, true); assertThrows(IllegalArgumentException.class, () -> load(root)); config.remove(tier);
        }
    }

    @Test void bindsParallelismToTheFormulaWithoutImplicitScaling() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("cp_size", 4).put("cp_enabled", true).put("tp_size", 4).put("dp_size", 1);
        config.putObject("compute").putObject("forward").put("expression", "cpSize + tpSize + dpSize + cpEnabled");
        var parsed = load(root);
        var request = new org.flexlb.mock.compute.ForwardBatch.Request("r", 2, 0, 1, 0, 1, 50, java.util.List.of());
        var batch = new org.flexlb.mock.compute.ForwardBatch("b", java.util.List.of(
            new org.flexlb.mock.compute.ForwardBatch.Partition(0, java.util.List.of(request))));
        assertEquals(java.time.Duration.ofMillis(10), parsed.compute().plan(parsed.role(), parsed.id(), batch).planned());
        assertEquals(4, parsed.effective().path("cp_size").intValue()); assertTrue(parsed.effective().path("cp_enabled").booleanValue());
        config.putObject("compute").putObject("forward").put("expression", "2");
        assertEquals(java.time.Duration.ofMillis(2), load(root).compute().plan(parsed.role(), parsed.id(), batch).planned());
    }

    @Test void deferredSettingsAreValidatedAndNeverReportedAsEffective() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("max_batch_tokens_without_cache", 50000).put("max_inited_kv_cache_streams", 128)
            .put("prefill_retry_times", 3).put("block_tree_device_evict_low_watermark_ratio", .82)
            .put("block_tree_device_evict_high_watermark_ratio", .9);
        var effective = load(root).effective();
        assertEquals(50000, effective.path("max_batch_tokens_without_cache").intValue());
        assertEquals(128, effective.path("max_inited_kv_cache_streams").intValue());
        assertFalse(effective.path("pending_config").has("max_inited_kv_cache_streams"));
        assertEquals(3, effective.path("pending_config").path("prefill_retry_times").intValue());
        assertEquals(0, effective.path("prefill_retry_times").intValue());
        assertFalse(effective.has("config"));
        config.put("max_inited_kv_cache_streams", -1); assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("max_inited_kv_cache_streams", 128).put("block_tree_device_evict_low_watermark_ratio", .99);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void rejectsTyposRemovedKnobsAndSettingsCopiedToTheWrongRole() throws Exception {
        for (String key : java.util.List.of("max_context_batch_size", "max_generate_batch_szie", "decode_retry_times", "load_cache_timeout_ms")) {
            var root = root(); ((ObjectNode) root.path("component").path("config")).put(key, 10);
            assertTrue(assertThrows(IllegalArgumentException.class, () -> load(root)).getMessage().contains(key));
        }
        var root = root(); root.put("component_role", "decode");
        ((ObjectNode) root.path("component").path("config")).put("prefill_retry_timeout_ms", 100);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }
    @Test void enablesHostAndReportsStorageSettingsAsEffective() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("enable_memory_cache", true).put("memory_cache_blocks", 32).put("memory_eviction_policy", "fifo");
        var parsed = load(root); var effective = parsed.effective();
        assertEquals(32, parsed.hostBlocks()); assertEquals("fifo", effective.path("memory_eviction_policy").asText());
        assertTrue(effective.path("enable_memory_cache").booleanValue());
        assertFalse(effective.path("pending_config").has("memory_cache_blocks"));
        assertFalse(effective.path("pending_config").has("memory_eviction_policy"));
        assertFalse(effective.path("pending_config").has("block_tree_device_evict_low_watermark_ratio"));
        config.put("memory_cache_blocks", 0); assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("memory_cache_blocks", 32).put("block_tree_memory_evict_low_watermark_ratio", .95);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

}
