package org.flexlb.mock.engine.scheduler;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.*;
import org.flexlb.mock.engine.config.RtpConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class RtpConfigTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    @TempDir Path directory;
    private RtpConfig load(ObjectNode value) throws Exception {
        var path = directory.resolve("component.json"); JSON.writeValue(path.toFile(), value); return RtpConfig.load(path);
    }
    private ObjectNode root() { return JSON.valueToTree(RtpNodeTest.configuration("prefill")); }

    @Test void schedulerModeSelectsPolicyAndRejectsUnknownValues() throws Exception {
        var root = root();
        var values = (ObjectNode) root.path("component").path("config");
        var defaults = load(root);
        try (var cache = new org.flexlb.mock.engine.cache.KvCache(16, 8, 0);
             var scheduler = AbstractRtpScheduler.factory(defaults).create(new Object(), cache, new org.flexlb.mock.engine.scheduler.TaskHistory())) {
            assertInstanceOf(RtpFifoScheduler.class, scheduler);
            assertEquals("fifo", defaults.effective().path("mock.scheduler.mode").asText());
        }
        values.put("mock.scheduler.mode", "group");
        var grouped = load(root);
        try (var cache = new org.flexlb.mock.engine.cache.KvCache(16, 8, 0);
             var scheduler = AbstractRtpScheduler.factory(grouped).create(new Object(), cache, new org.flexlb.mock.engine.scheduler.TaskHistory())) {
            assertInstanceOf(RtpPrefillGroupScheduler.class, scheduler);
            assertEquals("group", grouped.effective().path("mock.scheduler.mode").asText());
        }
        values.put("mock.scheduler.mode", "unknown");
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

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
        config.remove("SEQ_SIZE_PER_BLOCK"); config.put("seq_size_per_block", -1);
        assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("seq_size_per_block", 16).put("REUSE_CACHE", 1);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void unsupportedRealSettingsAreIgnoredWithoutValidatingTheirValues() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        var before = load(root).effective();
        for (String key : java.util.List.of("dp_rank", "enable_disk_cache", "memory_cache_size_mb",
                "prefill_retry_times", "load_cache_timeout_ms", "some_future_real_setting")) config.putObject(key).put("anything", true);
        assertEquals(before, load(root).effective());
        assertFalse(load(root).business().has("memory_cache_size_mb"));
    }

    @Test void rejectsOldComputeAndInvalidOrMisplacedModels() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        ((ObjectNode) root.path("environment")).putObject("compute");
        assertThrows(IllegalArgumentException.class, () -> load(root)); ((ObjectNode) root.path("environment")).remove("compute");
        var compute = (ObjectNode) config.path("mock.compute");
        compute.putObject("eos").put("probability", .1);
        assertThrows(Exception.class, () -> load(root));
        ((ObjectNode) compute.path("eos")).put("seed", 42);
        assertEquals(.1, load(root).compute().configuration().eos().probability());
        ((ObjectNode) compute.path("eos")).put("extra", 1); assertThrows(Exception.class, () -> load(root));
        compute.remove("eos"); compute.putObject("decode").put("tokens.per.step", 2.6);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void componentLoaderOwnsDuplicateFieldsAndEnvelopeValidation() throws Exception {
        var path = directory.resolve("duplicate.json");
        Files.writeString(path, JSON.writeValueAsString(root()).replace("\"mock.compute\":{", "\"mock.compute\":{},\"mock.compute\":{"));
        assertThrows(java.io.IOException.class, () -> RtpConfig.load(path));
    }

    @Test void defaultsBlockGeometryButRequiresMatchingRoleFormula() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.remove("seq_size_per_block"); assertEquals(64, load(root).blockSize());
        config.put("seq_size_per_block", 16); config.remove("mock.compute");
        assertThrows(Exception.class, () -> load(root));
    }
    @Test void fractionalDecodeModelLoadsAndItsEffectiveConfigurationCanBeReadBack() throws Exception {
        var root = root(); root.put("component_role", "decode");
        var config = (ObjectNode) root.path("component").path("config");
        ((ObjectNode) config.path("mock.compute")).putObject("decode").put("tokens.per.step", 2.6);
        var model = load(root);
        assertEquals(2.6, model.compute().configuration().decode().tokens_per_step());
        ((ObjectNode) root.path("component")).set("config", model.effective());
        assertEquals(2.6, load(root).compute().configuration().decode().tokens_per_step());
    }
    @Test void acceptsOnlyLocalLru() throws Exception {
        var root = root();
        var config = (ObjectNode) root.path("component").path("config");
        assertFalse(load(root).effective().has("device_eviction_policy"));
        config.put("device_eviction_policy", "lfu");
        assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("device_eviction_policy", "LRU");
        assertThrows(IllegalArgumentException.class, () -> load(root));
        config.remove("device_eviction_policy"); config.put("enable_memory_cache", true);
        assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("enable_memory_cache", false);
        assertFalse(load(root).effective().has("device_eviction_policy"));
    }

    @Test void bindsParallelismToTheFormulaWithoutImplicitScaling() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("cp_rotate_method", "ALL_GATHER").put("tp_size", 4).put("dp_size", 1);
        config.putObject("mock.compute").putObject("forward").put("expression", "cpSize + tpSize + dpSize + cpEnabled");
        var parsed = load(root);
        var request = new org.flexlb.mock.engine.forward.ForwardBatch.Request("r", 2, 0, 1, 0, 1, 50, java.util.List.of());
        var batch = new org.flexlb.mock.engine.forward.ForwardBatch("b", java.util.List.of(
            new org.flexlb.mock.engine.forward.ForwardBatch.Partition(0, java.util.List.of(request))));
        assertEquals(java.time.Duration.ofMillis(10), parsed.compute().plan(parsed.role(), parsed.id(), batch).planned());
        assertEquals(4, parsed.cpSize()); assertTrue(parsed.cpEnabled());
        assertEquals("ALL_GATHER", parsed.effective().path("cp_rotate_method").asText());
        config.putObject("mock.compute").putObject("forward").put("expression", "2");
        assertEquals(java.time.Duration.ofMillis(2), load(root).compute().plan(parsed.role(), parsed.id(), batch).planned());
    }

    @Test void nativeParallelismSelectsFormulaConstantsWithoutSpawningRanks() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("tp_size", 4).put("dp_size", 3);
        assertEquals(3, load(root).dpSize());
        assertFalse(load(root).cpEnabled()); assertEquals(1, load(root).cpSize());
        for (String method : java.util.List.of("ALL_GATHER", "ALL_GATHER_WITH_OVERLAP", "alltoall")) {
            config.put("cp_rotate_method", method);
            assertTrue(load(root).cpEnabled()); assertEquals(4, load(root).cpSize());
        }
        // The special PREFILL_CP path is distinct from general CP in real PrefillCPConfig.
        config.put("cp_rotate_method", "PREFILL_CP").put("prefill_cp_size", 2);
        assertFalse(load(root).cpEnabled()); assertEquals(1, load(root).cpSize());
        assertFalse(load(root).effective().has("prefill_cp_size"));
        config.put("cp_rotate_method", "misspelled");
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void implementedSettingsValidateWhileUnsupportedSettingsDoNotBecomeEffective() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("max_batch_tokens_without_cache", 50000).put("max_inited_kv_cache_streams", 128)
            .put("prefill_retry_times", 3).put("block_tree_device_evict_low_watermark_ratio", .82)
            .put("block_tree_device_evict_high_watermark_ratio", .9);
        var effective = load(root).effective();
        assertEquals(50000, effective.path("max_batch_tokens_without_cache").intValue());
        assertEquals(128, effective.path("max_inited_kv_cache_streams").intValue());
        assertFalse(effective.has("pending_config"));
        assertFalse(effective.has("prefill_retry_times"));
        assertFalse(effective.has("config"));
        config.put("max_inited_kv_cache_streams", -1); assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("max_inited_kv_cache_streams", 128).put("block_tree_device_evict_low_watermark_ratio", .99);
        assertFalse(load(root).effective().has("block_tree_device_evict_low_watermark_ratio"));
    }

    @Test void zeroTimeHandoffDoesNotReportAnIndependentLoadTimeoutAsEffective() throws Exception {
        var root = JSON.<ObjectNode>valueToTree(RtpNodeTest.configuration("decode"));
        ((ObjectNode) root.path("component").path("config")).put("load_cache_timeout_ms", 5000);
        var effective = load(root).effective();
        assertFalse(effective.has("load_cache_timeout_ms"));
        assertFalse(effective.has("pending_config"));
    }

    @Test void mockSettingsRejectTyposCasingNullsAndCollisions() throws Exception {
        for (String key : java.util.List.of("mock.compute.forward.expresson", "mock.Compute.forward.expression", "MOCK_COMPUTE_FORWARD_EXPRESSION", "mock.cache.memory_blocks")) {
            var root = root(); ((ObjectNode) root.path("component").path("config")).put(key, 10);
            assertThrows(IllegalArgumentException.class, () -> load(root));
        }
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("mock.compute.forward.expression", "2");
        assertThrows(IllegalArgumentException.class, () -> load(root));
        config.remove("mock.compute.forward.expression"); config.putNull("mock.cache.memory.blocks");
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void fileAndEnvironmentResolveToIdenticalModelsAndDefaults() throws Exception {
        for (String role : java.util.List.of("prefill", "decode")) {
            var root = JSON.<ObjectNode>valueToTree(RtpNodeTest.configuration(role));
            var config = (ObjectNode) root.path("component").path("config");
            config.put("cp_rotate_method", "ALL_GATHER").put("tp_size", 4).put("dp_size", 3);
            config.put("mock.scheduler.mode", "fifo");
            var parsed = load(root);
            var env = new java.util.HashMap<String, String>();
            parsed.business().fields().forEachRemaining(entry -> env.put(entry.getKey().toUpperCase(java.util.Locale.ROOT).replace('.', '_'), entry.getValue().asText()));
            env.put("CREDENTIAL", "never-record-this");
            assertEquals(parsed.effective(), RtpConfig.fromEnvironment(parsed.id(), parsed.role(), env).effective());
            assertEquals(32, parsed.maxGenerateBatchSize());
            assertFalse(parsed.effective().has("max_generate_batch_size"));
            env.put("MOCK_COMPUTE_UNKNOWN", "1");
            assertThrows(IllegalArgumentException.class, () -> RtpConfig.fromEnvironment(parsed.id(), parsed.role(), env));
        }
    }
    @Test void resolvesNativeAutomaticBudgetsAndPreservesExplicitOverrides() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.remove("max_batch_tokens_size");
        config.put("max_seq_len", 100).put("max_context_batch_size", 3).put("seq_size_per_block", 0);
        var parsed = load(root);
        assertEquals(300, parsed.maxBatchTokens()); assertEquals(64, parsed.blockSize());
        assertEquals(300, parsed.business().path("max_batch_tokens_size").intValue());
        assertEquals(parsed.effective(), RtpConfig.fromEnvironment(parsed.id(), parsed.role(), environmentOf(parsed.business())).effective());
        config.put("max_batch_tokens_size", 50); assertEquals(50, load(root).maxBatchTokens());
        config.put("max_batch_tokens_size", 0).put("max_seq_len", Integer.MAX_VALUE);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    private static java.util.Map<String, String> environmentOf(JsonNode config) {
        var env = new java.util.HashMap<String, String>();
        config.fields().forEachRemaining(e -> env.put(e.getKey().toUpperCase(java.util.Locale.ROOT).replace('.', '_'), e.getValue().asText()));
        return env;
    }

    @Test void usesExplicitRoutingIdentityWithoutReadingCheckpoint() throws Exception {
        var root = root();
        var config = (ObjectNode) root.path("component").path("config");
        config.remove("max_seq_len");
        config.put("checkpoint_path", "/missing/model");
        assertEquals(8192, load(root).maxSeqLen());
        config.put("max_seq_len", 131072).put("image_token_id", 1000)
            .put("cache_key_hash_seed", 0x4453563431494631L);
        var parsed = load(root);
        assertEquals(131072, parsed.maxSeqLen());
        assertEquals(1000, parsed.imageToken());
        assertEquals(0x4453563431494631L, parsed.cacheKeySeed());
        config.put("cache_key_hash_seed", "9223372036854775808");
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void nativeConcurrencyIsNotCappedByMockBookkeeping() throws Exception {
        var root = root();
        ((ObjectNode) root.path("component").path("config")).put("concurrency_limit", 2048);
        assertEquals(2048, load(root).maxGenerateBatchSize());
    }

    @Test void totalCacheCapacityDefaultsToDeviceAndHasOneFileAndEnvironmentSetting() throws Exception {
        var root = root(); var settings = (ObjectNode) root.path("component").path("config");
        var original = load(root);
        assertEquals(original.deviceBlocks(), original.totalCacheBlocks());
        int total = original.deviceBlocks() + 100;
        settings.put("mock.cache.total.blocks", total);
        var expanded = load(root);
        assertEquals(original.deviceBlocks(), expanded.deviceBlocks());
        assertEquals(original.reserveBlocks(), expanded.reserveBlocks());
        assertEquals(total, expanded.totalCacheBlocks());
        assertEquals(total, expanded.effective().path("mock.cache.total.blocks").intValue());
        assertEquals(expanded.effective(), RtpConfig.fromEnvironment(expanded.id(), expanded.role(), environmentOf(expanded.business())).effective());
        ((ObjectNode) root.path("component")).set("config", expanded.effective());
        assertEquals(expanded.effective(), load(root).effective());
        for (int invalid : java.util.List.of(0, original.deviceBlocks() - 1)) {
            ((ObjectNode) root.path("component").path("config")).put("mock.cache.total.blocks", invalid);
            assertThrows(IllegalArgumentException.class, () -> load(root));
        }
    }

}
