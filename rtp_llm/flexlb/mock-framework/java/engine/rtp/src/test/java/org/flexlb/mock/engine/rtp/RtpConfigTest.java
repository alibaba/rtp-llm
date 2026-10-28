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

    @Test void acceptsCopiedNativeEnvValuesAndPreservesNativePayloadWithoutRenaming() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.remove("seq_size_per_block"); config.put("SEQ_SIZE_PER_BLOCK", "32").put("REUSE_CACHE", "on")
            .put("TP_SIZE", "2").put("DISABLE_DECODE_FIRST_MALLOC_DEVICE_REUSE", "true").put("MODEL_NATIVE_JSON", "{unmodified native payload}");
        var parsed = load(root);
        assertEquals(32, parsed.blockSize()); assertTrue(parsed.reuseCache()); assertEquals(2, parsed.tpSize());
        assertTrue(parsed.disableDecodeFirstReuse());
        assertEquals("{unmodified native payload}", parsed.business().path("MODEL_NATIVE_JSON").textValue());
        assertEquals(config, parsed.business());
        ((ObjectNode) parsed.business()).removeAll(); assertFalse(parsed.business().isEmpty());
    }

    @Test void preservesNativeReservePercentageAndAllowsExplicitObservedReserveBlocks() throws Exception {
        var root = root(); var capacity = (ObjectNode) root.path("component").path("config").path("capacity");
        capacity.remove("reserve_blocks");
        assertEquals(1, load(root).reserveBlocks()); // native default 5%, rounded up for this mock pool
        capacity.put("reserve_blocks", 2); assertEquals(2, load(root).reserveBlocks());
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

    @Test void rejectsUnsupportedDpAndUnknownMockCapacityFields() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.put("dp_size", 2); assertThrows(IllegalArgumentException.class, () -> load(root));
        config.remove("dp_size"); ((ObjectNode) config.path("capacity")).put("device_block", 8);
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }

    @Test void requiresExplicitGeometryAndMatchingRoleFormula() throws Exception {
        var root = root(); var config = (ObjectNode) root.path("component").path("config");
        config.remove("seq_size_per_block"); assertThrows(IllegalArgumentException.class, () -> load(root));
        config.put("seq_size_per_block", 16); ((ObjectNode) root.path("environment").path("compute")).removeAll();
        assertThrows(IllegalArgumentException.class, () -> load(root));
    }
}
