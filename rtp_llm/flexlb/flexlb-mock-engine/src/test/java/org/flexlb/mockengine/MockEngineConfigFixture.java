package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;

final class MockEngineConfigFixture {
    static MockEngineConfig load() {
        try { return MockEngineConfig.load("../tools/mock_bundle/engine-config.json"); }
        catch (java.io.IOException error) { throw new java.io.UncheckedIOException(error); }
    }
    static MockEngineConfig pool(String role, int blocks, int blockSize) {
        ObjectNode document = (ObjectNode)load().snapshot();
        ((ObjectNode)document.path("roles").path(role).path("capacity")).put("device_blocks", blocks).put("seq_size_per_block", blockSize).put("reserve_blocks", (int)(blocks * document.path("roles").path(role).path("kv_cache").path("reserve_block_ratio").doubleValue() / 100));
        ((ObjectNode)document.path("roles").path(role).path("kv_cache")).put("seq_size_per_block", blockSize);
        return MockEngineConfig.parse(document);
    }
}
