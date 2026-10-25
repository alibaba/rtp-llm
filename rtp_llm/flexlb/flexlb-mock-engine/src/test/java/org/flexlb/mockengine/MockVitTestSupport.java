package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;

final class MockVitTestSupport {
    static MockVitService.Config config(com.fasterxml.jackson.databind.JsonNode document) {
        var vit = document.path("roles").path("vit");
        var defaults = MockVitService.Config.defaults();
        return new MockVitService.Config(defaults.hiddenSize(), vit.path("vit_concurrency").intValue(),
                vit.path("vit_max_queue_size").intValue(), vit.path("mm_cache_cpu_max_bytes").longValue(),
                vit.path("mm_hash_key_cache_max_bytes").longValue(), MockEngineConfig.vitImageBytes(vit),
                defaults.maxOutputBytes(), defaults.computeDelayMs());
    }
    static MockVitService service(MockVitService.Config config) { return service(config, null); }
    static MockVitService service(MockVitService.Config config, MockV41Image.Config model) {
        var document = (ObjectNode)MockEngineConfigFixture.load().snapshot();
        var vit = (ObjectNode)document.path("roles").path("vit");
        vit.put("vit_concurrency", config.concurrency()).put("vit_max_queue_size", config.queueCapacity())
                .put("mm_cache_gpu_max_bytes", 0).put("mm_cache_cpu_max_bytes", config.embeddingCacheBytes())
                .put("mm_hash_key_cache_max_bytes", config.hashCacheBytes());
        return new MockVitService(config, model, document);
    }
}
