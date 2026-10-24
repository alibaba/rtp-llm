package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.node.ObjectNode;

final class MockVitTestSupport {
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
