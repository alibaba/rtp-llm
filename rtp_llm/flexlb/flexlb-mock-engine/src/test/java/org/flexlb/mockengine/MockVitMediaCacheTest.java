package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.util.List;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(10)
class MockVitMediaCacheTest {
    @Test void itemLruTouchesHitsAndZeroDisables() {
        var cache = new MockVitMediaCache(2);
        byte[] original = {1, 2}; cache.put("a", original); cache.put("b", new byte[]{3});
        assertSame(original, cache.get("a")); cache.put("c", new byte[]{4});
        assertNull(cache.get("b")); assertEquals(2, cache.size()); cache.clear(); assertEquals(0, cache.size());
        var disabled = new MockVitMediaCache(0); disabled.put("a", original); assertNull(disabled.get("a"));
        var negative = new MockVitMediaCache(-1); negative.put("a", original); assertEquals(0, negative.size());
    }
    @Test void embeddingCacheDisabledStillReusesUrlBytesAndEvictsByUrlItemLimit() throws Exception {
        var image = MockVitServiceTest.image(56, 56, 123);
        var data = image.getMultimodalUrl();
        byte[] png = java.util.Base64.getDecoder().decode(data.substring(data.indexOf(',') + 1));
        var downloads = new java.util.concurrent.atomic.AtomicInteger();
        var server = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            downloads.incrementAndGet(); exchange.sendResponseHeaders(200, png.length);
            try (var body = exchange.getResponseBody()) { body.write(png); }
        });
        server.start();
        try {
            var document = (com.fasterxml.jackson.databind.node.ObjectNode)MockEngineConfigFixture.load().snapshot();
            ((com.fasterxml.jackson.databind.node.ObjectNode)document.path("roles").path("vit"))
                    .put("vit_concurrency", 1).put("vit_max_queue_size", 4).put("mm_cache_gpu_max_bytes", 0)
                    .put("mm_cache_cpu_max_bytes", 0).put("mm_hash_key_cache_max_bytes", 0).put("url_cache_item_num", 1);
            var config = new MockVitService.Config(8, 1, 4, 0, 0, 1 << 20, 1 << 20, 0);
            try (var service = new MockVitService(config, null, document)) {
                var a = image.toBuilder().setMultimodalUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/a").build();
                var b = a.toBuilder().setMultimodalUrl(a.getMultimodalUrl().replace("/a", "/b")).build();
                for (var input : List.of(a, a, b, a)) service.acquire(input, 3000).get(3, TimeUnit.SECONDS);
                assertEquals(4, service.computations.get()); assertEquals(3, downloads.get());
                assertEquals(1, service.snapshot().get("url_cache_items"));
                assertEquals(0, service.snapshot().get("embedding_bytes")); assertEquals(0, service.snapshot().get("hash_bytes"));
            }
            var rawCache = new MockVitMediaCache(1); rawCache.put("data:cached", png);
            var cached = image.toBuilder().setMultimodalUrl("data:cached").build();
            assertThrows(IllegalArgumentException.class, () -> MockVitImage.prepare(cached, 8, 1, 1 << 20, 1000, java.util.Map.of(), rawCache));
            assertEquals(3, downloads.get());
        } finally { server.stop(0); }
    }
}
