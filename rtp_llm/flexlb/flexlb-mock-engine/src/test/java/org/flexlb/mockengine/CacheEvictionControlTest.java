package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.httpGet;
import static org.flexlb.mockengine.MockEngineTestSupport.httpPostResponse;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.unary;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * KV observation/control capabilities backing the flexlb_ft KV case family
 * (tools/online_eval/flexlb_ft/cases/kv.py):
 *
 * <p>1. per-engine key-set exposure — every /snapshot engine dict carries
 * {@code cache_key_set} (the engine's full MockLruBlockCache key list,
 * sorted) next to the {@code cache_keys} count, on prefill AND decode
 * engines;
 *
 * <p>2. {@code POST /cache_evict {"engine", "keys"}} — force-evict the
 * named keys from one engine's LRU and bump that engine's cacheVersion so
 * the master's cache-status poll propagates the eviction (idempotent:
 * unknown keys are a no-op with NO version bump, so re-evictions never
 * trigger spurious master re-syncs).
 *
 * <p>Also pins the MockLruBlockCache unit contract: forced evictions count
 * in {@code evictions}, evicted keys no longer prefix-hit (first miss
 * truncates the run), and gRPC getCacheStatus — the master-facing sync
 * surface — serves the post-eviction key set under the bumped version.
 */
class CacheEvictionControlTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63900;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
        }
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  MockLruBlockCache unit contract (evict + prefixHitBlocks)
    // ════════════════════════════════════════════════════════════════

    @Test
    void blockCacheEvictCountsEvictionsAndTruncatesPrefixHits() {
        MockLruBlockCache cache = new MockLruBlockCache(16);
        assertTrue(cache.admit(List.of(101L, 102L, 103L)));
        assertEquals(3, cache.prefixHitBlocks(List.of(101L, 102L, 103L)));

        assertTrue(cache.evict(List.of(102L)), "evicting a present key must report changed");
        // First miss truncates: the gap at 102 stops the prefix run at 101.
        assertEquals(1, cache.prefixHitBlocks(List.of(101L, 102L, 103L)));
        assertEquals(1, cache.prefixHitBlocks(List.of(101L)));
        assertEquals(0, cache.prefixHitBlocks(List.of(102L)), "evicted key must no longer hit");
        assertEquals(1, cache.prefixHitBlocks(List.of(103L)), "untouched key still hits");
        assertEquals(1, cache.evictions(), "forced evictions must count in evictions");

        // Idempotency: re-evicting a removed key / evicting unknown keys is a no-op.
        assertFalse(cache.evict(List.of(102L, 999L)));
        assertEquals(1, cache.evictions());

        assertTrue(cache.evict(List.of(101L, 103L)));
        assertEquals(3, cache.evictions());
        assertTrue(cache.snapshotKeys().isEmpty());
        assertEquals(0, cache.prefixHitBlocks(List.of(101L, 102L, 103L)));
    }

    // ════════════════════════════════════════════════════════════════
    //  Capability 1: /snapshot cache_key_set exposure (P and D engines)
    // ════════════════════════════════════════════════════════════════

    @Test
    void snapshotExposesCacheKeySetForPrefillAndDecodeEngines() throws Exception {
        startCluster();
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);

        // Idle engines expose an EMPTY list — the field is always present.
        JsonNode idle = engineByName("prefill-0");
        assertTrue(idle.get("cache_key_set").isArray(), "cache_key_set must always be present");
        assertEquals(0, idle.get("cache_key_set").size());

        enqueue(prefill, batch(9600, slot(0, inputWithBlockKeys(
                9601, 10, decode.getGrpcPort(), List.of(101L, 102L, 103L)))));
        awaitCompleted(decode, 1, 10_000);
        awaitEngineCacheKeys("prefill-0", 3, 10_000);
        awaitEngineCacheKeys("decode-0", 3, 10_000);

        // Prefill engine: cache_keys count + the concrete sorted key list.
        JsonNode prefillEngine = engineByName("prefill-0");
        assertEquals(3, prefillEngine.get("cache_keys").asInt());
        JsonNode prefillKeys = prefillEngine.get("cache_key_set");
        assertTrue(prefillKeys.isArray(), "cache_key_set must be a list");
        assertEquals(3, prefillKeys.size());
        assertEquals(101L, prefillKeys.get(0).asLong());
        assertEquals(102L, prefillKeys.get(1).asLong());
        assertEquals(103L, prefillKeys.get(2).asLong());

        // Decode engine: the local cache does not drive routing, but the key
        // set is exposed all the same (observability by symmetry).
        JsonNode decodeEngine = engineByName("decode-0");
        assertEquals(3, decodeEngine.get("cache_keys").asInt());
        JsonNode decodeKeys = decodeEngine.get("cache_key_set");
        assertEquals(3, decodeKeys.size());
        assertEquals(103L, decodeKeys.get(2).asLong());
    }

    // ════════════════════════════════════════════════════════════════
    //  Capability 2: /cache_evict semantics
    // ════════════════════════════════════════════════════════════════

    @Test
    void cacheEvictRemovesKeysBumpsVersionAndPropagatesToCacheStatus() throws Exception {
        startCluster();
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);

        enqueue(prefill, batch(9610, slot(0, inputWithBlockKeys(
                9611, 10, decode.getGrpcPort(), List.of(201L, 202L)))));
        awaitCompleted(decode, 1, 10_000);
        awaitEngineCacheKeys("prefill-0", 2, 10_000);

        long versionAfterAdmit = cacheStatus(prefill).getVersion();
        long evictionsBefore = engineByName("prefill-0").get("cache_evictions").asLong();

        HttpResponse<String> response = httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\",\"keys\":[201]}");
        assertEquals(200, response.statusCode());
        JsonNode body = MAPPER.readTree(response.body());
        assertEquals("ok", body.get("status").asText());
        assertEquals("prefill-0", body.get("engine").asText());
        assertTrue(body.get("changed").asBoolean(), "evicting an admitted key must report changed=true");

        // The version bump is the master-side sync premise: the next
        // getCacheStatus poll sees a NEW version and re-pulls the key set.
        EngineRpcService.CacheStatusPB status = cacheStatus(prefill);
        assertEquals(versionAfterAdmit + 1, status.getVersion(),
                "evict must bump cacheVersion exactly once");
        assertFalse(status.getCacheKeysMap().containsKey(201L),
                "evicted key must disappear from the master-facing cache status");
        assertTrue(status.getCacheKeysMap().containsKey(202L));

        // Snapshot mirrors the eviction: key set, count, eviction counter.
        JsonNode engine = engineByName("prefill-0");
        assertEquals(1, engine.get("cache_keys").asInt());
        assertEquals(1, engine.get("cache_key_set").size());
        assertEquals(202L, engine.get("cache_key_set").get(0).asLong());
        assertEquals(evictionsBefore + 1, engine.get("cache_evictions").asLong());
    }

    @Test
    void cacheEvictIsIdempotentWithoutVersionBump() throws Exception {
        startCluster();
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);

        enqueue(prefill, batch(9620, slot(0, inputWithBlockKeys(
                9621, 10, decode.getGrpcPort(), List.of(301L, 302L)))));
        awaitCompleted(decode, 1, 10_000);
        awaitEngineCacheKeys("prefill-0", 2, 10_000);

        // First evict: changed=true, version +1.
        long versionBefore = cacheStatus(prefill).getVersion();
        HttpResponse<String> first = httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\",\"keys\":[301]}");
        assertEquals(200, first.statusCode());
        assertTrue(MAPPER.readTree(first.body()).get("changed").asBoolean());
        long versionAfterFirst = cacheStatus(prefill).getVersion();
        assertEquals(versionBefore + 1, versionAfterFirst);

        // Re-evicting the same key: no-op — changed=false, NO version bump,
        // so the master's cache-status poll never re-syncs on a stale evict.
        HttpResponse<String> second = httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\",\"keys\":[301]}");
        assertEquals(200, second.statusCode());
        JsonNode secondBody = MAPPER.readTree(second.body());
        assertFalse(secondBody.get("changed").asBoolean());
        assertEquals(versionAfterFirst, secondBody.get("cache_version").asLong());
        assertEquals(versionAfterFirst, cacheStatus(prefill).getVersion());

        // Unknown keys and the empty array are no-ops too.
        HttpResponse<String> unknown = httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\",\"keys\":[999, 1000]}");
        assertEquals(200, unknown.statusCode());
        assertFalse(MAPPER.readTree(unknown.body()).get("changed").asBoolean());

        HttpResponse<String> empty = httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\",\"keys\":[]}");
        assertEquals(200, empty.statusCode());
        assertFalse(MAPPER.readTree(empty.body()).get("changed").asBoolean());
        assertEquals(versionAfterFirst, cacheStatus(prefill).getVersion());
    }

    @Test
    void cacheEvictValidatesInputAndKeepsEngineIsolation() throws Exception {
        startCluster();
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);

        enqueue(prefill, batch(9630, slot(0, inputWithBlockKeys(
                9631, 10, decode.getGrpcPort(), List.of(401L, 402L)))));
        awaitCompleted(decode, 1, 10_000);
        awaitEngineCacheKeys("prefill-0", 2, 10_000);
        awaitEngineCacheKeys("decode-0", 2, 10_000);

        // Missing 'keys' -> 400.
        assertEquals(400, httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\"}").statusCode());
        // 'keys' not an array -> 400.
        assertEquals(400, httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\",\"keys\":401}").statusCode());
        // Non-integer entry -> 400 (Jackson would otherwise coerce to 0).
        assertEquals(400, httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\",\"keys\":[401,\"abc\"]}").statusCode());
        // Unknown engine -> 404; no engine/port -> 400 (dual addressing).
        assertEquals(404, httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-99\",\"keys\":[401]}").statusCode());
        assertEquals(400, httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"keys\":[401]}").statusCode());

        // Evicting on prefill-0 leaves decode-0's key set untouched.
        HttpResponse<String> response = httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"engine\":\"prefill-0\",\"keys\":[401]}");
        assertEquals(200, response.statusCode());
        assertTrue(MAPPER.readTree(response.body()).get("changed").asBoolean());
        assertEquals(1, engineByName("prefill-0").get("cache_key_set").size());
        assertEquals(2, engineByName("decode-0").get("cache_key_set").size());

        // Port addressing (original Java scheme) hits the same engine, and
        // integral decimal strings parse as block keys (strict parsing).
        HttpResponse<String> byPort = httpPostResponse(controlServer.getPort(), "/cache_evict",
                "{\"port\":" + prefill.getGrpcPort() + ",\"keys\":[\"402\"]}");
        assertEquals(200, byPort.statusCode());
        assertTrue(MAPPER.readTree(byPort.body()).get("changed").asBoolean());
        assertEquals(0, engineByName("prefill-0").get("cache_key_set").size());
    }

    // ──────────── Cluster setup (PythonCompatControlApiTest pattern) ────────────

    private void startCluster() throws Exception {
        MockPerformanceModel model = MockEngineTestSupport.performanceModel(tempDir, "10", 0.1);
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "mock-engine-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();

        int port = BASE_PORT;
        JavaMockEngineCluster.FastRpcService prefill = new JavaMockEngineCluster.FastRpcService(
                "prefill-0", "127.0.0.1", "prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                port, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(),
                JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS,
                JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY);
        services.put(port, prefill);
        prefillServices.add(prefill);

        JavaMockEngineCluster.FastRpcService decode = new JavaMockEngineCluster.FastRpcService(
                "decode-0", "127.0.0.1", "decode",
                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                port + 1, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(),
                JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS,
                JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY);
        services.put(port + 1, decode);
        decodeServices.add(decode);

        controlServer = new MockControlServer(services, new ConcurrentHashMap<>(),
                null, null, "127.0.0.1", 0);
        controlServer.start();
    }

    /** GenerateInputPB carrying explicit block_cache_keys (uniqueKey metadata). */
    private static EngineRpcService.GenerateInputPB inputWithBlockKeys(
            long requestId, int inputTokens, int decodePort, List<Long> blockKeys) {
        String uniqueKey;
        try {
            uniqueKey = MAPPER.writeValueAsString(Map.of("block_cache_keys", blockKeys));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        EngineRpcService.GenerateConfigPB.Builder config = EngineRpcService.GenerateConfigPB.newBuilder()
                .setMaxNewTokens(1)
                .setUniqueKey(uniqueKey)
                .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                        .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                        .setRoleStr("DECODE")
                        .setGrpcPort(decodePort)
                        .build());
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(config.build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    /** gRPC getCacheStatus with needCacheKeys — the master-facing sync surface. */
    private static EngineRpcService.CacheStatusPB cacheStatus(JavaMockEngineCluster.FastRpcService service) {
        return unary(observer -> service.getCacheStatus(
                EngineRpcService.CacheVersionPB.newBuilder().setNeedCacheKeys(true).build(),
                observer));
    }

    private JsonNode engineByName(String name) throws Exception {
        JsonNode engines = MAPPER.readTree(httpGet(controlServer.getPort(), "/snapshot")).get("engines");
        for (JsonNode engine : engines) {
            if (name.equals(engine.get("name").asText())) {
                return engine;
            }
        }
        fail("engine " + name + " not found in /snapshot");
        return null;
    }

    /** Poll until the engine's cache_keys count reaches expected (admits are async). */
    private void awaitEngineCacheKeys(String engineName, int expected, long timeoutMs) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (engineByName(engineName).get("cache_keys").asInt() >= expected) {
                return;
            }
            Thread.sleep(10);
        }
        fail("timeout waiting for " + expected + " cache keys on " + engineName
                + ", got " + engineByName(engineName).get("cache_keys").asInt());
    }

    private void awaitCompleted(JavaMockEngineCluster.FastRpcService service,
                                int expected, long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getCompletedCount() >= expected) {
                return;
            }
            Thread.sleep(5);
        }
        fail("timeout waiting for " + expected + " completions on port "
                + service.getGrpcPort() + ", got " + service.getCompletedCount());
    }
}
