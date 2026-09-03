package org.flexlb.mockengine;

import io.grpc.Server;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.unary;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Reported-total alignment with the built block pool (capacity model v2
 * reporting-caliber regression guard).
 *
 * <p>Regression fixed here: {@code --prefill-cache-blocks}/
 * {@code --decode-cache-blocks} shrink the pool the engine actually builds,
 * but the master-facing total used to stay at the per-role DEFAULT token
 * constant — so the master's used = total - available math read ~99.9% on a
 * 4-block decode pool, every decode engine tripped the KV-full gate, and the
 * whole KV case family parked structurally.
 *
 * <p>Production-caliber contract (the fix): every master-facing surface —
 * gRPC {@code getCacheStatus.totalKvCache}, gRPC
 * {@code getWorkerStatus.totalKvCache}, and HTTP {@code /snapshot}
 * {@code total_kv_tokens} — reports the allocator truth
 * {@code totalBlocks x blockSize}, the SAME value the pool is built from
 * ({@code MockLruBlockCache}), so {@code used = total - available} keeps its
 * production meaning at every pool size.
 */
class KvTotalReportAlignmentTest {

    private static final int SPB = 1024;
    private static final int BASE_PORT = 64_640;

    @TempDir
    Path tempDir;

    private EventLoopGroup bossGroup;
    private EventLoopGroup workerGroup;
    private ScheduledExecutorService scheduler;
    private Map<Integer, Server> serversByPort;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private MockPerformanceModel model;

    @BeforeEach
    void setUp() throws IOException {
        bossGroup = new NioEventLoopGroup(1);
        workerGroup = new NioEventLoopGroup(1);
        scheduler = Executors.newScheduledThreadPool(2, runnable -> {
            Thread thread = new Thread(runnable, "kv-total-report-test-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        serversByPort = new ConcurrentHashMap<>();
        services = new ConcurrentHashMap<>();
        // performanceModel() pins block_size = 1024 (SPB).
        model = MockEngineTestSupport.performanceModel(tempDir, "10", 1.0, 1.0);
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (Server server : serversByPort.values()) {
            server.shutdownNow();
        }
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
        bossGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        workerGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
    }

    /**
     * Start ONE engine through the production construction path
     * ({@code JavaMockEngineCluster.startEngine}) with a per-role
     * block-count override, exactly like the CLI flags do.
     */
    private JavaMockEngineCluster.FastRpcService startOne(
            String role, int blocksOverride, int grpcPort, long roleTotalKvTokens)
            throws IOException {
        JavaMockEngineCluster.Config config = new JavaMockEngineCluster.Config();
        config.host = "127.0.0.1";
        config.uniqueEngineIps = false;
        if (roleTotalKvTokens > 0) {
            if ("prefill".equals(role)) {
                config.prefillTotalKvTokens = roleTotalKvTokens;
            } else {
                config.decodeTotalKvTokens = roleTotalKvTokens;
            }
        }
        if ("prefill".equals(role)) {
            config.prefillCacheBlocks = blocksOverride;
        } else {
            config.decodeCacheBlocks = blocksOverride;
        }
        return JavaMockEngineCluster.startEngine(
                config, model, serversByPort, bossGroup, workerGroup,
                services, scheduler, new JavaMockEngineCluster.ClusterStats(),
                role, role + "-0", grpcPort, 0);
    }

    private static EngineRpcService.CacheStatusPB cacheStatus(
            JavaMockEngineCluster.FastRpcService service) {
        return unary(observer -> service.getCacheStatus(
                EngineRpcService.CacheVersionPB.newBuilder().build(),
                observer));
    }

    /** The master's KV gate math on the reported pair: used = total - available. */
    private static double usedRatio(long total, long available) {
        return total <= 0 ? 1.0 : (double) (total - available) / total;
    }

    // ───────── decode side: --decode-cache-blocks 4 ─────────

    @Test
    void decodeBlockOverrideShrinksReportedTotalOnEverySurface() throws Exception {
        JavaMockEngineCluster.FastRpcService decode =
                startOne("decode", 4, BASE_PORT, 0);
        long expectedTotal = 4L * SPB; // 4096 = the pool actually built

        // Every master-facing surface derives from the pool-aligned total.
        assertEquals(expectedTotal, decode.getTotalKvTokens(),
                "getTotalKvTokens must equal blocks x spb (the built pool)");
        EngineRpcService.CacheStatusPB cache = cacheStatus(decode);
        assertEquals(expectedTotal, cache.getTotalKvCache(),
                "getCacheStatus.totalKvCache must report the built pool, not the DEFAULT constant");
        assertTrue(cache.getAvailableKvCache() <= expectedTotal,
                "available must stay within the small pool caliber");
        assertEquals(SPB, cache.getBlockSize());
        EngineRpcService.WorkerStatusPB worker = workerStatus(decode, 0);
        assertEquals(expectedTotal, worker.getTotalKvCache(),
                "getWorkerStatus.totalKvCache must report the built pool too");
        assertTrue(worker.getAvailableKvCache() <= expectedTotal);
        assertEquals(expectedTotal,
                ((Number) decode.getSnapshot().get("total_kv_tokens")).longValue(),
                "/snapshot total_kv_tokens must report the built pool as well");

        // Master gate sanity on the reported pair: an idle small pool must
        // NOT read as KV-full.  (Pre-fix math: total = 4_194_304 while
        // available <= 4096 -> used ratio ~99.9% >= the 90% full gate ->
        // every decode engine parked.)
        double idleUsedRatio = usedRatio(cache.getTotalKvCache(), cache.getAvailableKvCache());
        assertEquals(0.0, idleUsedRatio, 1e-9,
                "an idle pool must report zero used tokens");
        assertTrue(idleUsedRatio < 0.9, "idle decode pool must stay under the KV-full gate");
    }

    // ───────── prefill side: --prefill-cache-blocks 4 (same lesion) ─────────

    @Test
    void prefillBlockOverrideShrinksReportedTotalOnEverySurface() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill =
                startOne("prefill", 4, BASE_PORT + 1, 0);
        long expectedTotal = 4L * SPB;

        assertEquals(expectedTotal, prefill.getTotalKvTokens());
        assertEquals(expectedTotal, cacheStatus(prefill).getTotalKvCache(),
                "getCacheStatus.totalKvCache (prefill side) must report the built pool");
        assertEquals(expectedTotal, workerStatus(prefill, 0).getTotalKvCache(),
                "getWorkerStatus.totalKvCache (prefill side) must report the built pool");
        assertEquals(expectedTotal,
                ((Number) prefill.getSnapshot().get("total_kv_tokens")).longValue());
    }

    // ───────── default pools: derived block count reports the same tokens ─────────

    @Test
    void defaultPoolReportsDerivedTokenTotal() throws Exception {
        // No block override: blocks = ceil(per-role default tokens / spb) —
        // both defaults divide by 1024, so the reported total equals the
        // per-role default exactly (no behavior change for default configs).
        JavaMockEngineCluster.FastRpcService decode =
                startOne("decode", 0, BASE_PORT + 2, 0);
        assertEquals(JavaMockEngineCluster.DEFAULT_DECODE_TOTAL_KV_TOKENS,
                decode.getTotalKvTokens());
        assertEquals(JavaMockEngineCluster.DEFAULT_DECODE_TOTAL_KV_TOKENS,
                cacheStatus(decode).getTotalKvCache());

        JavaMockEngineCluster.FastRpcService prefill =
                startOne("prefill", 0, BASE_PORT + 3, 0);
        assertEquals(JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS,
                prefill.getTotalKvTokens());
        assertEquals(JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS,
                cacheStatus(prefill).getTotalKvCache());
    }

    // ───────── token-config override: ceil-aligned to the pool ─────────

    @Test
    void tokenOverrideCeilsToPoolAlignedTotal() throws Exception {
        // --prefill-total-kv-tokens 5000: the pool rounds UP to 5 blocks, and
        // the reported total follows the POOL (5 x 1024 = 5120), mirroring
        // the production allocator's real capacity — not the raw config value.
        JavaMockEngineCluster.FastRpcService prefill =
                startOne("prefill", 0, BASE_PORT + 4, 5000L);
        assertEquals(5L * SPB, prefill.getTotalKvTokens(),
                "non-divisible token config must report the ceil-aligned pool total");
        assertEquals(5L * SPB, cacheStatus(prefill).getTotalKvCache());
    }

    // ───────── master gate math under partial load ─────────

    @Test
    void masterGateMathStaysHealthyUnderPressure() throws Exception {
        // A 4-block decode pool carrying a 2-block injected load: the master's
        // used = total - available must read ~50% — comfortably under the 90%
        // KV-full gate — and all three surfaces must agree on the pair.
        JavaMockEngineCluster.FastRpcService decode =
                startOne("decode", 4, BASE_PORT + 5, 0);
        long expectedTotal = 4L * SPB;
        long expectedAvailable = expectedTotal - 2L * SPB;

        decode.setFaultConfig(FaultInjectionConfig.builder()
                .kvPressureTokens(2L * SPB)
                .build());

        EngineRpcService.CacheStatusPB cache = cacheStatus(decode);
        EngineRpcService.WorkerStatusPB worker = workerStatus(decode, 0);
        long snapshotAvailable =
                ((Number) decode.getSnapshot().get("available_kv_tokens")).longValue();
        assertEquals(expectedAvailable, cache.getAvailableKvCache(),
                "getCacheStatus available = pool - pressure");
        assertEquals(expectedAvailable, worker.getAvailableKvCache(),
                "getWorkerStatus available must match getCacheStatus");
        assertEquals(expectedAvailable, snapshotAvailable,
                "/snapshot available must match getCacheStatus");
        assertEquals(expectedTotal, cache.getTotalKvCache());
        assertEquals(expectedTotal, worker.getTotalKvCache());

        double used = usedRatio(cache.getTotalKvCache(), cache.getAvailableKvCache());
        assertEquals(0.5, used, 1e-9);
        assertTrue(used < 0.9, "a half-loaded small pool must not read as KV-full");

        decode.clearFaultConfig();
        assertEquals(expectedTotal, cacheStatus(decode).getAvailableKvCache(),
                "clearing the pressure must restore full-pool availability");
    }
}
