package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.DpRankCacheStatus;
import org.flexlb.dao.master.DpRankStatus;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineStatusConverterTest {

    @Test
    void worker_status_with_dp_size_gt_one_carries_per_rank_breakdown() {
        EngineRpcService.WorkerStatusPB pb = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("prefill")
                .setDpSize(3)
                .setAvailableConcurrency(30)
                .addDpStatus(rankStatus("10.0.0.1", 9081, 10, 5, 2))
                .addDpStatus(rankStatus("10.0.0.1", 9082, 12, 3, 1))
                .addDpStatus(rankStatus("10.0.0.1", 9083, 8, 7, 4))
                .build();

        WorkerStatusResponse resp = EngineStatusConverter.convertToWorkerStatusResponse(pb);

        List<DpRankStatus> dp = resp.getDpStatuses();
        assertEquals(3, dp.size(), "dp_status[] must yield one DpRankStatus per rank");
        assertEquals(0, dp.get(0).dpRank());
        assertEquals(1, dp.get(1).dpRank());
        assertEquals(2, dp.get(2).dpRank());
        assertEquals("10.0.0.1", dp.get(0).ip());
        assertEquals(9082, dp.get(1).grpcPort());
        assertEquals(8, dp.get(2).availableConcurrency());
    }

    @Test
    void worker_status_with_dp_size_one_skips_per_rank_parsing() {
        EngineRpcService.WorkerStatusPB pb = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("prefill")
                .setDpSize(1)
                // Even if the engine accidentally emitted dp_status[], dp_size=1 ⇒ ignore.
                .addDpStatus(rankStatus("10.0.0.5", 9081, 1, 0, 0))
                .build();

        WorkerStatusResponse resp = EngineStatusConverter.convertToWorkerStatusResponse(pb);
        assertTrue(resp.getDpStatuses().isEmpty(),
                "dp_size==1 must short-circuit the per-rank parse to keep the legacy path zero-cost");
    }

    @Test
    void cache_status_with_dp_cache_carries_per_rank_keys() {
        EngineRpcService.CacheStatusPB pb = EngineRpcService.CacheStatusPB.newBuilder()
                .setAvailableKvCache(8000)
                .setTotalKvCache(12000)
                .setBlockSize(256)
                .putCacheKeys(1L, true)
                .putCacheKeys(2L, true)
                .addDpCache(EngineRpcService.CacheStatusPB.newBuilder()
                        .setIp("10.0.0.1").setGrpcPort(9081)
                        .setAvailableKvCache(4000).setTotalKvCache(6000).setBlockSize(256)
                        .putCacheKeys(1L, true).build())
                .addDpCache(EngineRpcService.CacheStatusPB.newBuilder()
                        .setIp("10.0.0.1").setGrpcPort(9082)
                        .setAvailableKvCache(4000).setTotalKvCache(6000).setBlockSize(256)
                        .putCacheKeys(2L, true).build())
                .build();

        CacheStatus cs = EngineStatusConverter.convertToCacheStatus(pb);
        List<DpRankCacheStatus> dp = cs.getDpCaches();
        assertEquals(2, dp.size());
        assertEquals(0, dp.get(0).dpRank());
        assertEquals(1, dp.get(1).dpRank());
        assertTrue(dp.get(0).cachedKeys().contains(1L));
        assertTrue(dp.get(1).cachedKeys().contains(2L));
        // Outer union view stays as the legacy "any-rank match" surface.
        assertEquals(2, cs.getCachedKeys().size());
    }

    @Test
    void cache_status_without_dp_cache_keeps_dpCaches_empty() {
        EngineRpcService.CacheStatusPB pb = EngineRpcService.CacheStatusPB.newBuilder()
                .setAvailableKvCache(8000)
                .setTotalKvCache(12000)
                .setBlockSize(256)
                .putCacheKeys(42L, true)
                .build();

        CacheStatus cs = EngineStatusConverter.convertToCacheStatus(pb);
        assertTrue(cs.getDpCaches().isEmpty(),
                "no dp_cache[] in payload ⇒ DTO list stays empty (legacy single-DP path)");
        assertTrue(cs.getCachedKeys().contains(42L));
    }

    private static EngineRpcService.WorkerStatusPB rankStatus(String ip, int port,
                                                              int avail, int running, int waiting) {
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setIp(ip)
                .setGrpcPort(port)
                .setAvailableConcurrency(avail)
                .setRunningQueryLen(running)
                .setWaitingQueryLen(waiting)
                .setAlive(true)
                .build();
    }
}
