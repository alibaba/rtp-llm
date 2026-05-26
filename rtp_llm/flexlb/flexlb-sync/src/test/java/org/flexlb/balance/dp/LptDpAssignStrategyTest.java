package org.flexlb.balance.dp;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class LptDpAssignStrategyTest {

    private CacheAwareService cacheAwareService;
    private LptDpAssignStrategy strategy;

    private static final ServerStatus PREFILL = prefillStatus("10.0.0.1", 8080);
    private static final ServerStatus DECODE = decodeStatus("10.0.0.2", 8081);

    @BeforeEach
    void setUp() {
        cacheAwareService = mock(CacheAwareService.class);
        when(cacheAwareService.findMatchingRanksInGroup(any(), any())).thenReturn(Collections.emptyMap());
        strategy = new LptDpAssignStrategy(cacheAwareService);
    }

    @Test
    void longest_jobs_assigned_to_different_ranks() {
        // 4 requests with seqLen: 1000, 900, 100, 50 → dpSize=2
        // LPT: 1000→rank0, 900→rank1, 100→rank1(900+100=1000 < 1000+100=1100 → actually 1000 vs 900 → rank1),
        // 50→rank? (depends on load: rank0=1000, rank1=1000 → tie, takes first)
        PrefillBatch batch = batch(2, 1,
                pending(1, 1000), pending(2, 900), pending(3, 100), pending(4, 50));

        List<RankAssignment> assignments = strategy.assign(batch);

        assertEquals(4, assignments.size());

        // Collect load per rank
        Map<Integer, Long> rankLoad = new HashMap<>();
        for (RankAssignment ra : assignments) {
            long seqLen = ra.request().ctx().getRequest().getSeqLen();
            rankLoad.merge(ra.dpRank(), seqLen, Long::sum);
        }

        // With LPT: rank0 gets 1000+50=1050, rank1 gets 900+100=1000 (or close)
        // The key invariant: both ranks should be loaded, and load should be roughly balanced
        assertTrue(rankLoad.containsKey(0));
        assertTrue(rankLoad.containsKey(1));
        long maxLoad = Collections.max(rankLoad.values());
        long minLoad = Collections.min(rankLoad.values());
        assertTrue(maxLoad - minLoad <= 100, "LPT should produce balanced load, got " + rankLoad);
    }

    @Test
    void all_equal_length_distributes_evenly() {
        PrefillBatch batch = batch(4, 1,
                pending(1, 500), pending(2, 500), pending(3, 500), pending(4, 500));

        List<RankAssignment> assignments = strategy.assign(batch);

        assertEquals(4, assignments.size());
        Map<Integer, Integer> rankCounts = new HashMap<>();
        for (RankAssignment ra : assignments) {
            rankCounts.merge(ra.dpRank(), 1, Integer::sum);
        }
        // Each rank should get exactly 1 request
        for (int rank = 0; rank < 4; rank++) {
            assertEquals(1, rankCounts.getOrDefault(rank, 0));
        }
    }

    @Test
    void single_request_goes_to_rank_0() {
        PrefillBatch batch = batch(4, 1, pending(1, 1000));
        List<RankAssignment> assignments = strategy.assign(batch);

        assertEquals(1, assignments.size());
        assertEquals(0, assignments.get(0).dpRank());
    }

    @Test
    void empty_batch_returns_empty() {
        PrefillBatch batch = new PrefillBatch(PREFILL, List.of(), 4, 1);
        List<RankAssignment> assignments = strategy.assign(batch);
        assertTrue(assignments.isEmpty());
    }

    @Test
    void invalid_dpSize_throws() {
        PrefillBatch batch = new PrefillBatch(PREFILL, List.of(pending(1, 100)), 0, 1);
        assertThrows(IllegalArgumentException.class, () -> strategy.assign(batch));
    }

    @Test
    void name_is_LPT() {
        assertEquals("LPT", strategy.name());
    }

    // ============== helpers ==============

    private static PendingRequest pending(long requestId, long seqLen) {
        BalanceContext ctx = new BalanceContext();
        Request r = new Request();
        r.setRequestId(requestId);
        r.setSeqLen(seqLen);
        ctx.setRequest(r);
        return new PendingRequest(ctx, PREFILL, DECODE, new CompletableFuture<>(), System.nanoTime() / 1000);
    }

    private static PrefillBatch batch(int dpSize, long blockSize, PendingRequest... requests) {
        return new PrefillBatch(PREFILL, List.of(requests), dpSize, blockSize);
    }

    private static ServerStatus prefillStatus(String ip, int port) {
        ServerStatus s = new ServerStatus();
        s.setRole(RoleType.PREFILL);
        s.setServerIp(ip);
        s.setHttpPort(port);
        s.setGrpcPort(port + 1000);
        s.setGroup("g1");
        s.setSuccess(true);
        return s;
    }

    private static ServerStatus decodeStatus(String ip, int port) {
        ServerStatus s = new ServerStatus();
        s.setRole(RoleType.DECODE);
        s.setServerIp(ip);
        s.setHttpPort(port);
        s.setGrpcPort(port + 1000);
        s.setGroup("g1");
        s.setSuccess(true);
        return s;
    }
}
