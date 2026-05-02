package org.flexlb.balance.dp;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

class RoundRobinAssignTest {

    private final RoundRobinAssign rr = new RoundRobinAssign();

    @Test
    void single_batch_size4_dpSize2_alternates() {
        List<Integer> ranks = ranksOf(rr.assign(makeBatch(4, 2)));
        assertEquals(List.of(0, 1, 0, 1), ranks);
    }

    @Test
    void single_batch_size4_dpSize4_each_rank_once() {
        List<Integer> ranks = ranksOf(rr.assign(makeBatch(4, 4)));
        assertEquals(List.of(0, 1, 2, 3), ranks);
    }

    @Test
    void single_batch_size3_dpSize4_partial_fill() {
        // Even when the batch isn't full, ranks must stay within [0, dpSize).
        List<Integer> ranks = ranksOf(rr.assign(makeBatch(3, 4)));
        assertEquals(List.of(0, 1, 2), ranks);
    }

    @Test
    void cursor_advances_across_batches() {
        // batch1 (4, dp=4) → [0,1,2,3], cursor → 4
        // batch2 (4, dp=4) → [0,1,2,3]  (4 mod 4 = 0)
        // batch3 (2, dp=4) → [0,1],     cursor → 10
        // batch4 (4, dp=4) → [2,3,0,1]  (10 mod 4 = 2)
        assertEquals(List.of(0, 1, 2, 3), ranksOf(rr.assign(makeBatch(4, 4))));
        assertEquals(List.of(0, 1, 2, 3), ranksOf(rr.assign(makeBatch(4, 4))));
        assertEquals(List.of(0, 1), ranksOf(rr.assign(makeBatch(2, 4))));
        assertEquals(List.of(2, 3, 0, 1), ranksOf(rr.assign(makeBatch(4, 4))));
    }

    @Test
    void concurrent_assign_each_batch_full_dp_yields_uniform_distribution() throws Exception {
        // Every batch is dp=batchSize (one batch fills exactly one DP cycle), so any
        // cursor starting point still produces a permutation of 0..dp-1. Even under
        // concurrency, the per-rank total count must remain strictly equal.
        int dpSize = 4;
        int batchesPerThread = 50;
        int threads = 8;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        List<Integer> allRanks = Collections.synchronizedList(new ArrayList<>());

        for (int t = 0; t < threads; t++) {
            pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < batchesPerThread; i++) {
                        for (RankAssignment ra : rr.assign(makeBatch(dpSize, dpSize))) {
                            allRanks.add(ra.dpRank());
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }
        start.countDown();
        pool.shutdown();
        assertTrue(pool.awaitTermination(5, TimeUnit.SECONDS));

        // Every rank must be in [0, dpSize)
        assertTrue(allRanks.stream().allMatch(r -> r >= 0 && r < dpSize));
        // Total = threads * batchesPerThread * dpSize
        assertEquals(threads * batchesPerThread * dpSize, allRanks.size());
        // Each rank count must be exactly equal: getAndAdd takes a contiguous dpSize
        // span of cursor values, and mod dpSize hands out exactly one of each.
        int[] counts = new int[dpSize];
        allRanks.forEach(r -> counts[r]++);
        for (int c : counts) {
            assertEquals(threads * batchesPerThread, c,
                    "ranks must be perfectly evenly distributed even under concurrency");
        }
    }

    @Test
    void invalid_dpSize_throws() {
        assertThrows(IllegalArgumentException.class, () -> rr.assign(makeBatch(2, 0)));
        assertThrows(IllegalArgumentException.class, () -> rr.assign(makeBatch(2, -1)));
    }

    @Test
    void name_is_RR() {
        assertEquals("RR", rr.name());
        assertEquals(RoundRobinAssign.NAME, rr.name());
    }

    // ============== helpers ==============

    private static List<Integer> ranksOf(List<RankAssignment> assignments) {
        return assignments.stream().map(RankAssignment::dpRank).toList();
    }

    private static PrefillBatch makeBatch(int size, int dpSize) {
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(9080);
        List<PendingRequest> reqs = IntStream.range(0, size)
                .mapToObj(i -> PendingRequest.of(null, prefill, null, new CompletableFuture<Response>()))
                .toList();
        return new PrefillBatch(prefill, reqs, dpSize);
    }
}
