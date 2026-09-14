package org.flexlb.balance.strategy.batch;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchCompletionSearchTest {

    @Test
    @DisplayName("Completion search uses the configured budget to repair an incomplete greedy plan")
    void usesTheConfiguredBudgetToRepairAnIncompletePlan() {
        List<BatchPlanningRequest> requests = List.of(
                request(candidate("w1")),
                request(candidate("w1"), candidate("w2")));

        assertArrayEquals(new int[] {0, -1},
                BatchCompletionSearch.complete(requests, new int[] {0, -1}, 1));
        assertArrayEquals(new int[] {0, 1},
                BatchCompletionSearch.complete(requests, new int[] {0, -1}, 4096));
    }

    @Test
    @DisplayName("Completion search stops exactly at the 4096-evaluation budget")
    void stopsExactlyAtTheConfiguredEvaluationLimit() {
        List<BatchPlanningRequest> requests = saturatedSearchRequests();
        BatchCompletionSearch.SearchResult result = BatchCompletionSearch.search(
                requests, new int[] {0, 0, 1, 2, 3, 4, 5, -1}, 4096);

        assertEquals(4096, result.evaluations());
        assertTrue(result.budgetExhausted());
        assertEquals(7L, Arrays.stream(result.plan())
                .filter(candidate -> candidate >= 0)
                .count());
    }

    @Test
    @Tag("performance-regression")
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    @EnabledIfSystemProperty(
            named = "flexlb.perf.completion-search.enabled",
            matches = "true")
    @DisplayName("Reports the local cost of a saturated 4096-evaluation completion search")
    void reportsSaturatedCompletionSearchLatency() {
        List<BatchPlanningRequest> requests = saturatedSearchRequests();
        int[] greedy = {0, 0, 1, 2, 3, 4, 5, -1};
        for (int warmup = 0; warmup < 200; warmup++) {
            assertEquals(4096, BatchCompletionSearch.search(
                    requests, greedy, 4096).evaluations());
        }

        long[] elapsedNanos = new long[500];
        long checksum = 0L;
        for (int measurement = 0; measurement < elapsedNanos.length; measurement++) {
            long started = System.nanoTime();
            BatchCompletionSearch.SearchResult result = BatchCompletionSearch.search(
                    requests, greedy, 4096);
            elapsedNanos[measurement] = System.nanoTime() - started;
            checksum += result.evaluations();
        }
        Arrays.sort(elapsedNanos);
        System.out.printf(
                "FlexLB completion-search performance: evaluations=4096 samples=%d "
                        + "p50_us=%d p95_us=%d checksum=%d%n",
                elapsedNanos.length,
                TimeUnit.NANOSECONDS.toMicros(elapsedNanos[elapsedNanos.length / 2]),
                TimeUnit.NANOSECONDS.toMicros(elapsedNanos[(elapsedNanos.length * 95) / 100]),
                checksum);
        assertTrue(checksum > 0L);
    }

    private static BatchPlanningRequest request(BatchCandidate... candidates) {
        return new BatchPlanningRequest(List.of(candidates), 0L, 0.0, 0L, false);
    }

    private static BatchCandidate candidate(String worker) {
        return new BatchCandidate(worker, 0L, 0L, 0L,
                0L, 0L, 1);
    }

    private static List<BatchPlanningRequest> saturatedSearchRequests() {
        List<BatchPlanningRequest> requests = new ArrayList<>();
        for (int requestIndex = 0; requestIndex < 8; requestIndex++) {
            List<BatchCandidate> candidates = new ArrayList<>();
            for (int workerIndex = 0; workerIndex < 6; workerIndex++) {
                candidates.add(new BatchCandidate(
                        "w" + workerIndex,
                        10L + workerIndex,
                        0L,
                        0L,
                        0L,
                        0L,
                        workerIndex == 0 ? 2 : 1));
            }
            requests.add(new BatchPlanningRequest(
                    candidates, 0L, 0.0, 0L, false));
        }
        return List.copyOf(requests);
    }
}
