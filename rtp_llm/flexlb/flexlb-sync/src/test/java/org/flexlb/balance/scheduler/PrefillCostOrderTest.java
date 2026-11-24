package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PrefillCostOrderTest {
    private record Request(int priority, int arrival, long input, long hits) {
        long compute() { return input - hits; }
    }

    private static final Comparator<Request> WORKER_ORDER = PrefillCostOrder.workerOrder(
            Request::priority, Request::input, Request::hits, Comparator.comparingInt(Request::arrival));

    @Test
    void closeRatiosAtLongMaxValueKeepTheirExactOrder() {
        var worse = new Request(50, 1, Long.MAX_VALUE, 1);
        var better = new Request(50, 2, Long.MAX_VALUE, 2);
        assertEquals(List.of(better, worse), List.of(worse, better).stream().sorted(WORKER_ORDER).toList());
    }

    @Test
    void workerOrderingMatchesExactUnitCostAcrossBacklogs() {
        Comparator<Request> byRatio = (a, b) ->
                BigInteger.valueOf(a.compute()).multiply(BigInteger.valueOf(b.input()))
                        .compareTo(BigInteger.valueOf(b.compute()).multiply(BigInteger.valueOf(a.input())));
        var reference = Comparator.comparingInt(Request::priority).reversed()
                .thenComparing(request -> request.input() == 0)
                .thenComparing(byRatio)
                .thenComparingLong(Request::compute)
                .thenComparingLong(Request::input)
                .thenComparingInt(Request::arrival);
        var random = new Random(20261114L);
        for (int round = 0; round < 50; round++) {
            var requests = new ArrayList<Request>();
            for (int i = 0; i < 100; i++) {
                long input = switch (i % 4) {
                    case 0 -> 0;
                    case 1 -> Long.MAX_VALUE - random.nextInt(100);
                    default -> 1000L * (1 + random.nextInt(10));
                };
                requests.add(new Request(random.nextInt(3), i, input, input / (1 + random.nextInt(10))));
            }
            assertEquals(requests.stream().sorted(reference).toList(),
                    requests.stream().sorted(WORKER_ORDER).toList(), "round=" + round);
        }
    }
}
