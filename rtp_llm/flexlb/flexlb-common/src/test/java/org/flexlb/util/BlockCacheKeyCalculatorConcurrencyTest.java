package org.flexlb.util;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BlockCacheKeyCalculatorConcurrencyTest {

    private static final int THREAD_COUNT = 16;
    private static final int REQUEST_COUNT = 320;
    private static final int INPUT_VARIANT_COUNT = 4;
    private static final int TOKEN_COUNT = 20_480;
    private static final int BLOCK_SIZE = 64;

    @Test
    void producesStableResultsUnderConcurrentLoad() throws Exception {
        List<int[]> inputs = new ArrayList<>(INPUT_VARIANT_COUNT);
        List<List<Long>> expectedResults = new ArrayList<>(INPUT_VARIANT_COUNT);
        for (int variant = 0; variant < INPUT_VARIANT_COUNT; variant++) {
            int firstTokenId = variant * 100_000;
            int[] inputIds = IntStream.range(firstTokenId, firstTokenId + TOKEN_COUNT).toArray();
            inputs.add(inputIds);
            expectedResults.add(BlockCacheKeyCalculator.calculate(inputIds, BLOCK_SIZE));
        }

        ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);
        CountDownLatch startGate = new CountDownLatch(1);
        List<Future<List<Long>>> futures = new ArrayList<>(REQUEST_COUNT);
        try {
            for (int requestIndex = 0; requestIndex < REQUEST_COUNT; requestIndex++) {
                int inputIndex = requestIndex % INPUT_VARIANT_COUNT;
                futures.add(executor.submit(() -> {
                    startGate.await();
                    return BlockCacheKeyCalculator.calculate(inputs.get(inputIndex), BLOCK_SIZE);
                }));
            }

            startGate.countDown();
            for (int requestIndex = 0; requestIndex < REQUEST_COUNT; requestIndex++) {
                int inputIndex = requestIndex % INPUT_VARIANT_COUNT;
                assertEquals(
                        expectedResults.get(inputIndex),
                        futures.get(requestIndex).get(30, TimeUnit.SECONDS));
            }
        } finally {
            startGate.countDown();
            executor.shutdownNow();
            executor.awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
