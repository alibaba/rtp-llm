package org.flexlb.balance.delivery;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchIdGeneratorTest {

    private static final long EPOCH_MILLIS = 1_577_836_800_000L;

    @Test
    void preservesTimestampAndProcessFieldsWithPositiveIds() {
        AtomicLong clock = new AtomicLong(EPOCH_MILLIS + 100_123L);
        BatchIdGenerator generator = new BatchIdGenerator(0xABCL, clock::get);

        long first = generator.nextId();
        long second = generator.nextId();
        assertEquals(100L, first >>> 33);
        assertEquals(0xABCL, (first >>> 21) & 0xFFFL);
        assertEquals(first + 1L, second);
        assertTrue(first > 0L);

        clock.addAndGet(1_000L);
        long nextSecond = generator.nextId();
        assertEquals(101L, nextSecond >>> 33);
        assertEquals(0L, nextSecond & 0x1F_FFFFL);
        assertTrue(nextSecond > second);
    }

    @Test
    void skipsTheZeroIdAtTheEpoch() {
        BatchIdGenerator generator = new BatchIdGenerator(0L, () -> EPOCH_MILLIS);

        assertEquals(1L, generator.nextId());
        assertEquals(2L, generator.nextId());
    }

    @Test
    void rejectsTimeOutsideThePositiveTimestampRangeWithoutChangingState() {
        AtomicLong clock = new AtomicLong(EPOCH_MILLIS - 1L);
        BatchIdGenerator generator = new BatchIdGenerator(4_095L, clock::get);

        assertThrows(IllegalStateException.class, generator::nextId);
        clock.set(EPOCH_MILLIS + 1_073_741_823_000L);
        long lastSecondId = generator.nextId();
        assertTrue(lastSecondId > 0L);
        assertEquals(1_073_741_823L, lastSecondId >>> 33);

        clock.addAndGet(1_000L);
        assertThrows(IllegalStateException.class, generator::nextId);
        clock.addAndGet(-1_000L);
        assertEquals(lastSecondId + 1L, generator.nextId());
    }

    @Test
    void rejectsExhaustionUntilTheClockAdvances() {
        AtomicLong clock = new AtomicLong(EPOCH_MILLIS + 1_000L);
        BatchIdGenerator generator = new BatchIdGenerator(7L, clock::get);
        long previous = 0L;
        for (int i = 0; i < 2_097_152; i++) {
            long current = generator.nextId();
            assertTrue(current > previous, "IDs must not repeat within one second");
            previous = current;
        }
        assertEquals(0x1F_FFFFL, previous & 0x1F_FFFFL);
        assertTimeoutPreemptively(Duration.ofSeconds(1L), () -> {
            assertThrows(IllegalStateException.class, generator::nextId);
            assertThrows(IllegalStateException.class, generator::nextId);
        });

        clock.addAndGet(1_000L);
        long recovered = generator.nextId();
        assertTrue(recovered > previous);
        assertEquals(0L, recovered & 0x1F_FFFFL);
    }

    @Test
    void rejectsTimestampRollbackAndContinuesWhenTimeRecovers() {
        AtomicLong clock = new AtomicLong(EPOCH_MILLIS + 5_000L);
        BatchIdGenerator generator = new BatchIdGenerator(8L, clock::get);
        long first = generator.nextId();

        clock.addAndGet(-1_000L);
        assertTimeoutPreemptively(Duration.ofSeconds(1L), () -> {
            assertThrows(IllegalStateException.class, generator::nextId);
            assertThrows(IllegalStateException.class, generator::nextId);
        });
        clock.addAndGet(1_000L);
        assertEquals(first + 1L, generator.nextId());

        clock.addAndGet(1_000L);
        assertTrue(generator.nextId() > first);
    }

    @Test
    void subsecondClockChangesKeepAllocatingDistinctIds() {
        AtomicLong clock = new AtomicLong(EPOCH_MILLIS + 5_999L);
        BatchIdGenerator generator = new BatchIdGenerator(8L, clock::get);
        long first = generator.nextId();

        clock.set(EPOCH_MILLIS + 5_000L);
        assertEquals(first + 1L, generator.nextId());
    }

    @Test
    void concurrentCallersReceiveDistinctPositiveIds() throws Exception {
        BatchIdGenerator generator = new BatchIdGenerator(9L, () -> EPOCH_MILLIS + 1_000L);
        int callers = 8;
        int idsPerCaller = 5_000;
        CountDownLatch ready = new CountDownLatch(callers);
        CountDownLatch start = new CountDownLatch(1);
        var executor = Executors.newFixedThreadPool(callers);
        try {
            List<Future<List<Long>>> results = new ArrayList<>();
            for (int i = 0; i < callers; i++) {
                results.add(executor.submit(() -> {
                    ready.countDown();
                    start.await();
                    List<Long> generated = new ArrayList<>(idsPerCaller);
                    for (int j = 0; j < idsPerCaller; j++) {
                        generated.add(generator.nextId());
                    }
                    return generated;
                }));
            }
            assertTrue(ready.await(10L, TimeUnit.SECONDS));
            start.countDown();
            HashSet<Long> generated = new HashSet<>();
            for (Future<List<Long>> result : results) {
                for (long id : result.get(10L, TimeUnit.SECONDS)) {
                    assertTrue(id > 0L);
                    assertTrue(generated.add(id), "concurrent callers received the same batch ID");
                }
            }
            assertEquals(callers * idsPerCaller, generated.size());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void generatorInstancesOwnTheirSequences() {
        BatchIdGenerator firstProcess = new BatchIdGenerator(10L, () -> EPOCH_MILLIS + 1_000L);
        BatchIdGenerator secondProcess = new BatchIdGenerator(11L, () -> EPOCH_MILLIS + 1_000L);

        firstProcess.nextId();
        firstProcess.nextId();
        assertEquals(0L, secondProcess.nextId() & 0x1F_FFFFL);
    }
}
