package org.flexlb.balance.endpoint;

import org.flexlb.balance.projection.WorkSnapshot;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillStateSnapshotTest {
    private final AtomicLong clock = new AtomicLong(100);
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillState state = new PrefillState(
            lock, PrefillActiveIndex.disabled(), clock::get, () -> { });

    @Test
    void concurrentReadersShareACompleteImmutableMaterialization() throws Exception {
        var second = state.tryRegisterDirect(2, 20, 0).reservation();
        var first = state.tryRegisterDirect(1, 10, 0).reservation();
        var captured = capture();
        try (var executor = Executors.newFixedThreadPool(8)) {
            CountDownLatch start = new CountDownLatch(1);
            List<Future<WorkSnapshot>> readers = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                readers.add(executor.submit(() -> {
                    assertTrue(start.await(5, TimeUnit.SECONDS));
                    return captured.work().materialize();
                }));
            }
            start.countDown();
            WorkSnapshot shared = readers.getFirst().get(5, TimeUnit.SECONDS);
            for (var reader : readers) {
                assertSame(shared, reader.get(5, TimeUnit.SECONDS));
            }
            assertEquals(List.of(1L, 2L), shared.requests().stream()
                    .map(WorkSnapshot.RequestWork::requestId).toList());
            second.close();
            first.close();
            assertEquals(2, shared.requests().size());
            assertTrue(capture().work().materialize().requests().isEmpty());
        }
    }

    @Test
    void clockRollbackRecapturesWorkWithoutMutatingEarlierSnapshots() {
        state.tryRegisterDirect(1, 10, 0);
        var original = capture();
        clock.set(101);
        assertSame(original.work(), capture().work());
        clock.set(90);
        var rebased = capture();
        assertNotSame(original.work(), rebased.work());
        assertEquals(100, original.work().materialize().capturedAtMs());
        assertEquals(90, rebased.work().materialize().capturedAtMs());
        assertEquals(rebased.capturedAtMs(), rebased.work().materialize().capturedAtMs());
        assertEquals(original.work().materialize().requests(), rebased.work().materialize().requests());
    }

    private PrefillState.Snapshot capture() {
        lock.lock();
        try {
            return state.snapshotUnderLock();
        } finally {
            lock.unlock();
        }
    }
}
