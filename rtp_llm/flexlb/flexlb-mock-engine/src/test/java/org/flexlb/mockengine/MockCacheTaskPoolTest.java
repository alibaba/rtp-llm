package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

class MockCacheTaskPoolTest {
    @Test void idlePoolsAndVirtualDrainLeaveCarriersAvailable(
            @org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
        var log = directory.resolve("carrier-probe.log");
        var child = new ProcessBuilder(System.getProperty("java.home") + "/bin/java",
                "-Djdk.virtualThreadScheduler.parallelism=2", "-Djdk.virtualThreadScheduler.maxPoolSize=4",
                "-cp", System.getProperty("java.class.path"), CarrierProbe.class.getName())
                .redirectErrorStream(true).redirectOutput(log.toFile()).start();
        try {
            assertTrue(child.waitFor(20, TimeUnit.SECONDS), "cache pools blocked the child JVM");
            assertEquals(0, child.exitValue(), () -> {
                try { return java.nio.file.Files.readString(log); }
                catch (java.io.IOException error) { return error.toString(); }
            });
        } finally {
            child.destroyForcibly();
            assertTrue(child.waitFor(5, TimeUnit.SECONDS), "carrier probe did not stop");
        }
    }

    public static final class CarrierProbe {
        public static void main(String[] args) throws Exception {
            var pools = new java.util.ArrayList<MockCacheTaskPool>();
            var release = new CountDownLatch(1);
            try {
                for (int i = 0; i < 8; i++) pools.add(new MockCacheTaskPool(1, 0));
                Thread.sleep(250); // Let idle workers enter their queue wait before probing unrelated work.
                var available = new java.util.concurrent.CompletableFuture<Void>();
                Thread.ofVirtual().start(() -> available.complete(null));
                available.get(2, TimeUnit.SECONDS);
                var entered = new CountDownLatch(4);
                for (var pool : pools.subList(0, 4)) pool.submit(true, () -> {
                        entered.countDown();
                        try { release.await(); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
                    }, Long.MAX_VALUE, () -> {});
                if (!entered.await(2, TimeUnit.SECONDS)) throw new AssertionError("cache work did not start");
                var drained = new CountDownLatch(4);
                var waiters = new java.util.ArrayList<Thread>();
                for (var pool : pools.subList(0, 4)) waiters.add(Thread.ofVirtual().start(() -> {
                    pool.waitForIdle(); drained.countDown();
                }));
                for (var waiter : waiters) {
                    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
                    while (waiter.getState() != Thread.State.WAITING && System.nanoTime() < deadline) Thread.sleep(1);
                    if (waiter.getState() != Thread.State.WAITING) throw new AssertionError("drain did not enter its wait");
                }
                Thread.ofVirtual().start(release::countDown);
                if (!drained.await(2, TimeUnit.SECONDS)) throw new AssertionError("virtual drain starved its release task");
            } finally {
                release.countDown();
                for (var pool : pools) pool.close();
            }
        }
    }

    @Test void reservesLoadSlotsAndDispatchesCompletionThenFourLoadsThenBackground() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var seen = new CopyOnWriteArrayList<String>(); var done = new CountDownLatch(71);
        try (var pool = new MockCacheTaskPool(1, 70)) {
            assertTrue(pool.submit(true, () -> { entered.countDown(); try { release.await(); } catch (InterruptedException error) { Thread.currentThread().interrupt(); } }, Long.MAX_VALUE, () -> {}));
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            for (int i = 0; i < 6; i++) assertTrue(pool.submit(false, () -> { seen.add("B"); done.countDown(); }, Long.MAX_VALUE, () -> {}));
            assertFalse(pool.submit(false, () -> fail("reserved slot"), Long.MAX_VALUE, () -> {}));
            for (int i = 0; i < 64; i++) assertTrue(pool.submit(true, () -> { seen.add("L"); done.countDown(); }, Long.MAX_VALUE, () -> {}));
            assertFalse(pool.submit(true, () -> fail("full"), Long.MAX_VALUE, () -> {}));
            assertTrue(pool.complete(() -> { seen.add("C"); done.countDown(); }));
            assertArrayEquals(new int[]{64, 6, 1}, pool.sizes());
            release.countDown();
            assertTrue(done.await(2, TimeUnit.SECONDS));
            // The running LOAD counted as the first in its burst.
            assertEquals(List.of("C", "L", "L", "L", "B"), seen.subList(0, 5));
        } finally { release.countDown(); }
    }
    @Test void deadlineExpiresBeforeExecutionAndUnboundedZeroAcceptsBothClasses() throws Exception {
        var expired = new CountDownLatch(1);
        try (var pool = new MockCacheTaskPool(1, 0)) {
            assertTrue(pool.submit(true, () -> fail("expired body ran"), System.nanoTime() - 1, expired::countDown));
            assertTrue(expired.await(1, TimeUnit.SECONDS));
            var done = new CountDownLatch(1);
            assertTrue(pool.submit(false, done::countDown, Long.MAX_VALUE, () -> {}));
            assertTrue(done.await(1, TimeUnit.SECONDS));
        }
    }
    @Test void syncTimeoutFailsWholeLoadAndQueuesRecover() throws Exception {
        try (var transfer = new MockHostTransfer(new MockHostTransfer.Policy(1, 1, 1, 1, 20))) {
            var failed = transfer.submit(true, 1, 50);
            assertThrows(java.util.concurrent.ExecutionException.class, () -> failed.get(1, TimeUnit.SECONDS));
            assertNull(transfer.submit(true, 1, 0).get(1, TimeUnit.SECONDS));
        }
    }
    @Test void stoppedAdmissionStillAcceptsAndDrainsSettlement() {
        try (var pool = new MockCacheTaskPool(1, 1)) {
            var settled = new java.util.concurrent.atomic.AtomicBoolean();
            pool.stopAdmission();
            assertFalse(pool.submit(true, () -> fail("stopped admission"), Long.MAX_VALUE, () -> {}));
            assertTrue(pool.complete(() -> settled.set(true)));
            pool.waitForIdle();
            assertTrue(settled.get());
        }
    }

    @Test void rawCloseDropsQueuedBodiesWithoutTimeoutAndJoinsRunningBody() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var interrupted = new java.util.concurrent.atomic.AtomicBoolean();
        var queuedRan = new java.util.concurrent.atomic.AtomicBoolean();
        var expired = new java.util.concurrent.atomic.AtomicBoolean();
        var pool = new MockCacheTaskPool(1, 1);
        var close = new java.util.concurrent.CompletableFuture<Void>();
        try {
            assertTrue(pool.submit(true, () -> {
                entered.countDown();
                try { release.await(); } catch (InterruptedException error) { interrupted.set(true); }
            }, Long.MAX_VALUE, () -> {}));
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            assertTrue(pool.submit(false, () -> queuedRan.set(true), Long.MAX_VALUE, () -> expired.set(true)));
            Thread.ofVirtual().start(() -> { pool.close(); close.complete(null); });
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (pool.sizes()[1] != 0 && System.nanoTime() < deadline) Thread.sleep(1);
            assertArrayEquals(new int[]{0, 0, 0}, pool.sizes());
            assertFalse(close.isDone(), "close must join the running task");
            assertFalse(interrupted.get());
            release.countDown(); close.get(2, TimeUnit.SECONDS);
            assertFalse(queuedRan.get()); assertFalse(expired.get());
        } finally { release.countDown(); pool.close(); }
    }

    @Test void ownerCloseDrainsTransfersAndTheirSettlementBeforeClosingPools() throws Exception {
        var transfer = new MockHostTransfer(new MockHostTransfer.Policy(1, 0, 0, 1, 3000));
        var load = transfer.submit(true, 3, 50);
        var store = transfer.submit(false, 3, 50);
        var closed = new java.util.concurrent.CompletableFuture<Void>();
        try {
            Thread.ofVirtual().start(() -> { transfer.close(); closed.complete(null); });
            closed.get(3, TimeUnit.SECONDS);
            assertNull(load.get()); assertNull(store.get());
            assertEquals(0, transfer.snapshot().get("in_flight"));
            assertTrue(transfer.submit(true, 1, 0).isCompletedExceptionally());
        } finally { transfer.close(); }
    }

}
