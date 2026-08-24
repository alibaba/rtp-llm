package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * 并发压力小测：多线程 register/observe/settle 全生命周期——
 * 无丢失、无泄漏，计数守恒（终态数 == 墓碑吸收数），
 * 快照零锁读在并发下可安全完成。
 */
class StateLedgerConcurrencyTest {

    @Test
    void concurrentLifecyclesConserveCounts() throws Exception {
        StateLedgerConfig cfg = new StateLedgerConfig(60_000L, 300_000L, 8);
        StateLedger ledger = new StateLedger(cfg);
        long pGen = ledger.newGeneration(TestEndpoints.ep(1L, StateRole.PREFILL, 0L));
        long dGen = ledger.newGeneration(TestEndpoints.ep(2L, StateRole.DECODE, 0L));
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);

        int n = 160;
        int threads = 8;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>(n);
        // 零锁快照并发读者：生命周期进行中持续读已发布快照（无锁无异常）
        CountDownLatch stopReader = new CountDownLatch(1);
        Future<?> reader = pool.submit(() -> {
            while (stopReader.getCount() > 0) {
                LedgerSnapshot s = ledger.snapshot();
                assertTrue(s.prefill().inflight() >= 0);
                assertTrue(s.decode().activeTotal() >= 0);
                ledger.crossSide();
            }
            return null;
        });

        for (int i = 0; i < n; i++) {
            final long id = 1_000L + i;
            final long batch = 50L + (i % 3);
            final GenerationTriple pBinding = new GenerationTriple(1, pGen, batch);
            final GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
            futures.add(pool.submit(() -> {
                start.await();
                // 请求内顺序推进（跨请求并发）
                assertEquals(RegisterResult.OK, ledger.prefill().register(id, batch));
                ledger.prefill().onQueued(id);
                ledger.prefill().onDispatching(id, batch);
                assertTrue(ledger.prefill().onDispatched(id, pBinding));
                ledger.observe(TestEndpoints.runningOnly(pEp, id, 1_000L + id,
                        TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, batch, 128L, 1L)));
                assertEquals(ReserveResult.OK, ledger.decode().reserve(id, 256L, 1024L, dBinding));
                assertTrue(ledger.decode().onDispatched(id, dBinding));
                ledger.observe(TestEndpoints.runningOnly(dEp, id, 2_000L + id,
                        TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 1024L, 1L)));
                ledger.observe(TestEndpoints.runningOnly(dEp, id, 3_000L + id,
                        TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, 1024L, 2L)));
                ledger.observe(TestEndpoints.finishedOnly(dEp, id, 4_000L + id,
                        TestEndpoints.finished(id, StateRole.DECODE, 0, 4_000L + id, 3L)));
                return null;
            }));
        }
        start.countDown();
        for (Future<?> f : futures) {
            f.get(60L, TimeUnit.SECONDS);
        }
        stopReader.countDown();
        reader.get(10L, TimeUnit.SECONDS);
        pool.shutdown();
        assertTrue(pool.awaitTermination(10L, TimeUnit.SECONDS));

        // 守恒：全部终态（D finished → 因果闭包收缩 P）——两侧活跃归零
        ledger.prefill().refreshSnapshot();
        ledger.decode().refreshSnapshot();
        assertEquals(0L, ledger.prefill().snapshot().inflight(), "P 侧应无泄漏");
        assertEquals(0L, ledger.decode().snapshot().activeTotal(), "D 侧应无泄漏");

        // 终态数 == 墓碑吸收数（两侧各 n）
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(n, s.prefillTombstones());
        assertEquals(n, s.decodeTombstones());

        // 无 unknown、无迟到、无跨代拒绝（请求内有序，请求间独立）
        assertEquals(0L, s.unknownRunningEvents());
        assertEquals(0L, s.unknownFinishedEvents());
        assertEquals(0L, s.lateEventsAbsorbed());
        assertEquals(0L, s.crossGenerationRejects());

        // 并发转换后：计数器增量账与全量重算一致（零锁快照一致性）
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /** 并发重复终局：settle CAS 单出口——同一请求多次 settle 只有一次生效。 */
    @Test
    void concurrentSettleIsCasSingleExit() throws Exception {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(TestEndpoints.ep(1L, StateRole.PREFILL, 0L));
        GenerationTriple pBinding = new GenerationTriple(1, pGen, -1L);
        long id = 42L;
        assertEquals(RegisterResult.OK, ledger.prefill().register(id, -1L));
        ledger.prefill().onQueued(id);
        ledger.prefill().onDispatching(id, -1L);
        assertTrue(ledger.prefill().onDispatched(id, pBinding));

        int threads = 8;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<Boolean>> results = new ArrayList<>(threads);
        CountDownLatch start = new CountDownLatch(1);
        TerminalOutcome outcome =
                new TerminalOutcome(TerminalState.COMPLETED, TerminalReason.SUCCEEDED, "");
        for (int i = 0; i < threads; i++) {
            results.add(pool.submit(() -> {
                start.await();
                return ledger.prefill().settle(id, outcome, SettleReason.ENGINE_FINISHED);
            }));
        }
        start.countDown();
        int successes = 0;
        for (Future<Boolean> f : results) {
            if (f.get(10L, TimeUnit.SECONDS)) {
                successes++;
            }
        }
        pool.shutdown();
        assertTrue(pool.awaitTermination(10L, TimeUnit.SECONDS));

        // CAS 单出口：恰好一次成功
        assertEquals(1, successes);
        assertTrue(ledger.prefill().get(id).isEmpty());
        assertEquals(1L, ledger.snapshot().prefillTombstones());
        ledger.prefill().refreshSnapshot();
        assertEquals(0L, ledger.prefill().snapshot().inflight());
        assertTrue(ledger.auditAndDrift().clean());
    }
}
