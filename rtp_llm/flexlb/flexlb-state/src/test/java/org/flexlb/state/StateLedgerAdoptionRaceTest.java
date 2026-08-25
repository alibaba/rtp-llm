package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * 收养路径并发回归（真机 master 重启演练 per-EP 计数簿漂移的竞态裁决）：
 * 多线程对同一批 requestId 交错执行「收养 + 相位观察 + 终局 + 重复收养」，
 * 结束后增量账与全量重算对账零偏差、活跃条目无泄漏、墓碑数守恒
 * （终局恰一次、每 requestId 恰一条墓碑、迟到事件被吸收而非新建条目）。
 *
 * <p>三阶段：</p>
 * <ol>
 *   <li><b>混沌交错</b>：6 线程 × 固定轮次，随机 requestId（新鲜/近期复用
 *       混合）上随机执行收养 / 重复收养 / 相位观察 / 引擎 finished 终局 /
 *       本地结算 / 迟到收养重放 / 释放（D 侧）小序列——覆盖收养可见窗口、
 *       观察入账窗口、终局移除窗口的任意交错。</li>
 *   <li><b>定向窗口竞速</b>：收养线程与出账线程（引擎 finished / 本地
 *       settle / 相位推进 / 主动释放 / 重复收养）对同一新鲜 requestId
 *       逐个对齐竞速，出账侧持续锤击直到条目可见即触发——最大化压缩
 *       「条目对外可见（putIfAbsent）与收养入账（端点簿建桶）之间」窗口
 *       的命中概率。该窗口内出账若先于入账执行，桶缺失 no-op 的出账
 *       语义下入账将永久悬挂（端点簿恒高/低固定值）。</li>
 *   <li><b>兜底收敛</b>：残留活跃条目确定性终局（CAS 单出口恰一胜者），
 *       随后做全量断言并打印 drift 明细。</li>
 * </ol>
 */
class StateLedgerAdoptionRaceTest {

    /** 混沌阶段线程数与每线程轮次（≥2000 轮）。 */
    private static final int CHAOS_THREADS = 6;
    private static final int CHAOS_ROUNDS_PER_THREAD = 4_000;

    /** 定向竞速每类 requestId 数（8 类 × 6000 = 48000 次窗口竞速）。 */
    private static final int RACE_IDS_PER_KIND = 6_000;

    /**
     * 出账锤击时间窗（防御性；条目出现后出账必胜，正常命中在微秒级）。
     * 时间窗兜底收养线程的调度延迟：次数上限会在收养条目尚未对外可见时
     * 被空转锤击提前耗尽（对不存在条目的锤击是 O(1) 快速失败）；窗内
     * 未命中时由主线程在收养完成后确定性补刀（幂等，不影响竞速覆盖）。
     */
    private static final long HAMMER_WINDOW_NANOS = TimeUnit.SECONDS.toNanos(5L);

    @Test
    void concurrentAdoptionRacesKeepBooksExact() throws Exception {
        // 大墓碑保持期：竞速全程墓碑不过期（过期会恢复收养、干扰守恒断言语义）
        StateLedgerConfig cfg = new StateLedgerConfig(600_000L, 300_000L, 8);
        StateLedger ledger = new StateLedger(cfg);
        long pGen1 = ledger.newGeneration(TestEndpoints.ep(1L, StateRole.PREFILL, 0L));
        long pGen3 = ledger.newGeneration(TestEndpoints.ep(3L, StateRole.PREFILL, 0L));
        long dGen2 = ledger.newGeneration(TestEndpoints.ep(2L, StateRole.DECODE, 0L));
        long dGen4 = ledger.newGeneration(TestEndpoints.ep(4L, StateRole.DECODE, 0L));
        TestEndpoints.Endpoint pEp1 = TestEndpoints.ep(1L, StateRole.PREFILL, pGen1);
        TestEndpoints.Endpoint pEp3 = TestEndpoints.ep(3L, StateRole.PREFILL, pGen3);
        TestEndpoints.Endpoint dEp2 = TestEndpoints.ep(2L, StateRole.DECODE, dGen2);
        TestEndpoints.Endpoint dEp4 = TestEndpoints.ep(4L, StateRole.DECODE, dGen4);

        AtomicLong versionGen = new AtomicLong(1_000_000L);
        AtomicLong pIdGen = new AtomicLong(100_000L);
        AtomicLong dIdGen = new AtomicLong(200_000L);
        Set<Long> pIds = ConcurrentHashMap.newKeySet();
        Set<Long> dIds = ConcurrentHashMap.newKeySet();

        ExecutorService pool = Executors.newFixedThreadPool(CHAOS_THREADS);
        try {
            // ---- 阶段一：混沌交错 ----
            CountDownLatch start = new CountDownLatch(1);
            List<Future<?>> workers = new ArrayList<>(CHAOS_THREADS);
            for (int t = 0; t < CHAOS_THREADS; t++) {
                workers.add(pool.submit(() -> {
                    start.await();
                    Deque<Long> recentP = new ArrayDeque<>();
                    Deque<Long> recentD = new ArrayDeque<>();
                    ThreadLocalRandom rnd = ThreadLocalRandom.current();
                    for (int round = 0; round < CHAOS_ROUNDS_PER_THREAD; round++) {
                        if (rnd.nextBoolean()) {
                            chaosRoundPrefill(ledger, pEp1, pEp3,
                                    pickId(pIdGen, pIds, recentP, rnd), versionGen, rnd);
                        } else {
                            chaosRoundDecode(ledger, dEp2, dEp4,
                                    pickId(dIdGen, dIds, recentD, rnd), versionGen, rnd);
                        }
                    }
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> w : workers) {
                w.get(120L, TimeUnit.SECONDS);
            }
            // 阶段一即时对账：活跃条目在场时增量账也必须与全量重算一致
            assertTrue(ledger.auditAndDrift().clean(),
                    () -> "混沌阶段 drift:\n" + ledger.auditAndDrift());

            long basePTombstones = ledger.snapshot().prefillTombstones();
            long baseDTombstones = ledger.snapshot().decodeTombstones();

            // ---- 阶段二：定向窗口竞速（P 侧四类 + D 侧四类） ----
            raceAdoptionVsFinishedObserve(ledger, pEp1, StateRole.PREFILL, 300_000L, pool, versionGen);
            raceAdoptionVsLocalSettle(ledger, pEp1, true, 310_000L, pool, versionGen);
            raceAdoptionVsPhaseAdvance(ledger, pEp1, StateRole.PREFILL, 320_000L, pool, versionGen);
            raceDoubleAdoption(ledger, pEp1, StateRole.PREFILL, 330_000L, pool, versionGen);
            raceAdoptionVsFinishedObserve(ledger, dEp2, StateRole.DECODE, 400_000L, pool, versionGen);
            raceAdoptionVsLocalSettle(ledger, dEp2, false, 410_000L, pool, versionGen);
            raceAdoptionVsPhaseAdvance(ledger, dEp2, StateRole.DECODE, 420_000L, pool, versionGen);
            raceAdoptionVsRelease(ledger, dEp2, 430_000L, pool, versionGen);

            long afterRacePTombstones = ledger.snapshot().prefillTombstones();
            long afterRaceDTombstones = ledger.snapshot().decodeTombstones();

            // ---- 阶段三：兜底收敛（混沌阶段残留条目确定性终局） ----
            long extraPTombstones = settleAllRemaining(ledger, pIds, true);
            long extraDTombstones = settleAllRemaining(ledger, dIds, false);

            // ---- 全量断言 ----
            CounterDriftReport drift = ledger.auditAndDrift();
            assertTrue(drift.clean(),
                    () -> "收养竞速 drift 明细（增量账 vs 全量重算）:\n" + drift);

            ledger.prefill().refreshSnapshot();
            ledger.decode().refreshSnapshot();
            assertEquals(0L, ledger.prefill().snapshot().inflight(), "P 侧活跃归零（无泄漏）");
            assertEquals(0L, ledger.decode().snapshot().activeTotal(), "D 侧活跃归零（无泄漏）");

            // 墓碑守恒：定向竞速每 id 恰一次终局恰一条墓碑
            // （P 四类：finished / 本地 settle / 推进+兜底 / 双收养+兜底；
            //   D 三类落墓碑 + 释放类不落墓碑）
            LedgerSnapshot s = ledger.snapshot();
            assertEquals(basePTombstones + 4L * RACE_IDS_PER_KIND + extraPTombstones,
                    s.prefillTombstones(), "P 侧墓碑守恒（终局恰一次、每 id 恰一条）");
            assertEquals(baseDTombstones + 3L * RACE_IDS_PER_KIND + extraDTombstones,
                    s.decodeTombstones(), "D 侧墓碑守恒（释放不落墓碑、终局恰一条）");
        } finally {
            pool.shutdown();
            assertTrue(pool.awaitTermination(30L, TimeUnit.SECONDS));
        }
    }

    // ---- 阶段一：混沌交错 ----

    /** id 选取：1/4 概率新鲜发号，其余从近期窗口复用（制造同 id 并发交错）。 */
    private static long pickId(AtomicLong idGen, Set<Long> ids, Deque<Long> recent,
                               ThreadLocalRandom rnd) {
        if (recent.isEmpty() || rnd.nextInt(4) == 0) {
            long fresh = idGen.incrementAndGet();
            ids.add(fresh);
            recent.addLast(fresh);
            if (recent.size() > 16) {
                recent.pollFirst();
            }
            return fresh;
        }
        Long[] window = recent.toArray(new Long[0]);
        return window[rnd.nextInt(window.length)];
    }

    private static void chaosRoundPrefill(StateLedger ledger, TestEndpoints.Endpoint epMain,
                                          TestEndpoints.Endpoint epAlt, long id,
                                          AtomicLong versionGen, ThreadLocalRandom rnd) {
        EnginePhase ph = randomPhase(rnd);
        // 少量走另一端点：世代/绑定不匹配 → 整报拒绝（REJECT_GENERATION），不动账
        TestEndpoints.Endpoint ep = rnd.nextInt(10) < 3 ? epAlt : epMain;
        TerminalOutcome completed = new TerminalOutcome(TerminalState.COMPLETED,
                TerminalReason.SUCCEEDED, "");
        switch (rnd.nextInt(7)) {
            case 0 -> adoptRunning(ledger, ep, id, StateRole.PREFILL, ph, versionGen);
            case 1 -> {
                adoptRunning(ledger, ep, id, StateRole.PREFILL, ph, versionGen);
                adoptRunning(ledger, ep, id, StateRole.PREFILL, randomPhase(rnd), versionGen);
            }
            case 2 -> {
                adoptRunning(ledger, ep, id, StateRole.PREFILL, ph, versionGen);
                observeRunning(ledger, ep, id, StateRole.PREFILL, EnginePhase.RUNNING, versionGen);
            }
            case 3 -> observeFinished(ledger, ep, id, StateRole.PREFILL, versionGen);
            case 4 -> {
                observeFinished(ledger, ep, id, StateRole.PREFILL, versionGen);
                adoptRunning(ledger, ep, id, StateRole.PREFILL, ph, versionGen);
            }
            case 5 -> {
                ledger.prefill().settle(id, completed, SettleReason.ENGINE_FINISHED);
                adoptRunning(ledger, ep, id, StateRole.PREFILL, ph, versionGen);
            }
            case 6 -> {
                observeRunning(ledger, ep, id, StateRole.PREFILL, ph, versionGen);
                observeFinished(ledger, ep, id, StateRole.PREFILL, versionGen);
            }
            default -> throw new AssertionError("unreachable");
        }
    }

    private static void chaosRoundDecode(StateLedger ledger, TestEndpoints.Endpoint epMain,
                                         TestEndpoints.Endpoint epAlt, long id,
                                         AtomicLong versionGen, ThreadLocalRandom rnd) {
        EnginePhase ph = randomPhase(rnd);
        TestEndpoints.Endpoint ep = rnd.nextInt(10) < 3 ? epAlt : epMain;
        TerminalOutcome completed = new TerminalOutcome(TerminalState.COMPLETED,
                TerminalReason.SUCCEEDED, "");
        switch (rnd.nextInt(9)) {
            case 0 -> adoptRunning(ledger, ep, id, StateRole.DECODE, ph, versionGen);
            case 1 -> {
                adoptRunning(ledger, ep, id, StateRole.DECODE, ph, versionGen);
                adoptRunning(ledger, ep, id, StateRole.DECODE, randomPhase(rnd), versionGen);
            }
            case 2 -> {
                adoptRunning(ledger, ep, id, StateRole.DECODE, ph, versionGen);
                observeRunning(ledger, ep, id, StateRole.DECODE, EnginePhase.RUNNING, versionGen);
            }
            case 3 -> observeFinished(ledger, ep, id, StateRole.DECODE, versionGen);
            case 4 -> {
                observeFinished(ledger, ep, id, StateRole.DECODE, versionGen);
                adoptRunning(ledger, ep, id, StateRole.DECODE, ph, versionGen);
            }
            case 5 -> {
                ledger.decode().settle(id, completed, SettleReason.ENGINE_FINISHED);
                adoptRunning(ledger, ep, id, StateRole.DECODE, ph, versionGen);
            }
            case 6 -> {
                observeRunning(ledger, ep, id, StateRole.DECODE, ph, versionGen);
                observeFinished(ledger, ep, id, StateRole.DECODE, versionGen);
            }
            case 7 -> {
                adoptRunning(ledger, ep, id, StateRole.DECODE, ph, versionGen);
                ledger.decode().release(id);
            }
            case 8 -> ledger.decode().release(id);
            default -> throw new AssertionError("unreachable");
        }
    }

    private static EnginePhase randomPhase(ThreadLocalRandom rnd) {
        EnginePhase[] phases = {EnginePhase.RECEIVED, EnginePhase.KV_ALLOCATED, EnginePhase.RUNNING};
        return phases[rnd.nextInt(phases.length)];
    }

    private static void adoptRunning(StateLedger ledger, TestEndpoints.Endpoint ep, long id,
                                     StateRole side, EnginePhase phase, AtomicLong versionGen) {
        // statusMs 用真实墙钟：墓碑过期判定基于真实时钟，假时间戳会让墓碑瞬间“过期”
        ledger.observeAdopting(TestEndpoints.runningOnly(ep, 1L, System.currentTimeMillis(),
                TestEndpoints.running(id, side, phase, -1L, 64L, versionGen.incrementAndGet())));
    }

    private static void observeRunning(StateLedger ledger, TestEndpoints.Endpoint ep, long id,
                                       StateRole side, EnginePhase phase, AtomicLong versionGen) {
        ledger.observe(TestEndpoints.runningOnly(ep, 2L, System.currentTimeMillis(),
                TestEndpoints.running(id, side, phase, -1L, 96L, versionGen.incrementAndGet())));
    }

    private static void observeFinished(StateLedger ledger, TestEndpoints.Endpoint ep, long id,
                                        StateRole side, AtomicLong versionGen) {
        ledger.observe(TestEndpoints.finishedOnly(ep, 3L, System.currentTimeMillis(),
                TestEndpoints.finished(id, side, 0, System.currentTimeMillis(), versionGen.incrementAndGet())));
    }

    // ---- 阶段二：定向窗口竞速 ----

    /**
     * 收养 vs 引擎 finished 锤击终局：收养线程收养（执行位）同时，终局
     * 线程持续 observe finished（条目出现前计 unknown 无害；出现后恰一次
     * 终局移除并落墓碑）——终局出账与收养入账的窗口竞速。
     */
    private void raceAdoptionVsFinishedObserve(StateLedger ledger, TestEndpoints.Endpoint ep,
                                               StateRole side, long baseId,
                                               ExecutorService pool, AtomicLong versionGen) throws Exception {
        boolean pSide = side == StateRole.PREFILL;
        for (int i = 0; i < RACE_IDS_PER_KIND; i++) {
            long id = baseId + i;
            CountDownLatch go = new CountDownLatch(1);
            Future<?> adopter = pool.submit(() -> {
                go.await();
                adoptRunning(ledger, ep, id, side, EnginePhase.RUNNING, versionGen);
                return null;
            });
            Future<?> settler = pool.submit(() -> {
                go.await();
                long deadline = System.nanoTime() + HAMMER_WINDOW_NANOS;
                while (ledger.terminalOutcomeOf(id, side).isEmpty()
                        && System.nanoTime() < deadline) {
                    observeFinished(ledger, ep, id, side, versionGen);
                }
                return null;
            });
            go.countDown();
            adopter.get(10L, TimeUnit.SECONDS);
            settler.get(10L, TimeUnit.SECONDS);
            // 补刀：收养完成后条目必在场（幂等——已终局则迟到 finished 被墓碑吸收）
            observeFinished(ledger, ep, id, side, versionGen);
            assertTrue(ledger.terminalOutcomeOf(id, side).isPresent(),
                    "收养条目必被 finished 终局（id=" + id + "）");
            assertTrue(pSide ? ledger.prefill().get(id).isEmpty() : ledger.decode().get(id).isEmpty(),
                    "终局后条目移除（id=" + id + "）");
        }
    }

    /**
     * 收养 vs 本地 settle 锤击终局：本地结算通道（门面 settle → CAS 单出口）
     * 与收养入账的窗口竞速——条目出现后 settle 必胜（无并发移除者）。
     */
    private void raceAdoptionVsLocalSettle(StateLedger ledger, TestEndpoints.Endpoint ep,
                                           boolean pSide, long baseId,
                                           ExecutorService pool, AtomicLong versionGen) throws Exception {
        StateRole side = pSide ? StateRole.PREFILL : StateRole.DECODE;
        TerminalOutcome completed = new TerminalOutcome(TerminalState.COMPLETED,
                TerminalReason.SUCCEEDED, "");
        for (int i = 0; i < RACE_IDS_PER_KIND; i++) {
            long id = baseId + i;
            CountDownLatch go = new CountDownLatch(1);
            Future<?> adopter = pool.submit(() -> {
                go.await();
                adoptRunning(ledger, ep, id, side, EnginePhase.RUNNING, versionGen);
                return null;
            });
            Future<?> settler = pool.submit(() -> {
                go.await();
                long deadline = System.nanoTime() + HAMMER_WINDOW_NANOS;
                while (System.nanoTime() < deadline) {
                    boolean ok = pSide
                            ? ledger.prefill().settle(id, completed, SettleReason.ENGINE_FINISHED)
                            : ledger.decode().settle(id, completed, SettleReason.ENGINE_FINISHED);
                    if (ok) {
                        return null;
                    }
                }
                return null;
            });
            go.countDown();
            adopter.get(10L, TimeUnit.SECONDS);
            settler.get(10L, TimeUnit.SECONDS);
            // 补刀：收养完成后条目必在场（幂等——已终局则结算让位返回 false）
            if (pSide) {
                ledger.prefill().settle(id, completed, SettleReason.ENGINE_FINISHED);
            } else {
                ledger.decode().settle(id, completed, SettleReason.ENGINE_FINISHED);
            }
            assertTrue(ledger.terminalOutcomeOf(id, side).isPresent(),
                    "本地 settle 必终局收养条目（id=" + id + "）");
            assertTrue(pSide ? ledger.prefill().get(id).isEmpty() : ledger.decode().get(id).isEmpty(),
                    "终局后条目移除（id=" + id + "）");
        }
    }

    /**
     * 收养 vs 相位推进锤击：收养线程收养（KV 装载位，留推进空间）同时，
     * 推进线程持续 observe running 执行位——桶迁移账（onPhaseTransition
     * 的出账读桶）与收养入账建桶的窗口竞速；竞速后兜底终局。
     */
    private void raceAdoptionVsPhaseAdvance(StateLedger ledger, TestEndpoints.Endpoint ep,
                                            StateRole side, long baseId,
                                            ExecutorService pool, AtomicLong versionGen) throws Exception {
        boolean pSide = side == StateRole.PREFILL;
        TerminalOutcome completed = new TerminalOutcome(TerminalState.COMPLETED,
                TerminalReason.SUCCEEDED, "");
        for (int i = 0; i < RACE_IDS_PER_KIND; i++) {
            long id = baseId + i;
            CountDownLatch go = new CountDownLatch(1);
            Future<?> adopter = pool.submit(() -> {
                go.await();
                adoptRunning(ledger, ep, id, side, EnginePhase.KV_ALLOCATED, versionGen);
                return null;
            });
            Future<?> advancer = pool.submit(() -> {
                go.await();
                // 固定锤击次数：条目出现即推进（首次推进后其余锤击为同相位重复观察）
                for (int attempt = 0; attempt < 64; attempt++) {
                    observeRunning(ledger, ep, id, side, EnginePhase.RUNNING, versionGen);
                }
                return null;
            });
            go.countDown();
            adopter.get(10L, TimeUnit.SECONDS);
            advancer.get(10L, TimeUnit.SECONDS);
            boolean settled = pSide
                    ? ledger.prefill().settle(id, completed, SettleReason.ENGINE_FINISHED)
                    : ledger.decode().settle(id, completed, SettleReason.ENGINE_FINISHED);
            assertTrue(settled, "兜底终局必胜（推进竞速不终局，id=" + id + "）");
            assertTrue(ledger.terminalOutcomeOf(id, side).isPresent());
        }
    }

    /**
     * 双收养对齐竞速：两个收养线程对同一 requestId 同时收养（不同观察
     * 相位与版本）——putIfAbsent 恰一胜者，入账恰一次；竞速后兜底终局。
     * （入账若未与胜者裁决绑定，重复收养会双重入账造成恒高 1 漂移。）
     */
    private void raceDoubleAdoption(StateLedger ledger, TestEndpoints.Endpoint ep,
                                    StateRole side, long baseId,
                                    ExecutorService pool, AtomicLong versionGen) throws Exception {
        boolean pSide = side == StateRole.PREFILL;
        TerminalOutcome completed = new TerminalOutcome(TerminalState.COMPLETED,
                TerminalReason.SUCCEEDED, "");
        for (int i = 0; i < RACE_IDS_PER_KIND; i++) {
            long id = baseId + i;
            CountDownLatch go = new CountDownLatch(1);
            Future<?> first = pool.submit(() -> {
                go.await();
                adoptRunning(ledger, ep, id, side, EnginePhase.RUNNING, versionGen);
                return null;
            });
            Future<?> second = pool.submit(() -> {
                go.await();
                adoptRunning(ledger, ep, id, side, EnginePhase.KV_ALLOCATED, versionGen);
                return null;
            });
            go.countDown();
            first.get(10L, TimeUnit.SECONDS);
            second.get(10L, TimeUnit.SECONDS);
            boolean settled = pSide
                    ? ledger.prefill().settle(id, completed, SettleReason.ENGINE_FINISHED)
                    : ledger.decode().settle(id, completed, SettleReason.ENGINE_FINISHED);
            assertTrue(settled, "兜底终局必胜（双收养恰一胜者存活，id=" + id + "）");
            assertTrue(ledger.terminalOutcomeOf(id, side).isPresent());
        }
    }

    /**
     * 收养 vs 主动释放锤击（D 侧特有移除通道）：释放不落墓碑（非终局），
     * 出账读终局时刻现态——与收养入账的窗口竞速；条目出现后释放必胜。
     */
    private void raceAdoptionVsRelease(StateLedger ledger, TestEndpoints.Endpoint ep,
                                       long baseId, ExecutorService pool,
                                       AtomicLong versionGen) throws Exception {
        for (int i = 0; i < RACE_IDS_PER_KIND; i++) {
            long id = baseId + i;
            CountDownLatch go = new CountDownLatch(1);
            Future<?> adopter = pool.submit(() -> {
                go.await();
                adoptRunning(ledger, ep, id, StateRole.DECODE, EnginePhase.RUNNING, versionGen);
                return null;
            });
            Future<?> releaser = pool.submit(() -> {
                go.await();
                long deadline = System.nanoTime() + HAMMER_WINDOW_NANOS;
                while (System.nanoTime() < deadline) {
                    if (ledger.decode().release(id)) {
                        return null;
                    }
                }
                return null;
            });
            go.countDown();
            adopter.get(10L, TimeUnit.SECONDS);
            releaser.get(10L, TimeUnit.SECONDS);
            // 补刀：收养完成后条目必在场（幂等——已释放则条目不在，no-op）
            ledger.decode().release(id);
            assertTrue(ledger.decode().get(id).isEmpty(), "释放必移除收养条目（id=" + id + "）");
            assertTrue(ledger.terminalOutcomeOf(id, StateRole.DECODE).isEmpty(),
                    "释放不是终局——不落墓碑（id=" + id + "）");
        }
    }

    // ---- 阶段三：兜底收敛 ----

    /** 残留活跃条目确定性终局（本地结算 CAS 单出口）；返回终局成功条数（墓碑增量）。 */
    private static long settleAllRemaining(StateLedger ledger, Set<Long> ids, boolean pSide) {
        TerminalOutcome completed = new TerminalOutcome(TerminalState.COMPLETED,
                TerminalReason.SUCCEEDED, "");
        long settled = 0L;
        for (Long id : ids) {
            boolean ok = pSide
                    ? ledger.prefill().settle(id, completed, SettleReason.ENGINE_FINISHED)
                    : ledger.decode().settle(id, completed, SettleReason.ENGINE_FINISHED);
            if (ok) {
                settled++;
            }
        }
        return settled;
    }
}
