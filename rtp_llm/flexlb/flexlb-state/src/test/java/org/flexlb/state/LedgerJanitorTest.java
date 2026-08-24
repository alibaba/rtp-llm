package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.flexlb.state.internal.FenceRegistry;
import org.flexlb.state.internal.LedgerJanitor;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * M4 清理层 LedgerJanitor 组件级测试：四通道（F1-F4）× 三护栏 + 迟到吸收 + 并发守恒。
 *
 * <p>覆盖矩阵：F2 证据通道（护栏 1 防抖/护栏 2 完整性/护栏 3 fence 豁免）、
 * F3 TTL（createdAt 不可续命 + 轮转分摊）、F4 hard cap（fence 不豁免决策 +
 * 优先于 TTL）、迟到 finished 墓碑吸收、janitor tick 与快路径 settle 并发
 * （CAS 单出口——无双重结算、终态守恒）。</p>
 */
class LedgerJanitorTest {

    private static final TestEndpoints.Endpoint P_EP0 = TestEndpoints.ep(1L, StateRole.PREFILL, 0L);
    private static final TestEndpoints.Endpoint D_EP0 = TestEndpoints.ep(2L, StateRole.DECODE, 0L);

    /** D 侧账本 + janitor 装配（helper）。 */
    private record DSetup(StateLedger ledger, LedgerJanitor janitor, TestEndpoints.Endpoint dEp,
                          GenerationTriple dBinding) {
    }

    private static DSetup decodeLedger(LedgerJanitorConfig cfg) {
        StateLedger ledger = new StateLedger();
        LedgerJanitor janitor = ledger.createJanitor(cfg);
        long dGen = ledger.newGeneration(D_EP0);
        return new DSetup(ledger, janitor,
                TestEndpoints.ep(2L, StateRole.DECODE, dGen), new GenerationTriple(2, dGen, -1L));
    }

    /** D 侧开账（reserve + dispatched）并以一轮 running 完成引擎确认（lastSeenRound=1）。 */
    private static void openConfirmedDecode(DSetup s, long id) {
        s.ledger().decode().reserve(id, 10L, 100L, s.dBinding());
        s.ledger().decode().onDispatched(id, s.dBinding());
        s.ledger().observe(TestEndpoints.runningOnly(s.dEp(), 1L, 1_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 100L, 1L)));
        assertEquals(1L, s.ledger().decode().get(id).orElseThrow().lastSeenRound(),
                "开账后条目须被引擎确认过（引擎上报观察轮次未建立）");
    }

    /** 该端点一轮完整空 tick（running/finished 均空，detailCount=0 完整）。 */
    private static void completeRound(DSetup s, long round) {
        s.ledger().observe(TestEndpoints.observation(s.dEp(), round, 1_000L + round, List.of(), List.of()));
    }

    /** 不完整 tick（detailCount 虚高模拟截断上报（上报完整性））。 */
    private static void truncatedRound(DSetup s, long round) {
        s.ledger().observe(new EngineObservation(s.dEp(), round, 1_000L + round, 10, List.of(), List.of()));
    }

    // ---- 1. F2 证据通道 ----

    /** 完整 tick 连续 staleRounds 轮缺席 → VANISHED settle（等价任务判定式 lastSeen &lt; round-N）。 */
    @Test
    void evidenceChannelVanishesAfterConsecutiveCompleteAbsence() {
        DSetup s = decodeLedger(new LedgerJanitorConfig(3, 300_000L, 900_000L, 4096));
        long id = 100L;
        openConfirmedDecode(s, id);

        for (long r = 2; r <= 4; r++) {
            completeRound(s, r);
            assertTrue(s.ledger().decode().get(id).isPresent(),
                    "跨度 " + (r - 2) + " < staleRounds=3 不应触发（护栏 1 防抖）");
        }
        completeRound(s, 5L); // 跨度 3 → 触发
        assertTrue(s.ledger().decode().get(id).isEmpty(), "连续 3 完整轮缺席应判死 VANISHED");

        TerminalOutcome outcome = s.ledger().terminalOutcomeOf(id, StateRole.DECODE).orElseThrow();
        assertEquals(TerminalState.SLO_TIMEOUT, outcome.state());
        assertEquals(TerminalReason.VANISHED, outcome.reason());
        LedgerJanitor.JanitorStats st = s.janitor().stats();
        assertEquals(1L, st.vanishedSettles());
        assertEquals(0L, st.errors());
        assertEquals(0, s.janitor().absentTracked(), "触发后缺席追踪须清理");
    }

    /** 护栏 2（上报完整性）：不完整 tick 绝不推进缺席计数——round 值大跳也不放大跨度。 */
    @Test
    void incompleteTicksNeverAdvanceAbsenceCounting() {
        DSetup s = decodeLedger(new LedgerJanitorConfig(3, 300_000L, 900_000L, 4096));
        long id = 101L;
        openConfirmedDecode(s, id);

        // 不完整 tick：round 值大跳（5→50→500）。若按 round 值差判定会被放大误触发；
        // 完整轮序数判定下这些 tick 全部丢弃。
        truncatedRound(s, 5L);
        truncatedRound(s, 50L);
        truncatedRound(s, 500L);
        assertTrue(s.ledger().decode().get(id).isPresent(), "不完整 tick 绝不推进缺席计数（护栏 2 上报完整性）");
        assertEquals(3L, s.janitor().stats().incompleteTicksSkipped());
        assertEquals(1L, s.janitor().stats().roundEndTicks(), "仅开账确认轮计入完整轮");

        // 之后完整轮从零起算：恰好 3 轮后触发（轮序数与 round 值解耦）
        for (long r = 2; r <= 4; r++) {
            completeRound(s, r);
            assertTrue(s.ledger().decode().get(id).isPresent(), "完整轮累计 r=" + r + " 跨度 " + (r - 2));
        }
        completeRound(s, 5L);
        assertTrue(s.ledger().decode().get(id).isEmpty(), "完整轮累计 3 轮后才触发");
        assertEquals(1L, s.janitor().stats().vanishedSettles());
    }

    /** 护栏 3（fence 豁免）：fenced 条目缺席触发被跳过并计 fence_hold；解除后从新起点重新累计。 */
    @Test
    void fencedEntrySkippedByEvidenceChannelWithFenceHoldCount() {
        DSetup s = decodeLedger(new LedgerJanitorConfig(3, 300_000L, 900_000L, 4096));
        long id = 102L;
        openConfirmedDecode(s, id);
        s.ledger().fences().fence("cancel-flow", id, FenceRegistry.FenceType.CANCEL);

        for (long r = 2; r <= 5; r++) {
            completeRound(s, r);
        }
        assertTrue(s.ledger().decode().get(id).isPresent(), "fenced 条目必须被豁免（护栏 3 fence 豁免）");
        assertEquals(1L, s.janitor().stats().fenceHoldSkips(), "fence 豁免须计数");
        assertEquals(0L, s.janitor().stats().vanishedSettles());

        // fence 解除后重新累计（Absent 已在触发尝试时移除——保守重新观察）
        s.ledger().fences().unfence(id);
        for (long r = 6; r <= 8; r++) {
            completeRound(s, r);
            assertTrue(s.ledger().decode().get(id).isPresent(), "解除后从新起点累计（r=" + r + "）");
        }
        completeRound(s, 9L); // 9-6=3 → 触发
        assertTrue(s.ledger().decode().get(id).isEmpty());
        assertEquals(TerminalReason.VANISHED,
                s.ledger().terminalOutcomeOf(id, StateRole.DECODE).orElseThrow().reason());
        assertEquals(1L, s.janitor().stats().vanishedSettles());
    }

    /** 护栏 1：缺席中途恢复（running 重新出现）→ 追踪清零重新起算，不触发。 */
    @Test
    void absenceRecoveredMidwayRestartsTracking() {
        DSetup s = decodeLedger(new LedgerJanitorConfig(3, 300_000L, 900_000L, 4096));
        long id = 103L;
        openConfirmedDecode(s, id);

        completeRound(s, 2L);
        completeRound(s, 3L);
        // round 4：running 重新出现——若未清零，下一轮（跨度 5-2=3）会误触发
        s.ledger().observe(TestEndpoints.runningOnly(s.dEp(), 4L, 1_004L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, 100L, 4L)));
        assertEquals(4L, s.ledger().decode().get(id).orElseThrow().lastSeenRound());

        for (long r = 5; r <= 7; r++) {
            completeRound(s, r);
            assertTrue(s.ledger().decode().get(id).isPresent(),
                    "恢复后重新起算（r=" + r + " 跨度 " + (r - 5) + " < 3，护栏 1）");
        }
        completeRound(s, 8L); // 8-5=3 → 触发
        assertTrue(s.ledger().decode().get(id).isEmpty());
        assertEquals(TerminalReason.VANISHED,
                s.ledger().terminalOutcomeOf(id, StateRole.DECODE).orElseThrow().reason());
    }

    // ---- 2. F3 TTL 通道 ----

    /** TTL 不可续命专项：TTL 按 createdAt 判定，持续 observe（lastSeenRound 新鲜）不续命。 */
    @Test
    void ttlExpiresOnCreatedAtNotRenewedByObserve() {
        DSetup s = decodeLedger(new LedgerJanitorConfig(3, 100L, 1_000_000L, 4096)); // ttl=100ms
        long id = 200L;
        s.ledger().decode().reserve(id, 10L, 100L, s.dBinding());
        long createdAt = s.ledger().decode().get(id).orElseThrow().createdAtMs();

        // 持续新鲜观察（20 轮 running——lastSeenRound 一路推进，若可续命则永不 TTL）
        for (long r = 1; r <= 20; r++) {
            s.ledger().observe(TestEndpoints.runningOnly(s.dEp(), r, 1_000L + r,
                    TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, 100L, r)));
        }
        assertEquals(20L, s.ledger().decode().get(id).orElseThrow().lastSeenRound());

        s.janitor().runMaintenanceTick(createdAt + 200L); // age=200 > ttl=100

        assertTrue(s.ledger().decode().get(id).isEmpty(), "TTL 按 createdAt 判定——observe 不续命（创建时刻固定）");
        TerminalOutcome outcome = s.ledger().terminalOutcomeOf(id, StateRole.DECODE).orElseThrow();
        assertEquals(TerminalState.SLO_TIMEOUT, outcome.state());
        assertEquals(TerminalReason.TTL_EXPIRED, outcome.reason());
        assertEquals(1L, s.janitor().stats().ttlSettles());
    }

    /** 轮转分摊：单 tick 预算内只清部分——unbound 优先逐个，绑定条目按 endpoint 轮转。 */
    @Test
    void ttlScanSpreadsAcrossTicksByBudgetAndRotation() {
        StateLedger ledger = new StateLedger();
        LedgerJanitor janitor = ledger.createJanitor(new LedgerJanitorConfig(3, 100L, 1_000_000L, 1));
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, -1L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        // 2 个 P unbound（排队中）+ 1 个 P 绑定 endpoint1 + 1 个 D 绑定 endpoint2，全部超 TTL
        ledger.prefill().register(1L, -1L);
        ledger.prefill().register(2L, -1L);
        ledger.prefill().register(3L, -1L);
        ledger.prefill().onQueued(3L);
        ledger.prefill().onDispatching(3L, -1L);
        ledger.prefill().onDispatched(3L, pBinding);
        ledger.decode().reserve(4L, 10L, 100L, dBinding);
        long now = System.currentTimeMillis() + 10_000L; // 全部远超 ttl=100

        janitor.runMaintenanceTick(now);
        assertEquals(3, aliveCount(ledger), "budget=1：每 tick 至多清 1 条 unbound（分摊）");
        janitor.runMaintenanceTick(now);
        assertEquals(2, aliveCount(ledger), "第二个 unbound");

        janitor.runMaintenanceTick(now); // unbound 耗尽 → 轮转第一个 endpoint
        assertEquals(1, aliveCount(ledger), "轮转到第一个 endpoint 名下条目");
        // 多 endpoint 轮转断言：剩下的绑定条目分属不同 endpoint——本 tick 只清其一
        boolean p3Alive = ledger.prefill().get(3L).isPresent();
        boolean d4Alive = ledger.decode().get(4L).isPresent();
        assertTrue(p3Alive ^ d4Alive, "一轮只清部分 endpoint 名下条目（轮转分摊）");

        janitor.runMaintenanceTick(now);
        assertEquals(0, aliveCount(ledger), "全部清空");
        assertEquals(4L, janitor.stats().ttlSettles());
        assertEquals(0L, janitor.stats().errors());
    }

    private static int aliveCount(StateLedger ledger) {
        int c = 0;
        for (long id = 1; id <= 3; id++) {
            if (ledger.prefill().get(id).isPresent()) {
                c++;
            }
        }
        if (ledger.decode().get(4L).isPresent()) {
            c++;
        }
        return c;
    }

    // ---- 3. F4 强制通道（hard cap）----

    /**
     * hard cap vs fence 决策：fenced 条目超硬上限也无条件清理（宁清勿留——
     * fence 超硬上限说明 fence 自身泄漏），reason=HARD_CAP + 告警计数翻倍（双计）。
     */
    @Test
    void hardCapSettlesUnconditionallyEvenWhenFenced() {
        DSetup s = decodeLedger(new LedgerJanitorConfig(3, 100L, 200L, 4096)); // ttl=100 hardCap=200
        long id = 300L;
        s.ledger().decode().reserve(id, 10L, 100L, s.dBinding());
        long createdAt = s.ledger().decode().get(id).orElseThrow().createdAtMs();
        s.ledger().fences().fence("preempt", id, FenceRegistry.FenceType.PREEMPT_UNSETTLED);

        s.janitor().runMaintenanceTick(createdAt + 250L); // age=250 > hardCap=200

        assertTrue(s.ledger().decode().get(id).isEmpty(), "hard cap 无条件——fence 不豁免（决策记录）");
        TerminalOutcome outcome = s.ledger().terminalOutcomeOf(id, StateRole.DECODE).orElseThrow();
        assertEquals(TerminalState.FAILED, outcome.state());
        assertEquals(TerminalReason.HARD_CAP, outcome.reason());
        LedgerJanitor.JanitorStats st = s.janitor().stats();
        assertEquals(1L, st.hardCapSettles());
        assertEquals(1L, st.hardCapFenceViolations(), "fenced 条目超硬上限 → 告警计数翻倍（双计）");
        assertEquals(0L, st.fenceHoldSkips(), "hard cap 通道不做 fence 预检跳过");
    }

    /** hard cap 先于 TTL 判定（更严兜底优先结算）。 */
    @Test
    void hardCapTakesPrecedenceOverTtl() {
        DSetup s = decodeLedger(new LedgerJanitorConfig(3, 100L, 150L, 4096));
        long id = 301L;
        s.ledger().decode().reserve(id, 10L, 100L, s.dBinding());
        long createdAt = s.ledger().decode().get(id).orElseThrow().createdAtMs();

        s.janitor().runMaintenanceTick(createdAt + 200L); // 同时超 ttl=100 与 hardCap=150

        assertEquals(TerminalReason.HARD_CAP,
                s.ledger().terminalOutcomeOf(id, StateRole.DECODE).orElseThrow().reason());
        assertEquals(1L, s.janitor().stats().hardCapSettles());
        assertEquals(0L, s.janitor().stats().ttlSettles(), "hard cap 优先于 TTL");
    }

    // ---- 4. 迟到事件吸收 ----

    /** VANISHED 后迟到 finished 被墓碑吸收（非 unknown），终态唯一。 */
    @Test
    void lateFinishedAfterVanishedAbsorbedByTombstone() {
        DSetup s = decodeLedger(new LedgerJanitorConfig(3, 300_000L, 900_000L, 4096));
        long id = 400L;
        openConfirmedDecode(s, id);
        for (long r = 2; r <= 5; r++) {
            completeRound(s, r);
        }
        assertTrue(s.ledger().decode().get(id).isEmpty(), "VANISHED 前置");

        // 迟到 finished（更高版本/轮次）被墓碑吸收
        s.ledger().observe(TestEndpoints.finishedOnly(s.dEp(), 99L, 1_099L,
                TestEndpoints.finished(id, StateRole.DECODE, 0, 1_099L, 99L)));

        LedgerSnapshot snap = s.ledger().snapshot();
        assertEquals(1L, snap.lateEventsAbsorbed(), "迟到 finished 被墓碑吸收");
        assertEquals(0L, snap.unknownFinishedEvents(), "已终局请求的迟到事件不是 unknown");
        assertEquals(1L, snap.decodeTombstones(), "终态唯一（无双重结算）");
    }

    // ---- 5. 并发（janitor tick × 快路径 settle）----

    /** 回归：负 endpointId（flexlb-sync 影子桥 ipPort 哈希可为负）条目同样入索引——缺席判定无盲区。 */
    @Test
    void negativeEndpointIdEntriesAreIndexedAndScanned() {
        StateLedger ledger = new StateLedger();
        LedgerJanitor janitor = ledger.createJanitor(new LedgerJanitorConfig(3, 300_000L, 900_000L, 4096));
        TestEndpoints.Endpoint negBase = TestEndpoints.ep(-7L, StateRole.DECODE, 0L);
        long dGen = ledger.newGeneration(negBase);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(-7L, StateRole.DECODE, dGen);
        GenerationTriple dBinding = new GenerationTriple(-7, dGen, -1L);
        long id = 500L;
        ledger.decode().reserve(id, 10L, 100L, dBinding);
        ledger.decode().onDispatched(id, dBinding);
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 1_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 100L, 1L)));

        for (long r = 2; r <= 5; r++) {
            ledger.observe(TestEndpoints.observation(dEp, r, 1_000L + r, List.of(), List.of()));
        }
        assertTrue(ledger.decode().get(id).isEmpty(), "负 endpointId 条目缺席判定不得成盲区");
        assertEquals(TerminalReason.VANISHED,
                ledger.terminalOutcomeOf(id, StateRole.DECODE).orElseThrow().reason());
        assertEquals(1L, janitor.stats().vanishedSettles());
    }

    /**
     * janitor 维护线程与快路径 observe-finished 并发竞争同一批条目：
     * CAS 单出口保证无双重结算（每请求恰好一个终态/墓碑）、无泄漏、无 unknown、
     * 计数守恒（janitor 通道胜者 ≤ 总数；超车败者计数不出错）。
     */
    @Test
    void concurrentFastPathSettleAndJanitorTicksConserveTerminals() throws Exception {
        StateLedger ledger = new StateLedger(new StateLedgerConfig(600_000L, 300_000L, 8));
        LedgerJanitor janitor = ledger.createJanitor(new LedgerJanitorConfig(3, 50L, 10_000_000L, 4096));
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        int n = 80;
        for (int i = 0; i < n; i++) {
            long id = 5_000L + i;
            ledger.decode().reserve(id, 10L, 100L, dBinding);
            ledger.decode().onDispatched(id, dBinding);
        }

        ExecutorService pool = Executors.newFixedThreadPool(2);
        CountDownLatch start = new CountDownLatch(1);
        Future<?> fastPath = pool.submit(() -> {
            start.await();
            for (int i = 0; i < n; i++) {
                long id = 5_000L + i;
                // statusMs/endTimeMs 用真实墙钟：墓碑 terminalAtMs 与 janitor 的 evictExpired(本地时钟)
                // 同基准——固定小值会被注入未来时刻的 tick 当场当过期清除（时钟域一致性）
                long now = System.currentTimeMillis();
                ledger.observe(TestEndpoints.finishedOnly(dEp, 100L + i, now,
                        TestEndpoints.finished(id, StateRole.DECODE, 0, now, 1L)));
            }
            return null;
        });
        Future<?> ticks = pool.submit(() -> {
            start.await();
            for (int i = 0; i < 200; i++) {
                janitor.runMaintenanceTick(System.currentTimeMillis() + 500L); // age > ttl → 竞争 settle
            }
            return null;
        });
        start.countDown();
        fastPath.get(30L, TimeUnit.SECONDS);
        ticks.get(30L, TimeUnit.SECONDS);
        pool.shutdown();
        assertTrue(pool.awaitTermination(10L, TimeUnit.SECONDS));

        // 排水：无论竞争谁赢，补一轮 tick 后全部终局
        janitor.runMaintenanceTick(System.currentTimeMillis() + 500L);

        ledger.decode().refreshSnapshot();
        assertEquals(0L, ledger.decode().snapshot().activeTotal(), "无泄漏");
        LedgerSnapshot snap = ledger.snapshot();
        assertEquals(n, snap.decodeTombstones(), "每请求恰好一个终态（CAS 单出口——无双重结算）");
        assertEquals(0L, snap.unknownFinishedEvents(), "竞争中被 janitor 抢先的 finished 走墓碑吸收，非 unknown");
        for (int i = 0; i < n; i++) {
            assertTrue(ledger.terminalOutcomeOf(5_000L + i, StateRole.DECODE).isPresent(),
                    "id=" + (5_000L + i) + " 须有终态");
        }
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());

        LedgerJanitor.JanitorStats st = janitor.stats();
        assertEquals(0L, st.errors(), "janitor 铁律：并发下绝不外抛");
        long janitorWins = st.ttlSettles() + st.vanishedSettles() + st.hardCapSettles();
        assertTrue(janitorWins <= n, "janitor 通道胜者数守恒（实际 " + janitorWins + "）");
        // 超车计数（TTL 通道）：janitor 败者 = 快路径胜 = 正常；此处只验证不越界
        assertTrue(st.lostToFastPath() >= 0);
        assertFalse(janitorWins > 0 && st.lostToFastPath() > n);
    }
}
