package org.flexlb.sync.shadow;

import org.flexlb.state.TerminalReason;
import org.flexlb.state.TerminalState;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * G1 影子对账单测：新旧终态一致/不一致/单侧缺失三情形计数 + 滑动窗口淘汰
 * + shutdown summary 单行聚合。
 */
class StateShadowDiffCollectorTest {

    // ---- 情形一：终态一致（等价类内）→ 只计 matched，零 diff ----

    @Test
    void shouldCountMatchOnly_whenTerminalsAgree() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        // 旧先到、新后到
        collector.recordOldTerminal(1L, "COMPLETED");
        collector.recordNewTerminal(1L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED);

        assertEquals(1L, collector.matchedCount());
        assertEquals(0L, collector.diffTerminalState());
        assertEquals(0L, collector.diffTerminalReason());
        assertEquals(0L, collector.diffMissingOnNew());
        assertEquals(0L, collector.diffMissingOnOld());
        assertEquals(0, collector.pendingOld());
        assertEquals(0, collector.pendingNew());
    }

    @Test
    void shouldCountMatchOnly_whenNewArrivesFirst() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        // 新先到、旧后到（顺序无关：先到入窗、后到 remove 比对双清）
        collector.recordNewTerminal(2L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED);
        collector.recordOldTerminal(2L, "COMPLETED");

        assertEquals(1L, collector.matchedCount());
        assertEquals(0L, collector.diffTerminalState());
        assertEquals(0L, collector.diffTerminalReason());
    }

    @Test
    void shouldTreatSloTimeoutAsEquivalentToTimedOut() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        // 跨值域等价类：旧 TIMED_OUT ↔ 新 SLO_TIMEOUT（+TTL_EXPIRED/VANISHED 同族 reason）
        collector.recordOldTerminal(3L, "TIMED_OUT");
        collector.recordNewTerminal(3L, TerminalState.SLO_TIMEOUT, TerminalReason.SLO_BUDGET_EXHAUSTED);

        assertEquals(1L, collector.matchedCount());
        assertEquals(0L, collector.diffTerminalState());
        assertEquals(0L, collector.diffTerminalReason());
    }

    @Test
    void shouldTreatCancelledFamilyReasonsAsEquivalent() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        collector.recordOldTerminal(4L, "CANCELLED");
        collector.recordNewTerminal(4L, TerminalState.CANCELLED, TerminalReason.CANCELLED_IMPLICIT);

        assertEquals(1L, collector.matchedCount());
        assertEquals(0L, collector.diffTerminalState());
        assertEquals(0L, collector.diffTerminalReason());
    }

    // ---- 情形二：终态不一致 → state/reason diff 计数 ----

    @Test
    void shouldCountStateDiff_whenTerminalStatesDisagree() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        collector.recordOldTerminal(5L, "COMPLETED");
        collector.recordNewTerminal(5L, TerminalState.FAILED, TerminalReason.ENGINE_FAILED);

        assertEquals(1L, collector.matchedCount());
        assertEquals(1L, collector.diffTerminalState());
        // 独立观测口径：旧终态为 COMPLETED，ENGINE_FAILED 不在 COMPLETED 等价 reason 集
        // → reason diff 同样计 1（state/reason 两个指标各自独立计数）
        assertEquals(1L, collector.diffTerminalReason());
    }

    @Test
    void shouldCountBothDiffs_whenPreemptedHasNoOldCounterpart() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        // PREEMPTED 为新语义回边态：state 必 diff（旧无对应）+ reason 必 diff（一律计，
        // 配对 reason 取强制收尾语义值——不在 COMPLETED 等价 reason 集内）
        collector.recordOldTerminal(6L, "COMPLETED");
        collector.recordNewTerminal(6L, TerminalState.PREEMPTED, TerminalReason.HARD_CAP);

        assertEquals(1L, collector.diffTerminalState());
        assertEquals(1L, collector.diffTerminalReason());
    }

    @Test
    void shouldCountReasonDiff_whenReasonOutsideEquivalentSet() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        // 旧 COMPLETED 但新 reason 是 TTL_EXPIRED（不属于 COMPLETED 等价集）
        collector.recordOldTerminal(7L, "COMPLETED");
        collector.recordNewTerminal(7L, TerminalState.COMPLETED, TerminalReason.TTL_EXPIRED);

        assertEquals(0L, collector.diffTerminalState());
        assertEquals(1L, collector.diffTerminalReason());
    }

    // ---- 情形三：单侧缺失（窗口过期淘汰结算 missing） ----

    @Test
    void shouldCountMissingOnNew_whenOnlyOldSideArrives() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null, 10_000L, 1024);

        collector.recordOldTerminal(8L, "COMPLETED");
        assertEquals(1, collector.pendingOld());

        // 窗口过期 → 旧侧条目淘汰 → missing_on_new
        collector.evictExpired(System.currentTimeMillis() + 11_000L);

        assertEquals(0, collector.pendingOld());
        assertEquals(1L, collector.diffMissingOnNew());
        assertEquals(0L, collector.diffMissingOnOld());
    }

    @Test
    void shouldCountMissingOnOld_whenOnlyNewSideArrives() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null, 10_000L, 1024);

        collector.recordNewTerminal(9L, TerminalState.FAILED, TerminalReason.ENGINE_FAILED);
        assertEquals(1, collector.pendingNew());

        collector.evictExpired(System.currentTimeMillis() + 11_000L);

        assertEquals(0, collector.pendingNew());
        assertEquals(1L, collector.diffMissingOnOld());
        assertEquals(0L, collector.diffMissingOnNew());
    }

    @Test
    void shouldDropOldestAndCountMissing_whenWindowCapacityExceeded() {
        // 容量 2：第 3 条进入时最早条目被淘汰并计 missing
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null, 600_000L, 2);

        collector.recordOldTerminal(10L, "COMPLETED");
        collector.recordOldTerminal(11L, "FAILED");
        collector.recordOldTerminal(12L, "CANCELLED");

        assertEquals(2, collector.pendingOld(), "容量上限 2：最早条目被淘汰");
        assertEquals(1L, collector.diffMissingOnNew(), "被淘汰的旧侧条目计 missing_on_new");
        assertEquals(1L, collector.windowOverflowDropped());
    }

    @Test
    void shouldNeverCountTwice_forConcurrentDoubleArrival() {
        // 同 requestId 的新侧到达两次：第二次防重跳过（每请求新侧终态恰好一次，
        // 终态唯一性由 ledger CAS 单出口保证——旧语义的第二次入窗正是迟到
        // 阶段终局滞留窗口的源头）
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        collector.recordOldTerminal(13L, "COMPLETED");
        collector.recordNewTerminal(13L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED);
        collector.recordNewTerminal(13L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED);

        assertEquals(1L, collector.matchedCount(), "只匹配一次");
        assertEquals(1L, collector.duplicateNewSuppressed(), "重复到达的第二次防重跳过");
        assertEquals(0, collector.pendingNew(), "防重后零滞留（旧语义第二次入窗是滞留源头）");
        assertEquals(0L, collector.diffTerminalState());
    }

    @Test
    void shouldRejectInvalidWindowConfig() {
        assertThrows(IllegalArgumentException.class,
                () -> new StateShadowDiffCollector(null, 0L, 100));
        assertThrows(IllegalArgumentException.class,
                () -> new StateShadowDiffCollector(null, 1000L, 0));
    }

    // ---- 惰性淘汰限频（热路径防护：高频 put 下过期扫描至多每秒一次） ----

    /**
     * 限频语义：首次 put 触发过期扫描（水位初始化远早于 now）；限频窗口内的
     * 后续 put 不再扫描——已过期条目仍在窗，直到下一轮扫描或显式 evictExpired。
     * （missing 计数的结算时机漂移在秒级，窗口 10 分钟≫限频间隔——正确性不变。）
     */
    @Test
    void evictSweepThrottledWithinInterval() throws InterruptedException {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null, 100L, 1024);

        collector.recordOldTerminal(30L, "COMPLETED"); // 首次 put：扫描（无过期）
        Thread.sleep(150L); // 让 30L 过期（windowMs=100）

        collector.recordOldTerminal(31L, "COMPLETED"); // 限频窗口（1s）内：不扫描
        assertEquals(2, collector.pendingOld(),
                "限频窗口内的 put 不触发过期扫描——过期条目仍驻窗（结算时机漂移可接受）");
        assertEquals(0L, collector.diffMissingOnNew(), "未扫描 → 不结算 missing");

        collector.evictExpired(System.currentTimeMillis()); // 显式驱动：补结算
        assertEquals(1, collector.pendingOld(), "31L 未过期仍驻窗");
        assertEquals(1L, collector.diffMissingOnNew(), "显式扫描结算过期条目 30L 的 missing");
    }

    // ---- 新侧终态恰好一次（跨侧阶段终局重复上报防重） ----

    /**
     * 真机轮第二根因回归：D 终局（请求级终态 #1）配对后，P 引擎迟到
     * finished 再次触发记录（#2）——防重跳过，窗口零滞留。否则第二次
     * 永久滞留（旧侧终态已消费），高频下窗口满载滚动。
     */
    @Test
    void duplicateNewTerminalSuppressedAfterMatched() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        collector.recordNewTerminal(21L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED); // #1 入窗
        assertEquals(1, collector.pendingNew());
        collector.recordOldTerminal(21L, "COMPLETED"); // 配对双清
        assertEquals(1L, collector.matchedCount());

        collector.recordNewTerminal(21L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED); // 迟到重复
        assertEquals(1L, collector.duplicateNewSuppressed(), "配对后的迟到重复防重跳过");
        assertEquals(1L, collector.matchedCount(), "matched 不变");
        assertEquals(0, collector.pendingNew(), "配对后迟到重复零滞留（旧语义永久滞留窗口）");
    }

    /** 入窗未配对时的重复上报同样防重（不翻倍不覆盖不产生比对）。 */
    @Test
    void duplicateNewTerminalSuppressedBeforeMatched() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        collector.recordNewTerminal(22L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED);
        collector.recordNewTerminal(22L, TerminalState.FAILED, TerminalReason.ENGINE_FAILED); // 重复且值不同

        assertEquals(1L, collector.duplicateNewSuppressed());
        assertEquals(1, collector.pendingNew(), "窗口内单条（首次记录，重复不覆盖）");
        assertEquals(0L, collector.diffTerminalState(), "防重跳过不产生比对");

        collector.recordOldTerminal(22L, "COMPLETED");
        assertEquals(1L, collector.matchedCount(), "首次记录正常配对");
    }

    /**
     * 去重集 FIFO 容量同 maxEntries：逐出后的 id 重复上报重新可记
     * （防重失效边界可观测——引擎迟到终局时间跨度≫容量时不误吞首记）。
     */
    @Test
    void dedupFifoEvictionAllowsRerecord() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null, 600_000L, 2);

        collector.recordNewTerminal(23L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED);
        collector.recordNewTerminal(24L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED);
        collector.recordNewTerminal(25L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED); // FIFO 逐出 23L

        collector.recordNewTerminal(23L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED); // 逐出后重新可记
        assertEquals(2, collector.pendingNew(), "逐出后的 id 重复上报重新入窗（防重失效边界）");
        assertEquals(0L, collector.duplicateNewSuppressed(), "首记/逐出后重记均不计防重");
    }

    // ---- shutdown summary：全部计数读口的单行聚合（日志即验收证据） ----

    @Test
    void shouldRenderSummaryLineWithAllCounters() {
        StateShadowDiffCollector collector = new StateShadowDiffCollector(null);

        collector.onEvent();
        collector.recordOldTerminal(20L, "COMPLETED");
        collector.recordNewTerminal(20L, TerminalState.COMPLETED, TerminalReason.SUCCEEDED);
        collector.recordOldTerminal(21L, "FAILED"); // 入窗未匹配 → pendingOld=1

        String line = collector.summaryLine();

        assertTrue(line.contains("event=1"), line);
        assertTrue(line.contains("error=0"), line);
        assertTrue(line.contains("matched=1"), line);
        assertTrue(line.contains("diffTerminalState=0"), line);
        assertTrue(line.contains("diffTerminalReason=0"), line);
        assertTrue(line.contains("missingOnNew=0"), line);
        assertTrue(line.contains("missingOnOld=0"), line);
        assertTrue(line.contains("overflowDropped=0"), line);
        assertTrue(line.contains("pendingOld=1"), line);
        assertTrue(line.contains("pendingNew=0"), line);
    }
}
