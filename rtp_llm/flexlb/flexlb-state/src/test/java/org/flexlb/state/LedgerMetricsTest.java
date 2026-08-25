package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * LedgerMetrics 指标采样一致性：sample 快照与账本事实逐项对账——
 * 相位人口直方图（全局级，带相位名展开）/端点分布与 per-EP 摘要（升序、
 * P 侧只含已绑定条目）/reason 计数账（settle 三联/双联、terminal、
 * transition 驱动力）/超车计数（advance CAS 败者）/janitor 挂载态/
 * drift 对账报告；collectAges 年龄抽样入桶与桶上界分位口径
 * （累积采样分布，跨 tick 累积）。
 */
class LedgerMetricsTest {

    private static final TestEndpoints.Endpoint P_EP0 = TestEndpoints.ep(1L, StateRole.PREFILL, 0L);
    private static final TestEndpoints.Endpoint D_EP0 = TestEndpoints.ep(2L, StateRole.DECODE, 0L);

    // ---- 1. 相位人口 + 端点分布 + per-EP 摘要与账本事实一致 ----

    @Test
    void sampleMatchesPhasePopulationAndEndpointSummaries() {
        StateLedger ledger = new StateLedger();
        Bindings b = new Bindings(ledger);

        // P 侧：id=100 推进到 P_RUNNING（观察时钟 10_000）、id=101 停在 QUEUED（未绑定，
        // 不进端点索引——排队/攒批窗口由派发编排侧覆盖）
        dispatchPrefill(ledger, b.pEp, b.pBinding, 100L);
        ledger.observe(TestEndpoints.runningOnly(b.pEp, 2L, 10_000L,
                TestEndpoints.running(100L, StateRole.PREFILL, EnginePhase.RUNNING, 77L, 128L, 2L)));
        ledger.prefill().register(101L, 77L);
        ledger.prefill().onQueued(101L);

        // D 侧：id=102 推进到 D_LOADING（kvTokens=512 引擎事实接管）、id=103 停在 RESERVED
        assertEquals(ReserveResult.OK, ledger.decode().reserve(102L, 128L, 256L, b.dBinding));
        assertTrue(ledger.decode().onDispatched(102L, b.dBinding));
        ledger.observe(TestEndpoints.runningOnly(b.dEp, 1L, 10_000L,
                TestEndpoints.running(102L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 1L)));
        assertEquals(ReserveResult.OK, ledger.decode().reserve(103L, 64L, 128L, b.dBinding));

        LedgerMetricsSample sample = ledger.metrics().sample(20_000L);

        // 全局级水位
        assertEquals(2L, sample.snapshot().prefill().inflight());
        assertEquals(2L, sample.snapshot().decode().activeTotal());
        assertEquals(20_000L, sample.sampleAtMs());

        // 相位人口直方图（全局级，含未绑定条目）
        assertEquals(1L, phasePopulation(sample, "P", "P_RUNNING"));
        assertEquals(1L, phasePopulation(sample, "P", "QUEUED"));
        assertEquals(0L, phasePopulation(sample, "P", "PREFILL_DONE"));
        assertEquals(1L, phasePopulation(sample, "D", "D_LOADING"));
        assertEquals(1L, phasePopulation(sample, "D", "RESERVED"));
        // 两侧全部相位都出现（无样本相位计 0，不缺席——消费侧免 ordinal 映射）
        assertEquals(10, countSide(sample, "P"));
        assertEquals(4, countSide(sample, "D"));

        // per-EP 摘要：升序；P 端点只含已绑定（QUEUED 不在）；D 端点含全相位
        List<LedgerMetricsSample.EndpointLedgerSummary> eps = sample.endpoints();
        assertEquals(2, eps.size());
        assertEquals(1, eps.get(0).endpointId());
        assertEquals(1L, eps.get(0).prefillActive());
        assertEquals(0L, eps.get(0).decodeActive());
        assertEquals(2, eps.get(1).endpointId());
        assertEquals(0L, eps.get(1).prefillActive());
        assertEquals(2L, eps.get(1).decodeActive());
        // D 端点未确认预占 KV：102 已确认撤出、103 未确认 128（reserve 声明 expectedKv）
        assertEquals(128L, eps.get(1).decodeReservedKv());
        assertEquals(512L, eps.get(1).decodeKvTokens());

        // 跨端点分布（P+D 合并口径）：端点活跃 [1, 2] → P50=1、P95=2、max=2
        LedgerMetricsSample.EndpointDistributionSummary dist = sample.endpointDistribution();
        assertEquals(2, dist.endpointCount());
        assertEquals(1L, dist.activeP50());
        assertEquals(2L, dist.activeP95());
        assertEquals(2L, dist.activeMax());
    }

    // ---- 2. reason 计数账与结算通道一致 ----

    @Test
    void reasonAccountsMatchSettleChannels() {
        StateLedger ledger = new StateLedger();
        Bindings b = new Bindings(ledger);

        // 通道一：本地取消（facade settle 双联——证据通道 + 终局受控原因）
        dispatchPrefill(ledger, b.pEp, b.pBinding, 200L);
        assertTrue(ledger.prefill().settle(200L,
                new TerminalOutcome(TerminalState.CANCELLED, TerminalReason.CANCELLED_IMPLICIT, ""),
                SettleReason.LOCAL_CANCEL));

        // 通道二：引擎 finished(success)（三联——证据 + 清理 + 终局）
        dispatchPrefill(ledger, b.pEp, b.pBinding, 201L);
        ledger.observe(TestEndpoints.runningOnly(b.pEp, 2L, 30_000L,
                TestEndpoints.running(201L, StateRole.PREFILL, EnginePhase.RUNNING, 77L, 128L, 2L)));
        ledger.observe(TestEndpoints.finishedOnly(b.pEp, 3L, 30_040L,
                TestEndpoints.finished(201L, StateRole.PREFILL, 0, 30_040L, 3L)));

        // 通道三：F1 因果闭包（D finished success ⇒ 同 tick 收缩 P 条目）
        dispatchPrefill(ledger, b.pEp, b.pBinding, 202L);
        assertEquals(ReserveResult.OK, ledger.decode().reserve(202L, 128L, 256L, b.dBinding));
        assertTrue(ledger.decode().onDispatched(202L, b.dBinding));
        ledger.observe(TestEndpoints.runningOnly(b.dEp, 1L, 30_080L,
                TestEndpoints.running(202L, StateRole.DECODE, EnginePhase.RUNNING, -1L, 512L, 1L)));
        ledger.observe(TestEndpoints.finishedOnly(b.dEp, 2L, 30_100L,
                TestEndpoints.finished(202L, StateRole.DECODE, 0, 30_100L, 2L)));

        LedgerMetricsSample sample = ledger.metrics().sample(31_000L);

        // settle 证据通道：本地取消 1 + 引擎 finished 2（P 201 + D 202）+ 因果闭包 1（P 202）
        assertEquals(1L, sample.settleReasonCounts().get(SettleReason.LOCAL_CANCEL));
        assertEquals(2L, sample.settleReasonCounts().get(SettleReason.ENGINE_FINISHED));
        assertEquals(1L, sample.settleReasonCounts().get(SettleReason.CAUSAL_CLOSURE));
        assertEquals(0L, sample.settleReasonCounts().get(SettleReason.TTL_CHANNEL));

        // 清理通道：finished 正常清理 3（P 201 + D 202 + P 202 闭包）
        assertEquals(3L, sample.cleanupReasonCounts().get(CleanupReason.FINISHED_REPORTED));
        assertEquals(0L, sample.cleanupReasonCounts().get(CleanupReason.TTL));

        // 终局受控原因：取消 1 + 完成 3（P 201 + D 202 + P 202 闭包）
        assertEquals(1L, sample.terminalReasonCounts().get(TerminalReason.CANCELLED_IMPLICIT));
        assertEquals(3L, sample.terminalReasonCounts().get(TerminalReason.SUCCEEDED));

        // 快路径 CAS 结算总数（含一切 settleRemove 胜者——正常通道净额由消费侧派生）
        assertEquals(3L, sample.prefillFastPathSettles());
        assertEquals(1L, sample.decodeFastPathSettles());

        // 转换驱动力：调度决策（queued/dispatching/dispatched）≥1、引擎观察（running/finished）≥1
        assertTrue(sample.transitionReasonCounts().get(TransitionReason.SCHEDULER_DECISION) >= 9,
                "3 条 P 全流程各 3 次 + D 1 次 = 至少 10 次调度决策推进，实际 "
                        + sample.transitionReasonCounts().get(TransitionReason.SCHEDULER_DECISION));
        assertTrue(sample.transitionReasonCounts().get(TransitionReason.ENGINE_OBSERVATION) >= 2);

        // 墓碑水位：P 3 条、D 1 条
        assertEquals(3L, sample.snapshot().prefillTombstones());
        assertEquals(1L, sample.snapshot().decodeTombstones());
    }

    // ---- 3. 超车计数（advance CAS 败者）----

    @Test
    void overtakenCountsAdvanceCasLosers() {
        StateLedger ledger = new StateLedger();
        Bindings b = new Bindings(ledger);

        dispatchPrefill(ledger, b.pEp, b.pBinding, 300L);
        // 相位回退（QUEUED < DISPATCHED）→ CAS 败 → 超车计数
        ledger.prefill().onQueued(300L);
        // 败者不产生 debug 转换日志也不记 transition 驱动力账
        ledger.prefill().onQueued(300L);

        LedgerMetricsSample sample = ledger.metrics().sample(40_000L);
        assertEquals(2L, sample.prefillOvertaken());
        assertEquals(0L, sample.decodeOvertaken());
        // 超车败者不记驱动力账（仅 3 次胜者推进：queued/dispatching/dispatched）
        assertEquals(3L, sample.transitionReasonCounts().get(TransitionReason.SCHEDULER_DECISION));
        assertEquals(0L, sample.transitionReasonCounts().get(TransitionReason.LOAD_TRANSFER));
    }

    // ---- 4. 年龄抽样：入桶 + 桶上界分位 + 跨 tick 累积 ----

    @Test
    void collectAgesBucketsAndQuantilesAccumulate() {
        StateLedger ledger = new StateLedger();
        Bindings b = new Bindings(ledger);

        // P 条目 10_000 进入 P_RUNNING（观察 statusMs 即相位进入时刻）
        dispatchPrefill(ledger, b.pEp, b.pBinding, 400L);
        ledger.observe(TestEndpoints.runningOnly(b.pEp, 2L, 10_000L,
                TestEndpoints.running(400L, StateRole.PREFILL, EnginePhase.RUNNING, 77L, 0L, 2L)));
        // D 条目 10_000 进入 D_LOADING（另一端点）
        assertEquals(ReserveResult.OK, ledger.decode().reserve(400L, 128L, 256L, b.dBinding));
        assertTrue(ledger.decode().onDispatched(400L, b.dBinding));
        ledger.observe(TestEndpoints.runningOnly(b.dEp, 1L, 10_000L,
                TestEndpoints.running(400L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 1L)));

        LedgerMetrics metrics = ledger.metrics();

        // tick 1：轮转游标指向端点 1（TreeSet 升序首个）→ 只抽 P 条目，
        // 年龄 3_000ms 落 (2048, 4096] 桶，桶上界口径返回 4096
        metrics.collectAges(13_000L);
        LedgerMetricsSample s1 = metrics.sample(13_000L);
        assertEquals(1L, s1.ageSamples(), "单 tick 只抽 1 个端点（轮转抽样纪律）");
        LedgerMetricsSample.PhaseAgeSummary pRun = ageOf(s1, "P", "P_RUNNING");
        assertEquals(1L, pRun.samples());
        assertEquals(4096L, pRun.p50Ms(), "3000ms 落 (2048, 4096] 桶，桶上界口径保守偏高");
        assertEquals(4096L, pRun.p95Ms());
        // 端点 2 的 D 条目尚未被轮到，无样本相位不出现
        assertNull(ageOf(s1, "D", "D_LOADING"));
        assertNull(ageOf(s1, "P", "QUEUED"));

        // tick 2：轮转到端点 2 → D 条目年龄 6_000ms → 桶上界 8192
        metrics.collectAges(16_000L);
        LedgerMetricsSample s2 = metrics.sample(16_000L);
        assertEquals(2L, s2.ageSamples());
        LedgerMetricsSample.PhaseAgeSummary dLoad = ageOf(s2, "D", "D_LOADING");
        assertEquals(1L, dLoad.samples());
        assertEquals(8192L, dLoad.p50Ms());
        // P 侧仍是 tick 1 的累积样本（跨 tick 累积，不因新 tick 重置）
        assertEquals(1L, ageOf(s2, "P", "P_RUNNING").samples());

        // tick 3：轮回端点 1 → P 条目年龄 6_500ms → 两样本 [3000, 6500]，
        // P50 = 第 1 样本桶上界 4096、P95 = 第 2 样本桶上界 8192
        metrics.collectAges(16_500L);
        LedgerMetricsSample s3 = metrics.sample(16_500L);
        assertEquals(3L, s3.ageSamples());
        pRun = ageOf(s3, "P", "P_RUNNING");
        assertEquals(2L, pRun.samples());
        assertEquals(4096L, pRun.p50Ms(), "P50 = 第 1 个样本所在桶上界（3000）");
        assertEquals(8192L, pRun.p95Ms(), "P95 = 第 2 个样本所在桶上界（6500）");

        // 采样 tick 同时刷新 drift 对账报告（正常操作零漂移）
        assertTrue(s3.drift().clean(), () -> "drift=" + s3.drift());
        assertTrue(metrics.lastDriftReport().clean());
    }

    // ---- 5. janitor 挂载态与未挂载态 ----

    @Test
    void janitorStatsExposedOnlyWhenMounted() {
        StateLedger ledger = new StateLedger();
        assertNull(ledger.metrics().sample(1L).janitorStats(), "未挂载 janitor 时 janitorStats 为 null");

        LedgerJanitorConfig janitorConfig = new LedgerJanitorConfig(
                3, 60_000L, 120_000L, LedgerJanitorConfig.DEFAULT_SCAN_BUDGET_PER_TICK);
        assertNotNull(ledger.createJanitor(janitorConfig));
        LedgerMetricsSample sample = ledger.metrics().sample(2L);
        assertNotNull(sample.janitorStats());
        assertEquals(0L, sample.janitorStats().vanishedSettles());
        assertEquals(0L, sample.janitorStats().maintenanceTicks());
    }

    // ---- 6. 空账本采样边界 ----

    @Test
    void emptyLedgerSampleIsZeroValuedButWellFormed() {
        StateLedger ledger = new StateLedger();
        LedgerMetricsSample sample = ledger.metrics().sample(5_000L);

        assertEquals(0L, sample.snapshot().prefill().inflight());
        assertEquals(0L, sample.snapshot().decode().activeTotal());
        assertEquals(0L, sample.ageSamples());
        assertTrue(sample.prefillAges().isEmpty());
        assertTrue(sample.decodeAges().isEmpty());
        assertTrue(sample.endpoints().isEmpty());
        assertEquals(0, sample.endpointDistribution().endpointCount());
        // 全部 reason 键位齐全（零计数），消费侧免 null 处理
        assertEquals(SettleReason.values().length, sample.settleReasonCounts().size());
        assertEquals(CleanupReason.values().length, sample.cleanupReasonCounts().size());
        assertEquals(TransitionReason.values().length, sample.transitionReasonCounts().size());
        assertEquals(TerminalReason.values().length, sample.terminalReasonCounts().size());
        // 未跑过 collectAges 时零漂移空报告
        assertTrue(sample.drift().clean());
    }

    // ---- helpers ----

    /** P 条目全流程推进到 DISPATCHED（绑定世代）。 */
    private static void dispatchPrefill(StateLedger ledger, TestEndpoints.Endpoint pEp,
                                        GenerationTriple pBinding, long id) {
        assertEquals(RegisterResult.OK, ledger.prefill().register(id, 77L));
        ledger.prefill().onQueued(id);
        ledger.prefill().onDispatching(id, 77L);
        assertTrue(ledger.prefill().onDispatched(id, pBinding), "id=" + id + " 派发绑定应成功");
    }

    private static long phasePopulation(LedgerMetricsSample sample, String side, String phaseName) {
        return sample.phasePopulation().stream()
                .filter(p -> p.side().equals(side) && p.phaseName().equals(phaseName))
                .mapToLong(LedgerMetricsSample.PhasePopulation::count)
                .findFirst().orElse(-1L);
    }

    private static long countSide(LedgerMetricsSample sample, String side) {
        return sample.phasePopulation().stream()
                .filter(p -> p.side().equals(side))
                .count();
    }

    private static LedgerMetricsSample.PhaseAgeSummary ageOf(LedgerMetricsSample sample,
                                                             String side, String phaseName) {
        return sample.prefillAges().stream()
                .filter(a -> a.side().equals(side) && a.phaseName().equals(phaseName))
                .findFirst()
                .orElseGet(() -> sample.decodeAges().stream()
                        .filter(a -> a.side().equals(side) && a.phaseName().equals(phaseName))
                        .findFirst()
                        .orElse(null));
    }

    /** 世代登记 + 绑定三元组（P/D 各一端点；构造时登记世代）。 */
    private static final class Bindings {
        final TestEndpoints.Endpoint pEp;
        final TestEndpoints.Endpoint dEp;
        final GenerationTriple pBinding;
        final GenerationTriple dBinding;

        Bindings(StateLedger ledger) {
            long pGen = ledger.newGeneration(P_EP0);
            long dGen = ledger.newGeneration(D_EP0);
            this.pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
            this.dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
            this.pBinding = new GenerationTriple(1, pGen, 77L);
            this.dBinding = new GenerationTriple(2, dGen, -1L);
        }
    }
}
