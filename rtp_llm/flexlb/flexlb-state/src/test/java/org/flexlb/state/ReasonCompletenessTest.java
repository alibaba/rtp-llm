package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.flexlb.state.internal.FenceRegistry;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * reason 枚举完备性契约（观测层守护；TerminalReason javadoc 引用本测试）：
 * 四类受控枚举（{@link SettleReason}/{@link CleanupReason}/{@link TransitionReason}/
 * {@link TerminalReason}）的每个值必须至少被一处产出路径使用——防死枚举
 * （无实现路径的预留值不进枚举）。
 *
 * <h2>验证方式：反射枚举值 × 行为化产出收集 × sync 侧白名单</h2>
 * <ul>
 *   <li><b>行为化收集</b>：驱动 flexlb-state 侧全部产出路径（引擎 finished
 *       成功/失败、F1 因果闭包、跨侧收缩、janitor 四通道缺席/fence/截断/TTL/
 *       硬上限），从 {@code metrics().sample()} 的 reason 计数账收集
 *       计数 &gt; 0 的值（多账本按通道分组隔离驱动，防通道间交叉触发）。</li>
 *   <li><b>sync 侧白名单</b>：flexlb-sync 适配层独有的产出点（caller-supplied
 *       reason 或 sync 决策分流）以白名单文档化——每项由 flexlb-sync 侧
 *       StateShadowBridgeTest 行为化验证闭环（本地取消两分流 + 旧路径超时）。</li>
 *   <li><b>断言</b>：每个枚举值 ∈ 行为收集集 ∪ 白名单；且白名单与行为收集集
 *       不相交（防白名单沦为本可行为验证之值的遮蔽，新增 state 侧产出点时
 *       必须同步裁剪白名单）。</li>
 * </ul>
 */
class ReasonCompletenessTest {

    private static final TestEndpoints.Endpoint P_EP0 = TestEndpoints.ep(1L, StateRole.PREFILL, 0L);
    private static final TestEndpoints.Endpoint D_EP0 = TestEndpoints.ep(2L, StateRole.DECODE, 0L);

    /**
     * flexlb-sync 适配层独有产出点白名单（key = 枚举全名，value = 产出点文档；
     * 闭环验证见 StateShadowBridgeTest#cancelledReasonSplitsByEngineSeen /
     * #timedOutSettlesBothSidesImmediately——sync 侧行为断言 reason 值）。
     */
    private static final Map<String, String> SYNC_PRODUCTION_POINTS = Map.of(
            "SettleReason.LOCAL_CANCEL",
            "StateShadowBridge#onOldTerminal/onOldTerminalAuthority 本地取消通道（caller-supplied）",
            "TerminalReason.CANCELLED_IMPLICIT",
            "StateShadowBridge#localCancelReason——取消时条目引擎已见分流",
            "TerminalReason.CANCELLED_NEVER_ARRIVED",
            "StateShadowBridge#localCancelReason——取消时条目从未到达引擎分流",
            "TerminalReason.SLO_BUDGET_EXHAUSTED",
            "StateShadowBridge#settleBothSidesAuthoritatively 旧路径 TIMED_OUT 通道");

    /** 四类枚举的行为化收集结果（计数 > 0 的值）。 */
    private record Collected(Set<SettleReason> settle, Set<CleanupReason> cleanup,
                             Set<TransitionReason> transition, Set<TerminalReason> terminal) {
    }

    @Test
    void everyReasonValueHasProductionPath() {
        Collected c = collectFromAllStateSidePaths();

        assertComplete(SettleReason.class, c.settle(), "settle");
        assertComplete(CleanupReason.class, c.cleanup(), "cleanup");
        assertComplete(TransitionReason.class, c.transition(), "transition");
        assertComplete(TerminalReason.class, c.terminal(), "terminal");
    }

    // ---- 行为化驱动（按通道分组账本，防交叉触发）----

    private static Collected collectFromAllStateSidePaths() {
        Map<SettleReason, Long> settleCounts = new HashMap<>();
        Map<CleanupReason, Long> cleanupCounts = new HashMap<>();
        Map<TransitionReason, Long> transitionCounts = new HashMap<>();
        Map<TerminalReason, Long> terminalCounts = new HashMap<>();

        mergeInto(driveEnginePaths(), settleCounts, cleanupCounts, transitionCounts, terminalCounts);
        mergeInto(driveJanitorEvidencePaths(), settleCounts, cleanupCounts, transitionCounts, terminalCounts);
        mergeInto(driveJanitorTtlPath(), settleCounts, cleanupCounts, transitionCounts, terminalCounts);
        mergeInto(driveJanitorHardCapPath(), settleCounts, cleanupCounts, transitionCounts, terminalCounts);

        return new Collected(
                positiveKeys(settleCounts, SettleReason.class),
                positiveKeys(cleanupCounts, CleanupReason.class),
                positiveKeys(transitionCounts, TransitionReason.class),
                positiveKeys(terminalCounts, TerminalReason.class));
    }

    /** 引擎侧路径：finished 成功/失败、F1 因果闭包、跨侧收缩、调度决策/引擎观察转换。 */
    private static LedgerMetricsSample driveEnginePaths() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        // 引擎 finished 成功 → ENGINE_FINISHED + FINISHED_REPORTED + SUCCEEDED
        dispatchPrefill(ledger, pBinding, 100L);
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, 1_000L,
                TestEndpoints.running(100L, StateRole.PREFILL, EnginePhase.RUNNING, 77L, 0L, 1L)));
        ledger.observe(TestEndpoints.finishedOnly(pEp, 2L, 1_040L,
                TestEndpoints.finished(100L, StateRole.PREFILL, 0, 1_040L, 2L)));

        // 引擎 finished 失败 → ENGINE_FAILED
        dispatchPrefill(ledger, pBinding, 101L);
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, 2_000L,
                TestEndpoints.running(101L, StateRole.PREFILL, EnginePhase.RUNNING, 77L, 0L, 1L)));
        ledger.observe(TestEndpoints.finishedOnly(pEp, 2L, 2_040L,
                TestEndpoints.finished(101L, StateRole.PREFILL, 5, 2_040L, 2L)));

        // F1 因果闭包：D finished 成功 ⇒ 同 tick 收缩存活 P 条目 → CAUSAL_CLOSURE
        dispatchPrefill(ledger, pBinding, 102L);
        ledger.decode().reserve(102L, 128L, 256L, dBinding);
        ledger.decode().onDispatched(102L, dBinding);
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 3_000L,
                TestEndpoints.running(102L, StateRole.DECODE, EnginePhase.RUNNING, -1L, 512L, 1L)));
        ledger.observe(TestEndpoints.finishedOnly(dEp, 2L, 3_040L,
                TestEndpoints.finished(102L, StateRole.DECODE, 0, 3_040L, 2L)));

        // 跨侧收缩：D KV_ALLOCATED 确认 ⇒ P（P_RECEIVED..P_WAITING_LOADED）闭包到
        // PREFILL_DONE → LOAD_TRANSFER
        dispatchPrefill(ledger, pBinding, 103L);
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, 4_000L,
                TestEndpoints.running(103L, StateRole.PREFILL, EnginePhase.RECEIVED, 77L, 0L, 1L)));
        ledger.decode().reserve(103L, 128L, 256L, dBinding);
        ledger.decode().onDispatched(103L, dBinding);
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 4_020L,
                TestEndpoints.running(103L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 1L)));
        assertTrue(ledger.prefill().get(103L).orElseThrow().phaseName().equals("PREFILL_DONE"),
                "D 确认点即 P 释放点——跨侧收缩应推进 P 到 PREFILL_DONE");

        return ledger.metrics().sample(5_000L);
    }

    /** janitor 证据通道路径：缺席判死（ABSENT_N_ROUNDS）、fence 豁免（FENCE_HOLD）、截断上报（TRUNCATED_REPORT_EXCLUDED）。 */
    private static LedgerMetricsSample driveJanitorEvidencePaths() {
        StateLedger ledger = new StateLedger();
        ledger.createJanitor(new LedgerJanitorConfig(3, 300_000L, 900_000L, 4096));
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        // id=200：引擎确认后连续 staleRounds 完整轮缺席 → VANISHED（EVIDENCE_CHANNEL）
        openConfirmedDecode(ledger, dEp, dBinding, 200L);
        for (long r = 2L; r <= 4L; r++) {
            completeRound(ledger, dEp, r);
        }
        completeRound(ledger, dEp, 5L); // 跨度 3 → 触发
        assertTrue(ledger.decode().get(200L).isEmpty(), "连续 3 完整轮缺席应判死 VANISHED");

        // id=201：fenced 条目缺席触发被豁免 → FENCE_HOLD
        openConfirmedDecode(ledger, dEp, dBinding, 201L);
        ledger.fences().fence("cancel-flow", 201L, FenceRegistry.FenceType.CANCEL);
        for (long r = 6L; r <= 9L; r++) {
            completeRound(ledger, dEp, r);
        }
        assertTrue(ledger.decode().get(201L).isPresent(), "fenced 条目必须被豁免（护栏 3）");

        // 截断上报（detailCount 虚高）→ TRUNCATED_REPORT_EXCLUDED
        ledger.observe(new EngineObservation(dEp, 20L, 10_020L, 10, List.of(), List.of()));

        return ledger.metrics().sample(11_000L);
    }

    /** janitor TTL 通道：createdAt 不可续命 → TTL_EXPIRED + TTL_CHANNEL + TTL。 */
    private static LedgerMetricsSample driveJanitorTtlPath() {
        StateLedger ledger = new StateLedger();
        org.flexlb.state.internal.LedgerJanitor janitor =
                ledger.createJanitor(new LedgerJanitorConfig(3, 100L, 1_000_000L, 4096));
        long dGen = ledger.newGeneration(D_EP0);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        ledger.decode().reserve(300L, 10L, 100L, dBinding);
        long createdAt = ledger.decode().get(300L).orElseThrow().createdAtMs();
        janitor.runMaintenanceTick(createdAt + 200L); // age=200 > ttl=100
        assertTrue(ledger.decode().get(300L).isEmpty(), "TTL 到期应被结算");

        return ledger.metrics().sample(createdAt + 300L);
    }

    /** janitor 强制通道：硬上限（优先于 TTL）→ HARD_CAP + FORCE_CHANNEL + HARD_CAP 清理账。 */
    private static LedgerMetricsSample driveJanitorHardCapPath() {
        StateLedger ledger = new StateLedger();
        org.flexlb.state.internal.LedgerJanitor janitor =
                ledger.createJanitor(new LedgerJanitorConfig(3, 100L, 150L, 4096));
        long dGen = ledger.newGeneration(D_EP0);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        ledger.decode().reserve(400L, 10L, 100L, dBinding);
        long createdAt = ledger.decode().get(400L).orElseThrow().createdAtMs();
        janitor.runMaintenanceTick(createdAt + 200L); // 同时超 ttl=100 与 hardCap=150 → 硬上限优先
        assertTrue(ledger.decode().get(400L).isEmpty(), "硬上限应被结算");

        return ledger.metrics().sample(createdAt + 300L);
    }

    // ---- 断言 ----

    private static <E extends Enum<E>> void assertComplete(Class<E> enumClass, Set<E> collected, String kind) {
        Set<E> all = EnumSet.allOf(enumClass);
        Set<String> whitelisted = whitelistValuesOf(enumClass);

        Set<E> uncovered = new HashSet<>(all);
        uncovered.removeAll(collected);
        Set<String> uncoveredNames = new HashSet<>();
        for (E v : uncovered) {
            String full = enumClass.getSimpleName() + "." + v.name();
            if (!whitelisted.contains(full)) {
                uncoveredNames.add(full);
            }
        }
        assertTrue(uncoveredNames.isEmpty(),
                () -> kind + " 枚举存在无产出路径的死值（防死枚举契约）: " + uncoveredNames
                        + "；行为收集=" + collected + "；sync 白名单=" + whitelisted);

        // 白名单与行为收集不相交：state 侧新增产出点时须同步裁剪白名单
        Set<String> stale = new HashSet<>();
        for (E v : collected) {
            String full = enumClass.getSimpleName() + "." + v.name();
            if (whitelisted.contains(full)) {
                stale.add(full);
            }
        }
        assertTrue(stale.isEmpty(),
                () -> kind + " 白名单存在已可由 state 侧行为产出的值（须裁剪白名单）: " + stale);
    }

    /** 白名单中属于该枚举的值（全名解析；含合法性校验——防白名单键拼写漂移）。 */
    private static <E extends Enum<E>> Set<String> whitelistValuesOf(Class<E> enumClass) {
        Set<String> values = new HashSet<>();
        for (Map.Entry<String, String> e : SYNC_PRODUCTION_POINTS.entrySet()) {
            String key = e.getKey();
            int dot = key.indexOf('.');
            String enumSimpleName = key.substring(0, dot);
            String valueName = key.substring(dot + 1);
            if (!enumSimpleName.equals(enumClass.getSimpleName())) {
                continue;
            }
            Enum.valueOf(enumClass, valueName); // 非法值名直接抛出（防拼写漂移）
            values.add(key);
        }
        return values;
    }

    // ---- helpers ----

    private static void dispatchPrefill(StateLedger ledger, GenerationTriple pBinding, long id) {
        ledger.prefill().register(id, 77L);
        ledger.prefill().onQueued(id);
        ledger.prefill().onDispatching(id, 77L);
        ledger.prefill().onDispatched(id, pBinding);
    }

    /** D 侧开账（reserve + dispatched）并以一轮 running 完成引擎确认（lastSeenRound=1）。 */
    private static void openConfirmedDecode(StateLedger ledger, TestEndpoints.Endpoint dEp,
                                            GenerationTriple dBinding, long id) {
        ledger.decode().reserve(id, 10L, 100L, dBinding);
        ledger.decode().onDispatched(id, dBinding);
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 1_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 100L, 1L)));
    }

    /** 该端点一轮完整空 tick（running/finished 均空，detailCount=0 完整）。 */
    private static void completeRound(StateLedger ledger, TestEndpoints.Endpoint dEp, long round) {
        ledger.observe(TestEndpoints.observation(dEp, round, 1_000L + round, List.of(), List.of()));
    }

    private static void mergeInto(LedgerMetricsSample sample,
                                  Map<SettleReason, Long> settleCounts,
                                  Map<CleanupReason, Long> cleanupCounts,
                                  Map<TransitionReason, Long> transitionCounts,
                                  Map<TerminalReason, Long> terminalCounts) {
        mergeMap(sample.settleReasonCounts(), settleCounts);
        mergeMap(sample.cleanupReasonCounts(), cleanupCounts);
        mergeMap(sample.transitionReasonCounts(), transitionCounts);
        mergeMap(sample.terminalReasonCounts(), terminalCounts);
    }

    private static <E extends Enum<E>> void mergeMap(Map<E, Long> from, Map<E, Long> into) {
        for (Map.Entry<E, Long> e : from.entrySet()) {
            into.merge(e.getKey(), e.getValue(), Long::sum);
        }
    }

    private static <E extends Enum<E>> Set<E> positiveKeys(Map<E, Long> counts, Class<E> type) {
        Set<E> out = EnumSet.noneOf(type);
        for (Map.Entry<E, Long> e : counts.entrySet()) {
            if (e.getValue() > 0L) {
                out.add(e.getKey());
            }
        }
        return out;
    }
}
