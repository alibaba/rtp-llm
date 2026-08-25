package org.flexlb.state;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.internal.LedgerJanitor;
import org.flexlb.state.internal.decode.DecodePhase;
import org.flexlb.state.internal.decode.DecodeRequestState;
import org.flexlb.state.internal.decode.DecodeSideStore;
import org.flexlb.state.internal.prefill.PrefillPhase;
import org.flexlb.state.internal.prefill.PrefillRequestState;
import org.flexlb.state.internal.prefill.PrefillSideStore;

/**
 * 账本观测采样器：相位年龄抽样入桶 + 只读 {@link LedgerMetricsSample} 出口。
 *
 * <h2>采样纪律（O(1) 化红线：禁止热路径全量遍历）</h2>
 * <ul>
 *   <li><b>相位年龄</b>：低频 tick（{@link #collectAges(long)}，flexlb-sync 侧
 *       5s 调度）轮转抽 1 个端点，每侧至多 {@value #AGE_SAMPLE_BUDGET_PER_ENDPOINT}
 *       条活跃条目读 {@code now - lastPhaseEnteredAtMs} 入对数桶（累积采样分布，
 *       跨 tick 累积——单 tick 成本 O(单端点活跃条目)，与账本总量无关）。
 *       年龄口径是<b>驻留中年龄</b>（右删视样本：仍在该相位的当前驻留时长，
 *       用于发现滞留相位；非已完成驻留时长分布）。未绑定条目（P 侧排队/
 *       攒批窗口）不在端点抽样口径内。</li>
 *   <li><b>drift</b>：collectAges tick 内更新最近一次
 *       {@link StateLedger#auditAndDrift()} 对账报告（全局 + 端点级；不静默
 *       修正语义保持——修正告警由消费侧决定）。</li>
 *   <li><b>sample</b>：{@link #sample(long)} 汇总全部观测读口（两侧 Store 的
 *       O(1) 计数读口 + reason 计数账快照 + janitor stats + 端点分布
 *       O(端点数) 轻扫），可随时调用。</li>
 * </ul>
 *
 * <h2>分位口径</h2>
 * 年龄分位为<b>桶上界口径</b>（保守偏高：落在第 i 桶的样本值 ∈
 * (bounds[i-1], bounds[i]]，分位返回 bounds[i]；开尾桶按下界近似——
 * 超出最大桶界的样本记为最大桶界值）。桶计数为弱一致读（观测语义允许）。
 *
 * <p>线程模型：collectAges 由单一调度线程串行调用（游标/桶写入无锁）；
 * sample 可被任意线程调用（只读 + volatile 快照读）。桶数组写读弱一致
 * ——统计指标允许近似，不影响账本正确性。</p>
 */
public final class LedgerMetrics {

    /**
     * 年龄对数桶上界（毫秒）：1, 2, 4, ..., 524288（约 8.7 分钟）；
     * 超出入开尾桶。桶数 = 本数组长度 + 1。
     */
    private static final long[] AGE_BUCKET_BOUNDS_MS = {
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1024, 2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144, 524288
    };

    /** 单端点单 tick 年龄抽样预算（两侧各自计）。 */
    private static final int AGE_SAMPLE_BUDGET_PER_ENDPOINT = 8;

    private final StateLedger ledger;

    /** [phaseOrdinal][bucket] 累积年龄样本计数（collectAges 单线程写）。 */
    private final long[][] prefillAgeBuckets;
    private final long[][] decodeAgeBuckets;
    private final LongAdder ageSamples = new LongAdder();

    /** 端点轮转游标（collectAges 单线程访问）。 */
    private int rotationCursor;

    /** 最近一次对账偏差报告（collectAges tick 内更新；空清单 = 零漂移）。 */
    private volatile CounterDriftReport lastDriftReport = new CounterDriftReport(List.of());

    LedgerMetrics(StateLedger ledger) {
        this.ledger = ledger;
        this.prefillAgeBuckets = new long[PrefillPhase.values().length][AGE_BUCKET_BOUNDS_MS.length + 1];
        this.decodeAgeBuckets = new long[DecodePhase.values().length][AGE_BUCKET_BOUNDS_MS.length + 1];
    }

    /**
     * 低频采样 tick（flexlb-sync 侧 5s 调度）：轮转抽 1 个端点的年龄样本
     * 入桶 + 刷新对账偏差报告。单 tick 成本：O(单端点活跃条目)（两侧各
     * 至多 {@value #AGE_SAMPLE_BUDGET_PER_ENDPOINT} 条）+ O(全量活跃条目)
     * （对账重算，与既有 10s 对账扫描同量级）。
     */
    public void collectAges(long nowMs) {
        List<Integer> endpoints = trackedEndpointsOrdered();
        if (!endpoints.isEmpty()) {
            if (rotationCursor >= endpoints.size()) {
                rotationCursor = 0;
            }
            int endpointId = endpoints.get(rotationCursor);
            rotationCursor = (rotationCursor + 1) % endpoints.size();
            samplePrefillAges(endpointId, nowMs);
            sampleDecodeAges(endpointId, nowMs);
        }
        lastDriftReport = ledger.auditAndDrift();
    }

    /**
     * 只读采样快照（可随时调用；不推进抽样游标）。显式刷新两侧已发布
     * volatile 快照（refreshSnapshot 不受转换间隔限制——观测出口给最新值）。
     */
    public LedgerMetricsSample sample(long nowMs) {
        PrefillSideStore pStore = ledger.pStoreView();
        DecodeSideStore dStore = ledger.dStoreView();
        pStore.refreshSnapshot();
        dStore.refreshSnapshot();
        LedgerSnapshot snapshot = ledger.snapshot();

        LedgerJanitor janitor = ledger.janitor();
        LedgerJanitor.JanitorStats janitorStats = janitor == null ? null : janitor.stats();

        List<LedgerMetricsSample.EndpointLedgerSummary> endpoints = endpointSummaries(pStore, dStore);
        LedgerMetricsSample.EndpointDistributionSummary distribution = distributionOf(endpoints);
        // endpoints 已按 endpointId 升序（trackedEndpointsOrdered 的 TreeSet 序）。

        return new LedgerMetricsSample(
                nowMs,
                snapshot,
                pStore.fastPathSettles(),
                dStore.fastPathSettles(),
                pStore.overtakenEvents(),
                dStore.overtakenEvents(),
                janitorStats,
                ledger.settleReasonCountsView(),
                ledger.cleanupReasonCountsView(),
                ledger.transitionReasonCountsView(),
                ledger.terminalReasonCountsView(),
                lastDriftReport,
                phasePopulationOf(snapshot),
                prefillAgeSummaries(),
                decodeAgeSummaries(),
                ageSamples.sum(),
                distribution,
                endpoints);
    }

    /** 最近一次对账偏差报告（未跑过 collectAges 时为零漂移空清单）。 */
    public CounterDriftReport lastDriftReport() {
        return lastDriftReport;
    }

    // ---- 年龄抽样 ----

    private void samplePrefillAges(int endpointId, long nowMs) {
        int budget = AGE_SAMPLE_BUDGET_PER_ENDPOINT;
        for (PrefillRequestState e : ledger.pStoreView().entriesByEndpoint(endpointId)) {
            if (budget-- <= 0) {
                return;
            }
            long ageMs = Math.max(nowMs - e.lastPhaseEnteredAtMs(), 0L);
            prefillAgeBuckets[e.phase().ordinal()][bucketIndex(ageMs)]++;
            ageSamples.increment();
        }
    }

    private void sampleDecodeAges(int endpointId, long nowMs) {
        int budget = AGE_SAMPLE_BUDGET_PER_ENDPOINT;
        for (DecodeRequestState e : ledger.dStoreView().entriesByEndpoint(endpointId)) {
            if (budget-- <= 0) {
                return;
            }
            long ageMs = Math.max(nowMs - e.lastPhaseEnteredAtMs(), 0L);
            decodeAgeBuckets[e.phase().ordinal()][bucketIndex(ageMs)]++;
            ageSamples.increment();
        }
    }

    private List<LedgerMetricsSample.PhaseAgeSummary> prefillAgeSummaries() {
        List<LedgerMetricsSample.PhaseAgeSummary> out = new ArrayList<>();
        for (PrefillPhase phase : PrefillPhase.values()) {
            long[] buckets = prefillAgeBuckets[phase.ordinal()];
            long samples = totalOf(buckets);
            if (samples == 0) {
                continue; // 无样本相位不出现
            }
            out.add(new LedgerMetricsSample.PhaseAgeSummary("P", phase.ordinal(), phase.name(),
                    samples, bucketQuantile(buckets, 0.50), bucketQuantile(buckets, 0.95)));
        }
        return out;
    }

    private List<LedgerMetricsSample.PhaseAgeSummary> decodeAgeSummaries() {
        List<LedgerMetricsSample.PhaseAgeSummary> out = new ArrayList<>();
        for (DecodePhase phase : DecodePhase.values()) {
            long[] buckets = decodeAgeBuckets[phase.ordinal()];
            long samples = totalOf(buckets);
            if (samples == 0) {
                continue;
            }
            out.add(new LedgerMetricsSample.PhaseAgeSummary("D", phase.ordinal(), phase.name(),
                    samples, bucketQuantile(buckets, 0.50), bucketQuantile(buckets, 0.95)));
        }
        return out;
    }

    // ---- 端点分布与全量 per-EP 摘要 ----

    private List<LedgerMetricsSample.EndpointLedgerSummary> endpointSummaries(
            PrefillSideStore pStore, DecodeSideStore dStore) {
        List<Integer> endpoints = trackedEndpointsOrdered();
        List<LedgerMetricsSample.EndpointLedgerSummary> out = new ArrayList<>(endpoints.size());
        for (int endpointId : endpoints) {
            PrefillEndpointCounters p = pStore.endpointCounters(endpointId);
            DecodeEndpointCounters d = dStore.endpointCounters(endpointId);
            out.add(new LedgerMetricsSample.EndpointLedgerSummary(
                    endpointId, p.activeTotal(), d.activeTotal(),
                    d.unconfirmedExpectedKv(), d.kvTokensReportedTotal()));
        }
        return out;
    }

    private static LedgerMetricsSample.EndpointDistributionSummary distributionOf(
            List<LedgerMetricsSample.EndpointLedgerSummary> endpoints) {
        if (endpoints.isEmpty()) {
            return LedgerMetricsSample.EndpointDistributionSummary.empty();
        }
        long[] active = endpoints.stream()
                .mapToLong(s -> s.prefillActive() + s.decodeActive())
                .sorted()
                .toArray();
        return new LedgerMetricsSample.EndpointDistributionSummary(
                endpoints.size(),
                quantileFromSorted(active, 0.50),
                quantileFromSorted(active, 0.95),
                active[active.length - 1]);
    }

    /** 相位人口直方图（全局级，带相位名——两侧 snapshot 的 phaseCounts 展开）。 */
    private static List<LedgerMetricsSample.PhasePopulation> phasePopulationOf(LedgerSnapshot snapshot) {
        List<LedgerMetricsSample.PhasePopulation> out = new ArrayList<>();
        List<Long> pCounts = snapshot.prefill().phaseCounts();
        PrefillPhase[] pPhases = PrefillPhase.values();
        for (int i = 0; i < pPhases.length && i < pCounts.size(); i++) {
            out.add(new LedgerMetricsSample.PhasePopulation("P", i, pPhases[i].name(), pCounts.get(i)));
        }
        List<Long> dCounts = snapshot.decode().phaseCounts();
        DecodePhase[] dPhases = DecodePhase.values();
        for (int i = 0; i < dPhases.length && i < dCounts.size(); i++) {
            out.add(new LedgerMetricsSample.PhasePopulation("D", i, dPhases[i].name(), dCounts.get(i)));
        }
        return out;
    }

    // ---- 桶工具 ----

    private static int bucketIndex(long ageMs) {
        for (int i = 0; i < AGE_BUCKET_BOUNDS_MS.length; i++) {
            if (ageMs <= AGE_BUCKET_BOUNDS_MS[i]) {
                return i;
            }
        }
        return AGE_BUCKET_BOUNDS_MS.length; // 开尾桶
    }

    /** 桶分位（桶上界口径，保守偏高；开尾桶按下界近似）。空桶集返回 0。 */
    private static long bucketQuantile(long[] buckets, double q) {
        long total = totalOf(buckets);
        if (total == 0) {
            return 0L;
        }
        long target = Math.max((long) Math.ceil(total * q), 1L);
        long acc = 0;
        for (int i = 0; i < buckets.length; i++) {
            acc += buckets[i];
            if (acc >= target) {
                return i < AGE_BUCKET_BOUNDS_MS.length
                        ? AGE_BUCKET_BOUNDS_MS[i]
                        : AGE_BUCKET_BOUNDS_MS[AGE_BUCKET_BOUNDS_MS.length - 1];
            }
        }
        return AGE_BUCKET_BOUNDS_MS[AGE_BUCKET_BOUNDS_MS.length - 1];
    }

    private static long totalOf(long[] buckets) {
        long total = 0;
        for (long c : buckets) {
            total += c;
        }
        return total;
    }

    /** 排序数组分位（上取整口径：第 ceil(n*q) 个值）。 */
    private static long quantileFromSorted(long[] sorted, double q) {
        if (sorted.length == 0) {
            return 0L;
        }
        int idx = (int) Math.min(Math.max(Math.ceil(sorted.length * q) - 1, 0), sorted.length - 1);
        return sorted[idx];
    }

    /** 两侧已登记端点并集（TreeSet 稳定轮转序）。 */
    private List<Integer> trackedEndpointsOrdered() {
        TreeSet<Integer> ids = new TreeSet<>();
        ids.addAll(ledger.pStoreView().trackedEndpointIds());
        ids.addAll(ledger.dStoreView().trackedEndpointIds());
        return new ArrayList<>(ids);
    }
}
