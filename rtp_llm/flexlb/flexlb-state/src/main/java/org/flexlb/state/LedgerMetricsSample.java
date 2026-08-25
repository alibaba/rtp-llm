package org.flexlb.state;

import java.util.List;
import java.util.Map;
import org.flexlb.state.internal.LedgerJanitor;

/**
 * 账本观测采样快照（不可变）：{@link LedgerMetrics#sample(long)} 的产物。
 *
 * <p>纯数据载体（flexlb-state 不依赖 metric 库——指标语义与上报通道由
 * flexlb-sync 侧适配器消费本快照决定）。两级聚合设计（防基数爆炸）：
 * 全局级（相位人口/墓碑水位/超车三分/reason 账/janitor 计数/drift） +
 * 端点级（跨端点活跃分布 + 全量 per-EP 摘要——端点 × 相位矩阵刻意不发）；
 * 相位年龄为累积采样分布（{@link LedgerMetrics#collectAges(long)} 低频
 * 轮转抽样入桶）。相位名/端点摘要均已展开为纯字符串/数值——消费侧
 * 零 internal 类型知识（除 janitorStats 透传 record）。</p>
 *
 * @param sampleAtMs             采样时刻（epoch 毫秒）
 * @param snapshot               全局聚合快照（相位人口/墓碑存量/迟到吸收/unknown/裁决计数）
 * @param prefillFastPathSettles P 侧快路径 CAS 结算总数（含 janitor 兜底胜者；
 *                               正常通道净额 = 本值 − janitor 通道结算数，由消费侧派生）
 * @param decodeFastPathSettles  D 侧快路径 CAS 结算总数（同上）
 * @param prefillOvertaken       P 侧相位推进 CAS 败者数（advance 超车）
 * @param decodeOvertaken        D 侧相位推进 CAS 败者数（advance 超车）
 * @param janitorStats           janitor 通道计数（兜底通道胜 = vanished/ttl/hardCap/causal 之和；
 *                               未挂载 janitor 时 null）
 * @param settleReasonCounts     settle 证据通道计数（全枚举值覆盖）
 * @param cleanupReasonCounts    清理通道计数（全枚举值覆盖）
 * @param transitionReasonCounts 相位转换驱动力计数（全枚举值覆盖）
 * @param terminalReasonCounts   终局受控原因计数（全枚举值覆盖）
 * @param drift                  最近一次对账偏差报告（collectAges tick 内更新；空清单 = 零漂移）
 * @param phasePopulation        相位人口直方图（全局级，带相位名——消费侧免 ordinal 映射）
 * @param prefillAges            P 侧各相位驻留时长分位（累积采样分布；样本数为 0 的相位不出现）
 * @param decodeAges             D 侧各相位驻留时长分位（同上）
 * @param ageSamples             相位年龄累积采样样本总数
 * @param endpointDistribution   跨端点活跃条目数分布（P+D 合并口径）
 * @param endpoints              全量 per-EP 摘要（按 endpointId 升序；消费方自行取 top-K）
 */
public record LedgerMetricsSample(
        long sampleAtMs,
        LedgerSnapshot snapshot,
        long prefillFastPathSettles,
        long decodeFastPathSettles,
        long prefillOvertaken,
        long decodeOvertaken,
        LedgerJanitor.JanitorStats janitorStats,
        Map<SettleReason, Long> settleReasonCounts,
        Map<CleanupReason, Long> cleanupReasonCounts,
        Map<TransitionReason, Long> transitionReasonCounts,
        Map<TerminalReason, Long> terminalReasonCounts,
        CounterDriftReport drift,
        List<PhasePopulation> phasePopulation,
        List<PhaseAgeSummary> prefillAges,
        List<PhaseAgeSummary> decodeAges,
        long ageSamples,
        EndpointDistributionSummary endpointDistribution,
        List<EndpointLedgerSummary> endpoints) {

    public LedgerMetricsSample {
        settleReasonCounts = Map.copyOf(settleReasonCounts);
        cleanupReasonCounts = Map.copyOf(cleanupReasonCounts);
        transitionReasonCounts = Map.copyOf(transitionReasonCounts);
        terminalReasonCounts = Map.copyOf(terminalReasonCounts);
        phasePopulation = List.copyOf(phasePopulation);
        prefillAges = List.copyOf(prefillAges);
        decodeAges = List.copyOf(decodeAges);
        endpoints = List.copyOf(endpoints);
    }

    /**
     * 单相位人口（全局级直方图条目，带相位名）。
     *
     * @param side        侧（"P" / "D"）
     * @param phaseOrdinal 相位序号（侧内枚举 ordinal）
     * @param phaseName   相位名
     * @param count      当前该相位活跃条目数
     */
    public record PhasePopulation(String side, int phaseOrdinal, String phaseName, long count) {
    }

    /**
     * 单相位驻留时长分位摘要（累积采样分布的桶分位估计——桶粒度见采样器边界）。
     *
     * @param side        侧（"P" / "D"）
     * @param phaseOrdinal 相位序号（侧内枚举 ordinal）
     * @param phaseName   相位名
     * @param samples     该相位累积采样样本数
     * @param p50Ms       驻留时长 P50（毫秒，桶上界口径——保守偏高）
     * @param p95Ms       驻留时长 P95（毫秒，桶上界口径——保守偏高）
     */
    public record PhaseAgeSummary(String side, int phaseOrdinal, String phaseName,
                                  long samples, long p50Ms, long p95Ms) {
    }

    /**
     * 跨端点活跃条目数分布（每 tick 全端点 O(1) 轻扫聚合）。
     *
     * @param endpointCount 参与统计的端点数（P/D 并集）
     * @param activeP50     端点活跃条目数（P+D 合并）P50
     * @param activeP95     端点活跃条目数 P95
     * @param activeMax     端点活跃条目数最大值
     */
    public record EndpointDistributionSummary(int endpointCount, long activeP50, long activeP95, long activeMax) {

        static EndpointDistributionSummary empty() {
            return new EndpointDistributionSummary(0, 0L, 0L, 0L);
        }
    }

    /**
     * 单端点账本摘要（全量 per-EP；P/D 合并一条）。
     *
     * @param endpointId       端点 ID
     * @param prefillActive    P 侧活跃条目数（已派发绑定）
     * @param decodeActive     D 侧活跃条目数
     * @param decodeReservedKv D 侧未确认影子预占 KV 合计
     * @param decodeKvTokens   D 侧引擎事实 KV 合计
     */
    public record EndpointLedgerSummary(int endpointId, long prefillActive, long decodeActive,
                                        long decodeReservedKv, long decodeKvTokens) {
    }
}
