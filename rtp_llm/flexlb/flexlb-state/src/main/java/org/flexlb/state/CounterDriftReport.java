package org.flexlb.state;

import java.util.List;

/**
 * 对账扫描报告（10s 周期 auditAndDrift 产物）：全量重算 vs 增量计数器的偏差清单。
 *
 * <p>只比对不修正（不静默修正，指标/告警由观测层接入）。</p>
 *
 * @param discrepancies 偏差描述（人类可读）；空列表 = 无 drift
 */
public record CounterDriftReport(List<String> discrepancies) {

    public CounterDriftReport {
        discrepancies = List.copyOf(discrepancies);
    }

    /** 是否零偏差。 */
    public boolean clean() {
        return discrepancies.isEmpty();
    }
}
