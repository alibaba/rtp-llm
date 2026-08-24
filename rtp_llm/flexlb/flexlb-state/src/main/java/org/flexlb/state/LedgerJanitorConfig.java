package org.flexlb.state;

/**
 * LedgerJanitor 配置（清理层四通道参数，构造注入）。
 *
 * <p>生产侧由 flexlb-sync 的 FlexlbConfig 映射（flexlbStateV2StaleRounds /
 * flexlbStateV2TtlMs / flexlbStateV2HardCapMs / flexlbStateV2JanitorIntervalMs，
 * env 覆盖 + 启动回显与事件泵装配同模式）；调度周期（janitorIntervalMs）属于
 * 调度方不在本配置内。</p>
 *
 * @param staleRounds        证据通道缺席阈值：完整 tick 连续缺席跨度
 *                           超过 N 轮（{@code round - lastSeenRound > N}）才触发
 *                           VANISHED——天然防抖（护栏 1）。默认 3。
 * @param ttlMs              时间通道 TTL：createdAtMs 基准（创建时刻固定不可续命——任何
 *                           touch/observe 都不刷新基准）。默认 300s（承接旧账本
 *                           TTL 兑底口径）。
 * @param hardCapMs          强制通道硬上限：createdAtMs + 上限，到期无条件清理
 *                           （fence 不豁免——宁清勿留决策，见 LedgerJanitor）。
 *                           <b>必须 &gt; ttlMs</b>。默认 900s。
 * @param scanBudgetPerTick  TTL/硬上限轮转扫描的每 tick 条目预算：未绑定条目
 *                           优先全扫，endpoint 名下条目按轮转游标分摊，
 *                           单 tick 至多扫描约本预算条目（预算内完成，
 *                           超出部分延后到后续 tick）。默认 4096。
 */
public record LedgerJanitorConfig(int staleRounds, long ttlMs, long hardCapMs, int scanBudgetPerTick) {

    /** 单 tick 扫描预算默认值（每 10s 一 tick × 4096 条 ≈ 40 万条/秒吞吐余量）。 */
    public static final int DEFAULT_SCAN_BUDGET_PER_TICK = 4096;

    public LedgerJanitorConfig {
        if (staleRounds < 1) {
            throw new IllegalArgumentException("staleRounds >= 1, actual: " + staleRounds);
        }
        if (ttlMs <= 0) {
            throw new IllegalArgumentException("ttlMs > 0, actual: " + ttlMs);
        }
        if (hardCapMs <= ttlMs) {
            throw new IllegalArgumentException("hardCapMs > ttlMs (hard cap must strictly exceed TTL), actual: "
                    + hardCapMs + " vs " + ttlMs);
        }
        if (scanBudgetPerTick < 1) {
            throw new IllegalArgumentException("scanBudgetPerTick >= 1, actual: " + scanBudgetPerTick);
        }
    }

    /** 生产默认（staleRounds=3 / ttl=300s / hardCap=900s / budget=4096）。 */
    public static LedgerJanitorConfig defaults() {
        return new LedgerJanitorConfig(3, 300_000L, 900_000L, DEFAULT_SCAN_BUDGET_PER_TICK);
    }
}
