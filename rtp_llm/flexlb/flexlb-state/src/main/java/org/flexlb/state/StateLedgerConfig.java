package org.flexlb.state;

/**
 * StateLedger 配置（构造注入）。
 *
 * @param tombstoneRetentionMs        墓碑保持期（默认 60s）：终态条目判重窗口
 * @param fenceTtlMs                  fence 过期 TTL（默认 300s，防永生）
 * @param snapshotIntervalTransitions 派生快照发布间隔：每 N 次相位转换发布一次
 *                                    volatile 快照（显式 refreshSnapshot 不受间隔限制）
 * @param debugTransitionLog          相位转换 debug 日志开关（默认 false——开启时
 *                                    每次 CAS 胜者的相位转换打一行 debug 日志：
 *                                    requestId/from/to/version/reason。日志量红线，
 *                                    仅排障时经 env FLEXLB_STATE_V2_DEBUG_TRANSITION_LOG
 *                                    开启，不常开）
 */
public record StateLedgerConfig(long tombstoneRetentionMs, long fenceTtlMs,
                                int snapshotIntervalTransitions, boolean debugTransitionLog) {

    public StateLedgerConfig {
        if (tombstoneRetentionMs < 0 || fenceTtlMs < 0 || snapshotIntervalTransitions < 1) {
            throw new IllegalArgumentException(
                    "tombstoneRetentionMs/fenceTtlMs >= 0 且 snapshotIntervalTransitions >= 1，实际: "
                            + tombstoneRetentionMs + "/" + fenceTtlMs + "/" + snapshotIntervalTransitions);
        }
    }

    /** 三参便捷构造（debug 转换日志开关默认关——既有调用点兼容）。 */
    public StateLedgerConfig(long tombstoneRetentionMs, long fenceTtlMs, int snapshotIntervalTransitions) {
        this(tombstoneRetentionMs, fenceTtlMs, snapshotIntervalTransitions, false);
    }

    /** 衍生配置（装配处便捷：以本配置为基，仅覆盖 debug 转换日志开关）。 */
    public StateLedgerConfig withDebugTransitionLog(boolean debugTransitionLog) {
        return new StateLedgerConfig(tombstoneRetentionMs, fenceTtlMs,
                snapshotIntervalTransitions, debugTransitionLog);
    }

    public static StateLedgerConfig defaults() {
        return new StateLedgerConfig(60_000L, 300_000L, 64, false);
    }
}
