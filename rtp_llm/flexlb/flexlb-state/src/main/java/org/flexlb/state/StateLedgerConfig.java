package org.flexlb.state;

/**
 * StateLedger 配置（构造注入）。
 *
 * @param tombstoneRetentionMs        墓碑保持期（默认 60s）：终态条目判重窗口
 * @param fenceTtlMs                  fence 过期 TTL（默认 300s，防永生）
 * @param snapshotIntervalTransitions 派生快照发布间隔：每 N 次相位转换发布一次
 *                                    volatile 快照（显式 refreshSnapshot 不受间隔限制）
 */
public record StateLedgerConfig(long tombstoneRetentionMs, long fenceTtlMs, int snapshotIntervalTransitions) {

    public StateLedgerConfig {
        if (tombstoneRetentionMs < 0 || fenceTtlMs < 0 || snapshotIntervalTransitions < 1) {
            throw new IllegalArgumentException(
                    "tombstoneRetentionMs/fenceTtlMs >= 0 且 snapshotIntervalTransitions >= 1，实际: "
                            + tombstoneRetentionMs + "/" + fenceTtlMs + "/" + snapshotIntervalTransitions);
        }
    }

    public static StateLedgerConfig defaults() {
        return new StateLedgerConfig(60_000L, 300_000L, 64);
    }
}
