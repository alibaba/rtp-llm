package org.flexlb.state.internal;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.InternalApi;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.TerminalState;

/**
 * 墓碑仓库：终态条目保持期（默认 60s，构造可配）内的判重与迟到事件吸收。
 *
 * <p>每侧一个实例（P/D 各一，取实现简洁者——key 空间即 requestId）。</p>
 */
@InternalApi
public final class TombstoneStore {

    /**
     * 单条墓碑（终态记录）：含终局时条目 trace 环快照（人类可读相位历史，
     * 最多 {@value EntryTraceRing#SLOTS} 条——per-request 诊断端点在墓碑保留期内的
     * 故事线数据源）。
     */
    public record Tombstone(long requestId, TerminalState state, String reason,
                            long terminalAtMs, List<String> entryTrace) {
    }

    private final ConcurrentHashMap<Long, Tombstone> tombstones = new ConcurrentHashMap<>();
    private final long retentionMs;
    private final LongAdder lateEventCount = new LongAdder();
    private final LongAdder lateCancelCount = new LongAdder();

    public TombstoneStore(long retentionMs) {
        if (retentionMs < 0) {
            throw new IllegalArgumentException("retentionMs >= 0: " + retentionMs);
        }
        this.retentionMs = retentionMs;
    }

    /** 是否命中墓碑（保持期内终态记录存在）。 */
    public boolean isTombstoned(long requestId) {
        Tombstone t = tombstones.get(requestId);
        return t != null && !expired(t, now());
    }

    /** 终态吸收（无 trace 快照的旧签名：幂等，同 requestId 已有墓碑时保留首条）。 */
    public void absorb(long requestId, TerminalOutcome outcome, long nowMs) {
        absorb(requestId, outcome, nowMs, List.of());
    }

    /**
     * 终态吸收（幂等：同 requestId 已有墓碑时保留首条——含首条 trace）。
     * trace 为终局胜者在 settleRemove 内取的条目 trace 环快照。
     */
    public void absorb(long requestId, TerminalOutcome outcome, long nowMs, List<String> entryTrace) {
        tombstones.putIfAbsent(requestId,
                new Tombstone(requestId, outcome.state(), outcome.reason().name(), nowMs, entryTrace));
    }

    /** 迟到事件吸收计数（对已墓碑条目的 running/finished 事件）。 */
    public void absorbLateEvent() {
        lateEventCount.increment();
    }

    /** 迟到取消吸收计数（对已墓碑条目的 cancel）。 */
    public void absorbLateCancel() {
        lateCancelCount.increment();
    }

    /** 过期清理：线性扫墓碑 map（量小可接受——设计否决了桶化）。返回清理条数。 */
    public int evictExpired(long nowMs) {
        int evicted = 0;
        for (Map.Entry<Long, Tombstone> e : tombstones.entrySet()) {
            if (expired(e.getValue(), nowMs)) {
                if (tombstones.remove(e.getKey(), e.getValue())) {
                    evicted++;
                }
            }
        }
        return evicted;
    }

    /** 便捷重载：当前时刻过期清理。 */
    public int evictExpired() {
        return evictExpired(now());
    }

    /** 墓碑存量（含未过期清理的过期条目；精确语义用 evictExpired 后再读）。 */
    public long size() {
        return tombstones.size();
    }

    public long lateEventCount() {
        return lateEventCount.sum();
    }

    public long lateCancelCount() {
        return lateCancelCount.sum();
    }

    /** 单条查询（调试/测试）。 */
    public Optional<Tombstone> get(long requestId) {
        return Optional.ofNullable(tombstones.get(requestId));
    }

    /** 清空（rebuild 用）。 */
    public void reset() {
        tombstones.clear();
        lateEventCount.reset();
        lateCancelCount.reset();
    }

    private boolean expired(Tombstone t, long nowMs) {
        return nowMs - t.terminalAtMs() >= retentionMs;
    }

    private static long now() {
        return System.currentTimeMillis();
    }
}
