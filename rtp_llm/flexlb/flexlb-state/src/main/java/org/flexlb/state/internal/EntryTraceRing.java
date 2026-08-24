package org.flexlb.state.internal;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * 8 槽相位进入历史环形缓冲（packed 编码，单条目零分配读取）。
 *
 * <h2>编码</h2>
 * 每条历史为一个 packed long：{@code packed = (phaseOrdinal << 56) | (dtMs & 0x00FF_FFFF_FFFF_FFFF)}
 * <ul>
 *   <li>高 8 位：相位序号（0..255，调用方传入其相位枚举的 ordinal）</li>
 *   <li>低 56 位：相对毫秒 {@code dtMs}（自请求创建起的进入时刻，非负，&lt; 2^56）</li>
 * </ul>
 *
 * <h2>线程安全</h2>
 * 写入：全局单调序号由 {@link AtomicInteger#getAndIncrement}（底层 CAS）分配，
 * 槽内写入以 {@code AtomicReferenceArray.compareAndSet} 提交 {@code [seq, phaseOrdinal, dtMs]}
 * 三元组——同槽竞争时仅允许更大序号覆盖更小序号，杜绝"旧写后落覆盖新写"的损坏。
 * 读取：{@link #drain()} 为尽力快照，返回当前可见条目（写入与 drain 并发时，
 * 已分配序号但未提交的条目视为不可见）。
 *
 * <h2>drain 人类可读格式</h2>
 * 每条形如 {@code "P3→P5 ageMs=17 atMs=42"}（最旧 → 最新有序）：
 * <ul>
 *   <li>{@code from→to}：相邻两条的相位序号（首条 from 记为 {@code "-"}）</li>
 *   <li>{@code ageMs}：在前一相位停留的时长 = 本条 atMs - 前条 atMs（首条为 0）</li>
 *   <li>{@code atMs}：进入本相位的相对毫秒（即 dtMs）</li>
 * </ul>
 * 相位序号到枚举名的解释由调用方完成（ring 不感知具体相位枚举类型）。
 */
final class EntryTraceRing {

    /** 槽数（设计：8 槽足够覆盖越级闭包补记的沿途相位数）。 */
    static final int SLOTS = 8;

    private static final int PHASE_SHIFT = 56;
    private static final long DT_MASK = (1L << PHASE_SHIFT) - 1;
    /** 槽内三元组布局：[0]=seq, [1]=phaseOrdinal, [2]=dtMs。 */
    private static final int IDX_SEQ = 0;
    private static final int IDX_PHASE = 1;
    private static final int IDX_DT = 2;

    private final AtomicReferenceArray<long[]> slots = new AtomicReferenceArray<>(SLOTS);
    private final AtomicInteger writeCount = new AtomicInteger();

    /**
     * 追加一条相位进入记录。
     *
     * @param phaseOrdinal 相位序号（0..255）
     * @param dtMs         进入该相位的相对毫秒（自请求创建起，非负且 &lt; 2^56）
     * @throws IllegalArgumentException 参数越界
     */
    void append(long phaseOrdinal, long dtMs) {
        if (phaseOrdinal < 0 || phaseOrdinal > 0xFF) {
            throw new IllegalArgumentException("phaseOrdinal must be in [0, 255]: " + phaseOrdinal);
        }
        if (dtMs < 0 || dtMs > DT_MASK) {
            throw new IllegalArgumentException("dtMs must be in [0, 2^56): " + dtMs);
        }
        long seq = writeCount.getAndIncrement();
        int idx = (int) Math.floorMod(seq, SLOTS);
        long[] entry = {seq, phaseOrdinal, dtMs};
        // CAS 提交：仅当槽内序号不大于本条序号才覆盖；已被更新序号占据则本条视为已被环形覆盖，放弃。
        while (true) {
            long[] current = slots.get(idx);
            if (current != null && current[IDX_SEQ] > seq) {
                return;
            }
            if (slots.compareAndSet(idx, current, entry)) {
                return;
            }
        }
    }

    /**
     * 快照读取：按写入顺序（最旧 → 最新）返回人类可读历史，最多 {@value SLOTS} 条。
     * 写满环形覆盖后仅保留最新 {@value SLOTS} 条。
     */
    List<String> drain() {
        int total = writeCount.get();
        int visible = Math.min(total, SLOTS);
        List<String> out = new ArrayList<>(visible);
        long prevPhase = -1;
        long prevDt = 0;
        for (int k = 0; k < visible; k++) {
            int seq = total - visible + k;
            long[] entry = slots.get((int) Math.floorMod(seq, SLOTS));
            // 与并发 append 竞争：序号已分配但未提交（槽内仍是上一轮值）→ 本条暂不可见。
            if (entry == null || entry[IDX_SEQ] != seq) {
                continue;
            }
            long phase = entry[IDX_PHASE];
            long dt = entry[IDX_DT];
            String from = prevPhase < 0 ? "-" : "P" + prevPhase;
            long ageMs = prevPhase < 0 ? 0 : dt - prevDt;
            out.add(from + "→P" + phase + " ageMs=" + ageMs + " atMs=" + dt);
            prevPhase = phase;
            prevDt = dt;
        }
        return List.copyOf(out);
    }
}
