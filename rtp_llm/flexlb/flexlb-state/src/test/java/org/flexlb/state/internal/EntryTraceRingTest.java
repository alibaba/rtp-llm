package org.flexlb.state.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * EntryTraceRing：并发写安全（无丢失/无损坏）、drain 顺序（最旧→最新）、环形覆盖语义。
 */
class EntryTraceRingTest {

    private static final Pattern ENTRY_FORMAT =
            Pattern.compile("^(-|P\\d+)→P(\\d+) ageMs=(-?\\d+) atMs=(\\d+)$");

    @Test
    void drainEmpty() {
        assertEquals(List.of(), new EntryTraceRing().drain());
    }

    /** drain 顺序与人类可读格式：from→to / ageMs（在前一相位停留时长）/ atMs（进入相对时刻）。 */
    @Test
    void drainOrderedHumanReadable() {
        EntryTraceRing ring = new EntryTraceRing();
        ring.append(0, 0);
        ring.append(3, 17);
        ring.append(5, 42);

        List<String> drained = ring.drain();
        assertEquals(List.of(
                "-→P0 ageMs=0 atMs=0",
                "P0→P3 ageMs=17 atMs=17",
                "P3→P5 ageMs=25 atMs=42"), drained);
    }

    /** drain 为快照：drain 不消费历史（重复 drain 结果一致）。 */
    @Test
    void drainIsSnapshotNotConsuming() {
        EntryTraceRing ring = new EntryTraceRing();
        ring.append(1, 5);
        ring.append(2, 9);
        assertEquals(ring.drain(), ring.drain());
    }

    /** 环形覆盖：写满 8 槽后继续写，仅保留最新 8 条，顺序仍为最旧→最新。
     *  首条（seq=4）的前驱（seq=3）已被覆盖，from 只能记为 "-"。 */
    @Test
    void ringOverwriteKeepsLatestEight() {
        EntryTraceRing ring = new EntryTraceRing();
        for (int i = 0; i < 12; i++) {
            ring.append(i % 256, i * 10L);
        }
        List<String> drained = ring.drain();
        assertEquals(8, drained.size(), "12 条写入后仅保留最新 8 条");
        // 最新 8 条 = seq ∈ [4, 11]，atMs = i*10，按物理写入序递增。
        for (int k = 0; k < 8; k++) {
            int i = 4 + k;
            String expectedFrom = k == 0 ? "-" : "P" + (i - 1);
            String expected = expectedFrom + "→P" + i
                    + " ageMs=" + (k == 0 ? 0 : 10) + " atMs=" + (i * 10);
            assertEquals(expected, drained.get(k));
        }
    }

    /** 并发写安全：8 线程 × 8 条 = 64 条并发 append（远超槽位数），join 后 drain 必须：
     *  <ul>
     *    <li>恰好 8 条（每槽恰好一条胜者，无丢失）；</li>
     *    <li>单条内 (phase, atMs) 自洽（phase == atMs/10 对 256 取模）——每次 append 的
     *    三元组原子提交，无撕裂损坏；</li>
     *    <li>atMs 互异且都属于写入值域——不同写入各自存活，无重复覆盖丢失。</li>
     *  </ul>
     *  注：并发下物理写入序号（CAS 分配）与逻辑序号 i 脱钩，故不断言 atMs 单调；
     *  顺序保持语义由单线程测试（ringOverwrite/drainOrdered）覆盖。 */
    @Test
    void concurrentAppendNoLossNoCorruption() throws Exception {
        EntryTraceRing ring = new EntryTraceRing();
        int threads = 8;
        int perThread = 8;
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final int tid = t;
                futures.add(pool.submit(() -> {
                    start.await();
                    for (int k = 0; k < perThread; k++) {
                        long i = tid * (long) perThread + k;
                        // 写入不变量：phase = i % 256，atMs = i * 10 —— 单条内两者同源。
                        ring.append(i % 256, i * 10);
                    }
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> f : futures) {
                f.get(30, TimeUnit.SECONDS);
            }
        } finally {
            pool.shutdownNow();
        }

        List<String> drained = ring.drain();
        assertEquals(EntryTraceRing.SLOTS, drained.size(), "64 条并发写后每槽恰一条胜者");
        java.util.Set<Long> seenAtMs = new java.util.HashSet<>();
        for (String entry : drained) {
            java.util.regex.Matcher matcher = ENTRY_FORMAT.matcher(entry);
            assertTrue(matcher.matches(), "条目格式合法（无损坏）: " + entry);
            long to = Long.parseLong(matcher.group(2));
            long atMs = Long.parseLong(matcher.group(4));
            assertEquals(Math.floorMod(atMs / 10, 256), to,
                    "phase 与 atMs 必须同源于同一次 append（三元组原子提交，无撕裂）: " + entry);
            assertEquals(0, atMs % 10, "atMs 必须是写入值域 i*10 之一: " + entry);
            assertTrue(atMs >= 0 && atMs < 64 * 10, "atMs 在写入值域 [0, 640) 内: " + entry);
            assertTrue(seenAtMs.add(atMs), "atMs 互异（不同写入各自存活，无重复覆盖丢失）: " + entry);
        }
    }

    /** 并发写安全（未满槽场景）：K ≤ 8 条并发 append，join 后全部可见（无丢失）。 */
    @Test
    void concurrentAppendUnderCapacityNoLoss() throws Exception {
        EntryTraceRing ring = new EntryTraceRing();
        int threads = 8;
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final int tid = t;
                futures.add(pool.submit(() -> {
                    start.await();
                    ring.append(tid, tid * 7L);
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> f : futures) {
                f.get(30, TimeUnit.SECONDS);
            }
        } finally {
            pool.shutdownNow();
        }
        assertEquals(8, ring.drain().size(), "8 条并发写（未触发覆盖）必须全部可见");
    }

    /** packed 编码参数校验：相位序号 0..255、dtMs 非负且 < 2^56。 */
    @Test
    void appendValidatesPackedRange() {
        EntryTraceRing ring = new EntryTraceRing();
        assertThrows(IllegalArgumentException.class, () -> ring.append(-1, 0));
        assertThrows(IllegalArgumentException.class, () -> ring.append(256, 0));
        assertThrows(IllegalArgumentException.class, () -> ring.append(0, -1));
        assertThrows(IllegalArgumentException.class, () -> ring.append(0, 1L << 56));
        // 边界值合法：
        ring.append(255, (1L << 56) - 1);
        ring.append(0, 0);
        assertEquals(2, ring.drain().size());
    }
}
