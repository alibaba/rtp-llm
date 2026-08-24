package org.flexlb.state.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * FenceRegistry 组件级：R4 驱逐断言（fenced 条目 canEvict 抛 IllegalStateException）、
 * unfence/TTL 过期防永生、dump 可读视图、重复登记覆盖刷新。
 */
class FenceRegistryTest {

    /** R4：fenced 条目驱逐断言拒绝；未 fence 条目正常放行。 */
    @Test
    void fencedRequestCannotBeEvicted() {
        FenceRegistry r = new FenceRegistry(60_000L);
        r.fence("cancel-flow", 42L, FenceRegistry.FenceType.CANCEL);
        assertTrue(r.isFenced(42L));

        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> r.canEvict(42L));
        assertTrue(ex.getMessage().contains("42"), ex.getMessage());
        assertTrue(ex.getMessage().contains("CANCEL"), ex.getMessage());

        // 未 fence 条目正常放行
        r.canEvict(43L);

        // 解除后放行
        r.unfence(42L);
        assertFalse(r.isFenced(42L));
        r.canEvict(42L);
    }

    /** TTL 过期防永生：过期后 isFenced false + canEvict 放行 + evictExpired 清库存。 */
    @Test
    void ttlExpiryPreventsImmortalFence() {
        FenceRegistry r = new FenceRegistry(100L);
        // 注入未来时刻登记（可注入时刻的包私有重载），当前时刻视为未过期
        long future = System.currentTimeMillis() + 10_000L;
        r.fence("rebuild", 1L, FenceRegistry.FenceType.REBUILDING, future);
        assertTrue(r.isFenced(1L));

        // TTL 到点：线性扫清理
        assertEquals(1, r.evictExpired(future + 100L));
        assertFalse(r.isFenced(1L));
        r.canEvict(1L); // 不抛
        assertEquals(0, r.size());
    }

    /** 重复登记覆盖并刷新 TTL 基准（同 requestId 单条 fence）。 */
    @Test
    void refenceOverwritesAndRefreshes() {
        FenceRegistry r = new FenceRegistry(60_000L);
        r.fence("owner-a", 7L, FenceRegistry.FenceType.CANCEL);
        r.fence("owner-b", 7L, FenceRegistry.FenceType.RECONCILE);
        assertEquals(1, r.size());
        List<String> dump = r.dump();
        assertEquals(1, dump.size());
        assertTrue(dump.get(0).contains("owner=owner-b"), dump.get(0));
        assertTrue(dump.get(0).contains("RECONCILE"), dump.get(0));
    }

    /** dump 可读视图：多条件目逐条列出。 */
    @Test
    void dumpShowsReadableView() {
        FenceRegistry r = new FenceRegistry(60_000L);
        r.fence("a", 1L, FenceRegistry.FenceType.CANCEL);
        r.fence("b", 2L, FenceRegistry.FenceType.PREEMPT_UNSETTLED);
        List<String> dump = r.dump();
        assertEquals(2, dump.size());
        assertTrue(dump.get(0).contains("request=1") && dump.get(0).contains("CANCEL"));
        assertTrue(dump.get(1).contains("request=2") && dump.get(1).contains("PREEMPT_UNSETTLED"));
    }

    /** 构造参数校验：负 TTL 拒绝。 */
    @Test
    void negativeTtlRejected() {
        assertThrows(IllegalArgumentException.class, () -> new FenceRegistry(-1L));
    }
}
