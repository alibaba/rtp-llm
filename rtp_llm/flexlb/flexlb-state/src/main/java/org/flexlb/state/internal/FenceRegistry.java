package org.flexlb.state.internal;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.flexlb.state.InternalApi;

/**
 * fence 登记：跨侧协调期间冻结相关条目的驱逐（fence 驱逐断言防线，canEvict 断言拒绝），
 * 带 TTL 防永生。
 *
 * <p>fence 是意图标记不是锁：登记后对应条目在意图完成前不可被驱逐路径移除
 * （janitor 建立在登记/查询/断言/dump 基座之上）。</p>
 */
@InternalApi
public final class FenceRegistry {

    /** fence 意图类型。 */
    public enum FenceType {

        /** 取消传播进行中。 */
        CANCEL,

        /** 对账/审计期间。 */
        RECONCILE,

        /** 抢占但未终局（回边态悬置）。 */
        PREEMPT_UNSETTLED,

        /** 重启重建期间。 */
        REBUILDING
    }

    /** 单条 fence 记录。 */
    public record Fence(String owner, long requestId, FenceType type, long createdAtMs) {
    }

    private final ConcurrentHashMap<Long, Fence> fences = new ConcurrentHashMap<>();
    private final long ttlMs;

    public FenceRegistry(long ttlMs) {
        if (ttlMs < 0) {
            throw new IllegalArgumentException("ttlMs >= 0: " + ttlMs);
        }
        this.ttlMs = ttlMs;
    }

    /** 登记 fence 意图（同 requestId 重复登记覆盖并刷新 TTL 基准）。 */
    public void fence(String owner, long requestId, FenceType type) {
        fence(owner, requestId, type, System.currentTimeMillis());
    }

    /** 测试/重建可注入时刻的重载。 */
    void fence(String owner, long requestId, FenceType type, long nowMs) {
        fences.put(requestId, new Fence(owner, requestId, type, nowMs));
    }

    /** 该请求是否被有效 fence 保护（过期视为无 fence）。 */
    public boolean isFenced(long requestId) {
        Fence f = fences.get(requestId);
        return f != null && !expired(f, System.currentTimeMillis());
    }

    /**
     * 驱逐前断言（fence 驱逐断言防线）：fenced 条目驱逐拒绝——直接抛 {@link IllegalStateException}
     * （这是编程错误：驱逐路径必须先等 fence 解除）。
     */
    public void canEvict(long requestId) {
        Fence f = fences.get(requestId);
        if (f != null && !expired(f, System.currentTimeMillis())) {
            throw new IllegalStateException(
                    "request " + requestId + " is fenced (" + f.type() + " by " + f.owner()
                            + " since " + f.createdAtMs() + "), eviction forbidden (fence guard)");
        }
    }

    /** 解除 fence（意图完成）。 */
    public void unfence(long requestId) {
        fences.remove(requestId);
    }

    /** fence 过期清理（防永生）。返回清理条数。 */
    public int evictExpired(long nowMs) {
        int evicted = 0;
        for (Map.Entry<Long, Fence> e : fences.entrySet()) {
            if (expired(e.getValue(), nowMs)) {
                if (fences.remove(e.getKey(), e.getValue())) {
                    evicted++;
                }
            }
        }
        return evicted;
    }

    /** 可读视图（调试/测试）。 */
    public List<String> dump() {
        List<String> out = new ArrayList<>(fences.size());
        for (Fence f : fences.values()) {
            out.add("request=" + f.requestId() + " type=" + f.type()
                    + " owner=" + f.owner() + " createdAtMs=" + f.createdAtMs()
                    + (expired(f, System.currentTimeMillis()) ? " [EXPIRED]" : ""));
        }
        return List.copyOf(out);
    }

    public int size() {
        return fences.size();
    }

    private boolean expired(Fence f, long nowMs) {
        return nowMs - f.createdAtMs() >= ttlMs;
    }
}
