package org.flexlb.state.internal;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.InternalApi;

/**
 * 世代追踪器（S8）：端点换代登记与世代屏障校验。
 *
 * <p>换代公式：{@code generation = max(进程 epoch 启动时间戳, 该 endpoint 上一代 + 1)}
 * ——单调递增，且 epoch 兜底防 master 重启归零（重启后 epoch 即当前墙钟，
 * 必然大于重启前任何代）。</p>
 */
@InternalApi
public final class GenerationTracker {

    private final long epochMs;
    private final ConcurrentHashMap<Long, Long> currentByEndpoint = new ConcurrentHashMap<>();
    private final LongAdder crossGenerationRejects = new LongAdder();
    private final LongAdder generationBumps = new LongAdder();

    /** @param epochMs 进程 epoch（通常为构造时刻墙钟；可注入以便测试）。 */
    public GenerationTracker(long epochMs) {
        this.epochMs = epochMs;
    }

    /**
     * EP 换代登记：分配并登记该端点的新一代。
     *
     * @return 新代际号（= max(epochMs, prev + 1)）
     */
    public long nextGeneration(long endpointId) {
        long prev = currentByEndpoint.getOrDefault(endpointId, 0L);
        long next = Math.max(epochMs, prev + 1);
        currentByEndpoint.put(endpointId, next);
        generationBumps.increment();
        return next;
    }

    /**
     * 登记已观察到的世代（rebuild 场景：重放历史时把见过的最大代登记进去，防归零）。
     * 取历史最大值 merge。
     */
    public void observeGeneration(long endpointId, long generation) {
        currentByEndpoint.merge(endpointId, generation, Math::max);
    }

    /** 世代屏障校验：triple 的世代是否为该端点当前登记代。 */
    public boolean isCurrent(GenerationTriple triple) {
        Long current = currentByEndpoint.get((long) triple.endpointId());
        return current != null && current == triple.generation();
    }

    /** 跨代拒绝计数（供观测：旧代事件整报 REJECT）。 */
    public void recordCrossGenerationReject() {
        crossGenerationRejects.increment();
    }

    public long crossGenerationRejects() {
        return crossGenerationRejects.sum();
    }

    public long generationBumps() {
        return generationBumps.sum();
    }

    /** 端点当前代（未登记返回 -1；调试/测试）。 */
    public long currentGeneration(long endpointId) {
        return currentByEndpoint.getOrDefault(endpointId, -1L);
    }

    /** 全部端点当前代快照（调试/测试）。 */
    public Map<Long, Long> currentGenerations() {
        return Map.copyOf(currentByEndpoint);
    }
}
