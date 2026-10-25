package org.flexlb.balance.strategy;

import org.flexlb.cache.domain.CacheMatchResult;

import java.util.List;

/**
 * worker 准备的完整 Prefill 候选材料。没有选中节点，也没有占用节点容量或持有 endpoint hold。
 */
public record PrefillInputs(CacheMatchResult cacheMatch,
                            List<PrefillCandidateWork> candidates)
        implements
            AutoCloseable {

    public PrefillInputs {
        candidates = List.copyOf(candidates);
    }

    @Override
    public void close() {
        // 只有计算材料，不需要归还节点资源。
    }
}
