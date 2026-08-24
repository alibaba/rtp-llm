package org.flexlb.state;

import java.util.List;

/**
 * 跨侧推导视图（只读推导，不参与裁决）。
 *
 * <p>KV_TRANSFERRING 推导：P 条目 phase ∈ {P_RUNNING} ∧ D 条目已报
 * （engineOwned）= 传输重叠（边算边传窗口）。</p>
 *
 * @param kvTransferringRequestIds 传输重叠中的请求 ID 列表
 */
public record CrossSideView(List<Long> kvTransferringRequestIds) {

    public CrossSideView {
        kvTransferringRequestIds = List.copyOf(kvTransferringRequestIds);
    }

    /** 传输重叠请求数。 */
    public int kvTransferringCount() {
        return kvTransferringRequestIds.size();
    }
}
