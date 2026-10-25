package org.flexlb.balance.schedulev2.decode;

import org.flexlb.balance.scheduler.decode.DecodeRequirement;

/**
 * Decode worker 准备的请求容量需求和评分公式，不包含节点选择或节点占用快照。
 */
public record DecodeInputs(DecodeRequirement requirement) implements AutoCloseable {

    @Override
    public void close() {
        // 只有请求材料，不持有节点资源。
    }
}
