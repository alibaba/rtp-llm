package org.flexlb.state.spi;

/**
 * 端点身份纯值抽象（接入契约）：state 组件<b>不持有任何 endpoint 引用</b>，
 * 只消费端点身份三元组（端点 ID + 角色 + 世代）。
 */
public interface StateEndpointRef {

    /** 端点 ID。 */
    long endpointId();

    /** 端点承担的状态侧角色。 */
    StateRole role();

    /** 端点世代号（S8：世代不匹配的观察整报拒绝）。 */
    long generation();
}
