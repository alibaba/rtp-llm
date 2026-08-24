package org.flexlb.state;

/**
 * D 侧预约结果。
 */
public enum ReserveResult {

    /** 预约成功（D① 影子预占已入账）。 */
    OK,

    /** 拒绝：同 requestId 存活条目仍在（重复预约）。 */
    DUPLICATE_ALIVE,

    /** 拒绝：同 requestId 命中墓碑。 */
    DUPLICATE_TOMBSTONE,

    /** 拒绝：容量闸门（M2 预留值，当前容量组件未接入，reserve 不会返回）。 */
    CAPACITY_REJECTED
}
