package org.flexlb.state;

/**
 * P 侧登记结果。
 */
public enum RegisterResult {

    /** 登记成功。 */
    OK,

    /** 拒绝：同 requestId 存活条目仍在（重复登记）。 */
    DUPLICATE_ALIVE,

    /** 拒绝：同 requestId 命中墓碑（终态保持期内重复登记）。 */
    DUPLICATE_TOMBSTONE,

    /** 拒绝：容量闸门（M2 预留值，当前容量组件未接入，register 不会返回）。 */
    CAPACITY_REJECTED
}
