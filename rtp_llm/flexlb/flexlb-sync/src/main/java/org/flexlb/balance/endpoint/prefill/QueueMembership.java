package org.flexlb.balance.endpoint.prefill;

/**
 * 请求和等待队列索引的关系，与预占、引擎执行无关。
 * WAITING：在等待队列索引里；STOP_DETACHED：停机时已从索引摘下，等停止回调确认；UNINDEXED：已离开索引（已提交或已退役）。
 */
public enum QueueMembership {
    UNINDEXED, WAITING, STOP_DETACHED
}
