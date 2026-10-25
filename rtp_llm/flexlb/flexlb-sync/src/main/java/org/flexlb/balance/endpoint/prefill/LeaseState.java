package org.flexlb.balance.endpoint.prefill;

/**
 * 一份 Prefill 预占的状态。
 * OPEN：已预占、还没提交，可以回滚；OWNED：已提交，容量由请求表持有，等请求结束时释放；CLOSED：已释放。
 */
enum LeaseState {
    OPEN, OWNED, CLOSED
}
