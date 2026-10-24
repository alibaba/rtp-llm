package org.flexlb.balance.queue.enqueue;

import org.flexlb.balance.endpoint.prefill.PrefillState;

/**
 * 一次持队列锁入队的结果。
 *
 * @param accepted 请求是否已放进队列
 * @param rollback 放锁后需要释放的 Prefill 资源预留；没有则为 null
 * @param failure  入队过程中遇到的异常，放锁后抛出；没有则为 null
 */
record EnqueueAttempt(boolean accepted,
                      PrefillState.RouteReservation rollback,
                      Throwable failure) {
}
