package org.flexlb.balance.queue;

/**
 * 每一轮从队列里选出要交付的请求并交给 {@link GroupCommitter} 交付。
 * <p>
 * 单请求模式一次只取队首；FIXED_WINDOW 模式会等一小段时间，把多个请求凑成一组再交付。
 */
interface QueuePicker {

    /**
     * 跑一轮：选请求、交付，或者告诉调度线程接下来要等什么。
     */
    ProcessQueueResult processQueue();
}
