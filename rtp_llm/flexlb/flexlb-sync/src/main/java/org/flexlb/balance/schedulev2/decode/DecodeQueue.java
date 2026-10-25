package org.flexlb.balance.schedulev2.decode;

import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestOrder;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ConcurrentSkipListSet;

/**
 * 全局 Decode 队列，Prefill dispatch 出去的请求都进这里，分三部分：
 * <ul>
 * <li>waiting：还没选 Decode 的请求（DECODE_WAITING）；</li>
 * <li>reserved：已经在某台 Decode 上 reserve、等这台 Decode 给 permit 的请求（DECODE_RESERVED）；</li>
 * <li>published：已经发布、在 Decode 上有 reserve、引擎还没上报的请求（PUBLISHED）。</li>
 * </ul>
 * waiting 和 reserved 按 {@link RequestOrder} 排序。只有 Decode 队列线程会出队，其他线程只会加入或移除 waiting。
 * published 只由 Decode 队列线程访问。
 * <p>
 * reserved 里的请求结束时，别的线程不把它移出 reserved，由 Decode 队列线程扫到后归还 reserve 再移除。
 * 这样 V2 对 Decode reserve 的操作（reserve、拿 permit、dispatch、归还）都在 Decode 队列线程里，互相不会撞上。
 */
public final class DecodeQueue {

    private final ConcurrentSkipListSet<Request> waiting;
    private final ConcurrentSkipListSet<Request> reserved;
    private final List<Request> published = new ArrayList<>();

    public DecodeQueue(boolean priorityOrdering) {
        this.waiting = new ConcurrentSkipListSet<>(RequestOrder.of(priorityOrdering));
        this.reserved = new ConcurrentSkipListSet<>(RequestOrder.of(priorityOrdering));
    }

    public void addWaiting(Request request) {
        waiting.add(request);
    }

    public void removeWaiting(Request request) {
        waiting.remove(request);
    }

    public Iterator<Request> waitingIterator() {
        return waiting.iterator();
    }

    public int waitingCount() {
        return waiting.size();
    }

    public void addReserved(Request request) {
        reserved.add(request);
    }

    public void removeReserved(Request request) {
        reserved.remove(request);
    }

    public Iterator<Request> reservedIterator() {
        return reserved.iterator();
    }

    public int reservedCount() {
        return reserved.size();
    }

    public void addPublished(Request request) {
        published.add(request);
    }

    public Iterator<Request> publishedIterator() {
        return published.iterator();
    }

    public int publishedCount() {
        return published.size();
    }
}
