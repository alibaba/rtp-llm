package org.flexlb.balance.schedulev2.request;

import java.util.Comparator;

/**
 * 主队列、Prefill 队列和 Decode 队列里请求的出队顺序。
 */
public final class RequestOrder {

    private RequestOrder() {
    }

    /**
     * 开启优先级排序时按优先级从高到低、同优先级按进主队列的顺序；否则只按进主队列的顺序。
     */
    public static Comparator<Request> of(boolean priorityOrdering) {
        Comparator<Request> bySeq = Comparator.comparingLong(Request::queueSeq);
        return priorityOrdering ? Comparator.comparingInt(Request::priority).reversed().thenComparing(bySeq) : bySeq;
    }
}
