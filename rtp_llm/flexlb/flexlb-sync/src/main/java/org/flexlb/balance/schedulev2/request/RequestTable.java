package org.flexlb.balance.schedulev2.request;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 新链路的请求表：requestId → Request，负责查找、去重和统计。
 */
public final class RequestTable {

    private final ConcurrentHashMap<String, Request> requests = new ConcurrentHashMap<>();

    /**
     * 登记请求。requestId 已存在时返回 false。
     */
    public boolean add(Request request) {
        return requests.putIfAbsent(request.requestId(), request) == null;
    }

    /**
     * 只移除这一个请求对象，同一个 requestId 的新请求不受影响。
     */
    public boolean remove(Request request) {
        return requests.remove(request.requestId(), request);
    }

    public Request get(String requestId) {
        return requestId == null ? null : requests.get(requestId);
    }

    public int size() {
        return requests.size();
    }

    /**
     * 还在等待发布的请求数。
     */
    public int waitingCount() {
        int count = 0;
        for (Request request : requests.values()) {
            if (request.stage().isWaiting()) {
                count++;
            }
        }
        return count;
    }

    public List<Request> list() {
        return new ArrayList<>(requests.values());
    }
}
