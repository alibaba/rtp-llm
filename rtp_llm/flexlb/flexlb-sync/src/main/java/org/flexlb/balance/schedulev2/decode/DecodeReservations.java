package org.flexlb.balance.schedulev2.decode;

import org.springframework.stereotype.Component;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * V2 仍在管理的 Decode 预占 ID。创建预占前登记，归还或交给引擎后移除。
 * 旧孤儿清理持有 endpoint 锁时只查这个并发表，不获取请求锁或队列锁。
 */
@Component
public final class DecodeReservations {

    private final Set<String> requests = ConcurrentHashMap.newKeySet();

    public void add(String requestId) {
        requests.add(requestId);
    }

    public void remove(String requestId) {
        requests.remove(requestId);
    }

    public boolean contains(String requestId) {
        return requests.contains(requestId);
    }
}
