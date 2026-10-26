package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.schedulev2.request.PrefillChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * 按 {@link PrefillEndpoint} 对象管理已发布节点的 Prefill 队列。
 * PrefillEndpoint 没有重写 equals，同一地址重新上线后是另一个对象，所以也是另一个队列。
 * <p>
 * NON_BATCH_V2 调度器提供循环工厂后，启动各节点的队列线程。
 */
@Component
public class PrefillQueueRegistry {

    private final ConcurrentHashMap<PrefillEndpoint, PrefillQueue> queues = new ConcurrentHashMap<>();
    private final Map<PrefillEndpoint, PrefillQueueLoop> loops = new HashMap<>();
    private Function<PrefillQueue, PrefillQueueLoop> loopFactory;
    private boolean closed;

    /**
     * 节点已经可选路，可以启动其队列线程。
     */
    public synchronized void start(PrefillEndpoint endpoint) {
        PrefillQueue queue = endpoint.prefillQueue();
        if (closed || endpoint.isRetired()) {
            return;
        }
        queues.put(endpoint, queue);
        startLoop(queue);
    }

    /**
     * 已启动的循环负责退回等待请求；未启动的队列直接移除。
     */
    public synchronized void retire(PrefillEndpoint endpoint) {
        PrefillQueue queue = queues.get(endpoint);
        if (queue == null) {
            return;
        }
        if (loops.containsKey(endpoint)) {
            queue.signal().wake();
        } else {
            queues.remove(endpoint, queue);
        }
    }

    /**
     * 把请求加进它选中的 Prefill 的已注册队列。
     * 和节点队列的移除互斥，已退役节点拒绝入队。
     */
    public synchronized void addWaiting(Request request) {
        if (closed) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill queues are shutting down");
            return;
        }
        PrefillChoice choice = request.prefillChoice();
        PrefillQueue queue = find(choice.endpoint());
        if (queue == null) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill endpoint queue has retired");
            return;
        }
        queue.addWaiting(request);
    }

    /**
     * 启动每台 Prefill 的独立循环。后续新建的队列也会自动启动，空闲线程阻塞等待自己的信号。
     */
    public synchronized void start(Function<PrefillQueue, PrefillQueueLoop> loopFactory) {
        this.loopFactory = loopFactory;
        for (PrefillQueue queue : queues.values()) {
            startLoop(queue);
        }
    }

    private void startLoop(PrefillQueue queue) {
        if (closed || loopFactory == null || queue.endpoint().isRetired() || loops.containsKey(queue.endpoint())) {
            return;
        }
        PrefillQueueLoop loop = loopFactory.apply(queue);
        loops.put(queue.endpoint(), loop);
        loop.start();
    }

    /**
     * 停止所有 Prefill 线程；调用方已先结束请求，循环不会再派发新请求。
     */
    public synchronized void close() {
        closed = true;
        loops.values().forEach(PrefillQueueLoop::close);
        loops.clear();
    }

    /**
     * 读取节点持有的队列，已退役节点返回 null。
     */
    public PrefillQueue find(PrefillEndpoint endpoint) {
        return endpoint.isRetired() ? null : endpoint.prefillQueue();
    }

    /**
     * 请求选中的 Prefill 的队列，没选或队列已移除时返回 null。
     */
    public PrefillQueue find(Request request) {
        PrefillChoice choice = request.prefillChoice();
        if (choice == null) {
            return null;
        }
        return find(choice.endpoint());
    }

    public List<PrefillQueue> list() {
        return new ArrayList<>(queues.values());
    }

    /**
     * 移除这个队列对象。返回后不会再有请求加进它。
     */
    public synchronized boolean remove(PrefillQueue queue) {
        if (!queues.remove(queue.endpoint(), queue)) {
            return false;
        }
        PrefillQueueLoop loop = loops.remove(queue.endpoint());
        if (loop != null) {
            loop.close();
        }
        return true;
    }
}
