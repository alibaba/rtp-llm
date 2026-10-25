package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.schedulev2.request.PrefillChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * NON_BATCH_V2 里所有 Prefill 队列，按 {@link PrefillEndpoint} 对象查找。
 * PrefillEndpoint 没有重写 equals，同一地址重新上线后是另一个对象，所以也是另一个队列。
 * <p>
 * 只有 NON_BATCH_V2 会往里加队列，其他模式下一直是空的。
 */
@Component
public class PrefillQueueRegistry {

    private final ConcurrentHashMap<PrefillEndpoint, PrefillQueue> queues = new ConcurrentHashMap<>();
    private final ConfigService configService;
    private final Map<PrefillEndpoint, PrefillQueueLoop> loops = new HashMap<>();
    private Function<PrefillQueue, PrefillQueueLoop> loopFactory;
    private boolean closed;

    public PrefillQueueRegistry(ConfigService configService) {
        this.configService = configService;
    }

    /**
     * 把请求加进它选中的 Prefill 的队列，队列不存在时创建。
     * 和 {@link #remove(PrefillQueue)} 互斥：要么加进将被移除的队列、由移除方处理，要么加进新建的队列。
     */
    public synchronized void addWaiting(Request request) {
        if (closed) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill queues are shutting down");
            return;
        }
        PrefillChoice choice = request.prefillChoice();
        queues.compute(choice.endpoint(), (endpoint, queue) -> {
            PrefillQueue target = queue == null
                    ? new PrefillQueue(endpoint, configService.loadBalanceConfig().isPriorityOrdering())
                    : queue;
            target.addWaiting(request);
            return target;
        });
        startLoop(queues.get(choice.endpoint()));
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
        if (closed || loopFactory == null || loops.containsKey(queue.endpoint())) {
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
     * 这台 Prefill 的队列，没有时返回 null。
     */
    public PrefillQueue find(PrefillEndpoint endpoint) {
        return queues.get(endpoint);
    }

    /**
     * 请求选中的 Prefill 的队列，没选或队列已移除时返回 null。
     */
    public PrefillQueue find(Request request) {
        PrefillChoice choice = request.prefillChoice();
        if (choice == null) {
            return null;
        }
        return queues.get(choice.endpoint());
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
