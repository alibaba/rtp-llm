package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.schedulev2.PrefillChoice;
import org.flexlb.balance.schedulev2.Request;
import org.flexlb.config.ConfigService;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

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

    public PrefillQueueRegistry(ConfigService configService) {
        this.configService = configService;
    }

    /**
     * 把请求加进它选中的 Prefill 的队列，队列不存在时创建。
     * 和 {@link #remove(PrefillQueue)} 互斥：要么加进将被移除的队列、由移除方处理，要么加进新建的队列。
     */
    public void addWaiting(Request request) {
        PrefillChoice choice = request.prefillChoice();
        queues.compute(choice.endpoint(), (endpoint, queue) -> {
            PrefillQueue target = queue == null
                    ? new PrefillQueue(endpoint, configService.loadBalanceConfig().isPriorityOrdering())
                    : queue;
            target.addWaiting(request);
            return target;
        });
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
    public boolean remove(PrefillQueue queue) {
        return queues.remove(queue.endpoint(), queue);
    }
}
