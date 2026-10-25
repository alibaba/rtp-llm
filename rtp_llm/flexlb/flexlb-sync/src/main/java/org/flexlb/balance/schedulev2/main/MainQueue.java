package org.flexlb.balance.schedulev2.main;

import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.Request;
import org.flexlb.balance.schedulev2.Stage;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.Comparator;
import java.util.Iterator;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 主队列：还没选 Prefill 的请求在这里排队。
 * <p>
 * 开启优先级排序时按优先级从高到低、同优先级先进先出；否则只按先进先出。
 * 入队时检查排队总数（主队列 + 各 Prefill 队列里还没 dispatch 的请求），
 * 总数用 {@link QueueCapacity} 记账，请求从 Prefill 队列 dispatch 或结束时减掉。
 */
public final class MainQueue {

    public static final String QUEUE_FULL_MESSAGE = "Too many requests. Your requests are being throttled"
            + " due to system capacity limits. Please try again later.";

    /**
     * 队满时最多挤掉几个低优先级请求再重试，避免并发入队时一直互相抢。
     */
    private static final int MAX_ADD_ATTEMPTS = 3;

    private final QueueCapacity queueCapacity;
    private final LoopSignal loopSignal;
    private final ConcurrentSkipListSet<Request> requests;
    private final AtomicInteger size = new AtomicInteger();
    private final AtomicLong nextSeq = new AtomicLong();

    public MainQueue(QueueCapacity queueCapacity, LoopSignal loopSignal, boolean priorityOrdering) {
        this.queueCapacity = queueCapacity;
        this.loopSignal = loopSignal;
        Comparator<Request> bySeq = Comparator.comparingLong(Request::queueSeq);
        this.requests = new ConcurrentSkipListSet<>(
                priorityOrdering ? Comparator.comparingInt(Request::priority).reversed().thenComparing(bySeq) : bySeq);
    }

    /**
     * 请求入队。排队总数已满时：
     * <ul>
     * <li>没开优先级排序，返回 QUEUE_FULL；</li>
     * <li>请求优先级低于高优先级阈值，返回 QUEUE_FULL_LOW_PRIORITY_REJECTED；</li>
     * <li>否则挤掉主队列里一个更低优先级的请求（最低优先级里最晚入队的）再入队，
     * 找不到可挤的请求时返回 QUEUE_FULL_HIGH_PRIORITY_REJECTED。</li>
     * </ul>
     * 入队成功返回 null。
     */
    public StrategyErrorType add(Request request, FlexlbConfig config) {
        long limit = config.queueScheduler().getMaxQueuedRequests();
        for (int attempt = 0; attempt < MAX_ADD_ATTEMPTS; attempt++) {
            if (queueCapacity.tryIncreaseTotalQueueCount(request.future(), limit)) {
                request.setQueueSeq(nextSeq.incrementAndGet());
                if (requests.add(request)) {
                    size.incrementAndGet();
                }
                // 入队期间请求可能已结束，结束时的清理没看到它，这里补上
                if (request.stage() != Stage.MAIN_WAITING) {
                    remove(request);
                    queueCapacity.decreaseTotalQueueCount(request.future());
                }
                loopSignal.wake();
                return null;
            }
            if (!config.isPriorityOrdering()) {
                return StrategyErrorType.QUEUE_FULL;
            }
            int threshold = config.priorityOrdering().getHighPriorityThreshold();
            if (request.priority() < threshold) {
                return StrategyErrorType.QUEUE_FULL_LOW_PRIORITY_REJECTED;
            }
            Request victim = findLowerPriorityRequest(request.priority());
            if (victim == null) {
                return StrategyErrorType.QUEUE_FULL_HIGH_PRIORITY_REJECTED;
            }
            // 挤掉的请求结束时会同步减掉它占的排队总数，下一轮重试就能计入
            victim.fail(StrategyErrorType.queueRejectionFor(victim.priority(), threshold), QUEUE_FULL_MESSAGE);
        }
        return StrategyErrorType.QUEUE_FULL_HIGH_PRIORITY_REJECTED;
    }

    private Request findLowerPriorityRequest(int priority) {
        for (Request queued : requests.descendingSet()) {
            if (queued.priority() >= priority) {
                return null;
            }
            if (queued.stage() == Stage.MAIN_WAITING) {
                return queued;
            }
        }
        return null;
    }

    /**
     * 只从主队列里拿掉这个请求，不改阶段，也不动排队总数和名额。
     */
    public void remove(Request request) {
        if (requests.remove(request)) {
            size.decrementAndGet();
        }
    }

    /**
     * 按出队顺序遍历；遍历期间的增删不会抛异常。
     */
    public Iterator<Request> iterator() {
        return requests.iterator();
    }

    public int size() {
        return size.get();
    }
}
