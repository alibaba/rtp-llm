package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.QueueBlockReason;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.RequestStatusTimeout;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.request.PrefillChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.V2QueueReporter;
import org.flexlb.util.Logger;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * 一台 Prefill 的独立队列线程。不同机器互不等待，同一机器的容量检查和 dispatch 串行执行。
 * <p>
 * 每一轮处理自己的队列：
 * <ol>
 * <li>机器已下线或同一地址已换成重新上线的机器时，把等待的请求退回主队列，丢掉这个队列；</li>
 * <li>有新的引擎上报时更新已 dispatch 记录，并移除发布后超时还没被引擎上报的记录；</li>
 * <li>按顺序 dispatch 等待的请求。在途满时停在队首；只有 KV 不够时，允许后面较小的请求先走，
 * 直到队首等到饥饿阈值，这台机器改为严格按顺序。</li>
 * </ol>
 * dispatch 出去的请求进入 DECODE_WAITING，带着名额交给 Decode 队列，发布后才归还名额和排队总数。
 */
public final class PrefillQueueLoop {

    /**
     * 有请求在等机器空出来时，隔这么久看一次引擎有没有新的上报。
     */
    static final long BLOCKED_WAIT_MS = 5L;
    /**
     * 没有等待的请求时最多等这么久，有新请求或记录减少时会被提前唤醒。
     */
    static final long IDLE_WAIT_MS = 50L;

    private final PrefillQueueRegistry queues;
    private final PrefillQueue queue;
    private final LoopSignal loopSignal;
    private final RequestCleanup cleanup;
    private final MainQueue mainQueue;
    private final EndpointRegistry endpointRegistry;
    private final ConfigService configService;
    private final RequestSchedulerReporter reporter;
    private final V2QueueReporter queueReporter;
    /**
     * dispatch 出去的请求交给 Decode 队列。
     */
    private final Consumer<Request> sendToDecodeQueue;
    private final AtomicBoolean running = new AtomicBoolean();
    private final Thread thread;

    public PrefillQueueLoop(PrefillQueue queue,
                            PrefillQueueRegistry queues,
                            RequestCleanup cleanup,
                            MainQueue mainQueue,
                            EndpointRegistry endpointRegistry,
                            ConfigService configService,
                            RequestSchedulerReporter reporter,
                            Consumer<Request> sendToDecodeQueue,
                            V2QueueReporter queueReporter) {
        this.queueReporter = queueReporter;
        this.queues = queues;
        this.queue = queue;
        this.loopSignal = queue.signal();
        this.cleanup = cleanup;
        this.mainQueue = mainQueue;
        this.endpointRegistry = endpointRegistry;
        this.configService = configService;
        this.reporter = reporter;
        this.sendToDecodeQueue = sendToDecodeQueue;
        this.thread = new Thread(this::run, "flexlb-v2-prefill-" + queue.endpoint().ipPort());
        this.thread.setDaemon(true);
    }

    public void start() {
        if (running.compareAndSet(false, true)) {
            thread.start();
        }
    }

    public void close() {
        if (running.compareAndSet(true, false)) {
            loopSignal.wake();
        }
    }

    private void run() {
        while (running.get()) {
            boolean hasWaiting;
            try {
                hasWaiting = runOnce();
            } catch (Throwable failure) {
                Logger.error("V2 prefill queue loop round failed", failure);
                hasWaiting = true;
            }
            loopSignal.await(hasWaiting ? BLOCKED_WAIT_MS : IDLE_WAIT_MS);
        }
    }

    /**
     * 处理自己这台机器的队列，返回还有没有请求在等 dispatch。机器下线时退回请求并停止这个线程。
     */
    boolean runOnce() {
        int maxInflight = configService.loadBalanceConfig().getDispatcher().getMaxInflightPerPrefillWorker();
        long requestUpdateStatusTimeoutMs = RequestStatusTimeout.timeoutMs(configService);
        long nowMs = System.currentTimeMillis();
        long starvationThresholdMs =
                configService.loadBalanceConfig().getDispatcher().getDispatchStarvationThresholdMs();
        returnMarkedRequests();
        if (!isCurrent(queue)) {
            queues.remove(queue);
            sendBackToMainQueue(queue);
            close();
            return false;
        }
        queue.removeRequestStatusExpired(nowMs, requestUpdateStatusTimeoutMs, reporter);
        dispatchWaiting(queue, maxInflight, starvationThresholdMs, nowMs);
        return queue.waitingCount() > 0;
    }

    /**
     * 退回处理不受 Prefill 在途或 KV 容量限制，由同一个线程串行处理退回和下发。
     */
    private void returnMarkedRequests() {
        Request request;
        while ((request = queue.pollPendingReturn()) != null) {
            returnToMain(request);
        }
    }

    private void returnToMain(Request request) {
        if (!request.moveTo(Stage.PREFILL_WAITING, Stage.MAIN_WAITING)) {
            return;
        }
        Integer preemptorPriority = request.queueProgress().prefillReturnPriority();
        queue.removeWaiting(request);
        request.setPrefillChoice(null);
        request.setPrefillDecidedAtMs(0L);
        request.setPrefillQueuedAtMs(0L);
        cleanup.releaseQueueSlot(request);
        if (preemptorPriority != null) {
            queueReporter.preemptedReturn(preemptorPriority, request.priority(), queue.endpoint().ipPort());
        }
        mainQueue.addBack(request);
    }

    /**
     * 队列对应的机器是否还在：注册表里这个地址还是同一个 PrefillEndpoint 对象。
     */
    private boolean isCurrent(PrefillQueue queue) {
        PrefillEndpoint endpoint = queue.endpoint();
        return !endpoint.isRetired()
                && endpointRegistry.get(endpoint.getStatus().getRole(), endpoint.ipPort()) == endpoint;
    }

    private void dispatchWaiting(PrefillQueue queue, int maxInflight, long starvationThresholdMs, long nowMs) {
        Request head = null;
        Iterator<Request> iterator = queue.waitingIterator();
        while (iterator.hasNext()) {
            Request request = iterator.next();
            if (request.stage() != Stage.PREFILL_WAITING) {
                queue.removeWaiting(request);
                continue;
            }
            if (request.queueProgress().prefillReturnPriority() != null) {
                continue;
            }
            long kvTokens = request.prefillChoice().kvTokens(request.seqLen());
            if (!queue.hasDispatchSlot(maxInflight)) {
                queueReporter.blocked(request, Stage.PREFILL_WAITING, QueueBlockReason.PREFILL_INFLIGHT_FULL,
                        queue.endpoint().ipPort());
                return;
            }
            if (!queue.canDispatch(kvTokens, maxInflight)) {
                queueReporter.blocked(request, Stage.PREFILL_WAITING, QueueBlockReason.PREFILL_KV_FULL,
                        queue.endpoint().ipPort());
                if (head == null) {
                    head = request;
                }
                if (starvationThresholdMs == 0L || nowMs - head.prefillQueuedAtMs() >= starvationThresholdMs) {
                    return;
                }
                continue;
            }
            dispatch(queue, request, kvTokens);
        }
    }

    private void dispatch(PrefillQueue queue, Request request, long kvTokens) {
        PrefillChoice choice = request.prefillChoice();
        // 先加记录再推进阶段：推进成功后请求随时可能结束，结束时的清理要能找到这条记录
        queue.addDispatched(new DispatchedRequest(request, kvTokens, choice.predictedMs()));
        if (!request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING)) {
            queue.removeDispatched(request);
            queue.removeWaiting(request);
            return;
        }
        queue.removeWaiting(request);
        request.setPrefillDecidedAtMs(System.currentTimeMillis());
        reportSelection(choice);
        sendToDecodeQueue.accept(request);
    }

    private static void reportSelection(PrefillChoice choice) {
        try {
            choice.commitTelemetry().run();
        } catch (RuntimeException failure) {
            Logger.warn("V2 prefill selection telemetry failed: engine={}", choice.endpoint().ipPort(), failure);
        }
    }

    /**
     * 机器下线后，把还在等待的请求退回主队列重新选 Prefill。已 dispatch 的请求不受影响。
     */
    private void sendBackToMainQueue(PrefillQueue queue) {
        Iterator<Request> iterator = queue.waitingIterator();
        while (iterator.hasNext()) {
            Request request = iterator.next();
            returnToMain(request);
        }
    }
}
