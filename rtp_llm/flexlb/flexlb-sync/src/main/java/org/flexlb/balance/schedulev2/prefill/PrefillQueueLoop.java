package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.schedulev2.DispatchConfirmTimeout;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.request.PrefillChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Prefill 队列线程：轮流处理每台 Prefill 的队列。
 * <p>
 * 每一轮对每个队列：
 * <ol>
 * <li>机器已下线或同一地址已换成重新上线的机器时，把等待的请求退回主队列，丢掉这个队列；</li>
 * <li>有新的引擎上报时更新已 dispatch 记录，并移除发布后超时还没被引擎上报的记录；</li>
 * <li>按顺序 dispatch 等待的请求，队首放不下时这台机器停在队首，等记录减少或下一次上报。</li>
 * </ol>
 * dispatch 出去的请求进入 DECODE_WAITING，归还主队列名额和排队总数，交给 Decode 队列。
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
    private final LoopSignal loopSignal;
    private final RequestCleanup cleanup;
    private final MainQueue mainQueue;
    private final EndpointRegistry endpointRegistry;
    private final ConfigService configService;
    private final RequestSchedulerReporter reporter;
    /**
     * dispatch 出去的请求交给 Decode 队列。
     */
    private final Consumer<Request> sendToDecodeQueue;
    private final AtomicBoolean running = new AtomicBoolean();
    private final Thread thread;

    public PrefillQueueLoop(PrefillQueueRegistry queues,
                            LoopSignal loopSignal,
                            RequestCleanup cleanup,
                            MainQueue mainQueue,
                            EndpointRegistry endpointRegistry,
                            ConfigService configService,
                            RequestSchedulerReporter reporter,
                            Consumer<Request> sendToDecodeQueue) {
        this.queues = queues;
        this.loopSignal = loopSignal;
        this.cleanup = cleanup;
        this.mainQueue = mainQueue;
        this.endpointRegistry = endpointRegistry;
        this.configService = configService;
        this.reporter = reporter;
        this.sendToDecodeQueue = sendToDecodeQueue;
        this.thread = new Thread(this::run, "flexlb-v2-prefill-queue");
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
     * 处理一轮所有队列，返回处理完后还有没有请求在等 dispatch。
     */
    boolean runOnce() {
        int maxInflight = configService.loadBalanceConfig().getDispatcher().getMaxInflightPerPrefillWorker();
        long confirmTimeoutMs = DispatchConfirmTimeout.timeoutMs(configService);
        long nowMs = System.currentTimeMillis();
        boolean hasWaiting = false;
        for (PrefillQueue queue : queues.list()) {
            if (!isCurrent(queue)) {
                queues.remove(queue);
                sendBackToMainQueue(queue);
                continue;
            }
            queue.updateFromEngine();
            removeExpiredDispatched(queue, nowMs, confirmTimeoutMs);
            dispatchWaiting(queue, maxInflight);
            hasWaiting |= queue.waitingCount() > 0;
        }
        return hasWaiting;
    }

    private void removeExpiredDispatched(PrefillQueue queue, long nowMs, long confirmTimeoutMs) {
        int removed = queue.removeExpiredDispatched(nowMs, confirmTimeoutMs);
        for (int i = 0; i < removed; i++) {
            reporter.reportDispatchConfirmTimeout();
        }
    }

    /**
     * 队列对应的机器是否还在：注册表里这个地址还是同一个 PrefillEndpoint 对象。
     */
    private boolean isCurrent(PrefillQueue queue) {
        PrefillEndpoint endpoint = queue.endpoint();
        return endpointRegistry.get(endpoint.getStatus().getRole(), endpoint.ipPort()) == endpoint;
    }

    private void dispatchWaiting(PrefillQueue queue, int maxInflight) {
        Iterator<Request> iterator = queue.waitingIterator();
        while (iterator.hasNext()) {
            Request request = iterator.next();
            if (request.stage() != Stage.PREFILL_WAITING) {
                queue.removeWaiting(request);
                continue;
            }
            long kvTokens = request.prefillChoice().kvTokens(request.seqLen());
            if (!queue.canDispatch(kvTokens, maxInflight)) {
                return;
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
        cleanup.releaseAtPrefillDispatch(request);
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
            queue.removeWaiting(request);
            if (!request.moveTo(Stage.PREFILL_WAITING, Stage.MAIN_WAITING)) {
                continue;
            }
            request.setPrefillChoice(null);
            cleanup.releaseQueueSlot(request);
            mainQueue.addBack(request);
        }
    }
}
