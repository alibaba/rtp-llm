package org.flexlb.balance.schedulev2.main;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.ProcessWorkers;
import org.flexlb.balance.schedulev2.QueueBlockReason;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.SelectionResult;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.schedulev2.prefill.PrefillQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.PrefillChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.balance.strategy.PrefillInputs;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.V2QueueReporter;
import org.flexlb.util.Logger;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * 多个 worker 并行准备缓存匹配和请求自身耗时，主队列调度线程用最新负载作最终选择。
 * 每处理一份材料，紧接着占子队列名额、抢占并交给 Prefill 队列，再处理下一份材料。
 * <p>
 * 名额上限 = min(maxQueuedRequests, Prefill 机器数 × maxInflightPerPrefillWorker × prefillSubqueueLimitMultiplier)，
 * 名额满时，高优先级请求尝试挤掉选中 Prefill 队列里的低优先级请求。名额保留到发布时才归还。
 * 某个 group 暂时没有可选的 Prefill 或拿不到名额时，这一轮跳过同 group 的请求，
 * 其他 group 的请求照常出队。
 */
public final class MainQueueLoop {

    /**
     * 没事可做时最多等这么久再扫一遍，兜底没有发出变化通知的情况。
     */
    static final long IDLE_WAIT_MS = 50L;
    /**
     * 每一轮最多为这么多个请求选机器；直接跳过的请求不占次数，避免后面的 group 一直扫不到。
     */
    static final int MAX_SCAN_PER_ROUND = 256;

    private final MainQueue mainQueue;
    private final LoopSignal loopSignal;
    private final RequestCleanup cleanup;
    private final QueueCapacity queueCapacity;
    private final EndpointRegistry endpointRegistry;
    private final PlacementAvailability placementAvailability;
    private final ConfigService configService;
    private final PrefillPicker prefillPicker;
    private final PrefillQueueRegistry prefillQueues;
    private final ProcessWorkers<PrefillInputs> processWorkers;
    private final V2QueueReporter queueReporter;
    /**
     * 这个部署里 Prefill 用的角色：PD 分离时是 PREFILL，PD 合一时是 PDFUSION。
     */
    private final RoleType prefillRole;
    /**
     * 选好 Prefill 后把请求交给 Prefill 队列，请求上已经带着选好的 {@link PrefillChoice}。
     */
    private final Consumer<Request> sendToPrefillQueue;
    /**
     * Prefill 机器上下线或容量变化时唤醒循环，让等待中的请求马上重试。
     */
    private final PlacementAvailability.Listener prefillChangeListener = this::handleAvailabilityChanged;
    private final AtomicBoolean running = new AtomicBoolean();
    private final Thread thread;

    public MainQueueLoop(MainQueue mainQueue,
                         LoopSignal loopSignal,
                         RequestCleanup cleanup,
                         QueueCapacity queueCapacity,
                         EndpointRegistry endpointRegistry,
                         PlacementAvailability placementAvailability,
                         ConfigService configService,
                         PrefillPicker prefillPicker,
                         PrefillQueueRegistry prefillQueues,
                         RoleType prefillRole,
                         Consumer<Request> sendToPrefillQueue,
                         ProcessWorkers<PrefillInputs> processWorkers,
                         V2QueueReporter queueReporter) {
        this.queueReporter = queueReporter;
        this.mainQueue = mainQueue;
        this.loopSignal = loopSignal;
        this.cleanup = cleanup;
        this.queueCapacity = queueCapacity;
        this.endpointRegistry = endpointRegistry;
        this.placementAvailability = placementAvailability;
        this.configService = configService;
        this.prefillPicker = prefillPicker;
        this.prefillQueues = prefillQueues;
        this.processWorkers = processWorkers;
        this.prefillRole = prefillRole;
        this.sendToPrefillQueue = sendToPrefillQueue;
        this.thread = new Thread(this::run, "flexlb-v2-main-queue");
        this.thread.setDaemon(true);
    }

    public void start() {
        if (running.compareAndSet(false, true)) {
            placementAvailability.addListener(prefillChangeListener);
            thread.start();
        }
    }

    public void close() {
        processWorkers.close();
        if (running.compareAndSet(true, false)) {
            placementAvailability.removeListener(prefillChangeListener);
            loopSignal.wake();
        }
    }

    private void handleAvailabilityChanged(PlacementAvailability.Event event) {
        if (event.key().role() == prefillRole) {
            loopSignal.wake();
        }
    }

    private void run() {
        while (running.get()) {
            boolean moved;
            try {
                moved = runOnce();
            } catch (Throwable failure) {
                Logger.error("V2 main queue loop round failed", failure);
                moved = false;
            }
            if (!moved) {
                loopSignal.await(IDLE_WAIT_MS);
            }
        }
    }

    /**
     * 扫一轮主队列，返回这一轮有没有请求离开主队列。
     */
    boolean runOnce() {
        FlexlbConfig config = configService.loadBalanceConfig();
        long cacheTtlMs = config.getDispatcher().getCacheMatchResultTtlMs();
        processWorkers.removeExpired(cacheTtlMs);
        boolean noPrefill = endpointRegistry.getEndpointCount(prefillRole) == 0;
        Set<String> blockedGroups = new HashSet<>();
        boolean moved = false;
        int scanned = 0;
        Iterator<Request> iterator = mainQueue.iterator();
        while (iterator.hasNext() && scanned < MAX_SCAN_PER_ROUND) {
            Request request = iterator.next();
            if (request.stage() != Stage.MAIN_WAITING) {
                mainQueue.remove(request);
                continue;
            }
            String group = prefillPicker.groupOf(request.ctx());
            if (blockedGroups.contains(group)) {
                continue;
            }
            scanned++;
            if (noPrefill) {
                queueReporter.blocked(request, Stage.MAIN_WAITING, QueueBlockReason.NO_AVAILABLE_PREFILL, "all");
                blockedGroups.add(group);
                continue;
            }
            SelectionResult<PrefillInputs> completed = processWorkers.get(request, cacheTtlMs);
            if (completed == null) {
                processWorkers.submit(request, () -> prefillPicker.prepare(request.ctx(), prefillRole, group));
                completed = processWorkers.get(request, cacheTtlMs);
            }
            if (completed == null) {
                queueReporter.blocked(request, Stage.MAIN_WAITING, QueueBlockReason.PREPARING, "all");
                continue;
            }
            SelectionResult<PrefillInputs> result = completed;
            try {
                if (request.stage() != Stage.MAIN_WAITING) {
                    continue;
                }
                if (result.failure() != null) {
                    request.fail(StrategyErrorType.DISPATCH_FAILED,
                            "prefill selection failed: " + result.failure().getMessage());
                    moved = true;
                } else if (!prefillPicker.inputsAreCurrent(result.value(), prefillRole, group)) {
                    queueReporter.blocked(request, Stage.MAIN_WAITING, QueueBlockReason.CANDIDATES_CHANGED, "all");
                    processWorkers.discard(request);
                } else if (pickPrefill(request, group, result.value(), config)) {
                    moved = true;
                } else {
                    blockedGroups.add(group);
                }
            } catch (Throwable failure) {
                request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill selection failed: " + failure.getMessage());
                Logger.error("V2 prefill selection result failed", failure);
                moved = true;
            } finally {
                if (request.stage() != Stage.MAIN_WAITING) {
                    processWorkers.discard(request);
                }
            }
        }
        return moved;
    }

    /**
     * 选 Prefill 并尝试占子队列名额。返回 false 表示暂时没有机器或名额，请求留在主队列。
     */
    private boolean pickPrefill(Request request, String group, PrefillInputs inputs, FlexlbConfig config) {
        PlacementResult<SelectedRole, RoleType> result = prefillPicker.pick(request, prefillRole, group, inputs);
        switch (result.status()) {
            case SUCCESS -> {
                try (SelectedRole selected = result.value()) {
                    return sendPicked(request, selected, config);
                }
            }
            case BLOCKED -> {
                queueReporter.blocked(request, Stage.MAIN_WAITING, QueueBlockReason.NO_AVAILABLE_PREFILL, "all");
                return false;
            }
            default -> {
                if (result.failure() == null) {
                    request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill selection " + result.status());
                } else {
                    request.fail(result.failure());
                }
                return true;
            }
        }
    }

    private boolean sendPicked(Request request, SelectedRole selected, FlexlbConfig config) {
        PrefillChoice choice;
        try {
            choice = PrefillChoice.from(selected, request.seqLen());
        } catch (RuntimeException failure) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill selection failed: " + failure.getMessage());
            return true;
        }
        if (request.stage() != Stage.MAIN_WAITING) {
            return true;
        }
        if (!reserveQueueSlot(request, choice, config)) {
            queueReporter.blocked(request, Stage.MAIN_WAITING, QueueBlockReason.SHARED_QUEUE_FULL,
                    choice.endpoint().ipPort());
            return false;
        }
        request.setPrefillChoice(choice);
        if (!request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING)) {
            // 选的过程中请求已经结束
            cleanup.releaseQueueSlot(request);
            return true;
        }
        mainQueue.remove(request);
        sendToPrefillQueue.accept(request);
        return true;
    }

    /**
     * 先尝试占共享名额；满了只在选中的 Prefill 队列里挤掉一个低优先级请求，再重试一次。
     */
    private boolean reserveQueueSlot(Request request, PrefillChoice choice, FlexlbConfig config) {
        int endpointCount = endpointRegistry.getEndpointCount(prefillRole);
        if (!queueCapacity.tryReserveEngineSubQueueSlot(config, prefillRole, endpointCount)) {
            if (!config.isPriorityOrdering()
                    || request.priority() < config.priorityOrdering().getHighPriorityThreshold()) {
                return false;
            }
            PrefillQueue queue = prefillQueues.find(choice.endpoint());
            if (queue == null || !queue.preemptWaiting(request.priority(),
                    config.priorityOrdering().getHighPriorityThreshold())) {
                return false;
            }
            if (!queueCapacity.tryReserveEngineSubQueueSlot(config, prefillRole, endpointCount)) {
                return false;
            }
        }
        request.holdQueueSlot(prefillRole);
        return true;
    }
}
