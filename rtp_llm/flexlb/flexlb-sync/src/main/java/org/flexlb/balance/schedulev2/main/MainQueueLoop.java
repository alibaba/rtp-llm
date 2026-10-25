package org.flexlb.balance.schedulev2.main;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.schedulev2.request.PrefillChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * 主队列出队线程：按顺序取请求，先占一个主队列名额，再选 Prefill，然后交给 Prefill 队列。
 * <p>
 * 名额上限 = min(maxQueuedRequests, Prefill 机器数 × maxInflightPerPrefillWorker × prefillSubqueueLimitMultiplier)，
 * 名额满时整个主队列等名额归还。某个 group 暂时没有可选的 Prefill 时，这一轮跳过同 group 的请求，
 * 其他 group 的请求照常出队。
 */
public final class MainQueueLoop {

    /**
     * 没事可做时最多等这么久再扫一遍，兜底没有发出变化通知的情况。
     */
    static final long IDLE_WAIT_MS = 50L;
    /**
     * 每一轮最多看这么多个请求，避免某个 group 一直选不到时每轮都扫完整个队列。
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
                         RoleType prefillRole,
                         Consumer<Request> sendToPrefillQueue) {
        this.mainQueue = mainQueue;
        this.loopSignal = loopSignal;
        this.cleanup = cleanup;
        this.queueCapacity = queueCapacity;
        this.endpointRegistry = endpointRegistry;
        this.placementAvailability = placementAvailability;
        this.configService = configService;
        this.prefillPicker = prefillPicker;
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
            int endpointCount = endpointRegistry.getEndpointCount(prefillRole);
            if (!queueCapacity.tryReserveEngineSubQueueSlot(config, prefillRole, endpointCount)) {
                return moved;
            }
            request.holdQueueSlot(prefillRole);
            if (pickPrefill(request, group)) {
                moved = true;
            } else {
                blockedGroups.add(group);
            }
        }
        return moved;
    }

    /**
     * 给已占名额的请求选 Prefill。返回 false 表示这个 group 暂时没有可选的机器，请求留在主队列。
     */
    private boolean pickPrefill(Request request, String group) {
        PlacementResult<SelectedRole, RoleType> result;
        try {
            result = prefillPicker.pick(request.ctx(), prefillRole, group);
        } catch (RuntimeException failure) {
            cleanup.releaseQueueSlot(request);
            request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill selection failed: " + failure.getMessage());
            return true;
        }
        switch (result.status()) {
            case SUCCESS -> {
                sendPicked(request, result.value());
                return true;
            }
            case BLOCKED -> {
                cleanup.releaseQueueSlot(request);
                return false;
            }
            default -> {
                cleanup.releaseQueueSlot(request);
                if (result.failure() == null) {
                    request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill selection " + result.status());
                } else {
                    request.fail(result.failure());
                }
                return true;
            }
        }
    }

    private void sendPicked(Request request, SelectedRole selected) {
        PrefillChoice choice;
        try {
            choice = PrefillChoice.from(selected, request.seqLen());
        } catch (RuntimeException failure) {
            cleanup.releaseQueueSlot(request);
            request.fail(StrategyErrorType.DISPATCH_FAILED, "prefill selection failed: " + failure.getMessage());
            return;
        }
        request.setPrefillChoice(choice);
        if (!request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING)) {
            // 选的过程中请求已经结束
            cleanup.releaseQueueSlot(request);
            return;
        }
        mainQueue.remove(request);
        sendToPrefillQueue.accept(request);
    }
}
