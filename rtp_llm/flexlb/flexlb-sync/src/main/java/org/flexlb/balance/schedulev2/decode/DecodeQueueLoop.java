package org.flexlb.balance.schedulev2.decode;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.decode.DecodeRequirement;
import org.flexlb.balance.schedulev2.DecodeChoice;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.Request;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.Stage;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Decode 队列线程，每一轮做两件事：
 * <ol>
 * <li>reserved 的请求按顺序向 reserve 的那台 Decode 要 permit，拿到后发布结果（PUBLISHED）。
 * 某台 Decode 给不出 permit 时，这一轮跳过 reserve 在它上面的其他请求，等它有空位时被唤醒；</li>
 * <li>waiting 的请求按顺序选 Decode 并 reserve，成功后进入 DECODE_RESERVED。reserve 带容量上限，放不下就不 reserve。
 * 某个 group 没有放得下的 Decode 时，请求留在 waiting，这一轮跳过同 group 的请求，等 Decode 归还容量时被唤醒。</li>
 * </ol>
 * reserve 的 Decode 下线或 reserve 丢失时，请求退回 waiting 重新选。
 */
public final class DecodeQueueLoop {

    /**
     * 没事可做时最多等这么久再扫一遍，兜底没有发出变化通知的情况。
     */
    static final long IDLE_WAIT_MS = 50L;
    /**
     * 每一轮最多给这么多个 waiting 请求选 Decode，避免某个 group 一直选不到时每轮都扫完整个队列。
     */
    static final int MAX_SCAN_PER_ROUND = 256;

    private final DecodeQueue decodeQueue;
    private final LoopSignal loopSignal;
    private final RequestCleanup cleanup;
    private final PlacementAvailability placementAvailability;
    private final DecodeSelector decodeSelector;
    private final DefaultRouter router;
    /**
     * Decode 上下线或 reserve 容量变化时唤醒循环，让 waiting 的请求马上重试。
     */
    private final PlacementAvailability.Listener decodeChangeListener = this::handleAvailabilityChanged;
    /**
     * Decode 有 permit 空出来时唤醒循环，让 reserved 的请求马上重试。
     */
    private final Runnable capacityListener;
    /**
     * 已经注册了 capacityListener 的 Decode。只在循环线程里访问；弱引用，不阻止下线的 Decode 被回收。
     */
    private final Set<DecodeEndpoint> listenedEndpoints = Collections.newSetFromMap(new WeakHashMap<>());
    private final AtomicBoolean running = new AtomicBoolean();
    private final Thread thread;

    public DecodeQueueLoop(DecodeQueue decodeQueue,
                           LoopSignal loopSignal,
                           RequestCleanup cleanup,
                           PlacementAvailability placementAvailability,
                           DecodeSelector decodeSelector,
                           DefaultRouter router) {
        this.decodeQueue = decodeQueue;
        this.loopSignal = loopSignal;
        this.cleanup = cleanup;
        this.placementAvailability = placementAvailability;
        this.decodeSelector = decodeSelector;
        this.router = router;
        this.capacityListener = loopSignal::wake;
        this.thread = new Thread(this::run, "flexlb-v2-decode-queue");
        this.thread.setDaemon(true);
    }

    public void start() {
        if (running.compareAndSet(false, true)) {
            placementAvailability.addListener(decodeChangeListener);
            thread.start();
        }
    }

    public void close() {
        if (running.compareAndSet(true, false)) {
            placementAvailability.removeListener(decodeChangeListener);
            loopSignal.wake();
        }
    }

    private void handleAvailabilityChanged(PlacementAvailability.Event event) {
        if (event.key().role() == RoleType.DECODE) {
            loopSignal.wake();
        }
    }

    private void run() {
        while (running.get()) {
            boolean moved;
            try {
                moved = runOnce();
            } catch (RuntimeException failure) {
                Logger.error("V2 decode queue loop round failed", failure);
                moved = false;
            }
            if (!moved) {
                loopSignal.await(IDLE_WAIT_MS);
            }
        }
        for (DecodeEndpoint endpoint : listenedEndpoints) {
            endpoint.removeEngineDispatchCapacityListener(capacityListener);
        }
        listenedEndpoints.clear();
    }

    /**
     * 处理一轮，返回这一轮有没有请求发布、换了状态或离开队列。
     */
    boolean runOnce() {
        boolean moved = dispatchReserved();
        moved |= reserveWaiting();
        return moved;
    }

    private boolean dispatchReserved() {
        Set<DecodeEndpoint> fullEndpoints = Collections.newSetFromMap(new IdentityHashMap<>());
        boolean moved = false;
        Iterator<Request> iterator = decodeQueue.reservedIterator();
        while (iterator.hasNext()) {
            Request request = iterator.next();
            DecodeChoice choice = request.decodeChoice();
            if (request.stage() != Stage.DECODE_RESERVED || choice == null) {
                decodeQueue.removeReserved(request);
                continue;
            }
            if (fullEndpoints.contains(choice.endpoint())) {
                continue;
            }
            if (dispatch(request, choice)) {
                moved = true;
            } else {
                fullEndpoints.add(choice.endpoint());
            }
        }
        return moved;
    }

    /**
     * 向 reserve 的 Decode 要 permit 并发布。返回 false 表示这台 Decode 暂时给不出 permit，请求留在 reserved。
     */
    private boolean dispatch(Request request, DecodeChoice choice) {
        DecodeEndpoint endpoint = choice.endpoint();
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition;
        try {
            acquisition = endpoint.acquireDispatchPermit(choice.reservation(), choice.requirement().capacity());
        } catch (RuntimeException failure) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "decode dispatch failed: " + failure.getMessage());
            return true;
        }
        switch (acquisition.status()) {
            case SUCCESS, ALREADY_ACCEPTED -> {
                publish(request, choice, acquisition.permit());
                return true;
            }
            case CAPACITY_FULL -> {
                listenCapacity(endpoint);
                return false;
            }
            case NOT_OWNED, NOT_QUEUED, ENDPOINT_RETIRED -> {
                sendBackToWaiting(request);
                return true;
            }
            default -> {
                request.fail(StrategyErrorType.DISPATCH_FAILED, "decode dispatch permit " + acquisition.status());
                return true;
            }
        }
    }

    private void publish(Request request, DecodeChoice choice, DecodeEndpoint.EngineDispatchPermit permit) {
        DecodeEndpoint.EngineDispatchPermitTransferStatus transfer;
        try {
            transfer = permit.dispatch();
        } catch (RuntimeException failure) {
            permit.release();
            request.fail(StrategyErrorType.DISPATCH_FAILED, "decode dispatch failed: " + failure.getMessage());
            return;
        }
        if (transfer != DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED) {
            sendBackToWaiting(request);
            return;
        }
        decodeQueue.removeReserved(request);
        // 发布失败说明请求已经结束，结束时的清理会归还 reserve
        if (request.publish(Stage.DECODE_RESERVED,
                List.of(request.prefillChoice().serverStatus(), choice.serverStatus()))) {
            cleanup.releasePublished(request);
        }
    }

    /**
     * reserve 的 Decode 下线或 reserve 丢失：归还 reserve，请求退回 waiting 重新选 Decode。
     */
    private void sendBackToWaiting(Request request) {
        decodeQueue.removeReserved(request);
        if (!request.moveTo(Stage.DECODE_RESERVED, Stage.DECODE_WAITING)) {
            return;
        }
        cleanup.releaseDecodeReservation(request);
        decodeQueue.addWaiting(request);
    }

    private void listenCapacity(DecodeEndpoint endpoint) {
        if (listenedEndpoints.add(endpoint)) {
            endpoint.addEngineDispatchCapacityListener(capacityListener);
        }
    }

    private boolean reserveWaiting() {
        Set<String> blockedGroups = new HashSet<>();
        boolean moved = false;
        int scanned = 0;
        Iterator<Request> iterator = decodeQueue.waitingIterator();
        while (iterator.hasNext() && scanned < MAX_SCAN_PER_ROUND) {
            Request request = iterator.next();
            scanned++;
            if (request.stage() != Stage.DECODE_WAITING) {
                decodeQueue.removeWaiting(request);
                continue;
            }
            String group = groupOf(request);
            if (blockedGroups.contains(group)) {
                continue;
            }
            if (pickDecode(request, group)) {
                moved = true;
            } else {
                blockedGroups.add(group);
            }
        }
        return moved;
    }

    /**
     * 按流量策略算出的 group；没配策略时和选中的 Prefill 在同一个 group。
     */
    private String groupOf(Request request) {
        String group = router.resolvePolicyGroup(request.ctx());
        if (StringUtils.isBlank(group)) {
            return request.prefillChoice().serverStatus().getGroup();
        }
        return group;
    }

    /**
     * 给请求选 Decode 并 reserve。返回 false 表示这个 group 暂时没有可选的 Decode，请求留在 waiting。
     */
    private boolean pickDecode(Request request, String group) {
        DecodeRequirement requirement;
        PlacementResult<SelectedRole, RoleType> result;
        try {
            requirement = DecodeRequirement.capture(request.ctx());
            result = decodeSelector.select(request.ctx(), requirement, group);
        } catch (RuntimeException failure) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "decode selection failed: " + failure.getMessage());
            return true;
        }
        switch (result.status()) {
            case SUCCESS -> {
                return reserve(request, result.value(), requirement);
            }
            case BLOCKED -> {
                return false;
            }
            default -> {
                if (result.failure() == null) {
                    request.fail(StrategyErrorType.DISPATCH_FAILED, "decode selection " + result.status());
                } else {
                    request.fail(result.failure());
                }
                return true;
            }
        }
    }

    /**
     * 在选中的 Decode 上 reserve。返回 false 表示这台 Decode 放不下，请求留在 waiting。
     * 选择器在没有放得下的 Decode 时也会选一台忙的，所以放不下由 reserve 带容量上限在锁里判断，不会超额。
     */
    private boolean reserve(Request request, SelectedRole selected, DecodeRequirement requirement) {
        DecodeChoice choice;
        try (selected; WorkerEndpoint.EndpointHold hold = selected.takeEndpointHold()) {
            DecodeEndpoint endpoint = (DecodeEndpoint) hold.endpoint();
            DecodeEndpoint.ReservationHandle reservation =
                    endpoint.reserve(hold, request.requestId(), requirement.inputKvTokens(),
                            requirement.inputAndMaxOutputKvTokens(), requirement.priority(), requirement.capacity());
            if (reservation == null) {
                if (endpoint.reservationHandle(request.requestId()) == null) {
                    return false;
                }
                request.fail(StrategyErrorType.DISPATCH_FAILED,
                        "request id is already reserved on decode " + endpoint.ipPort());
                return true;
            }
            choice = new DecodeChoice(endpoint, reservation, selected.serverStatus(), requirement);
        } catch (RuntimeException failure) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "decode reserve failed: " + failure.getMessage());
            return true;
        }
        // 先设置再推进阶段：推进成功后请求随时可能结束，结束时的清理要能取到 reserve
        request.setDecodeChoice(choice);
        decodeQueue.removeWaiting(request);
        if (!request.moveTo(Stage.DECODE_WAITING, Stage.DECODE_RESERVED)) {
            // 选的过程中请求已经结束
            cleanup.releaseDecodeReservation(request);
            return true;
        }
        decodeQueue.addReserved(request);
        return true;
    }
}
