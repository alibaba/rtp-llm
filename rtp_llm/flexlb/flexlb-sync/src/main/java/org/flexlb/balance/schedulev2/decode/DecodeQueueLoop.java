package org.flexlb.balance.schedulev2.decode;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeEndpointSnapshot;
import org.flexlb.balance.eviction.DecodeEvictionProposal;
import org.flexlb.balance.eviction.DecodePreemptionCoordinator;
import org.flexlb.balance.eviction.EvictionPlanner;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.decode.DecodeMode;
import org.flexlb.balance.scheduler.decode.DecodeRequirement;
import org.flexlb.balance.schedulev2.DispatchConfirmTimeout;
import org.flexlb.balance.schedulev2.LoopSignal;
import org.flexlb.balance.schedulev2.PrefillPublishCheck;
import org.flexlb.balance.schedulev2.QueueBlockReason;
import org.flexlb.balance.schedulev2.RequestCleanup;
import org.flexlb.balance.schedulev2.request.DecodeChoice;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.V2QueueReporter;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Decode 队列线程，每一轮做三件事：
 * <ol>
 * <li>检查 published 的请求：引擎上报了，reserve 就交给 Decode 的状态同步管，不再跟踪；
 * 发布后超过 {@link DispatchConfirmTimeout} 引擎还没上报，说明调用方没把请求发给引擎，按 EXPIRED 归还 reserve；</li>
 * <li>reserved 的请求按顺序向 reserve 的那台 Decode 要 permit，拿到后发布结果（PUBLISHED）。
 * 发布后在 Decode 上有 reserve 的请求进入 published。扫到已经结束的请求就归还它的 reserve 并移出 reserved。
 * 某台 Decode 给不出 permit 时，这一轮跳过 reserve 在它上面的其他请求，等它有空位时被唤醒；</li>
 * <li>调度线程直接计算 waiting 请求的容量需求，读取最新节点负载、选择节点并立即 reserve。
 * reserve、空机器放行、抢占和归还都只在调度线程执行。成功后进入 DECODE_RESERVED。
 * 某个 group 排在最前面的请求放不下时，它留在 waiting。是否跳过同 group 后面的请求见 {@link #reserveWaiting}。</li>
 * </ol>
 * reserve 的 Decode 下线或 reserve 丢失时，请求退回 waiting 重新选。
 * <p>
 * 请求放不下时（选中的 Decode reserve 不上，或者选择器说都放不下、请求超过所有 Decode 的额度），
 * 先在同 group 里找空的 Decode（{@link EmptyDecodeFinder}），有就不看容量直接放过去并发布。
 * 没有空的 Decode 时，选中的 Decode reserve 不上才尝试抢占，其余情况请求留在 waiting。
 * <p>
 * 抢占模式下，选中的 Decode 放不下时，尝试抢占这台 Decode 上优先级更低、已经发给引擎的请求。
 * V2 只支持 RETURN 方式：被抢占请求的 id 放进响应的 preemptRequestIds，由调用方去取消，flexlb 不发 cancel。
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
    private final DecodePreemptionCoordinator preemptionCoordinator;
    private final EmptyDecodeFinder emptyDecodeFinder;
    private final ConfigService configService;
    private final RequestSchedulerReporter reporter;
    private final PrefillPublishCheck prefillCheck;
    private final DecodeReservations reservations;
    private final V2QueueReporter queueReporter;
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
                           DefaultRouter router,
                           DecodePreemptionCoordinator preemptionCoordinator,
                           EmptyDecodeFinder emptyDecodeFinder,
                           ConfigService configService,
                           RequestSchedulerReporter reporter,
                           PrefillPublishCheck prefillCheck,
                           DecodeReservations reservations,
                           V2QueueReporter queueReporter) {
        this.queueReporter = queueReporter;
        this.decodeQueue = decodeQueue;
        this.loopSignal = loopSignal;
        this.cleanup = cleanup;
        this.placementAvailability = placementAvailability;
        this.decodeSelector = decodeSelector;
        this.router = router;
        this.preemptionCoordinator = preemptionCoordinator;
        this.emptyDecodeFinder = emptyDecodeFinder;
        this.configService = configService;
        this.reporter = reporter;
        this.prefillCheck = prefillCheck;
        this.reservations = reservations;
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
            } catch (Throwable failure) {
                Logger.error("V2 decode queue loop round failed", failure);
                moved = false;
            }
            if (!moved) {
                loopSignal.await(IDLE_WAIT_MS);
            }
        }
        releaseEndedReserved();
        for (Iterator<Request> iterator = decodeQueue.publishedIterator(); iterator.hasNext();) {
            reservations.remove(iterator.next().requestId());
            iterator.remove();
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
        boolean moved = checkPublished();
        moved |= dispatchReserved();
        moved |= reserveWaiting();
        return moved;
    }

    /**
     * 拿走请求在 Decode 上的 reserve 并归还。没有 reserve 时什么也不做。
     */
    private void releaseNow(Request request) {
        DecodeChoice choice = request.takeDecodeChoice();
        try {
            if (choice != null) {
                release(choice);
            }
        } finally {
            reservations.remove(request.requestId());
        }
    }

    /**
     * 归还还没发布的请求在 Decode 上的 reserve。结果没发给调用方，所以按 NOT_SENT 归还。
     * 抢占来的 reserve 还挂着抢占记录，NOT_SENT 不归还（返回 CONFLICT），要按 LOCAL_ROLLBACK 撤销抢占。
     * 归还和 dispatch 都在这个线程里做，归还时 reserve 一定还没 dispatch；
     * 已经 dispatch 的请求不会走到这里（发布成功的不归还，发布失败的抢占记录在 dispatch 时已经清掉）。
     */
    private void release(DecodeChoice choice) {
        try {
            DecodeEndpoint.ReservationReleaseResult result =
                    choice.endpoint().release(choice.reservation(), DecodeEndpoint.ReleaseReason.NOT_SENT);
            if (result == DecodeEndpoint.ReservationReleaseResult.CONFLICT) {
                choice.endpoint().release(choice.reservation(), DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
            }
        } catch (RuntimeException failure) {
            Logger.error("V2 decode reservation release failed: request_id={}", choice.reservation().requestId(),
                    failure);
        }
    }

    /**
     * 检查已发布请求在 Decode 上的 reserve，返回有没有按超时归还 reserve。
     * <p>
     * 引擎已经上报、reserve 已经不在（引擎做完了或被别的清理删了）、Decode 已经下线时，不再跟踪；
     * 发布后超时引擎还没上报时，按 EXPIRED 归还 reserve。
     */
    private boolean checkPublished() {
        long confirmTimeoutMs = DispatchConfirmTimeout.timeoutMs(configService);
        long nowMs = System.currentTimeMillis();
        boolean released = false;
        Iterator<Request> iterator = decodeQueue.publishedIterator();
        while (iterator.hasNext()) {
            Request request = iterator.next();
            DecodeChoice choice = request.decodeChoice();
            DecodeEndpoint endpoint = choice.endpoint();
            DecodeEndpoint.ReservationHandle reservation = choice.reservation();
            if (endpoint.isRetired() || endpoint.isAcceptedByEngine(reservation)
                    || !reservation.equals(endpoint.reservationHandle(request.requestId()))) {
                iterator.remove();
                reservations.remove(request.requestId());
                continue;
            }
            if (DispatchConfirmTimeout.expired(request, nowMs, confirmTimeoutMs)) {
                iterator.remove();
                expire(choice);
                reservations.remove(request.requestId());
                reporter.reportDispatchConfirmTimeout();
                released = true;
            }
        }
        return released;
    }

    private static void expire(DecodeChoice choice) {
        try {
            choice.endpoint().release(choice.reservation(), DecodeEndpoint.ReleaseReason.EXPIRED);
        } catch (RuntimeException failure) {
            Logger.error("V2 decode reservation expire failed: request_id={}", choice.reservation().requestId(),
                    failure);
        }
    }

    private boolean dispatchReserved() {
        Set<DecodeEndpoint> fullEndpoints = Collections.newSetFromMap(new IdentityHashMap<>());
        boolean moved = false;
        Iterator<Request> iterator = decodeQueue.reservedIterator();
        while (iterator.hasNext()) {
            Request request = iterator.next();
            if (request.stage().isEnded()) {
                releaseNow(request);
                decodeQueue.removeReserved(request);
                moved = true;
                continue;
            }
            DecodeChoice choice = request.decodeChoice();
            if (request.stage() != Stage.DECODE_RESERVED || choice == null) {
                decodeQueue.removeReserved(request);
                continue;
            }
            if (!prefillCheck.isAvailable(request)) {
                returnToMain(request);
                moved = true;
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
                reportCapacityBlock(request, endpoint, choice.requirement(), true);
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
        try (WorkerEndpoint.EndpointHold hold = prefillCheck.hold(request)) {
            if (hold == null) {
                if (permit != null) {
                    permit.release();
                }
                returnToMain(request);
                return;
            }
            publishResult(request, choice, permit);
        }
    }

    private void publishResult(Request request, DecodeChoice choice, DecodeEndpoint.EngineDispatchPermit permit) {
        DecodeEndpoint.EngineDispatchPermitTransferStatus transfer;
        try {
            transfer =
                    permit == null ? DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED : permit.dispatch();
        } catch (RuntimeException failure) {
            if (permit != null) {
                permit.release();
            }
            request.fail(StrategyErrorType.DISPATCH_FAILED, "decode dispatch failed: " + failure.getMessage());
            return;
        }
        if (transfer != DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED) {
            sendBackToWaiting(request);
            return;
        }
        decodeQueue.removeReserved(request);
        if (request.publish(Stage.DECODE_RESERVED,
                List.of(request.prefillChoice().serverStatus(), choice.serverStatus()))) {
            cleanup.releasePublished(request);
            decodeQueue.addPublished(request);
            for (DecodeEndpoint.DecodeRequestView victim : choice.preemptedRequests()) {
                queueReporter.decodePreempted(request.priority(), victim.priority(), victim.phase(),
                        choice.endpoint().ipPort());
            }
        } else {
            // 请求已经结束，结果没发给调用方，归还 reserve
            releaseNow(request);
        }
    }

    /**
     * reserve 的 Decode 下线或 reserve 丢失：归还 reserve，请求退回 waiting 重新选 Decode。
     * 请求已经结束时只归还 reserve。
     */
    private void sendBackToWaiting(Request request) {
        decodeQueue.removeReserved(request);
        releaseNow(request);
        if (request.moveTo(Stage.DECODE_RESERVED, Stage.DECODE_WAITING)) {
            decodeQueue.addWaiting(request);
        }
    }

    /**
     * Prefill 已不可用，先在 Decode 线程归还预占，再退回主队列重新选 Prefill。
     */
    private void returnToMain(Request request) {
        Stage expected = request.stage();
        decodeQueue.removeWaiting(request);
        decodeQueue.removeReserved(request);
        releaseNow(request);
        if (expected == Stage.DECODE_WAITING || expected == Stage.DECODE_RESERVED) {
            cleanup.returnToMain(request, expected);
        }
    }

    /**
     * 线程退出前归还 reserved 里已经结束的请求的 reserve。
     */
    private void releaseEndedReserved() {
        Iterator<Request> iterator = decodeQueue.reservedIterator();
        while (iterator.hasNext()) {
            Request request = iterator.next();
            if (request.stage().isEnded()) {
                releaseNow(request);
                decodeQueue.removeReserved(request);
            }
        }
    }

    private void listenCapacity(DecodeEndpoint endpoint) {
        if (listenedEndpoints.add(endpoint)) {
            endpoint.addEngineDispatchCapacityListener(capacityListener);
        }
    }

    /**
     * waiting 的请求按顺序选 Decode 并 reserve。
     * <p>
     * 每个 group 排在最前面、放不下的请求叫队头。队头等得不久时，同 group 后面放得下的请求可以先走；
     * 队头从 Prefill dispatch 起等到 {@code dispatcher.dispatchStarvationThresholdMs}，这一轮跳过同 group 后面的请求，
     * 让归还的容量先留给队头。阈值是 0 时一直严格按顺序。
     */
    private boolean reserveWaiting() {
        long starvationThresholdMs =
                configService.loadBalanceConfig().getDispatcher().getDispatchStarvationThresholdMs();
        long now = System.currentTimeMillis();
        Set<String> groupsWithHead = new HashSet<>();
        Set<String> blockedGroups = new HashSet<>();
        boolean moved = false;
        int scanned = 0;
        Iterator<Request> iterator = decodeQueue.waitingIterator();
        while (iterator.hasNext() && scanned < MAX_SCAN_PER_ROUND) {
            Request request = iterator.next();
            if (request.stage() != Stage.DECODE_WAITING) {
                decodeQueue.removeWaiting(request);
                continue;
            }
            if (!prefillCheck.isAvailable(request)) {
                returnToMain(request);
                moved = true;
                continue;
            }
            String group = groupOf(request);
            if (blockedGroups.contains(group)) {
                continue;
            }
            scanned++;
            try {
                if (request.stage() != Stage.DECODE_WAITING) {
                    continue;
                }
                // 只有 token 加法和配置读取，没有缓存查询或网络 IO，直接在当前线程计算。
                DecodeRequirement requirement = DecodeRequirement.capture(request.ctx());
                if (pickDecode(request, group, requirement)) {
                    moved = true;
                } else if (groupsWithHead.add(group) && isStarved(request, starvationThresholdMs, now)) {
                    blockedGroups.add(group);
                }
            } catch (Throwable failure) {
                request.fail(StrategyErrorType.DISPATCH_FAILED, "decode selection failed: " + failure.getMessage());
                Logger.error("V2 decode selection result failed", failure);
                moved = true;
            }
        }
        return moved;
    }

    /**
     * 队头是否等得太久，需要同 group 严格按顺序。阈值是 0 时总是返回 true。
     */
    private static boolean isStarved(Request request, long starvationThresholdMs, long now) {
        return starvationThresholdMs == 0L || now - request.prefillDecidedAtMs() >= starvationThresholdMs;
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
    private boolean pickDecode(Request request, String group, DecodeRequirement requirement) {
        reservations.add(request.requestId());
        try {
            return selectAndReserve(request, group, requirement);
        } finally {
            if (request.decodeChoice() == null) {
                reservations.remove(request.requestId());
            }
        }
    }

    private boolean selectAndReserve(Request request, String group, DecodeRequirement requirement) {
        PlacementResult<SelectedRole, RoleType> result = decodeSelector.select(request.ctx(), requirement, group);
        switch (result.status()) {
            case SUCCESS -> {
                return reserve(request, result.value(), requirement, group);
            }
            case BLOCKED, REJECTED -> {
                // REJECTED 是请求超过了所有 Decode 的额度，等到有空的 Decode 时放过去
                if (placeOnEmptyDecode(request, requirement, group)) {
                    return true;
                }
                queueReporter.blocked(request, Stage.DECODE_WAITING,
                        result.status() == PlacementResult.Status.REJECTED
                                ? QueueBlockReason.REQUEST_EXCEEDS_KV_BUDGET
                                : QueueBlockReason.NO_AVAILABLE_DECODE,
                        "all");
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
     * 在选中的 Decode 上 reserve。返回 false 表示这台 Decode 放不下、没有空的 Decode、也抢占不到，请求留在 waiting。
     * 选择器在没有放得下的 Decode 时也会选一台忙的，所以放不下由 reserve 带容量上限在锁里判断，不会超额。
     */
    private boolean reserve(Request request, SelectedRole selected, DecodeRequirement requirement, String group) {
        DecodeChoice choice;
        List<DecodeEndpoint.DecodeRequestView> preemptedRequests = new ArrayList<>();
        try (selected; WorkerEndpoint.EndpointHold hold = selected.takeEndpointHold()) {
            DecodeEndpoint endpoint = (DecodeEndpoint) hold.endpoint();
            if (endpoint.isRetired()) {
                queueReporter.blocked(request, Stage.DECODE_WAITING, QueueBlockReason.ENDPOINT_RETIRED,
                        endpoint.ipPort());
                return false;
            }
            DecodeEndpoint.ReservationHandle reservation =
                    endpoint.reserve(hold, request.requestId(), requirement.inputKvTokens(),
                            requirement.inputAndMaxOutputKvTokens(), requirement.priority(), requirement.capacity());
            if (reservation == null) {
                if (placeOnEmptyDecode(request, requirement, group)) {
                    return true;
                }
                reservation = preempt(request, hold, requirement, selected.serverStatus(), preemptedRequests);
                if (reservation == null) {
                    reportCapacityBlock(request, endpoint, requirement, false);
                    return false;
                }
            }
            choice = new DecodeChoice(endpoint, reservation, selected.serverStatus(), requirement, preemptedRequests);
        } catch (RuntimeException failure) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "decode reserve failed: " + failure.getMessage());
            return true;
        }
        // 先设置再推进阶段：推进成功后请求随时可能结束，结束时的清理要能取到 reserve
        request.setDecodeChoice(choice);
        decodeQueue.removeWaiting(request);
        if (!request.moveTo(Stage.DECODE_WAITING, Stage.DECODE_RESERVED)) {
            // 选的过程中请求已经结束
            releaseNow(request);
            return true;
        }
        decodeQueue.addReserved(request);
        return true;
    }

    /**
     * 找一台空的 Decode，不看容量直接放过去并发布。返回 false 表示没有空的 Decode，请求留在 waiting。
     * <p>
     * reserve 按已经发给引擎记（不进排队，也不再要 permit），请求的 KV 照样记到这台 Decode 上，
     * 后面的请求看到的就不是空机器了。发布后和普通请求一样进 published，等引擎上报或超时归还。
     */
    private boolean placeOnEmptyDecode(Request request, DecodeRequirement requirement, String group) {
        DecodeChoice choice;
        try {
            SelectedRole selected = emptyDecodeFinder.find(group, request.requestId());
            if (selected == null) {
                return false;
            }
            try (selected; WorkerEndpoint.EndpointHold hold = selected.takeEndpointHold()) {
                DecodeEndpoint endpoint = (DecodeEndpoint) hold.endpoint();
                DecodeEndpoint.ReservationHandle reservation = endpoint.reserveUnqueued(hold, request.requestId(),
                        requirement.inputKvTokens(), requirement.inputAndMaxOutputKvTokens(), requirement.priority());
                choice = new DecodeChoice(endpoint, reservation, selected.serverStatus(), requirement);
            }
        } catch (RuntimeException failure) {
            request.fail(StrategyErrorType.DISPATCH_FAILED, "empty decode reserve failed: " + failure.getMessage());
            return true;
        }
        // 先设置再发布：发布前请求随时可能结束，结束时的清理要能看到它有 reserve
        request.setDecodeChoice(choice);
        decodeQueue.removeWaiting(request);
        try (WorkerEndpoint.EndpointHold hold = prefillCheck.hold(request)) {
            if (hold == null) {
                returnToMain(request);
                return true;
            }
            if (request.publish(Stage.DECODE_WAITING,
                    List.of(request.prefillChoice().serverStatus(), choice.serverStatus()))) {
                cleanup.releasePublished(request);
                decodeQueue.addPublished(request);
            } else {
                // 请求已经结束，结果没发给调用方，归还 reserve
                releaseNow(request);
            }
        }
        return true;
    }

    /**
     * 选中的 Decode 放不下时，尝试抢占这台 Decode 上优先级更低、已经发给引擎的请求。
     * 成功时返回新请求在这台 Decode 上的 reserve，并把被抢占请求的 id 写进返回给调用方的 Decode 信息；
     * 不能抢占或没有可抢占的请求时返回 null。
     */
    private DecodeEndpoint.ReservationHandle preempt(Request request,
                                                     WorkerEndpoint.EndpointHold hold,
                                                     DecodeRequirement requirement,
                                                     ServerStatus serverStatus,
                                                     List<DecodeEndpoint.DecodeRequestView> preemptedRequests) {
        PreemptionConfig policy = returnPreemptionPolicy(request, requirement);
        if (policy == null) {
            return null;
        }
        DecodeEndpoint endpoint = (DecodeEndpoint) hold.endpoint();
        DecodeEndpointSnapshot snapshot = DecodeEndpointSnapshot.capture(endpoint, requirement.capacity());
        DecodeEvictionProposal proposal =
                EvictionPlanner.planDecode(requirement.priority(), requirement.inputKvTokens(),
                        requirement.inputAndMaxOutputKvTokens(), List.of(snapshot), policy, null, new HashMap<>());
        if (proposal == null || !proposal.requiresEngineCancel()) {
            return null;
        }
        DecodePreemptionCoordinator.PreemptionCommand command = new DecodePreemptionCoordinator.PreemptionCommand(
                endpoint, request.requestId(), requirement.inputKvTokens(), requirement.inputAndMaxOutputKvTokens(),
                requirement.priority(), requirement.capacity(), proposal.victims(), 0L, policy.getTimeoutMs(),
                () -> !request.stage().isEnded(), "v2_decode_queue");
        // RETURN 方式不发 cancel，结果同步就有
        DecodePreemptionCoordinator.PreemptionResult result =
                preemptionCoordinator.prepareReturnedPreemption(command).join();
        if (!result.committed()) {
            Logger.debug("V2 decode preemption not committed: request_id={} detail={}", request.requestId(),
                    result.detail());
            return null;
        }
        DecodeEndpoint.ReservationHandle reservation = endpoint.reservationHandle(request.requestId());
        if (reservation == null) {
            return null;
        }
        // 抢占装上的 reserve 还没进排队，要先标记才能拿 permit
        if (!endpoint.markQueued(hold, reservation)) {
            endpoint.release(reservation, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
            return null;
        }
        serverStatus.setPreemptRequestIds(
                proposal.victims().stream().map(DecodeEndpoint.DecodeRequestView::requestId).toList());
        preemptedRequests.addAll(proposal.victims());
        return reservation;
    }

    /**
     * 失败后读容量视图，复用 reserve/permit 的容量公式。此时已恢复容量则记为状态变化，不误报 KV 不足。
     */
    private void reportCapacityBlock(Request request,
                                     DecodeEndpoint endpoint,
                                     DecodeRequirement requirement,
                                     boolean dispatch) {
        var view = endpoint.routingView();
        var deficit = requirement.capacity()
                .evaluate(dispatch ? view.dispatchUsage() : view.placementUsage(), requirement.inputKvTokens(),
                        requirement.inputAndMaxOutputKvTokens());
        queueReporter.blocked(request, dispatch ? Stage.DECODE_RESERVED : Stage.DECODE_WAITING,
                QueueBlockReason.decodeCapacity(deficit), endpoint.ipPort());
    }

    /**
     * V2 用的抢占策略：只抢已经发给引擎的请求，并且固定用 RETURN 方式。
     * 不是抢占模式、请求没有优先级或配置不允许抢引擎上的请求时返回 null。
     */
    private static PreemptionConfig returnPreemptionPolicy(Request request, DecodeRequirement requirement) {
        if (requirement.mode() != DecodeMode.PREEMPT_AT_PLACEMENT
                || !PriorityNormalizer.hasPriority(requirement.priority())) {
            return null;
        }
        PreemptionConfig configured =
                request.ctx().getConfig().queueScheduler().getOrdering().preemptionPolicy().orElse(null);
        if (configured == null || !configured.allows(VictimStage.DECODE_ENGINE_OWNED)) {
            return null;
        }
        PreemptionConfig policy = new PreemptionConfig();
        policy.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_ENGINE_OWNED));
        policy.setTimeoutMs(configured.getTimeoutMs());
        policy.getEngineCancellation().setMode(EngineCancellationConfig.Mode.RETURN);
        return policy;
    }
}
