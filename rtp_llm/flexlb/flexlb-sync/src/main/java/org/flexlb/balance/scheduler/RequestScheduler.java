package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFallback;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Objects;
import java.util.TreeSet;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;

/** Public routing facade and owner of requests which have not chosen a worker yet. */
@Component
public final class RequestScheduler implements RequestShutdownOrchestrator.Placement {
    private static final int RETRY_PARALLELISM =
            Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors()));
    private static final long MIN_BACKOFF_MS = 2L;
    private static final long MAX_BACKOFF_MS = 50L;
    private static final long FALLBACK_PROBE_INTERVAL_NANOS =
            TimeUnit.MILLISECONDS.toNanos(2L);
    private static final int COMMIT_TURN_STRIPES = 256;
    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;
    private final AdmissionFallback admissionFallback;
    private final RequestLifecycleCoordinator lifecycle;
    private final ScheduledThreadPoolExecutor executor;
    private final AtomicInteger waitingCount = new AtomicInteger();
    private final AtomicLong sequence = new AtomicLong();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final ReentrantReadWriteLock attemptGate = new ReentrantReadWriteLock(true);
    private final ReentrantLock[] commitTurns = new ReentrantLock[COMMIT_TURN_STRIPES];
    private final Object laneLock = new Object();
    private final Map<WaitResource, WaitLane> lanes = new HashMap<>();
    private long nextFallbackProbeNanos;
    private int scheduledFallbackProbes;
    @Autowired
    RequestScheduler(ConfigService configService, Router router,
            EndpointRegistry endpointRegistry, BatchSchedulerReporter reporter,
            AdmissionFallback admissionFallback, RequestLifecycleCoordinator lifecycle) {
        this.configService = Objects.requireNonNull(configService, "configService");
        this.router = Objects.requireNonNull(router, "router");
        this.endpointRegistry = Objects.requireNonNull(endpointRegistry, "endpointRegistry");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.admissionFallback = Objects.requireNonNull(admissionFallback, "admissionFallback");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        for (int index = 0; index < commitTurns.length; index++) {
            commitTurns[index] = new ReentrantLock();
        }
        AtomicInteger threadId = new AtomicInteger();
        executor = new ScheduledThreadPoolExecutor(RETRY_PARALLELISM, task -> {
            Thread thread = new Thread(task, "request-placement-" + threadId.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        });
        executor.setRemoveOnCancelPolicy(true);
        executor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
    }

    /** Register once; retries retain the exact context and public future. */
    public CompletableFuture<Response> submit(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            return CompletableFuture.completedFuture(error(StrategyErrorType.INVALID_REQUEST, null));
        }
        FlexlbConfig config;
        try {
            config = configService.loadBalanceConfig();
        } catch (Throwable failure) {
            return CompletableFuture.completedFuture(error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Failed to load scheduler configuration: " + failure.getMessage()));
        }
        CompletableFuture<Response> future = lifecycle.register(
                context, config.queueScheduler().getCapacity().getMaxOutstandingRequestsGlobal());
        if (future.isDone()) { return future; }
        Waiter waiter = new Waiter(
                context, future, sequence.incrementAndGet(), config.isPriorityOrdering());
        waitingCount.incrementAndGet();
        future.whenComplete((ignored, failure) -> endPlacement(waiter));
        if (closed.get()) { stop(waiter, null, shutdownError()); }
        else { attempt(waiter, null); }
        return future;
    }
    private void signal(WaitLane lane) {
        synchronized (laneLock) {
            if (closed.get() || lanes.get(lane.key) != lane || lane.signalled) { return; }
            lane.signalled = true;
            lane.backoffMs = MIN_BACKOFF_MS;
            if (lane.active == null) {
                cancelProbe(lane);
                scheduleProbe(lane, 0L);
            }
        }
    }
    private void attempt(Waiter waiter, WaitLane source) {
        attemptGate.readLock().lock();
        try {
            if (closed.get()) { stop(waiter, source, shutdownError()); }
            else { attemptOwned(waiter, source); }
        } finally {
            attemptGate.readLock().unlock();
        }
    }
    private void attemptOwned(Waiter waiter, WaitLane source) {
        if (!waiter.owned.get() || waiter.future.isDone()) {
            park(waiter, source, null, false); return;
        }
        if (closed.get()) { stop(waiter, source, shutdownError()); return; }
        if (waiter.context.requestExpired(System.currentTimeMillis())) {
            timeout(waiter, source); return;
        }
        RequestLifecycleCoordinator.AdmissionScope mutation = lifecycle.beginAdmission(
                waiter.context.getRequestId(), waiter.future);
        if (mutation == null) {
            if (source != null && waitingForPlacement(waiter)) {
                park(waiter, source, source.key, false);
            } else {
                stop(waiter, source, null);
            }
            return;
        }
        WaitResource deferred = null;
        Response terminal = null;
        BatchItem committed = null;
        WaitLane fallbackTurn = null;
        boolean parked = false;
        try (mutation) {
            QueueRoutingResult result = router.routeForQueue(waiter.context);
            if (result instanceof QueueRoutingResult.Admitted admitted) {
                QueueRouteAdmission admission = admitted.admission();
                CommitTurn turn = tryEnterCommitTurn(admission.response());
                try (turn; admission) {
                    if (!turn.acquired()) {
                        deferred = turn.blockedResource();
                        park(waiter, source, deferred, false);
                        parked = true;
                    } else {
                        deferred = blockedResource(waiter, source, admission.response());
                        if (deferred == null
                                && admission.prepareDecode(waiter.context)
                                        == QueueRouteAdmission.DecodePrepareStatus.CAPACITY_FULL) {
                            deferred = selectedDecode(admission.response());
                        }
                        if (deferred == null) {
                            BatchItem item = admission.buildItem(
                                    waiter.context, waiter.future, System.currentTimeMillis());
                            waiter.context.setRouteSubmittedNanos(System.nanoTime());
                            QueueRouteAdmission.CommitStatus status = admission.commitTo(
                                    lifecycle, item, false, mutation.exact());
                            switch (status) {
                                case COMMITTED -> committed = item;
                                case PREFILL_FULL ->
                                        deferred = selectedPrefill(admission.response());
                                case STALE -> deferred =
                                        selectedPrefill(admission.response()).generic();
                                case REQUEST_CLOSED -> {
                                    if (waitingForPlacement(waiter)) {
                                        deferred = selectedPrefill(
                                                admission.response()).generic();
                                    }
                                }
                            }
                        }
                        if (deferred != null) {
                            fallbackTurn = park(
                                    waiter, source, deferred, true);
                            parked = true;
                        }
                    }
                }
            } else if (result instanceof QueueRoutingResult.Deferred wait) {
                deferred = new WaitResource(wait.role(), wait.group(), null);
            } else { terminal = ((QueueRoutingResult.Rejected) result).response(); }
        } catch (Throwable failure) {
            Logger.error("Request placement failed for request id: {}",
                    waiter.context.getRequestId(), failure);
            terminal = error(StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Submit failed: " + failure.getMessage());
        }
        if (committed != null) {
            reportRouteSubmitted(waiter.context, committed);
            stop(waiter, source, null);
        } else if (terminal != null) {
            stop(waiter, source, terminal);
        } else if (waiter.context.requestExpired(System.currentTimeMillis())) {
            timeout(waiter, source);
        } else if (!lifecycle.isAdmissionOpen(waiter.context.getRequestId(), waiter.future)) {
            stop(waiter, source, null);
        } else if (parked) {
            if (fallbackTurn != null) {
                runFallback(waiter, fallbackTurn, deferred);
            }
        } else {
            retryOrPark(waiter, source, deferred);
        }
    }
    private void retryOrPark(Waiter waiter, WaitLane source, WaitResource deferred) {
        WaitLane turn = park(waiter, source, deferred, true);
        if (turn == null) { return; }
        runFallback(waiter, turn, deferred);
    }
    private void runFallback(Waiter waiter, WaitLane turn, WaitResource deferred) {
        try {
            admissionFallback.tryAdmit(waiter.context, waiter.future);
        } catch (Throwable failure) {
            Logger.error("Admission fallback failed for request id: {}",
                    waiter.context.getRequestId(), failure);
            stop(waiter, turn, error(StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Admission fallback failed: " + failure.getMessage()));
            return;
        }
        if (waitingForPlacement(waiter)) { park(waiter, turn, deferred, false); }
        else { stop(waiter, turn, null); }
    }
    private boolean waitingForPlacement(Waiter waiter) {
        return lifecycle.isWaitingForPlacement(waiter.context.getRequestId(), waiter.future);
    }

    /**
     * Try to own the short, side-effecting part of an exact P/D placement.
     * Selection remains fully concurrent; contenders never queue on these
     * locks and instead become visible in the matching wait lane.
     */
    private CommitTurn tryEnterCommitTurn(Response response) {
        List<WaitResource> resources = new ArrayList<>(2);
        if (response.getServerStatus() != null) {
            for (ServerStatus status : response.getServerStatus()) {
                if (status != null && (status.getRole() == RoleType.PREFILL
                        || status.getRole() == RoleType.PDFUSION
                        || status.getRole() == RoleType.DECODE)) {
                    resources.add(selected(status));
                }
            }
        }
        resources.sort(Comparator.comparingInt(this::commitTurnIndex));
        List<ReentrantLock> acquired = new ArrayList<>(resources.size());
        int previousIndex = -1;
        for (WaitResource resource : resources) {
            int index = commitTurnIndex(resource);
            if (index == previousIndex) {
                continue;
            }
            ReentrantLock lock = commitTurns[index];
            if (!lock.tryLock()) {
                for (int held = acquired.size() - 1; held >= 0; held--) {
                    acquired.get(held).unlock();
                }
                return new CommitTurn(List.of(), resource);
            }
            acquired.add(lock);
            previousIndex = index;
        }
        return new CommitTurn(acquired, null);
    }

    private int commitTurnIndex(WaitResource resource) {
        return Math.floorMod(resource.hashCode(), commitTurns.length);
    }

    /** The lane handoff lock linearizes Deferred publication against later route claims. */
    private WaitResource blockedResource(Waiter waiter, WaitLane source, Response response) {
        synchronized (laneLock) {
            List<ServerStatus> statuses = response.getServerStatus();
            if (statuses == null) { return null; }
            if (source != null && source.active == waiter && !source.waiters.isEmpty()
                    && source.waiters.first() != waiter
                    && statuses.stream().anyMatch(status -> overlaps(source.key, status))) {
                return source.key;
            }
            for (ServerStatus status : statuses) {
                WaitResource exact = selected(status);
                WaitLane blocked = blockedLane(exact, source);
                if (blocked == null && exact.endpointAddress() != null) {
                    blocked = blockedLane(new WaitResource(exact.role(), exact.group(), null), source);
                }
                if (blocked == null && exact.group() != null) {
                    blocked = blockedLane(new WaitResource(exact.role(), null, null), source);
                }
                if (blocked != null) { return blocked.key; }
            }
            return null;
        }
    }
    private WaitLane blockedLane(WaitResource key, WaitLane source) {
        WaitLane lane = lanes.get(key);
        return lane == null || lane == source || lane.waiters.isEmpty() ? null : lane;
    }
    private static boolean overlaps(WaitResource key, ServerStatus status) {
        WaitResource selected = selected(status);
        return key.role() == selected.role()
                && (key.group() == null || key.group().equals(selected.group()))
                && (key.endpointAddress() == null
                        || key.endpointAddress().equals(selected.endpointAddress()));
    }
    /** Publish a blocker before fallback; only the current lane head gets that turn. */
    private WaitLane park(
            Waiter waiter, WaitLane source, WaitResource deferred, boolean fallbackTurn) {
        synchronized (laneLock) {
            if (source != null && source.active == waiter
                    && (!fallbackTurn || !source.key.equals(deferred))) {
                source.active = null;
            }
            if (!waiter.owned.get() || deferred == null) {
                detach(waiter);
                advance(source);
                return null;
            }
            WaitLane current = waiter.lane;
            if (current == null || !current.key.equals(deferred)) {
                detach(waiter);
                current = lanes.computeIfAbsent(deferred,
                        key -> newLane(key, waiter.priorityOrdering));
                current.waiters.add(waiter);
                waiter.lane = current;
            }
            if (source != current) { advance(source); }
            if (fallbackTurn && current.waiters.first() == waiter
                    && (current.active == null || current.active == waiter)) {
                cancelProbe(current);
                current.active = waiter;
                return current;
            }
            if (current.active == waiter) { current.active = null; }
            boolean displaced = current.waiters.first() != waiter;
            if (current.active == null && !hasProbe(current)) {
                long delay = current.signalled || displaced ? 0L : current.backoffMs;
                current.signalled = false;
                if (delay > 0L) {
                    current.backoffMs = Math.min(MAX_BACKOFF_MS, current.backoffMs * 2L);
                }
                scheduleProbe(current, delay);
            }
            return null;
        }
    }
    private WaitLane newLane(WaitResource key, boolean priorityOrdering) {
        WaitLane lane = new WaitLane(key, priorityOrdering);
        if (key.endpointAddress() == null) { return lane; }
        WorkerEndpoint endpoint = endpointRegistry.get(key.role(), key.endpointAddress());
        Runnable listener = () -> signal(lane);
        // The first immediate recheck closes release-before-listener races.
        if (endpoint instanceof PrefillEndpoint prefill) {
            var availability = prefill.offerAvailability();
            availability.addListener(listener);
            lane.unsubscribe = () -> availability.removeListener(listener);
            lane.signalled = true;
        } else if (endpoint instanceof DecodeEndpoint decode) {
            decode.addEngineDispatchCapacityListener(listener);
            lane.unsubscribe = () -> decode.removeEngineDispatchCapacityListener(listener);
            lane.signalled = true;
        }
        return lane;
    }
    private void advance(WaitLane lane) {
        if (lane == null) { return; }
        if (lane.waiters.isEmpty() && lane.active == null) {
            cancelProbe(lane);
            if (lanes.remove(lane.key, lane)) { lane.unsubscribe.run(); }
        } else if (lane.active == null && !hasProbe(lane)) {
            scheduleProbe(lane, 0L);
        }
    }
    private void scheduleProbe(WaitLane lane, long delayMs) {
        if (closed.get() || hasProbe(lane) || lane.active != null
                || lane.waiters.isEmpty()) { return; }
        long delayNanos = 0L;
        if (delayMs > 0L) {
            long now = System.nanoTime();
            long locallyReady = now + TimeUnit.MILLISECONDS.toNanos(delayMs);
            long globallyReady = Math.max(locallyReady, nextFallbackProbeNanos);
            nextFallbackProbeNanos = globallyReady + FALLBACK_PROBE_INTERVAL_NANOS;
            delayNanos = Math.max(0L, globallyReady - now);
            lane.fallbackProbe = true;
            scheduledFallbackProbes++;
        }
        long probe = ++lane.probeGeneration;
        try {
            lane.probe = executor.schedule(() -> fireProbe(lane, probe),
                    delayNanos, TimeUnit.NANOSECONDS);
        } catch (RejectedExecutionException stopped) {
            lane.probe = null;
            releaseFallbackReservation(lane);
        }
    }
    private boolean hasProbe(WaitLane lane) {
        return lane.probe != null;
    }
    private void fireProbe(WaitLane lane, long probe) {
        Waiter head;
        synchronized (laneLock) {
            if (lane.probeGeneration != probe || closed.get()) { return; }
            lane.probe = null;
            releaseFallbackReservation(lane);
            if (lane.active != null || lane.waiters.isEmpty()) {
                advance(lane); return;
            }
            head = lane.waiters.first();
            lane.active = head;
            lane.signalled = false;
        }
        attempt(head, lane);
    }
    private void cancelProbe(WaitLane lane) {
        ScheduledFuture<?> probe = lane.probe;
        lane.probe = null;
        lane.probeGeneration++;
        releaseFallbackReservation(lane);
        if (probe != null) { probe.cancel(false); }
    }
    private void releaseFallbackReservation(WaitLane lane) {
        if (!lane.fallbackProbe) { return; }
        lane.fallbackProbe = false;
        if (--scheduledFallbackProbes == 0) {
            nextFallbackProbeNanos = System.nanoTime();
        }
    }
    private void stop(Waiter waiter, WaitLane source, Response terminal) {
        if (terminal != null) { waiter.future.complete(terminal); }
        endPlacement(waiter);
        park(waiter, source, null, false);
    }
    private void timeout(Waiter waiter, WaitLane source) {
        lifecycle.cancelRequest(waiter.context.getRequestId(), 0L, CancelReason.DEADLINE_EXCEEDED);
        stop(waiter, source, null);
    }
    private void endPlacement(Waiter waiter) {
        if (!waiter.owned.compareAndSet(true, false)) { return; }
        waitingCount.decrementAndGet();
        synchronized (laneLock) {
            WaitLane lane = waiter.lane;
            detach(waiter);
            if (lane != null && lane.active != waiter) { advance(lane); }
        }
    }
    private void detach(Waiter waiter) {
        if (waiter.lane != null) {
            waiter.lane.waiters.remove(waiter);
            waiter.lane = null;
        }
    }
    private static WaitResource selectedPrefill(Response response) {
        if (response.getServerStatus() != null) {
            for (ServerStatus status : response.getServerStatus()) {
                if (status.getRole() == RoleType.PREFILL || status.getRole() == RoleType.PDFUSION) {
                    return selected(status);
                }
            }
        }
        return new WaitResource(RoleType.PREFILL, null, null);
    }
    private static WaitResource selectedDecode(Response response) {
        if (response.getServerStatus() != null) {
            for (ServerStatus status : response.getServerStatus()) {
                if (status.getRole() == RoleType.DECODE) {
                    return selected(status);
                }
            }
        }
        return new WaitResource(RoleType.DECODE, null, null);
    }
    private static WaitResource selected(ServerStatus status) {
        String address = status.getServerIp() == null ? null
                : status.getServerIp() + ":" + status.getHttpPort();
        return new WaitResource(status.getRole(), status.getGroup(), address);
    }
    private void reportRouteSubmitted(BalanceContext context, BatchItem item) {
        try {
            reporter.reportRouteSubmitTimeMs(RoleType.PREFILL.name(), item.prefillEp().getIp(),
                    System.currentTimeMillis() - context.getStartTime());
        } catch (RuntimeException failure) {
            Logger.warn("Failed to record route-submit telemetry: request_id={}",
                    context.getRequestId(), failure);
        }
    }
    @Override
    public void closePlacement() {
        if (!closed.compareAndSet(false, true)) { return; }
        attemptGate.writeLock().lock();
        try {
            synchronized (laneLock) {
                lanes.values().forEach(lane -> { cancelProbe(lane); lane.unsubscribe.run(); });
                lanes.clear();
            }
            executor.shutdownNow();
        } finally {
            attemptGate.writeLock().unlock();
        }
    }
    public RequestLifecycleSnapshot cancelRequest(
            long requestId, long expectedBatchId, CancelReason reason) {
        return lifecycle.cancelRequest(requestId, expectedBatchId, reason);
    }
    public int getInflightSize() { return lifecycle.getInflightSize(); }
    public int getQueuedRequestCount() {
        long queued = waitingCount.get();
        for (PrefillEndpoint endpoint : endpointRegistry.snapshotPrefillEndpoints().values()) {
            queued += endpoint.queuedRequestCount();
            if (queued >= Integer.MAX_VALUE) { return Integer.MAX_VALUE; }
        }
        return (int) queued;
    }
    public List<RequestLifecycleSnapshot> snapshotActiveRequests() { return lifecycle.snapshotActiveRequests(); }
    public RequestLifecycleSnapshot getRequestState(long id, long batchId) {
        return lifecycle.getRequestState(id, batchId);
    }
    public boolean ownsRequestGeneration(long requestId) { return lifecycle.ownsRequestGeneration(requestId); }
    private static Response error(StrategyErrorType type, String detail) {
        return RequestLifecycleCoordinator.buildErrorResponse(type, detail);
    }
    private static Response shutdownError() {
        return error(StrategyErrorType.BATCH_DISPATCH_FAILED,
                "request scheduler is shutting down");
    }
    private record WaitResource(RoleType role, String group, String endpointAddress) {
        private WaitResource {
            group = group == null || group.isBlank() ? null : group;
            endpointAddress = endpointAddress == null || endpointAddress.isBlank() ? null : endpointAddress;
        }
        private WaitResource generic() { return new WaitResource(role, group, null); }
    }
    private record CommitTurn(
            List<ReentrantLock> acquiredLocks,
            WaitResource blockedResource) implements AutoCloseable {
        private CommitTurn {
            acquiredLocks = List.copyOf(acquiredLocks);
        }
        private boolean acquired() { return blockedResource == null; }
        @Override
        public void close() {
            for (int index = acquiredLocks.size() - 1; index >= 0; index--) {
                acquiredLocks.get(index).unlock();
            }
        }
    }
    private static final class Waiter {
        private final BalanceContext context;
        private final CompletableFuture<Response> future;
        private final long sequence;
        private final int priority;
        private final boolean priorityOrdering;
        private final AtomicBoolean owned = new AtomicBoolean(true);
        private WaitLane lane;
        private Waiter(BalanceContext context, CompletableFuture<Response> future,
                       long sequence, boolean priorityOrdering) {
            this.context = context;
            this.future = future; this.sequence = sequence;
            this.priority = context.getPriority();
            this.priorityOrdering = priorityOrdering;
        }
    }
    private static final class WaitLane {
        private final WaitResource key;
        private final NavigableSet<Waiter> waiters;
        private Waiter active;
        private ScheduledFuture<?> probe;
        private long probeGeneration;
        private boolean fallbackProbe;
        private boolean signalled;
        private long backoffMs = MIN_BACKOFF_MS;
        private Runnable unsubscribe = () -> { };
        private WaitLane(WaitResource key, boolean priorityOrdering) {
            this.key = key;
            Comparator<Waiter> arrival = Comparator.comparingLong(waiter -> waiter.sequence);
            waiters = new TreeSet<>(priorityOrdering
                    ? Comparator.<Waiter>comparingInt(waiter -> waiter.priority)
                            .reversed().thenComparing(arrival)
                    : arrival);
        }
    }
}
