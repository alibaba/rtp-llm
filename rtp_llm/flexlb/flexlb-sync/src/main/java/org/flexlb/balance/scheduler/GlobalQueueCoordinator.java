package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.InternalRuntimeSettings;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The single QUEUE admission owner for one FlexLB model.
 *
 * <p>This queue is the model's ordered placement boundary. A request is
 * selected exactly once from the complete live candidate fleet before the
 * endpoint runtime receives it. A bounded planning frontier controls how many
 * independent routes can be prepared together; request collection and grouping
 * remain exclusively endpoint-runtime concerns. The queue lock protects only
 * index operations; route projection and RPCs are never performed while it is
 * held.</p>
 *
 * <p>Plans are committed in queue order unless the exact endpoint selected by
 * one request is locally full. In that case the request is parked against that
 * endpoint and a later plan may commit only when it does not use the parked
 * endpoint. A selector-level miss has no concrete endpoint and therefore still
 * stops its ordered suffix. Planning concurrency is bounded by aggregate
 * delivery credits, while {@link WorkerBatcher} remains the sole SINGLE or
 * FIXED_WINDOW group owner. This keeps cache/KV
 * projections adjacent to each exact reservation while retaining planner
 * parallelism where the policy permits it.</p>
 */
final class GlobalQueueCoordinator implements AutoCloseable {

    private static final int PRIORITY_LEVELS = PriorityNormalizer.MAX_PRIORITY + 1;
    private static final int MIN_PLANNER_THREADS = 1;
    private static final int MIN_PLANNING_FRONTIER_SIZE = 1;

    private final DefaultRouter router;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;
    private final EvictionManager evictionManager;
    private final RequestRegistry lifecycle;
    private final PlacementAvailability availability;
    private final boolean priorityOrdering;
    private final int plannerCount;
    private final ConfigService configService;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private final ArrayDeque<Entry> fifo = new ArrayDeque<>();
    @SuppressWarnings("unchecked")
    private final ArrayDeque<Entry>[] priorityBuckets =
            (ArrayDeque<Entry>[]) new ArrayDeque<?>[PRIORITY_LEVELS];
    private final BitSet nonEmptyPriorities = new BitSet(PRIORITY_LEVELS);
    private final ExecutorService planners;
    private final Thread decisionThread;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final PlacementAvailability.Listener availabilityListener =
            this::onAvailabilityChanged;
    /** Requests blocked by an exact endpoint and eligible for endpoint bypass. */
    private final Set<Entry> endpointBlocked =
            Collections.newSetFromMap(new IdentityHashMap<>());
    private int size;
    /** A selector-level miss has no concrete route and cannot be bypassed. */
    private Entry frontierBlocked;

    GlobalQueueCoordinator(
            ConfigService configService,
            DefaultRouter router,
            EndpointRegistry endpointRegistry,
            BatchSchedulerReporter reporter,
            EvictionManager evictionManager,
            RequestRegistry lifecycle,
            PlacementAvailability availability) {
        ConfigService checkedConfig = Objects.requireNonNull(
                configService, "configService");
        this.configService = checkedConfig;
        this.router = Objects.requireNonNull(router, "router");
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.evictionManager = Objects.requireNonNull(
                evictionManager, "evictionManager");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.availability = Objects.requireNonNull(availability, "availability");
        this.priorityOrdering = resolvePriorityOrdering(checkedConfig);

        AtomicInteger plannerId = new AtomicInteger();
        ThreadFactory plannerFactory = task -> {
            Thread thread = new Thread(task,
                    "flexlb-global-planner-" + plannerId.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        };
        this.plannerCount = Math.max(MIN_PLANNER_THREADS,
                checkedConfig.loadBalanceConfig().getInternalRuntime()
                        .getQueuePlannerThreads());
        this.planners = Executors.newFixedThreadPool(this.plannerCount, plannerFactory);
        this.decisionThread = new Thread(this::runDecisionLoop,
                "flexlb-global-decision");
        this.decisionThread.setDaemon(true);
        this.decisionThread.setUncaughtExceptionHandler((thread, failure) -> {
            Logger.error("Global queue decision thread failed", failure);
            close();
        });
        availability.addListener(availabilityListener);
        decisionThread.start();
    }

    /** Enqueue without selecting an endpoint on the ingress thread. */
    boolean offer(
            BalanceContext context,
            CompletableFuture<Response> future,
            int priority) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(future, "future");
        Entry entry = new Entry(context, future, normalizePriority(priority));
        lock.lock();
        try {
            if (closed.get()) {
                return false;
            }
            if (priorityOrdering) {
                ArrayDeque<Entry> bucket = priorityBuckets[entry.priority];
                if (bucket == null) {
                    bucket = new ArrayDeque<>();
                    priorityBuckets[entry.priority] = bucket;
                }
                bucket.addLast(entry);
                nonEmptyPriorities.set(entry.priority);
            } else {
                fifo.addLast(entry);
            }
            size++;
            // Completion is also a queue tombstone. Marking it is O(1); the
            // physical deque unlink happens only when the tombstone reaches
            // the head, so cancellation never scans a large queue.
            future.whenComplete((ignored, failure) -> markCompleted(entry));
            changed.signal();
            return true;
        } finally {
            lock.unlock();
        }
    }

    int size() {
        lock.lock();
        try {
            return size;
        } finally {
            lock.unlock();
        }
    }

    private void runDecisionLoop() {
        try {
            while (!closed.get()) {
                List<Entry> window;
                try {
                    window = nextWindow();
                } catch (Throwable failure) {
                    Logger.error("Global queue window capture failed", failure);
                    continue;
                }
                if (window.isEmpty()) {
                    continue;
                }
                PlanWindow plans = planWindow(window);
                boolean retry = false;
                for (int planIndex = 0; planIndex < plans.size(); planIndex++) {
                    Plan plan = awaitPlan(
                            plans.future(planIndex), window.get(planIndex));
                    if (closed.get()) {
                        closePlan(plan);
                        closePendingPlans(plans, window, planIndex + 1);
                        break;
                    }
                    if (!isQueued(plan.entry)) {
                        closePlan(plan);
                        continue;
                    }
                    WorkerEndpoint conflictingEndpoint =
                            conflictingEndpoint(plan);
                    if (conflictingEndpoint != null) {
                        park(plan.entry, keyFor(conflictingEndpoint),
                                conflictingEndpoint);
                        closePlan(plan);
                        continue;
                    }
                    Outcome outcome;
                    try {
                        outcome = commit(plan);
                    } catch (Throwable failure) {
                        closePlan(plan);
                        remove(plan.entry);
                        completeDecisionResponse(plan.entry, error(
                                StrategyErrorType.BATCH_DISPATCH_FAILED,
                                "Placement failed: " + failure.getMessage()));
                        Logger.error("Global queue commit failed: request_id={}",
                                plan.entry.context.getRequestId(), failure);
                        continue;
                    }
                    if (outcome == Outcome.RETRY) {
                        closePendingPlans(plans, window, planIndex + 1);
                        retry = true;
                        break;
                    }
                    if (outcome == Outcome.REPLAN) {
                        // The winner became stale at endpoint-local commit.
                        // Keep later independent plans moving; retry this exact
                        // request from a fresh fleet snapshot next pass.
                        retry = true;
                        continue;
                    }
                    if (outcome == Outcome.BLOCKED) {
                        PlacementKey blocker = plan.blocker();
                        if (blocker == null) {
                            throw new IllegalStateException(
                                    "blocked placement has no capacity domain");
                        }
                        WorkerEndpoint blockedEndpoint = plan.blockedEndpoint();
                        if (capacityChangedSince(plan, blocker)) {
                            // The capacity edge may have been published after
                            // this planner captured its snapshot but before
                            // the request was parked. Do not lose that wakeup;
                            // retry against the newer endpoint state once.
                            closePlan(plan);
                            closePendingPlans(plans, window, planIndex + 1);
                            retry = true;
                            break;
                        }
                        if (blockedEndpoint == null) {
                            // There is no exact engine identity to compare. Keep
                            // the old strict frontier semantics for this case.
                            parkFrontier(plan.entry, blocker);
                            closePlan(plan);
                            closePendingPlans(plans, window, planIndex + 1);
                            break;
                        }
                        park(plan.entry, blocker, blockedEndpoint);
                        closePlan(plan);
                        // The next plan can use another engine immediately. Plans
                        // on this same engine are parked as well by the conflict
                        // check above, so no repeated failed reservation occurs.
                        continue;
                    }
                }
                if (retry) {
                    signal();
                }
            }
        } finally {
            drainOnClose();
        }
    }

    private Plan awaitPlan(
            CompletableFuture<Plan> future,
            Entry entry) {
        try {
            return future.join();
        } catch (Throwable failure) {
            Logger.error("Global queue planner failed: request_id={}",
                    entry.context.getRequestId(), failure);
            return Plan.failure(entry, failure, availability.sequence());
        }
    }

    private void closePendingPlans(
            PlanWindow plans,
            List<Entry> entries,
            int fromIndex) {
        for (int index = fromIndex; index < plans.size(); index++) {
            final int pendingIndex = index;
            CompletableFuture<Plan> future = plans.submittedFuture(index);
            if (future == null) {
                continue;
            }
            future.whenComplete((plan, failure) -> {
                if (failure != null) {
                    Logger.error(
                            "Global queue abandoned planner failed: request_id={}",
                            entries.get(pendingIndex).context.getRequestId(),
                            failure);
                }
                if (plan != null) {
                    closePlan(plan);
                }
            });
        }
    }

    private static void closePlan(Plan plan) {
        try {
            plan.close();
        } catch (Throwable failure) {
            Logger.warn("Failed to close global queue route plan", failure);
        }
    }

    private List<Entry> nextWindow() {
        lock.lock();
        try {
            while (!closed.get()) {
                pruneHeadTombstones();
                if (peekHead() == null) {
                    awaitChanged();
                    continue;
                }
                if (frontierBlocked != null
                        && !isQueued(frontierBlocked)) {
                    frontierBlocked = null;
                }
                int frontierSize = planningFrontierSize();
                List<Entry> window = snapshotPrefix(frontierSize);
                if (window.isEmpty()) {
                    // Every queued entry is parked, or the first otherwise
                    // eligible entry is the selector-level frontier. A
                    // capacity event, a cancellation, or a newly inserted
                    // higher-priority request will wake the coordinator.
                    awaitChanged();
                    continue;
                }
                return window;
            }
            return List.of();
        } finally {
            lock.unlock();
        }
    }

    private List<Entry> snapshotPrefix(int limit) {
        if (limit <= 0) {
            return List.of();
        }
        List<Entry> result = new ArrayList<>(Math.min(limit, size));
        if (priorityOrdering) {
            for (int priority = nonEmptyPriorities.previousSetBit(PRIORITY_LEVELS - 1);
                    priority >= 0 && result.size() < limit;
                    priority = nonEmptyPriorities.previousSetBit(priority - 1)) {
                ArrayDeque<Entry> bucket = priorityBuckets[priority];
                if (bucket == null) {
                    continue;
                }
                for (Entry entry : bucket) {
                    if (entry == frontierBlocked) {
                        return result;
                    }
                    if (isEligible(entry)) {
                        result.add(entry);
                        if (result.size() == limit) {
                            break;
                        }
                    }
                }
            }
        } else {
            for (Entry entry : fifo) {
                if (entry == frontierBlocked) {
                    return result;
                }
                if (isEligible(entry)) {
                    result.add(entry);
                    if (result.size() == limit) {
                        break;
                    }
                }
            }
        }
        return result;
    }

    private boolean isEligible(Entry entry) {
        return !entry.removed && !entry.future.isDone()
                && !endpointBlocked.contains(entry);
    }

    /**
     * Bounds speculative work to the planner pool. A suffix is submitted only
     * as earlier plans are consumed, so a blocked head can abandon at most the
     * in-flight planner set rather than the complete planning frontier.
     */
    private PlanWindow planWindow(List<Entry> entries) {
        return new PlanWindow(entries, plannerCount);
    }

    private Plan plan(Entry entry) {
        long availabilitySequence = availability.sequence();
        if (entry.removed || entry.future.isDone()) {
            return Plan.done(entry, availabilitySequence);
        }
        // Give every admitted request one authoritative route attempt.  The
        // lifecycle registration already rejects a request which is expired
        // at ingress; this second-attempt guard prevents a deadline update or
        // timer race from repeatedly invoking the selector.
        if (entry.attempted
                && entry.context.requestExpired(System.currentTimeMillis())) {
            lifecycle.cancelRequest(
                    entry.context.getRequestId(),
                    0L,
                    CancelReason.DEADLINE_EXCEEDED);
            return Plan.done(entry, availabilitySequence);
        }
        entry.attempted = true;
        AdmissionMutation mutation = lifecycle.claimAdmissionMutation(
                entry.context.getRequestId(), entry.future);
        if (mutation == null) {
            return Plan.done(entry, availabilitySequence);
        }
        try {
            PlacementResult<QueueRouteAdmission, PlacementKey> result =
                    router.routeForQueue(entry.context);
            if (result.status() == PlacementResult.Status.SUCCESS) {
                return Plan.success(
                        entry, mutation, result.value(), availabilitySequence);
            }
            mutation.close();
            return Plan.result(entry, result, availabilitySequence);
        } catch (Throwable failure) {
            mutation.close();
            return Plan.failure(entry, failure, availabilitySequence);
        }
    }

    private Outcome commit(Plan plan) {
        Entry entry = plan.entry;
        if (entry.removed || entry.future.isDone()) {
            plan.close();
            remove(entry);
            return Outcome.DONE;
        }
        if (plan.failure != null) {
            remove(entry);
            completeDecisionResponse(entry, error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Placement failed: " + plan.failure.getMessage()));
            return Outcome.DONE;
        }
        PlacementResult<QueueRouteAdmission, PlacementKey> result = plan.result;
        if (result.status() == PlacementResult.Status.REJECTED) {
            remove(entry);
            completeDecisionResponse(entry, result.rejection());
            return Outcome.DONE;
        }
        if (result.status() == PlacementResult.Status.BLOCKED) {
            entry.blockedKey = result.blocker();
            if (tryPriorityRescue(entry)) {
                remove(entry);
                return Outcome.DONE;
            }
            return Outcome.BLOCKED;
        }
        if (result.status() == PlacementResult.Status.CLOSED) {
            remove(entry);
            return Outcome.DONE;
        }
        if (result.status() == PlacementResult.Status.LIMIT_REACHED) {
            remove(entry);
            completeAcceptanceLimit(entry);
            return Outcome.DONE;
        }
        QueueRouteAdmission admission = plan.admission;
        try {
            PlacementResult<ScheduledRequest, PlacementKey> publication =
                    admission.tryPublish(entry.context, entry.future, lifecycle);
            if (publication.status() == PlacementResult.Status.SUCCESS) {
                remove(entry);
                reportRouteSubmitted(entry.context, publication.value());
                return Outcome.DONE;
            }
            if (publication.status() == PlacementResult.Status.LIMIT_REACHED) {
                remove(entry);
                completeAcceptanceLimit(entry);
                return Outcome.DONE;
            }
            if (publication.status() == PlacementResult.Status.REJECTED
                    || publication.status() == PlacementResult.Status.CLOSED) {
                remove(entry);
                return Outcome.DONE;
            }
            entry.blockedKey = publication.blocker();
            // The ordinary route still owns the admission mutation here.
            // Eviction is an independent takeover transaction and must start
            // only after that mutation and its provisional route are closed;
            // otherwise RequestRegistry correctly rejects the nested claim.
            plan.close();
            if (tryPriorityRescue(entry)) {
                remove(entry);
                return Outcome.DONE;
            }
            if (!entry.commitConflictRetried) {
                entry.commitConflictRetried = true;
                return Outcome.REPLAN;
            }
            return Outcome.BLOCKED;
        } finally {
            plan.close();
        }
    }

    /**
     * Returns the endpoint which makes this plan unsafe to commit while an
     * earlier request is parked. A route can contain both Prefill and Decode;
     * sharing either exact endpoint is a local capacity conflict.
     */
    private WorkerEndpoint conflictingEndpoint(Plan plan) {
        QueueRouteAdmission admission = plan.admission;
        if (admission == null) {
            return null;
        }
        lock.lock();
        try {
            for (Entry blocked : endpointBlocked) {
                WorkerEndpoint endpoint = blocked.blockedEndpoint;
                if (endpoint != null && admission.usesEndpoint(endpoint)) {
                    return endpoint;
                }
            }
            return null;
        } finally {
            lock.unlock();
        }
    }

    private boolean tryPriorityRescue(Entry entry) {
        return priorityOrdering && !entry.future.isDone()
                && evictionManager.tryAdmit(entry.context, entry.future);
    }

    private static boolean isQueued(Entry entry) {
        return !entry.removed && !entry.future.isDone();
    }

    /**
     * Detects a capacity publication which raced with route planning. Without
     * this check, a release occurring immediately before {@link #park} would
     * be consumed before the request entered the parked index and leave it
     * waiting for a second release that might never arrive.
     */
    private boolean capacityChangedSince(Plan plan, PlacementKey blocker) {
        return availability.lastChangedSequence(blocker)
                > plan.availabilitySequence;
    }

    private void park(
            Entry entry,
            PlacementKey blocker,
            WorkerEndpoint endpoint) {
        lock.lock();
        try {
            if (!isQueued(entry)) {
                return;
            }
            entry.blockedKey = Objects.requireNonNull(blocker, "blocker");
            entry.blockedEndpoint = Objects.requireNonNull(endpoint, "endpoint");
            endpointBlocked.add(entry);
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void parkFrontier(Entry entry, PlacementKey blocker) {
        lock.lock();
        try {
            if (!isQueued(entry)) {
                return;
            }
            entry.blockedKey = Objects.requireNonNull(blocker, "blocker");
            frontierBlocked = entry;
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private static PlacementKey keyFor(WorkerEndpoint endpoint) {
        return PlacementKey.exact(
                endpoint.getStatus().getRole(),
                endpoint.getStatus().topologySnapshot().group(),
                endpoint.ipPort());
    }

    private void remove(Entry entry) {
        lock.lock();
        try {
            if (entry.removed) {
                return;
            }
            entry.removed = true;
            endpointBlocked.remove(entry);
            if (frontierBlocked == entry) {
                frontierBlocked = null;
            }
            if (priorityOrdering) {
                ArrayDeque<Entry> bucket = priorityBuckets[entry.priority];
                if (bucket != null && bucket.peekFirst() == entry) {
                    bucket.removeFirst();
                }
                if (bucket == null || bucket.isEmpty()) {
                    nonEmptyPriorities.clear(entry.priority);
                }
            } else if (fifo.peekFirst() == entry) {
                fifo.removeFirst();
            }
            size = Math.max(0, size - 1);
            entry.blockedKey = null;
            entry.blockedEndpoint = null;
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    /** Mark a completed request without scanning its bucket/deque. */
    private void markCompleted(Entry entry) {
        lock.lock();
        try {
            if (entry.removed) {
                return;
            }
            entry.removed = true;
            size = Math.max(0, size - 1);
            endpointBlocked.remove(entry);
            if (frontierBlocked == entry) {
                frontierBlocked = null;
            }
            entry.blockedKey = null;
            entry.blockedEndpoint = null;
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void pruneHeadTombstones() {
        if (priorityOrdering) {
            while (!nonEmptyPriorities.isEmpty()) {
                int priority = nonEmptyPriorities.previousSetBit(PRIORITY_LEVELS - 1);
                ArrayDeque<Entry> bucket = priorityBuckets[priority];
                while (bucket != null && !bucket.isEmpty()
                        && (bucket.peekFirst().removed
                        || bucket.peekFirst().future.isDone())) {
                    Entry tombstone = bucket.peekFirst();
                    boolean accounted = tombstone.removed;
                    bucket.removeFirst();
                    if (!accounted) {
                        size = Math.max(0, size - 1);
                    }
                }
                if (bucket == null || bucket.isEmpty()) {
                    nonEmptyPriorities.clear(priority);
                } else {
                    break;
                }
            }
        } else {
            while (!fifo.isEmpty()
                    && (fifo.peekFirst().removed
                    || fifo.peekFirst().future.isDone())) {
                Entry tombstone = fifo.peekFirst();
                boolean accounted = tombstone.removed;
                fifo.removeFirst();
                if (!accounted) {
                    size = Math.max(0, size - 1);
                }
            }
        }
    }

    private Entry peekHead() {
        if (priorityOrdering) {
            int priority = nonEmptyPriorities.previousSetBit(PRIORITY_LEVELS - 1);
            ArrayDeque<Entry> bucket = priority < 0 ? null : priorityBuckets[priority];
            return bucket == null ? null : bucket.peekFirst();
        }
        return fifo.peekFirst();
    }

    private static boolean resolvePriorityOrdering(ConfigService configService) {
        try {
            return configService.loadBalanceConfig().isPriorityOrdering();
        } catch (Throwable failure) {
            Logger.warn("Unable to read queue ordering; retaining FIFO", failure);
            return false;
        }
    }

    /**
     * Resolve the independent requests admitted to one planning pass. The
     * endpoint runtime owns dispatcher-specific credit accounting; this
     * global pump neither groups requests nor interprets decision policy.
     */
    private int planningFrontierSize() {
        try {
            var activeConfig = configService.loadBalanceConfig();
            RoleType admissionRole = router.requiredRoles().stream()
                    .filter(role -> role == RoleType.PREFILL
                            || role == RoleType.PDFUSION)
                    .findFirst()
                    .orElse(RoleType.PREFILL);
            long globalLimit = Math.max(
                    MIN_PLANNING_FRONTIER_SIZE,
                    activeConfig.queueScheduler().getCapacity()
                            .getMaxOutstandingRequestsGlobal());
            long available = endpointRegistry
                    .availablePrefillDeliveryCredits(admissionRole);
            if (available <= 0L) {
                // With a live fleet, zero aggregate credit is authoritative:
                // wait for an endpoint edge instead of selecting requests
                // which cannot be committed. With no endpoint yet, route one
                // request so the selector can establish its role/group wait.
                return endpointRegistry.getEndpointCount(admissionRole) == 0
                        ? MIN_PLANNING_FRONTIER_SIZE : 0;
            }
            available = Math.min(available, globalLimit);
            return available >= Integer.MAX_VALUE
                    ? Integer.MAX_VALUE : (int) available;
        } catch (Throwable failure) {
            Logger.warn("Unable to read queue decision policy; retaining single decision", failure);
            return MIN_PLANNING_FRONTIER_SIZE;
        }
    }

    private void awaitChanged() {
        try {
            changed.await();
        } catch (InterruptedException interruption) {
            if (closed.get()) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void signal() {
        lock.lock();
        try {
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void onAvailabilityChanged(PlacementKey key, long ignoredSequence) {
        lock.lock();
        try {
            if (frontierBlocked != null
                    && isRelevant(frontierBlocked.blockedKey, key)) {
                frontierBlocked.blockedKey = null;
                frontierBlocked.commitConflictRetried = false;
                frontierBlocked = null;
            }
            for (var iterator = endpointBlocked.iterator(); iterator.hasNext();) {
                Entry entry = iterator.next();
                if (isRelevant(entry.blockedKey, key)) {
                    entry.blockedKey = null;
                    entry.blockedEndpoint = null;
                    entry.commitConflictRetried = false;
                    iterator.remove();
                }
            }
            // Aggregate planning credits can become available even when no
            // parked entry matches this exact edge.
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private static boolean isRelevant(PlacementKey blocker, PlacementKey event) {
        if (blocker == null || event == null || blocker.role() != event.role()) {
            return false;
        }
        if (blocker.endpoint() != null) {
            return Objects.equals(blocker.endpoint(), event.endpoint());
        }
        return Objects.equals(blocker.group(), event.group())
                || blocker.group() == null;
    }

    private void completeAcceptanceLimit(Entry entry) {
        BalanceContext context = entry.context;
        int limit = context.getConfig().queueScheduler().getLifecycle()
                .getMaxDeliveredNotAcceptedRequestsGlobal();
        String detail = "admission capacity is temporarily exhausted"
                + "; active_admissions=" + lifecycle.decodeAcceptanceCount()
                + " limit=" + limit;
        Response failure = Response.error(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        failure.setErrorMessage(
                StrategyErrorType.RESOURCE_EXHAUSTED.buildErrorMessage(detail));
        completeDecisionResponse(entry, failure);
    }

    private void completeDecisionResponse(Entry entry, Response response) {
        try {
            lifecycle.publishQueueDecisionResponseAsync(
                    entry.context.getRequestId(), entry.future, response);
        } catch (Throwable failure) {
            Logger.error(
                    "Global queue response publication failed: request_id={}",
                    entry.context.getRequestId(), failure);
        }
    }

    private void reportRouteSubmitted(
            BalanceContext context,
            ScheduledRequest item) {
        try {
            reporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    item.prefillEp().getIp(),
                    System.currentTimeMillis() - context.getStartTime());
        } catch (Throwable failure) {
            Logger.warn("Failed to record route-submit telemetry", failure);
        }
    }

    private void drainOnClose() {
        List<Entry> abandoned = new ArrayList<>();
        lock.lock();
        try {
            if (priorityOrdering) {
                for (ArrayDeque<Entry> bucket : priorityBuckets) {
                    if (bucket != null) {
                        abandoned.addAll(bucket);
                    }
                }
            } else {
                abandoned.addAll(fifo);
            }
            fifo.clear();
            for (int i = 0; i < priorityBuckets.length; i++) {
                priorityBuckets[i] = null;
            }
            nonEmptyPriorities.clear();
            endpointBlocked.clear();
            frontierBlocked = null;
            size = 0;
        } finally {
            lock.unlock();
        }
        for (Entry entry : abandoned) {
            entry.removed = true;
            entry.blockedKey = null;
            entry.blockedEndpoint = null;
            completeDecisionResponse(entry, error(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "request scheduler is shutting down"));
        }
    }

    private static int normalizePriority(int priority) {
        return PriorityNormalizer.isValid(priority)
                ? priority : PriorityNormalizer.DEFAULT_PRIORITY;
    }

    private static Response error(StrategyErrorType type, String detail) {
        return RequestRegistry.buildErrorResponse(type, detail);
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        availability.removeListener(availabilityListener);
        signal();
        planners.shutdownNow();
        if (Thread.currentThread() != decisionThread) {
            try {
                decisionThread.join(decisionThreadJoinTimeoutMs());
            } catch (InterruptedException interruption) {
                Thread.currentThread().interrupt();
            }
        }
        awaitPlannerTermination();
    }

    private void awaitPlannerTermination() {
        try {
            planners.awaitTermination(
                    decisionThreadJoinTimeoutMs(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
        }
    }

    private long decisionThreadJoinTimeoutMs() {
        try {
            return Math.max(0L, configService.loadBalanceConfig()
                    .getInternalRuntime()
                    .getQueueDecisionThreadJoinTimeoutMs());
        } catch (Throwable failure) {
            Logger.warn("Unable to read queue shutdown timeout; using runtime default", failure);
            return new InternalRuntimeSettings()
                    .getQueueDecisionThreadJoinTimeoutMs();
        }
    }

    private enum Outcome {
        DONE,
        BLOCKED,
        RETRY,
        REPLAN
    }

    private static final class Entry {
        private final BalanceContext context;
        private final CompletableFuture<Response> future;
        private final int priority;
        private volatile boolean removed;
        private volatile boolean attempted;
        private volatile boolean commitConflictRetried;
        private volatile PlacementKey blockedKey;
        private volatile WorkerEndpoint blockedEndpoint;

        private Entry(
                BalanceContext context,
                CompletableFuture<Response> future,
                int priority) {
            this.context = context;
            this.future = future;
            this.priority = priority;
        }
    }

    /** Ordered, bounded submission view over one captured planning frontier. */
    private final class PlanWindow {
        private final List<Entry> entries;
        private final List<CompletableFuture<Plan>> futures;
        private final int maxInFlight;
        private int nextToSubmit;

        private PlanWindow(
                List<Entry> entries,
                int maxInFlight) {
            this.entries = List.copyOf(entries);
            this.futures = new ArrayList<>(entries.size());
            for (int index = 0; index < entries.size(); index++) {
                futures.add(null);
            }
            this.maxInFlight = Math.max(MIN_PLANNER_THREADS, maxInFlight);
        }

        private int size() {
            return entries.size();
        }

        private CompletableFuture<Plan> future(int index) {
            if (index < 0 || index >= entries.size()) {
                throw new IndexOutOfBoundsException(index);
            }
            int targetExclusive = Math.min(
                    entries.size(), index + maxInFlight);
            while (nextToSubmit < targetExclusive) {
                futures.set(nextToSubmit, submit(entries.get(nextToSubmit)));
                nextToSubmit++;
            }
            return futures.get(index);
        }

        private CompletableFuture<Plan> submittedFuture(int index) {
            if (index < 0 || index >= futures.size()) {
                throw new IndexOutOfBoundsException(index);
            }
            return futures.get(index);
        }

        private CompletableFuture<Plan> submit(Entry entry) {
            try {
                return CompletableFuture.supplyAsync(
                        () -> plan(entry), planners);
            } catch (Throwable failure) {
                Logger.error("Global queue planner submission failed", failure);
                return CompletableFuture.completedFuture(
                        Plan.failure(entry, failure, availability.sequence()));
            }
        }
    }

    private static final class Plan implements AutoCloseable {
        private final Entry entry;
        private final AdmissionMutation mutation;
        private final QueueRouteAdmission admission;
        private final PlacementResult<QueueRouteAdmission, PlacementKey> result;
        private final Throwable failure;
        private final long availabilitySequence;

        private Plan(
                Entry entry,
                AdmissionMutation mutation,
                QueueRouteAdmission admission,
                PlacementResult<QueueRouteAdmission, PlacementKey> result,
                Throwable failure,
                long availabilitySequence) {
            this.entry = entry;
            this.mutation = mutation;
            this.admission = admission;
            this.result = result;
            this.failure = failure;
            this.availabilitySequence = availabilitySequence;
        }

        static Plan success(Entry entry, AdmissionMutation mutation,
                            QueueRouteAdmission admission,
                            long availabilitySequence) {
            return new Plan(entry, mutation, admission,
                    PlacementResult.success(admission), null,
                    availabilitySequence);
        }

        static Plan result(Entry entry,
                           PlacementResult<QueueRouteAdmission, PlacementKey> result,
                           long availabilitySequence) {
            return new Plan(entry, null, null, result, null,
                    availabilitySequence);
        }

        static Plan done(Entry entry, long availabilitySequence) {
            return new Plan(entry, null, null,
                    PlacementResult.closed(), null, availabilitySequence);
        }

        static Plan failure(Entry entry, Throwable failure,
                            long availabilitySequence) {
            return new Plan(entry, null, null, null, failure,
                    availabilitySequence);
        }

        PlacementKey blocker() {
            if (result != null && result.status() == PlacementResult.Status.BLOCKED) {
                return result.blocker();
            }
            return entry.blockedKey;
        }

        WorkerEndpoint blockedEndpoint() {
            if (admission != null) {
                WorkerEndpoint endpoint = admission.blockedEndpoint();
                if (endpoint != null) {
                    return endpoint;
                }
            }
            return entry.blockedEndpoint;
        }

        @Override
        public void close() {
            if (admission != null) {
                try {
                    admission.close();
                } catch (Throwable failure) {
                    Logger.warn("Failed to close abandoned route plan", failure);
                }
            }
            if (mutation != null) {
                mutation.close();
            }
        }
    }
}
