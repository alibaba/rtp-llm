package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.WorkerAdmissionCapacity;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Tracks capacity waits by role/group and charges retries when the scheduler creates a task.
 * The scheduling queue owns ordering. Notifications and periodic checks update retry eligibility;
 * failed retries and newly parked requests wait for the next finite pass.
 * All operations run under the scheduler lock.
 */
public final class CapacityWaitQueue {
    private final Map<CapacityKey, Domain> domains = new HashMap<>();
    private final Map<RequestContext, Waiter> membership = new IdentityHashMap<>();
    // Worker generation IDs are unique across roles within this process.
    private final Map<Long, RetryBudget> workerBudgets = new HashMap<>();
    private long workerAdditionVersion;

    /**
     * Joins the next pass for the latest blocker. Returns true if a newer budget reopened the pass.
     */
    public boolean park(
            RequestContext request, CapacityKey blocker, Set<Long> candidateGenerations) {
        var key = blocker.waitingDomain();
        var waiter = membership.get(request);
        if (waiter == null || !waiter.domain.key.equals(key)) {
            remove(request);
            waiter = new Waiter(domains.computeIfAbsent(key, Domain::new));
            membership.put(request, waiter);
            waiter.candidateGenerations = Set.copyOf(candidateGenerations);
            waiter.domain.deferred.add(request);
            return false;
        }
        waiter.candidateGenerations = Set.copyOf(candidateGenerations);
        return defer(request, waiter);
    }

    public void remove(RequestContext request) {
        var waiter = membership.remove(request);
        if (waiter == null) return;
        var domain = waiter.domain;
        if (waiter.budget != null) {
            finishRetry(waiter, false);
        } else if (!domain.deferred.remove(request)) {
            domain.currentPassCount--;
        }
        if (domain.currentPassCount == 0
                && domain.deferred.isEmpty()
                && domain.unfinishedRetries == 0) domains.remove(domain.key, domain);
    }

    /**
     * Applies a newer worker budget and returns whether the scheduler should rescan.
     */
    public boolean capacityChanged(WorkerAdmissionCapacity capacity) {
        var budget = workerBudgets.get(capacity.generation());
        if (budget != null && capacity.revision() <= budget.workerCapacityRevision) return false;
        if (budget == null) {
            budget = new RetryBudget();
            workerBudgets.put(capacity.generation(), budget);
            workerAdditionVersion++;
        }
        budget.refreshWorker(capacity);
        boolean rescan = false;
        for (var domain : domains.values()) {
            if (!budget.matches(domain.key)) {
                domain.workers.remove(budget);
                continue;
            }
            if (!domain.workers.contains(budget)) domain.workers.add(budget);
            if (domain.controlCheck != null && budget.retryLimit > 0L)
                domain.controlCheck.retryLimit = 0L;
            rescan |= reopen(domain) || (budget.hasRetrySlots() && domain.currentPassCount > 0);
        }
        return rescan;
    }

    public long workerAdditionVersion() {
        return workerAdditionVersion;
    }

    /**
     * Refreshes local capacity and permits one control check per full role/group for preemption.
     * The control check is not an Engine admission slot. Worker budgets remain shared across groups.
     */
    public void checkCapacity(List<WorkerAdmissionCapacity> capacities, long sampledVersion) {
        if (domains.isEmpty() && capacities.isEmpty()) {
            workerBudgets.clear();
            return;
        }
        var currentWorkers = new HashSet<Long>();
        for (var capacity : capacities) {
            long generation = capacity.generation();
            currentWorkers.add(generation);
            var budget = workerBudgets.computeIfAbsent(generation, ignored -> new RetryBudget());
            if (capacity.revision() >= budget.workerCapacityRevision)
                budget.refreshWorker(capacity);
        }
        // A new worker notification makes absence from this sample inconclusive.
        if (sampledVersion == workerAdditionVersion)
            workerBudgets
                    .entrySet()
                    .removeIf(
                            entry -> {
                                if (currentWorkers.contains(entry.getKey())) return false;
                                var budget = entry.getValue();
                                if (budget.unfinishedRetries == 0) return true;
                                budget.refresh(0L);
                                return false;
                            });
        for (var domain : domains.values()) {
            domain.workers.clear();
            for (var budget : workerBudgets.values())
                if (budget.matches(domain.key)) domain.workers.add(budget);
            boolean full = domain.workers.stream().noneMatch(budget -> budget.retryLimit > 0L);
            if (full && domain.controlCheck == null) domain.controlCheck = new RetryBudget();
            if (domain.controlCheck != null) domain.controlCheck.refresh(full ? 1L : 0L);
            reopen(domain);
        }
    }

    public boolean hasWaitingRequests() {
        return !domains.isEmpty();
    }

    /**
     * Tests eligibility during the existing ordered scan without charging a budget.
     */
    public boolean canSchedule(RequestContext request) {
        var waiter = membership.get(request);
        return waiter == null || availableBudget(request, waiter) != null;
    }

    /**
     * Charges one retry after ordering selects the request and before the task is recorded.
     */
    public boolean tryStartRetry(RequestContext request) {
        var waiter = membership.get(request);
        if (waiter == null) return true;
        var budget = availableBudget(request, waiter);
        if (budget == null) return false;
        waiter.domain.currentPassCount--;
        waiter.domain.unfinishedRetries++;
        budget.chargedRetries++;
        budget.unfinishedRetries++;
        waiter.budget = budget;
        waiter.budgetRoundAtDispatch = budget.budgetRound;
        return true;
    }

    /**
     * Settles an attempt that ended without removing the request or reporting a new blocker.
     */
    public boolean retry(RequestContext request) {
        var waiter = membership.get(request);
        return waiter != null && waiter.budget != null && defer(request, waiter);
    }

    public void clear() {
        domains.clear();
        membership.clear();
        workerBudgets.clear();
    }

    private boolean defer(RequestContext request, Waiter waiter) {
        if (!waiter.domain.deferred.add(request)) return false;
        if (waiter.budget == null) {
            waiter.domain.currentPassCount--;
            return false;
        }
        return finishRetry(waiter, true) && reopen(waiter.domain);
    }

    private boolean finishRetry(Waiter waiter, boolean retryDidNotAcquireCapacity) {
        var budget = waiter.budget;
        budget.unfinishedRetries--;
        waiter.domain.unfinishedRetries--;
        boolean refreshedSinceDispatch = waiter.budgetRoundAtDispatch != budget.budgetRound;
        if (refreshedSinceDispatch && retryDidNotAcquireCapacity) budget.chargedRetries--;
        waiter.budget = null;
        return refreshedSinceDispatch;
    }

    private boolean reopen(Domain domain) {
        if (domain.currentPassCount != 0 || domain.deferred.isEmpty()) return false;
        domain.currentPassCount = domain.deferred.size();
        domain.deferred.clear();
        return true;
    }

    private RetryBudget availableBudget(RequestContext request, Waiter waiter) {
        var domain = waiter.domain;
        if (waiter.budget != null || domain.deferred.contains(request)) return null;
        for (long generation : waiter.candidateGenerations) {
            var budget = workerBudgets.get(generation);
            if (budget != null && budget.matches(domain.key) && budget.hasRetrySlots())
                return budget;
        }
        for (var budget : domain.workers) if (budget.hasRetrySlots()) return budget;
        return domain.controlCheck != null && domain.controlCheck.hasRetrySlots()
                ? domain.controlCheck
                : null;
    }

    /**
     * Keeps the retry pass and the capacity sources for one role/group.
     */
    private static final class Domain {
        private final CapacityKey key;
        private final List<RetryBudget> workers = new ArrayList<>();
        private final Set<RequestContext> deferred = new HashSet<>();
        private RetryBudget controlCheck;

        private int currentPassCount;
        private int unfinishedRetries;

        private Domain(CapacityKey key) {
            this.key = key;
        }
    }

    /**
     * Retains a request's budget claim until its retry finishes or moves to another group.
     */
    private static final class Waiter {
        private final Domain domain;
        private RetryBudget budget;
        private long budgetRoundAtDispatch;
        private Set<Long> candidateGenerations = Set.of();

        private Waiter(Domain domain) {
            this.domain = domain;
        }
    }

    /**
     * Counts retries granted by a worker snapshot or a group probe.
     * A refresh starts a new round even if the worker revision did not change.
     * Unfinished retries count against the new round until their outcome is known.
     */
    private static final class RetryBudget {
        // Set only for worker budgets. Domain probes cannot participate in worker matching.
        private CapacityKey workerKey;
        private long workerCapacityRevision = -1L;
        private long retryLimit;
        private long chargedRetries;
        private long budgetRound;
        private int unfinishedRetries;

        private void refreshWorker(WorkerAdmissionCapacity observation) {
            workerKey = observation.key();
            workerCapacityRevision = observation.revision();
            refresh(observation.availableRequests());
        }

        private void refresh(long retryLimit) {
            this.retryLimit = retryLimit;
            chargedRetries = unfinishedRetries;
            budgetRound++;
        }

        private boolean hasRetrySlots() {
            return retryLimit > chargedRetries;
        }

        private boolean matches(CapacityKey waiter) {
            return workerKey.role() == waiter.role()
                    && (waiter.group() == null || waiter.group().equals(workerKey.group()));
        }
    }
}
