package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Selects and registers one role, owning candidate exclusion and bounded version-conflict retries.
 * Subclasses provide ranking and role-specific registration; the loop returns only registered selections.
 */
public abstract class AbstractWorkerSelectionStrategy<CandidateT, SelectionT> {
    private static final int MAX_VERSION_CONFLICTS = 8;

    public abstract SelectionResult<RoleSelection, CapacityKey> selectAndRegister(
            RequestContext context, String group);

    protected final SelectionResult<RoleSelection, CapacityKey> selectCandidates(
            RequestContext context,
            CapacityKey scope,
            RegistrationCandidates<CandidateT> candidates,
            Function<List<CandidateT>, SelectionResult<SelectionT, CapacityKey>> choose) {
        if (candidates.remaining.isEmpty())
            return SelectionResult.blocked(
                    scope, Response.error(StrategyErrorType.forRole(scope.role())), null);

        int conflicts = 0;
        SelectionResult<RoleSelection, CapacityKey> lastFull = null;
        while (!candidates.remaining.isEmpty()) {
            var choice = choose.apply(Collections.unmodifiableList(candidates.remaining));
            if (choice.status() != SelectionResult.Status.SUCCESS) {
                if (lastFull != null && choice.status() == SelectionResult.Status.REJECTED)
                    return lastFull;
                return choice.asFailure();
            }
            var selected = choice.value();
            var original = candidate(selected);
            int index = 0;
            while (index < candidates.remaining.size()
                    && candidates.remaining.get(index) != original) index++;
            checkArgument(
                    index < candidates.remaining.size(),
                    "strategy must select an original candidate");

            var accepted = register(context, scope.role(), scope.group(), selected);
            switch (accepted.status()) {
                case REGISTERED -> {
                    return SelectionResult.success(accepted.value());
                }
                case UNAVAILABLE -> candidates.mutableRemaining().remove(index);
                case CAPACITY_FULL -> {
                    lastFull = capacityFull(selected, scope);
                    candidates.recordFull(index);
                }
                case VERSION_CONFLICT, ALREADY_REGISTERED -> {
                    if (++conflicts >= MAX_VERSION_CONFLICTS)
                        return SelectionResult.retry(
                                Response.error(StrategyErrorType.SCHEDULER_PLAN_CONFLICT));
                }
            }
        }
        return lastFull != null ? lastFull : SelectionResult.retry();
    }

    protected abstract CandidateT candidate(SelectionT selected);

    protected abstract WorkerEndpoint.Registration<RoleSelection> register(
            RequestContext context,
            org.flexlb.config.service.RoleType role,
            String group,
            SelectionT selected);

    protected SelectionResult<RoleSelection, CapacityKey> capacityFull(
            SelectionT selected, CapacityKey scope) {
        return SelectionResult.blocked(
                scope, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED), null);
    }

    /**
     * Tracks registration outcomes for one role selection. Full workers remain eligible for a
     * capacity wait, even though they cannot be registered again in the same attempt.
     */
    protected static final class RegistrationCandidates<T> {
        private final List<T> initial;

        private List<T> remaining;
        private List<T> full;

        public RegistrationCandidates(List<T> initial) {
            this.initial = initial;
            this.remaining = initial;
        }

        boolean onlyFullWorkersRemain() {
            return remaining.isEmpty() && full != null;
        }

        Iterable<T> includingFull() {
            return full == null
                    ? remaining
                    : com.google.common.collect.Iterables.concat(remaining, full);
        }

        private void recordFull(int index) {
            if (full == null) full = new ArrayList<>();
            full.add(mutableRemaining().remove(index));
        }

        private List<T> mutableRemaining() {
            if (remaining == initial) remaining = new ArrayList<>(initial);
            return remaining;
        }
    }
}
