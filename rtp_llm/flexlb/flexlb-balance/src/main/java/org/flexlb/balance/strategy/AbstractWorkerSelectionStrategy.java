package org.flexlb.balance.strategy;

import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Selects and registers one role, owning candidate refresh, exclusion and bounded version-conflict retries.
 * Subclasses provide ranking and role-specific registration; the loop returns only registered selections.
 */
public abstract class AbstractWorkerSelectionStrategy<I, C> {
    private static final int MAX_VERSION_CONFLICTS = 8;

    protected final RoutingResult<RoleSelection, CapacityKey> selectCandidates(
            org.flexlb.balance.schedule.RequestContext context,
            CapacityKey scope,
            List<I> initial,
            Function<List<I>, RoutingResult<C, CapacityKey>> choose) {
        if (initial.isEmpty())
            return RoutingResult.blocked(
                    scope,
                    Response.error(
                            org.flexlb.worker.endpoint.StrategyErrorType.forRole(scope.role())),
                    null);
        List<I> candidates = initial;
        int conflicts = 0;
        RoutingResult<RoleSelection, CapacityKey> lastFull = null;
        while (true) {
            if (candidates.isEmpty() && candidates != initial)
                return lastFull == null ? RoutingResult.retry() : lastFull;
            var choice = choose.apply(Collections.unmodifiableList(candidates));
            if (choice.status() != RoutingResult.Status.SUCCESS) return choice.asFailure();
            C selected = choice.value();
            I original = candidate(selected);
            int index = -1;
            for (int i = 0; i < candidates.size(); i++) {
                if (candidates.get(i) == original) {
                    index = i;
                    break;
                }
            }
            checkArgument(index >= 0, "strategy must select an original candidate");
            var accepted = register(context, scope.role(), scope.group(), selected);
            if (accepted.status() == WorkerEndpoint.RegistrationStatus.REGISTERED)
                return RoutingResult.success(accepted.value());
            if (accepted.status() == WorkerEndpoint.RegistrationStatus.CAPACITY_FULL) {
                lastFull = capacityFull(selected, scope);
                if (!tryOtherWhenFull()) return lastFull;
            }
            if (candidates == initial) candidates = new ArrayList<>(initial);
            switch (accepted.status()) {
                case VERSION_CONFLICT, ALREADY_REGISTERED -> {
                    if (++conflicts >= MAX_VERSION_CONFLICTS)
                        return RoutingResult.retry(
                                Response.error(StrategyErrorType.SCHEDULER_PLAN_CONFLICT));
                    I refreshed = refresh(original);
                    if (refreshed == null) candidates.remove(index);
                    else candidates.set(index, refreshed);
                }
                case CAPACITY_FULL, UNAVAILABLE -> candidates.remove(index);
                case REGISTERED ->
                        throw new IllegalStateException("registered selection already returned");
            }
        }
    }

    protected abstract I candidate(C selected);

    protected abstract WorkerEndpoint.Registration<RoleSelection> register(
            org.flexlb.balance.schedule.RequestContext context,
            org.flexlb.config.service.RoleType role,
            String group,
            C selected);

    protected RoutingResult<RoleSelection, CapacityKey> capacityFull(
            C selected, CapacityKey scope) {
        return RoutingResult.blocked(
                scope, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED), null);
    }

    protected boolean tryOtherWhenFull() {
        return true;
    }

    protected abstract I refresh(I candidate);
}
