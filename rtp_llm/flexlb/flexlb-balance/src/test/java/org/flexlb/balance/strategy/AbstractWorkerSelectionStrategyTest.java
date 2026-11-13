package org.flexlb.balance.strategy;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;

/**
 * Verifies bounded conflicts, candidate exclusion and successful resource handoff through the common flow.
 */
class AbstractWorkerSelectionStrategyTest {
    private final CapacityKey scope = new CapacityKey(RoleType.DECODE, "g", null);
    private final RoleSelection registered = mock(RoleSelection.class);

    @Test
    void conflictRefreshesOnlyTheSelectedCandidate() {
        Object stale = new Object(), fresh = new Object();
        var calls = new AtomicInteger();
        var result =
                (new AbstractWorkerSelectionStrategy<Object, Object>() {
                            public Object candidate(Object selected) {
                                return selected;
                            }

                            public WorkerEndpoint.Registration<RoleSelection> register(
                                    org.flexlb.balance.schedule.RequestContext context,
                                    RoleType role,
                                    String group,
                                    Object selected) {
                                calls.incrementAndGet();
                                return selected == stale
                                        ? WorkerEndpoint.Registration.rejected(
                                                WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT)
                                        : new WorkerEndpoint.Registration<>(
                                                WorkerEndpoint.RegistrationStatus.REGISTERED,
                                                registered);
                            }

                            public Object refresh(Object candidate) {
                                assertSame(stale, candidate);
                                return fresh;
                            }

                            @Override
                            protected boolean tryOtherWhenFull() {
                                return false;
                            }
                        })
                        .selectCandidates(
                                null,
                                scope,
                                List.of(stale),
                                list -> RoutingResult.success(list.getFirst()));
        assertSame(registered, result.value());
        assertEquals(2, calls.get());
    }

    @Test
    void fullCandidatesDoNotConsumeTheVersionConflictBudgetOrModifyTheSourceList() {
        List<Integer> source = java.util.stream.IntStream.range(0, 20).boxed().toList();
        var calls = new AtomicInteger();
        var result =
                (new AbstractWorkerSelectionStrategy<Integer, Integer>() {
                            public Integer candidate(Integer selected) {
                                return selected;
                            }

                            public WorkerEndpoint.Registration<RoleSelection> register(
                                    org.flexlb.balance.schedule.RequestContext context,
                                    RoleType role,
                                    String group,
                                    Integer selected) {
                                calls.incrementAndGet();
                                return selected == 19
                                        ? new WorkerEndpoint.Registration<>(
                                                WorkerEndpoint.RegistrationStatus.REGISTERED,
                                                registered)
                                        : WorkerEndpoint.Registration.rejected(
                                                WorkerEndpoint.RegistrationStatus.CAPACITY_FULL);
                            }

                            public Integer refresh(Integer candidate) {
                                throw new AssertionError("full candidate cannot be refreshed");
                            }
                        })
                        .selectCandidates(
                                null,
                                scope,
                                source,
                                list -> RoutingResult.success(list.getFirst()));
        assertSame(registered, result.value());
        assertEquals(20, calls.get());
        assertEquals(20, source.size());
    }

    @Test
    void repeatedConflictsReturnToSchedulerAfterEightRegistrations() {
        var calls = new AtomicInteger();
        Object candidate = new Object();
        var result =
                (new AbstractWorkerSelectionStrategy<Object, Object>() {
                            public Object candidate(Object selected) {
                                return selected;
                            }

                            public WorkerEndpoint.Registration<RoleSelection> register(
                                    org.flexlb.balance.schedule.RequestContext context,
                                    RoleType role,
                                    String group,
                                    Object selected) {
                                calls.incrementAndGet();
                                return WorkerEndpoint.Registration.rejected(
                                        WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT);
                            }

                            public Object refresh(Object original) {
                                return original;
                            }

                            @Override
                            protected boolean tryOtherWhenFull() {
                                return false;
                            }
                        })
                        .selectCandidates(
                                null,
                                scope,
                                List.of(candidate),
                                list -> RoutingResult.success(list.getFirst()));
        assertEquals(RoutingResult.Status.RETRY, result.status());
        assertEquals(8, calls.get());
    }

    @Test
    void rejectedSelectionNeverRegistersResources() {
        var calls = new AtomicInteger();
        var result =
                (new AbstractWorkerSelectionStrategy<Object, Object>() {
                            public Object candidate(Object selected) {
                                throw new AssertionError();
                            }

                            public WorkerEndpoint.Registration<RoleSelection> register(
                                    org.flexlb.balance.schedule.RequestContext context,
                                    RoleType role,
                                    String group,
                                    Object selected) {
                                calls.incrementAndGet();
                                throw new AssertionError();
                            }

                            public Object refresh(Object original) {
                                throw new AssertionError();
                            }

                            @Override
                            protected boolean tryOtherWhenFull() {
                                return false;
                            }
                        })
                        .selectCandidates(
                                null, scope, List.of(), list -> RoutingResult.blocked(scope));
        assertEquals(RoutingResult.Status.BLOCKED, result.status());
        assertEquals(0, calls.get());
    }
}
