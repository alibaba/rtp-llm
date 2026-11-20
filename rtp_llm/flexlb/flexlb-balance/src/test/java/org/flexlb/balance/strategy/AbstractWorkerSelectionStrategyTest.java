package org.flexlb.balance.strategy;

import org.flexlb.balance.schedule.RequestContext;
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
    void conflictRerunsSelectionWithTheSameCandidate() {
        var worker = mock(WorkerEndpoint.class);
        var registrations = new AtomicInteger();
        var selections = new AtomicInteger();
        var result =
                (new CandidateLoopStrategy() {
                            public WorkerEndpoint candidate(WorkerEndpoint selected) {
                                return selected;
                            }

                            public WorkerEndpoint.Registration<RoleSelection> register(
                                    RequestContext context,
                                    RoleType role,
                                    String group,
                                    WorkerEndpoint selected) {
                                assertSame(worker, selected);
                                return registrations.incrementAndGet() == 1
                                        ? WorkerEndpoint.Registration.rejected(
                                                WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT)
                                        : new WorkerEndpoint.Registration<>(
                                                WorkerEndpoint.RegistrationStatus.REGISTERED,
                                                registered);
                            }
                        })
                        .selectCandidates(
                                null,
                                scope,
                                new AbstractWorkerSelectionStrategy.RegistrationCandidates<>(
                                        List.of(worker)),
                                list -> {
                                    selections.incrementAndGet();
                                    return SelectionResult.success(list.getFirst());
                                });
        assertSame(registered, result.value());
        assertEquals(2, registrations.get());
        assertEquals(2, selections.get());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {true, false})
    void fullCandidatesDoNotConsumeTheVersionConflictBudgetOrModifyTheSourceList(
            boolean lastWorkerFits) {
        List<WorkerEndpoint> source =
                java.util.stream.IntStream.range(0, 20)
                        .mapToObj(i -> mock(WorkerEndpoint.class))
                        .toList();
        var calls = new AtomicInteger();
        var result =
                (new CandidateLoopStrategy() {
                            public WorkerEndpoint candidate(WorkerEndpoint selected) {
                                return selected;
                            }

                            public WorkerEndpoint.Registration<RoleSelection> register(
                                    RequestContext context,
                                    RoleType role,
                                    String group,
                                    WorkerEndpoint selected) {
                                calls.incrementAndGet();
                                return lastWorkerFits && selected == source.getLast()
                                        ? new WorkerEndpoint.Registration<>(
                                                WorkerEndpoint.RegistrationStatus.REGISTERED,
                                                registered)
                                        : WorkerEndpoint.Registration.rejected(
                                                WorkerEndpoint.RegistrationStatus.CAPACITY_FULL);
                            }
                        })
                        .selectCandidates(
                                null,
                                scope,
                                new AbstractWorkerSelectionStrategy.RegistrationCandidates<>(
                                        source),
                                list -> SelectionResult.success(list.getFirst()));
        if (lastWorkerFits) assertSame(registered, result.value());
        else {
            assertEquals(SelectionResult.Status.BLOCKED, result.status());
            assertEquals(scope, result.blocker());
        }
        assertEquals(20, calls.get());
        assertEquals(20, source.size());
    }

    @Test
    void repeatedConflictsReturnToSchedulerAfterEightRegistrations() {
        var calls = new AtomicInteger();
        WorkerEndpoint candidate = mock(WorkerEndpoint.class);
        var result =
                (new CandidateLoopStrategy() {
                            public WorkerEndpoint candidate(WorkerEndpoint selected) {
                                return selected;
                            }

                            public WorkerEndpoint.Registration<RoleSelection> register(
                                    RequestContext context,
                                    RoleType role,
                                    String group,
                                    WorkerEndpoint selected) {
                                calls.incrementAndGet();
                                return WorkerEndpoint.Registration.rejected(
                                        WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT);
                            }
                        })
                        .selectCandidates(
                                null,
                                scope,
                                new AbstractWorkerSelectionStrategy.RegistrationCandidates<>(
                                        List.of(candidate)),
                                list -> SelectionResult.success(list.getFirst()));
        assertEquals(SelectionResult.Status.RETRY, result.status());
        assertEquals(8, calls.get());
    }

    @Test
    void rejectedSelectionNeverRegistersResources() {
        var calls = new AtomicInteger();
        var result =
                (new CandidateLoopStrategy() {
                            public WorkerEndpoint candidate(WorkerEndpoint selected) {
                                throw new AssertionError();
                            }

                            public WorkerEndpoint.Registration<RoleSelection> register(
                                    RequestContext context,
                                    RoleType role,
                                    String group,
                                    WorkerEndpoint selected) {
                                calls.incrementAndGet();
                                throw new AssertionError();
                            }
                        })
                        .selectCandidates(
                                null,
                                scope,
                                new AbstractWorkerSelectionStrategy.RegistrationCandidates<>(
                                        List.of()),
                                list -> SelectionResult.blocked(scope));
        assertEquals(SelectionResult.Status.BLOCKED, result.status());
        assertEquals(0, calls.get());
    }

    /**
     * Exercises the shared candidate loop without constructing a full role selection.
     */
    private abstract static class CandidateLoopStrategy
            extends AbstractWorkerSelectionStrategy<WorkerEndpoint, WorkerEndpoint> {
        @Override
        public SelectionResult<RoleSelection, CapacityKey> selectAndRegister(
                RequestContext context, String group) {
            throw new UnsupportedOperationException("test calls the candidate loop directly");
        }
    }
}
