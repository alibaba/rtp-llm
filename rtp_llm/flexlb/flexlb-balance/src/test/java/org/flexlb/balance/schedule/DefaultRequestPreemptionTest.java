package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.EndpointTestSupport;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.TaskPhase;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks resource waits against a real ledger, including cancellation and competition after worker release.
 */
class DefaultRequestPreemptionTest {
    @Test
    void noPreemptionWaitersReuseEmptyResults() {
        try (var f = new Fixture()) {
            org.junit.jupiter.api.Assertions.assertSame(
                    f.preemption.readyWork(), f.preemption.readyWork());
            org.junit.jupiter.api.Assertions.assertSame(
                    f.preemption.waitingRequests(), f.preemption.waitingRequests());
        }
    }

    @Test
    void cancellingARequestWithoutPreemptionDoesNotWakeCapacityWaiters() {
        try (var f = new Fixture()) {
            var wakeups = new java.util.concurrent.atomic.AtomicInteger();
            f.preemption.setWakeup(wakeups::incrementAndGet);
            f.preemption.cancel(f.request);
            assertEquals(0, wakeups.get());
        }
    }

    @Test
    void cancelAckDoesNotCompleteWaitAndLowerPriorityCannotTakeReleasedResources() {
        try (var f = new Fixture()) {
            assertTrue(f.preemption.start(f.selection, f.endpoint, f.acquired::set));
            assertNotNull(f.request.currentResourceWait());
            verify(f.victimScheduler).cancel(f.victim, CancelReason.PRIORITY_PREEMPTED);
            f.runChecks();
            assertNull(f.acquired.get());
            assertFalse(f.preemption.isIdle());
            f.releaseWorker();
            assertNull(f.register("b", 20));
            var c = f.register("c", 95);
            assertNotNull(c);
            f.runChecks();
            assertNull(f.acquired.get());
            f.endpoint.rollbackSelection(c);
            f.runChecks();
            assertNotNull(f.acquired.get());
            assertNull(f.request.currentResourceWait());
            assertTrue(f.preemption.isIdle());
            assertEquals(1, f.endpoint.selectionSnapshot().usage().occupiedRequests());
        }
    }

    @Test
    void requestCancellationImmediatelyInvalidatesWaitBeforePhysicalRemoval() {
        try (var f = new Fixture()) {
            assertTrue(f.preemption.start(f.selection, f.endpoint, f.acquired::set));
            assertTrue(
                    f.request.requestCancel(
                            CancelReason.CLIENT_CANCELLED,
                            false,
                            System.currentTimeMillis(),
                            StrategyErrorType.REQUEST_CANCELLED,
                            "cancelled",
                            null));
            assertNull(f.request.currentResourceWait());
            f.releaseWorker();
            assertNotNull(f.register("b", 20));
            f.runChecks();
            assertNull(f.acquired.get());
            assertTrue(f.preemption.isIdle());
            verify(f.victimScheduler, times(1)).cancel(f.victim, CancelReason.PRIORITY_PREEMPTED);
        }
    }

    @Test
    void stoppingEndsWaitWithoutWaitingForTheCancelledWorker() {
        try (var f = new Fixture()) {
            f.preemption.start(f.selection, f.endpoint, f.acquired::set);
            f.preemption.stopAdmission();
            assertNull(f.request.currentResourceWait());
            f.runChecks();
            assertTrue(f.preemption.isIdle());
            assertEquals(1, f.endpoint.selectionSnapshot().usage().occupiedRequests());
        }
    }

    /**
     * Prepares a Decode victim and waiter, then releases their resources after each preemption check.
     */
    private static final class Fixture implements AutoCloseable {
        private final RequestRepository repository = mock(RequestRepository.class);
        private final AbstractRequestScheduler victimScheduler =
                mock(AbstractRequestScheduler.class);
        private final RequestContext victim = mock(RequestContext.class);
        private final DecodeResources.AdmissionCapacity capacity =
                new DecodeResources.AdmissionCapacity(1, 100);
        private final DecodeEndpoint endpoint =
                EndpointTestSupport.decode(
                        EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8000, 8001),
                        repository);
        private final RequestContext request;
        private final RequestContext.SelectionAttempt selection;
        private final DefaultRequestPreemption preemption;
        private final AtomicReference<RoleSelection> acquired = new AtomicReference<>();
        private final DecodeEndpoint.RequestRegistration victimRegistration;

        private Fixture() {
            var config = SchedulingTestConfig.newConfig();
            SchedulingTestConfig.usePriorityQueue(config);
            SchedulingTestConfig.allowVictim(config, PreemptionStage.DECODE_ENGINE_OWNED);
            update(Map.of());
            victimRegistration = register("victim", 20);
            assertNotNull(victimRegistration);
            var task = new TaskInfo();
            task.setRequestId("victim");
            task.setInputLength(10L);
            task.setPhase(TaskPhase.RUNNING);
            update(Map.of("victim", task));
            var route = mock(RequestRoute.class);
            var snapshot = mock(RequestContext.Snapshot.class);
            when(victim.snapshot()).thenReturn(snapshot);
            when(snapshot.route()).thenReturn(route);
            when(victim.getRequestScheduler()).thenReturn(victimScheduler);
            when(route.decodeEndpoint()).thenReturn(endpoint);
            when(route.decodeRegistration()).thenReturn(victimRegistration);
            doReturn(victim).when(repository).findActive("victim");
            request = RequestProtocolTestSupport.context(config, 99001L);
            request.setSchedulingMetadata(SchedulingMetadata.of(90, null, Long.MAX_VALUE, 50));
            selection = RequestProtocolTestSupport.admissionHandle(request);
            preemption =
                    new DefaultRequestPreemption(
                            repository,
                            mock(org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                            SchedulingTestConfig.configService(config),
                            capacity,
                            new DecodePreemptionPlanner());
        }

        @Override
        public void close() {
            preemption.stopAdmission();
            runChecks();
            if (acquired.get() != null)
                ((org.flexlb.worker.endpoint.SelectionRollback) acquired.get().endpoint())
                        .rollbackSelection(acquired.get().registration());
            preemption.close();
            EndpointTestSupport.closeAndWait(endpoint);
            RequestProtocolTestSupport.close(request.getRequestScheduler());
        }

        private void runChecks() {
            preemption.requestPeriodicCheck();
            for (var work : preemption.readyWork()) work.action().run();
        }

        private DecodeEndpoint.RequestRegistration register(String id, int priority) {
            return endpoint.tryRegisterSelection(
                            id, endpoint.selectionSnapshot(), 10, 10, priority, capacity)
                    .value();
        }

        private void releaseWorker() {
            endpoint.releaseRequest(victimRegistration);
            update(Map.of());
        }

        private void update(Map<String, TaskInfo> tasks) {
            var status = new WorkerStatusResponse();
            status.setTotalKvCacheTokens(100000L);
            status.setAvailableKvCacheTokens(100000L);
            status.setRunningTaskInfo(tasks);
            EndpointTestSupport.applyStatus(endpoint, status).run();
        }
    }
}
