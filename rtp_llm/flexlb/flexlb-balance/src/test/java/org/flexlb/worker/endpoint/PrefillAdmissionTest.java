package org.flexlb.worker.endpoint;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks atomic request admission, Engine headroom and visibility prediction.
 */
class PrefillAdmissionTest {
    @Test
    void largeEngineWatermarkCannotOverflowTheRetryBudget() {
        var config = new FlexlbConfig();
        var prefill = config.getRouter().getRoles().getPrefill();
        prefill.getSelection().setType(RoutingConfig.PrefillSelectionType.TOKEN_BALANCE);
        var queue = new RoutingConfig.EngineQueueConfig();
        queue.setWaitingThreshold(Long.MAX_VALUE);
        prefill.getAdmission().setEngineQueue(queue);
        var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8081);
        var endpoint = new PrefillEndpoint(status, config, new CapacityNotifier());
        try {
            report(endpoint, 0, 0);
            assertEquals(128L, endpoint.requestCapacity().availableRequests());
            var registration =
                    endpoint.tryRegisterSelection("first", endpoint.selectionSnapshot(), 0, -1L)
                            .value();
            assertEquals(127L, endpoint.requestCapacity().availableRequests());
            endpoint.rollbackSelection(registration);
            assertEquals(128L, endpoint.requestCapacity().availableRequests());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @Test
    void schedulingCapacityUsesCountAndEngineQueueHeadroom() {
        var endpoint = endpoint(128, true);
        try {
            report(endpoint, 10, 0);
            var available = endpoint.requestCapacity();
            assertEquals(1L, available.availableRequests());
            var registration =
                    endpoint.tryRegisterSelection(
                                    "last-headroom", endpoint.selectionSnapshot(), 0, -1L)
                            .value();
            assertEquals(0L, endpoint.requestCapacity().availableRequests());
            assertTrue(endpoint.requestCapacity().revision() > available.revision());
            endpoint.rollbackSelection(registration);
            assertEquals(1L, endpoint.requestCapacity().availableRequests());
            report(endpoint, -1, 127);
            assertEquals(1L, endpoint.requestCapacity().availableRequests());
            report(endpoint, -1, 128);
            assertEquals(0L, endpoint.requestCapacity().availableRequests());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @Test
    void concurrentZeroCostRequestsStillConsumeOneSlotEach() throws Exception {
        var endpoint = endpoint(1, false);
        try (var workers = Executors.newFixedThreadPool(2)) {
            var snapshot = endpoint.selectionSnapshot();
            var start = new CountDownLatch(1);
            var first =
                    workers.submit(
                            () -> {
                                assertTrue(start.await(2, TimeUnit.SECONDS));
                                return endpoint.tryRegisterSelection("first", snapshot, 0, -1L);
                            });
            var second =
                    workers.submit(
                            () -> {
                                assertTrue(start.await(2, TimeUnit.SECONDS));
                                return endpoint.tryRegisterSelection("second", snapshot, 0, -1L);
                            });
            start.countDown();
            var a = first.get(2, TimeUnit.SECONDS);
            var b = second.get(2, TimeUnit.SECONDS);
            assertEquals(
                    1,
                    java.util.stream.Stream.of(a, b)
                            .filter(r -> r.status() == WorkerEndpoint.RegistrationStatus.REGISTERED)
                            .count());
            assertEquals(1, endpoint.selectionSnapshot().outstandingRequests());
            assertTrue(endpoint.selectionSnapshot().admissionBlocked());
            assertTrue(endpoint.rollbackSelection(a.value() != null ? a.value() : b.value()));
            assertFalse(endpoint.selectionSnapshot().admissionBlocked());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @Test
    void unobservedRegistrationConsumesEngineWaitingHeadroom() {
        var endpoint = endpoint(128, true);
        try {
            report(endpoint, 10, 0);
            var first =
                    endpoint.tryRegisterSelection("first", endpoint.selectionSnapshot(), 0, -1L);
            assertEquals(WorkerEndpoint.RegistrationStatus.REGISTERED, first.status());
            assertEquals(
                    WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                    endpoint.tryRegisterSelection("second", endpoint.selectionSnapshot(), 0, -1L)
                            .status());
            endpoint.rollbackSelection(first.value());
            assertFalse(endpoint.selectionSnapshot().admissionBlocked());
            report(endpoint, -1, 0);
            for (int i = 0; i < 12; i++) {
                assertEquals(
                        WorkerEndpoint.RegistrationStatus.REGISTERED,
                        endpoint.tryRegisterSelection(
                                        "unknown-" + i, endpoint.selectionSnapshot(), 0, -1L)
                                .status());
            }
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @Test
    void engineRequestsWithoutLocalRecordsConsumeCountCapacity() {
        var endpoint = endpoint(2, false);
        try {
            report(endpoint, 0, 2);
            assertTrue(endpoint.selectionSnapshot().admissionBlocked());
            report(endpoint, 0, 1);
            assertFalse(endpoint.selectionSnapshot().admissionBlocked());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @Test
    void identicalReportsRenewActivityWithoutCapacityNotifications() {
        for (boolean queue : new boolean[] {false, true}) {
            var endpoint = endpoint(128, queue);
            var changes = new java.util.concurrent.atomic.AtomicInteger();
            var activity = new java.util.concurrent.atomic.AtomicInteger();
            var notifier =
                    (CapacityNotifier)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    endpoint, "capacityNotifier");
            notifier.addListener(ignored -> changes.incrementAndGet());
            try {
                report(endpoint, 0, 0);
                assertEquals(0, changes.get());
                var registration =
                        endpoint.tryRegisterSelection(
                                        "active", endpoint.selectionSnapshot(), 100, 100)
                                .value();
                assertEquals(
                        WorkerEndpoint.CommitStatus.COMMITTED,
                        endpoint.confirmSelection(
                                registration, event -> activity.incrementAndGet(), 1));
                var task = new org.flexlb.worker.status.TaskInfo();
                task.setRequestId("active");
                task.setInputLength(100);
                task.setPhase(org.flexlb.worker.status.TaskPhase.RUNNING);
                var response = new WorkerStatusResponse();
                response.setAlive(true);
                response.setRunningQueryLen(1);
                response.setRunningTaskInfo(java.util.Map.of("active", task));
                EndpointTestSupport.applyStatus(endpoint, response).run();
                changes.set(0);
                activity.set(0);
                var snapshot = endpoint.selectionSnapshot();
                for (int i = 0; i < 5; i++)
                    EndpointTestSupport.applyStatus(endpoint, response).run();
                assertEquals(0, changes.get());
                assertEquals(5, activity.get());
                org.junit.jupiter.api.Assertions.assertSame(snapshot, endpoint.selectionSnapshot());
            } finally {
                EndpointTestSupport.closeAndWait(endpoint);
            }
        }
    }

    @Test
    void additionalQueueHeadroomNotifiesWhileAdmissionRemainsOpen() {
        var endpoint = endpoint(128, true);
        var changes = new java.util.concurrent.atomic.AtomicInteger();
        var notifier =
                (CapacityNotifier)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                endpoint, "capacityNotifier");
        try {
            report(endpoint, 10, 1);
            notifier.addListener(ignored -> changes.incrementAndGet());
            var before = endpoint.requestCapacity();
            report(endpoint, 9, 2);
            assertFalse(endpoint.selectionSnapshot().admissionBlocked());
            assertEquals(1L, before.availableRequests());
            assertEquals(2L, endpoint.requestCapacity().availableRequests());
            assertTrue(endpoint.requestCapacity().revision() > before.revision());
            assertEquals(1, changes.get());
            report(endpoint, 9, 2);
            assertEquals(1, changes.get());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @Test
    void waitingWatermarkChangesNotifyEvenWhenTotalRequestsStayTheSame() {
        var endpoint = endpoint(128, true);
        var changes = new java.util.concurrent.atomic.AtomicInteger();
        var notifier =
                (CapacityNotifier)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                endpoint, "capacityNotifier");
        try {
            report(endpoint, 10, 1);
            notifier.addListener(ignored -> changes.incrementAndGet());
            assertFalse(endpoint.selectionSnapshot().admissionBlocked());
            report(endpoint, 11, 0);
            assertTrue(endpoint.selectionSnapshot().admissionBlocked());
            assertEquals(1, changes.get());
            report(endpoint, 11, 0);
            assertEquals(1, changes.get());
            report(endpoint, 10, 1);
            assertFalse(endpoint.selectionSnapshot().admissionBlocked());
            assertEquals(2, changes.get());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @Test
    void completionDurationIncludesPrecedingWorkAndIgnoresLaterArrivals() {
        var endpoint = endpoint(128, false);
        try {
            var first =
                    endpoint.tryRegisterSelection("first", endpoint.selectionSnapshot(), 100, 100)
                            .value();
            var second =
                    endpoint.tryRegisterSelection("second", endpoint.selectionSnapshot(), 200, 200)
                            .value();
            endpoint.tryRegisterSelection("later", endpoint.selectionSnapshot(), 500, 500);
            assertEquals(300L, endpoint.estimatedCompletionDurationMs(second, 1000).orElseThrow());
            endpoint.rollbackSelection(first);
            assertEquals(200L, endpoint.estimatedCompletionDurationMs(second, 1000).orElseThrow());
            report(endpoint, 0, 1);
            assertTrue(endpoint.estimatedCompletionDurationMs(second, 1000).isEmpty());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @Test
    void engineWorkCorrectsCacheOverestimatesWithoutDoubleCharging() {
        var endpoint = endpoint(128, false);
        try {
            var registration =
                    endpoint.tryRegisterSelection("known", endpoint.selectionSnapshot(), 20, -1L)
                            .value();
            endpoint.confirmSelection(registration, ignored -> {}, 1);
            var task = new org.flexlb.worker.status.TaskInfo();
            task.setRequestId("known");
            task.setInputLength(100);
            task.setPrefixLengthValid(true);
            task.setPrefixLength(40);
            task.setPhase(org.flexlb.worker.status.TaskPhase.RUNNING);
            var report = new WorkerStatusResponse();
            report.setAlive(true);
            report.setRunningQueryLen(1);
            report.setRunningTaskInfo(java.util.Map.of("known", task));
            EndpointTestSupport.applyStatus(endpoint, report).run();
            assertEquals(60, endpoint.selectionSnapshot().computeTokens());
            var snapshot = endpoint.selectionSnapshot();
            EndpointTestSupport.applyStatus(endpoint, report).run();
            org.junit.jupiter.api.Assertions.assertSame(snapshot, endpoint.selectionSnapshot());
            task.setPrefixLength(90);
            EndpointTestSupport.applyStatus(endpoint, report).run();
            assertEquals(
                    20,
                    endpoint.selectionSnapshot().computeTokens(),
                    "local charge remains a lower bound");
            task.setPrefixLengthValid(false);
            EndpointTestSupport.applyStatus(endpoint, report).run();
            assertEquals(
                    100,
                    endpoint.selectionSnapshot().computeTokens(),
                    "invalid prefix cannot reduce work");
            task.setInputLength(0);
            EndpointTestSupport.applyStatus(endpoint, report).run();
            assertEquals(
                    20,
                    endpoint.selectionSnapshot().computeTokens(),
                    "missing input retains local estimate");
            task.setInputLength(100);
            EndpointTestSupport.applyStatus(endpoint, report).run();
            assertTrue(endpoint.releaseRequest(registration));
            assertEquals(0, endpoint.selectionSnapshot().computeTokens());
            assertFalse(endpoint.releaseRequest(registration));
            EndpointTestSupport.applyStatus(endpoint, report).run();
            assertEquals(
                    100,
                    endpoint.selectionSnapshot().computeTokens(),
                    "Engine-only work still counts");
            report(endpoint, 0, 0);
            assertEquals(0, endpoint.selectionSnapshot().computeTokens());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    private static PrefillEndpoint endpoint(int limit, boolean queue) {
        var config = new FlexlbConfig();
        var admission = config.getRouter().getRoles().getPrefill().getAdmission();
        admission.setMaxInflightRequests(limit);
        if (queue)
            config.getRouter()
                    .getRoles()
                    .getPrefill()
                    .getSelection()
                    .setType(RoutingConfig.PrefillSelectionType.TOKEN_BALANCE);
        var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8081);
        var endpoint = new PrefillEndpoint(status, config, new CapacityNotifier());
        report(endpoint, 0, 0);
        return endpoint;
    }

    private static void report(PrefillEndpoint endpoint, int waiting, int running) {
        var report = new WorkerStatusResponse();
        report.setAlive(true);
        report.setWaitingQueryLen(waiting);
        report.setRunningQueryLen(running);
        EndpointTestSupport.applyStatus(endpoint, report).run();
    }
}
