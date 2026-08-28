package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contract tests for the redesigned {@link WorkerStatus} status-commit model.
 *
 * <p>The previous transaction API ({@code applyResponseFields} /
 * {@code commitAppliedStatus} / {@code appliedStatusSnapshot}) and the
 * in-status resource-availability hysteresis were both removed. Status is now
 * published through the {@code freeze -> prepare -> publish} transaction under
 * the per-generation lock, with a CAS on the single committed holder. The
 * resource-availability/hysteresis decision moved out of {@code WorkerStatus}
 * to the resource-measure / routing layer, so its coverage is intentionally not
 * re-created here; it belongs with its new owner.
 */
@DisplayName("WorkerStatus status-commit contract")
class WorkerStatusTest {

    private static WorkerStatus discovered() {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group-a", "10.0.0.1", 8080, 9090, "site-a");
    }

    /**
     * Run one whole status transaction exactly as the production reducers do:
     * freeze the RPC response, prepare a strictly-newer committed holder under
     * the generation lock, then publish it.
     */
    private static void publish(WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            WorkerStatus.StatusObservation observation =
                    status.freezeStatusResponse(response);
            WorkerStatus.PreparedStatus prepared =
                    status.prepareNewStatus(observation);
            status.publishPreparedStatus(prepared);
        } finally {
            status.lock.unlock();
        }
    }

    /**
     * Build a response whose every Engine field is a deterministic function of
     * {@code marker}, so a torn read is detectable by any field disagreeing.
     * {@code statusVersion} equals the marker and must strictly increase across
     * publishes.
     */
    private static WorkerStatusResponse responseWithMarker(long marker) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAvailableConcurrency(marker);
        response.setAlive(true);
        response.setAvailableKvCacheTokens(100L + marker);
        response.setTotalKvCacheTokens(1_000L + marker);
        response.setStepLatencyMs(marker + 0.25);
        response.setIterateCount(marker);
        response.setDpSize(10L + marker);
        response.setTpSize(20L + marker);
        response.setDpRank(30L + marker);
        response.setMaxSeqLen(40L + marker);
        response.setMaxBatchTokensSize(50L + marker);
        response.setStatusVersion(marker);
        response.setLatestFinishedVersion(marker);
        TaskInfo task = new TaskInfo();
        task.setRequestId(marker);
        response.setRunningTaskInfo(Map.of(Long.toString(marker), task));
        return response;
    }

    private static void assertCoherent(WorkerStatus.CommittedWorkerStatus committed) {
        WorkerStatus.EngineObservation fields = committed.fields();
        long marker = fields.iterateCount();
        assertTrue(marker >= 1L, "marker must be a published version");
        // Fields and cursor are one immutable record swapped atomically; every
        // field must belong to the same marker as the committed cursor.
        assertEquals(marker, committed.cursor().statusVersion());
        assertEquals(marker, fields.availableConcurrency().longValue());
        assertEquals(100L + marker, fields.availableKvCacheTokens());
        assertEquals(1_000L + marker, fields.totalKvCacheTokens());
        assertEquals(marker + 0.25, fields.stepLatencyMs());
        assertEquals(10L + marker, fields.dpSize());
        assertEquals(20L + marker, fields.tpSize());
        assertEquals(30L + marker, fields.dpRank());
        assertEquals(40L + marker, fields.maxSeqLen());
        assertEquals(50L + marker, fields.maxBatchTokensSize());
        assertEquals(marker,
                fields.runningTaskList().get(Long.toString(marker)).requestId());
    }

    @Nested
    @DisplayName("Atomic publication")
    class AtomicPublication {

        @Test
        @DisplayName("Readers never observe a torn field/cursor pair")
        void readersObserveOneWholePublishedSnapshot() throws Exception {
            WorkerStatus status = discovered();
            publish(status, responseWithMarker(1L));

            int readers = 4;
            int observationsPerReader = 20_000;
            AtomicLong nextVersion = new AtomicLong(2L);
            CountDownLatch ready = new CountDownLatch(readers + 1);
            CountDownLatch start = new CountDownLatch(1);
            ExecutorService executor = Executors.newFixedThreadPool(readers + 1);
            List<Future<?>> futures = new ArrayList<>();
            try {
                futures.add(executor.submit(() -> {
                    ready.countDown();
                    assertTrue(start.await(5, TimeUnit.SECONDS));
                    for (int i = 0; i < observationsPerReader; i++) {
                        publish(status, responseWithMarker(
                                nextVersion.getAndIncrement()));
                    }
                    return null;
                }));
                for (int reader = 0; reader < readers; reader++) {
                    futures.add(executor.submit(() -> {
                        ready.countDown();
                        assertTrue(start.await(5, TimeUnit.SECONDS));
                        for (int i = 0; i < observationsPerReader; i++) {
                            assertCoherent(status.committedWorkerStatus());
                        }
                        return null;
                    }));
                }

                assertTrue(ready.await(5, TimeUnit.SECONDS));
                start.countDown();
                for (Future<?> future : futures) {
                    future.get(10, TimeUnit.SECONDS);
                }
            } finally {
                start.countDown();
                executor.shutdownNow();
                assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
            }
        }

        @Test
        @DisplayName("Published running tasks are a frozen immutable copy")
        void publishedRunningTasksDoNotRetainMutableMapStructure() {
            WorkerStatus status = discovered();
            Map<String, TaskInfo> tasks = new HashMap<>();
            TaskInfo task = new TaskInfo();
            task.setRequestId(7L);
            tasks.put("7", task);
            WorkerStatusResponse response = responseWithMarker(7L);
            response.setRunningTaskInfo(tasks);

            publish(status, response);
            tasks.clear();

            Map<String, WorkerStatus.TaskObservation> published =
                    status.committedEngineObservation().runningTaskList();
            assertEquals(1, published.size());
            assertEquals(7L, published.get("7").requestId());
            assertThrows(UnsupportedOperationException.class,
                    () -> published.remove("7"));
        }
    }

    @Nested
    @DisplayName("Prepare/publish transaction invariants")
    class TransactionInvariants {

        @Test
        @DisplayName("prepareNewStatus rejects a non-advancing status version")
        void prepareRejectsNonStrictlyNewerVersion() {
            WorkerStatus status = discovered();
            publish(status, responseWithMarker(5L));

            WorkerStatusResponse stale = responseWithMarker(5L); // equal version
            status.lock.lock();
            try {
                WorkerStatus.StatusObservation observation =
                        status.freezeStatusResponse(stale);
                assertThrows(IllegalArgumentException.class,
                        () -> status.prepareNewStatus(observation));
            } finally {
                status.lock.unlock();
            }
        }

        @Test
        @DisplayName("prepareNewStatus requires the generation lock")
        void prepareRequiresGenerationLock() {
            WorkerStatus status = discovered();
            WorkerStatus.StatusObservation observation =
                    status.freezeStatusResponse(responseWithMarker(1L));
            assertThrows(IllegalStateException.class,
                    () -> status.prepareNewStatus(observation));
        }

        @Test
        @DisplayName("publishPreparedStatus requires the generation lock")
        void publishRequiresGenerationLock() {
            WorkerStatus status = discovered();
            WorkerStatus.PreparedStatus prepared;
            status.lock.lock();
            try {
                prepared = status.prepareNewStatus(
                        status.freezeStatusResponse(responseWithMarker(1L)));
            } finally {
                status.lock.unlock();
            }
            assertThrows(IllegalStateException.class,
                    () -> status.publishPreparedStatus(prepared));
        }

        @Test
        @DisplayName("A stale prepared status loses the publish CAS")
        void stalePreparedStatusIsRejectedByCas() {
            WorkerStatus status = discovered();
            publish(status, responseWithMarker(1L));

            // Two transactions prepared from the same committed base, mirroring
            // two overlapping status rounds. Only the first publish may win.
            WorkerStatus.PreparedStatus first;
            WorkerStatus.PreparedStatus second;
            status.lock.lock();
            try {
                first = status.prepareNewStatus(
                        status.freezeStatusResponse(responseWithMarker(2L)));
                second = status.prepareNewStatus(
                        status.freezeStatusResponse(responseWithMarker(3L)));
            } finally {
                status.lock.unlock();
            }

            status.lock.lock();
            try {
                status.publishPreparedStatus(first);
                assertThrows(IllegalStateException.class,
                        () -> status.publishPreparedStatus(second));
            } finally {
                status.lock.unlock();
            }
            assertEquals(2L, status.appliedStatusCursor().statusVersion());
        }

        @Test
        @DisplayName("A retiring generation refuses further publication")
        void retiringGenerationRefusesPublication() {
            WorkerStatus status = discovered();
            publish(status, responseWithMarker(1L));

            status.lock.lock();
            try {
                assertTrue(status
                        .beginRetirementAfterEndpointGateClosed());
                WorkerStatus.StatusObservation observation =
                        status.freezeStatusResponse(responseWithMarker(2L));
                assertThrows(IllegalStateException.class,
                        () -> status.prepareNewStatus(observation));
            } finally {
                status.lock.unlock();
            }
            assertFalse(status.isActiveGeneration());
        }
    }

    @Nested
    @DisplayName("Cursor merge")
    class CursorMerge {

        @Test
        @DisplayName("Finished-task version never regresses across publishes")
        void finishedVersionIsMonotonic() {
            WorkerStatus status = discovered();
            long[] finishedVersions = {5L, 3L, 10L, 2L};
            long[] expectedHighWater = {5L, 5L, 10L, 10L};
            for (int round = 0; round < finishedVersions.length; round++) {
                WorkerStatusResponse response =
                        responseWithMarker(round + 1L); // strictly-newer status version
                response.setLatestFinishedVersion(finishedVersions[round]);
                publish(status, response);
                assertEquals(expectedHighWater[round],
                        status.appliedStatusCursor().latestFinishedTaskVersion());
            }
        }

        @Test
        @DisplayName("Serialized publishers advance the cursor exactly once each")
        void serializedPublishersAdvanceCursorMonotonically() throws Exception {
            WorkerStatus status = discovered();
            int publishers = 32;
            AtomicInteger version = new AtomicInteger(1);
            CountDownLatch ready = new CountDownLatch(publishers);
            CountDownLatch start = new CountDownLatch(1);
            ExecutorService executor = Executors.newFixedThreadPool(publishers);
            List<Future<?>> futures = new ArrayList<>(publishers);
            try {
                for (int i = 0; i < publishers; i++) {
                    futures.add(executor.submit(() -> {
                        ready.countDown();
                        assertTrue(start.await(5, TimeUnit.SECONDS));
                        status.lock.lock();
                        try {
                            long next = version.getAndIncrement();
                            WorkerStatus.StatusObservation observation =
                                    status.freezeStatusResponse(
                                            responseWithMarker(next));
                            status.publishPreparedStatus(
                                    status.prepareNewStatus(observation));
                        } finally {
                            status.lock.unlock();
                        }
                        return null;
                    }));
                }
                assertTrue(ready.await(5, TimeUnit.SECONDS));
                start.countDown();
                for (Future<?> future : futures) {
                    future.get(5, TimeUnit.SECONDS);
                }
                assertEquals(publishers,
                        status.appliedStatusCursor().statusVersion());
            } finally {
                start.countDown();
                executor.shutdownNow();
                assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
            }
        }
    }

    @Nested
    @DisplayName("Generation identity and topology")
    class GenerationIdentity {

        @Test
        @DisplayName("Each discovered generation has a distinct identity")
        void generationsHaveDistinctIdentity() {
            WorkerStatus first = discovered();
            WorkerStatus second = discovered();
            assertNotEquals(first.getGenerationId(), second.getGenerationId());
        }

        @Test
        @DisplayName("Discovery labels are refreshed under the generation lock")
        void discoveryLabelsHaveExplicitPublication() {
            WorkerStatus status = discovered();
            status.lock.lock();
            try {
                status.updateDiscoveryLabels("site-b", "group-b");
            } finally {
                status.lock.unlock();
            }
            WorkerStatus.TopologySnapshot topology = status.topologySnapshot();
            assertEquals("site-b", topology.site());
            assertEquals("group-b", topology.group());
            // Address identity is discovery-owned and must survive a label update.
            assertEquals("10.0.0.1", topology.ip());
            assertEquals(8080, topology.port());
        }

        @Test
        @DisplayName("updateDiscoveryLabels requires the generation lock")
        void discoveryLabelUpdateRequiresLock() {
            WorkerStatus status = discovered();
            assertThrows(IllegalStateException.class,
                    () -> status.updateDiscoveryLabels("site-x", "group-x"));
        }
    }

    @Nested
    @DisplayName("Poll health")
    class PollHealthContract {

        @Test
        @DisplayName("A successful poll publishes reported liveness")
        void successfulPollRecordsLiveness() {
            WorkerStatus status = discovered();
            status.lock.lock();
            try {
                WorkerStatus.PollHealth health = status.recordSuccessfulPoll(true);
                assertTrue(health.reportedAlive());
                assertEquals(0L, health.consecutiveTransportFailures());
            } finally {
                status.lock.unlock();
            }
        }

        @Test
        @DisplayName("Transport failures accumulate without rewriting liveness")
        void transportFailuresAccumulate() {
            WorkerStatus status = discovered();
            status.lock.lock();
            try {
                status.recordSuccessfulPoll(true);
                WorkerStatus.PollHealth afterFirst = status.recordTransportFailure();
                WorkerStatus.PollHealth afterSecond = status.recordTransportFailure();
                assertEquals(1L, afterFirst.consecutiveTransportFailures());
                assertEquals(2L, afterSecond.consecutiveTransportFailures());
                assertTrue(afterSecond.reportedAlive());
            } finally {
                status.lock.unlock();
            }
            assertSame(status.pollHealth(), status.pollHealth());
        }
    }
}
