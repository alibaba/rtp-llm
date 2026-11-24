package org.flexlb.balance.scheduler;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeMode;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.testfixture.TaskInfo;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.flexlb.testfixture.WorkerStatusTestFixtures;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.slf4j.LoggerFactory;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/** External contract for temporary P/D capacity pressure. */
class TransientCapacityQueueContractTest {

    @ParameterizedTest
    @CsvSource({"false,false,50,false", "false,true,50,false", "true,false,50,false",
            "true,true,50,false", "false,false,80,false", "true,true,80,false",
            "false,false,40,false", "true,true,40,false", "false,false,50,true", "true,true,50,true"})
    @Timeout(15)
    void newDecodeOnlyRunsTheRankedHeadWhenItFits(
            boolean priorityOrdering, boolean single, int priority, boolean decodeRecovers) throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(priorityOrdering
                ? QueueOrderingConfig.priority() : new QueueOrderingConfig());
        if (single) {
            config.queueScheduler().setDecision(org.flexlb.config.DecisionPolicyConfig.single());
        }
        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            var a = fixture.runtime.scheduler().submit(fixture.context(996_001L, 50, 512L));
            awaitCondition(() -> fixture.prefillEndpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock() != null, 2_000L);
            var d2 = publishDecode(fixture, "127.0.0.2", 256L, decodeRecovers);
            var b = fixture.runtime.scheduler().submit(fixture.context(996_002L, priority));
            if (decodeRecovers) {
                var recovered = statusResponse(RoleType.DECODE, 2L, false);
                recovered.setAvailableKvCacheTokens(256L);
                fixture.runtime.applyStatus(d2.getStatus(), recovered);
            }
            if (priorityOrdering && priority > 50) {
                Response response = b.get(2, TimeUnit.SECONDS);
                assertTrue(response.isSuccess());
                assertEquals("127.0.0.2:18081", decodeAddress(response));
                assertFalse(a.isDone());
                fixture.releaseCapacity();
                assertTrue(a.get(2, TimeUnit.SECONDS).isSuccess());
                assertEquals(List.of(996_002L, 996_001L), fixture.submission.requestIds());
            } else {
                assertFalse(fixture.submission.awaitCommands(1, 100, TimeUnit.MILLISECONDS),
                        "the spare Decode fits B but may not reorder the Prefill queue");
                assertEquals(0, d2.routingView().engineCapacityUsed());
                if (priority < 50) {
                    fixture.runtime.scheduler().cancelRequest(996_001L, 0L, CancelReason.CLIENT_CANCELLED);
                    assertFalse(a.get(2, TimeUnit.SECONDS).isSuccess());
                } else {
                    fixture.releaseCapacity();
                    assertTrue(a.get(2, TimeUnit.SECONDS).isSuccess());
                }
                Response response = b.get(2, TimeUnit.SECONDS);
                assertTrue(response.isSuccess());
                assertEquals("127.0.0.2:18081", decodeAddress(response));
                assertEquals(priority < 50 ? List.of(996_002L) : List.of(996_001L, 996_002L),
                        fixture.submission.requestIds());
            }
            assertEquals(0, fixture.prefillEndpoint.queuedRequestCount());
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    @Timeout(15)
    void fifoRetainsMiddleRequestWhenOnlySuccessorFitsTheNewDecode(
            boolean middleDecodeFull, boolean single) throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        if (single) {
            config.queueScheduler().setDecision(org.flexlb.config.DecisionPolicyConfig.single());
        }
        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            var a = fixture.runtime.scheduler().submit(fixture.context(999_001L, 50, 512L));
            var b = fixture.runtime.scheduler().submit(fixture.context(999_002L, 40, 512L));
            var c = fixture.runtime.scheduler().submit(fixture.context(999_003L, 80, 128L));
            awaitCondition(() -> fixture.runtime.scheduler().getQueuedRequestCount() == 3
                    && fixture.prefillEndpoint.captureRouteProjectionInputs().queue().admissionBlock() != null, 2_000L);
            var d2 = publishDecode(fixture, "127.0.0.2", 256L, middleDecodeFull);
            assertFalse(fixture.submission.awaitCommands(1, 100, TimeUnit.MILLISECONDS));
            fixture.runtime.scheduler().cancelRequest(999_001L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(a.get(2, TimeUnit.SECONDS).isSuccess());
            assertFalse(fixture.submission.awaitCommands(1, 100, TimeUnit.MILLISECONDS));
            fixture.runtime.scheduler().cancelRequest(999_002L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(b.get(2, TimeUnit.SECONDS).isSuccess());
            if (middleDecodeFull) {
                assertFalse(c.isDone());
                fixture.runtime.applyStatus(d2.getStatus(), statusResponse(RoleType.DECODE, 2L, false));
            }
            Response response = c.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals("127.0.0.2:18081", decodeAddress(response));
            assertEquals(List.of(999_003L), fixture.submission.requestIds());
            assertEquals(0, fixture.prefillEndpoint.queuedRequestCount());
        }
    }

    @ParameterizedTest
    @CsvSource({"false,recover", "true,recover", "false,cancel_waiter", "true,cancel_waiter",
            "false,stop", "true,stop"})
    @Timeout(15)
    void waitingForPrefillKvRemainsIndependentOfDecodeUpdates(boolean single, String recovery) throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        if (single) {
            config.queueScheduler().setDecision(org.flexlb.config.DecisionPolicyConfig.single());
        }
        try (Fixture fixture = new Fixture(null, config)) {
            fixture.submission.blockPreparation();
            var a = fixture.runtime.scheduler().submit(fixture.context(993_101L, 50, 50L));
            assertTrue(fixture.submission.awaitPreparation(2, TimeUnit.SECONDS));
            fixture.runtime.applyStatus(fixture.decodeStatus, statusResponse(RoleType.DECODE, 2L, true));
            var d2 = (DecodeEndpoint) publishEndpoint(fixture.runtime.endpointRegistry(), RoleType.DECODE,
                    "127.0.0.2:18081", initializedStatus(RoleType.DECODE, "127.0.0.2", 18081));
            var b = fixture.runtime.scheduler().submit(fixture.context(993_102L, 50, 500L));
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 2, 2_000L);
            var lowKv = statusResponse(RoleType.PREFILL, 2L, false);
            lowKv.setAvailableKvCacheTokens(100L);
            fixture.runtime.applyStatus(fixture.prefillStatus, lowKv);
            fixture.submission.unblockPreparation();
            awaitCondition(() -> "Prefill KV capacity exhausted".equals(
                    fixture.prefillEndpoint.queueWaitDiagnostics().get("cause")), 2_000L);
            var before = fixture.prefillEndpoint.captureRouteProjectionInputs();
            assertEquals(null, before.queue().admissionBlock(), "a KV wait is not a delivery admission block");
            for (int version = 3; version <= 12; version++) {
                fixture.runtime.applyStatus(fixture.decodeStatus,
                        statusResponse(RoleType.DECODE, version, true));
            }
            org.junit.jupiter.api.Assertions.assertSame(before,
                    fixture.prefillEndpoint.captureRouteProjectionInputs(),
                    "unchanged Decode heartbeats must not invalidate Prefill work prediction");
            Response first = a.get(2, TimeUnit.SECONDS);
            assertTrue(first.isSuccess());
            assertEquals("127.0.0.2:18081", decodeAddress(first));
            assertFalse(b.isDone());
            assertEquals(List.of(993_101L), fixture.submission.requestIds());

            if (recovery.equals("stop")) {
                fixture.runtime.close();
                assertFalse(b.get(2, TimeUnit.SECONDS).isSuccess());
                assertEquals(0, fixture.prefillEndpoint.queuedRequestCount());
                assertEquals(0, fixture.decodeEndpoint.resourceSnapshot().reserved().size());
                assertNull(d2.reservationHandle(993_102L));
                return;
            }
            if (recovery.equals("cancel_waiter")) {
                fixture.runtime.scheduler().cancelRequest(993_102L, 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(b.get(2, TimeUnit.SECONDS).isSuccess());
                assertNull(d2.reservationHandle(993_102L));
            }
            fixture.runtime.applyStatus(fixture.decodeStatus,
                    statusResponse(RoleType.DECODE, 13L, false));
            assertEquals("127.0.0.2:18081", decodeAddress(first));
            assertEquals(List.of(993_101L), fixture.submission.requestIds());
            if (recovery.equals("recover")) {
                assertFalse(b.isDone());
                fixture.runtime.applyStatus(fixture.prefillStatus,
                        statusResponse(RoleType.PREFILL, 3L, false));
                assertTrue(b.get(2, TimeUnit.SECONDS).isSuccess());
                assertEquals(List.of(993_101L, 993_102L), fixture.submission.requestIds());
            }
            assertEquals(0, fixture.prefillEndpoint.queuedRequestCount());
        }
    }

    @Test
    @Timeout(15)
    void decodeHeartbeatsDoNotInvalidateSharedPrefillCapacityWait() throws Exception {
        FlexlbConfig config = config();
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(64L);
        try (Fixture fixture = new Fixture(null, config)) {
            var a = fixture.runtime.scheduler().submit(fixture.context(999_101L, 50));
            assertTrue(a.get(2, TimeUnit.SECONDS).isSuccess());
            var b = fixture.runtime.scheduler().submit(fixture.context(999_102L, 50));
            awaitCondition(() -> fixture.prefillEndpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock() != null, 2_000L);
            var before = fixture.prefillEndpoint.captureRouteProjectionInputs();
            var semantics = before.queue().admissionBlock().semantics();
            assertTrue(semantics == null || semantics.afterProbe()
                    == org.flexlb.balance.projection.RouteProjection.AfterProbeAdmission.BLOCKED);
            for (int version = 2; version <= 11; version++) {
                fixture.runtime.applyStatus(fixture.decodeStatus,
                        statusResponse(RoleType.DECODE, version, false));
            }
            org.junit.jupiter.api.Assertions.assertSame(before,
                    fixture.prefillEndpoint.captureRouteProjectionInputs(),
                    "Decode heartbeats cannot invalidate a shared Prefill capacity wait");
            assertFalse(b.isDone());
            fixture.runtime.scheduler().cancelRequest(999_102L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(b.get(2, TimeUnit.SECONDS).isSuccess());
        }
    }

    @Test
    @Timeout(15)
    void simultaneousDecodeRecoveryPreservesPrefillHeadOrder() throws Exception {
        try (Fixture fixture = new Fixture(null, config())) {
            fixture.submission.blockPreparation();
            var a = fixture.runtime.scheduler().submit(fixture.context(998_001L, 50));
            assertTrue(fixture.submission.awaitPreparation(2, TimeUnit.SECONDS));
            fixture.runtime.applyStatus(fixture.decodeStatus, statusResponse(RoleType.DECODE, 2L, true));
            fixture.submission.unblockPreparation();
            awaitCondition(() -> fixture.prefillEndpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock() != null, 2_000L);
            fixture.submission.blockPreparation();
            var d2 = publishDecode(fixture, "127.0.0.2", 1_000L, false);
            var b = fixture.runtime.scheduler().submit(fixture.context(998_002L, 50));
            assertTrue(fixture.submission.awaitPreparation(2, TimeUnit.SECONDS));
            fixture.runtime.applyStatus(fixture.decodeStatus, statusResponse(RoleType.DECODE, 3L, false));
            fixture.submission.unblockPreparation();
            assertTrue(a.get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(b.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(List.of(998_001L, 998_002L), fixture.submission.requestIds());
            assertEquals(1, d2.routingView().totalLoad());
            assertEquals(0, fixture.prefillEndpoint.queuedRequestCount());
            assertEquals(0, d2.resourceSnapshot().activeDispatchPermits());
        }
    }

    @ParameterizedTest
    @CsvSource({"50,false", "80,false", "80,true"})
    @Timeout(15)
    void fifoCancellationAndDecodeWakePreserveAllLivePositions(int middlePriority, boolean cancelMiddle) throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            var noPrefillKv = statusResponse(RoleType.PREFILL, 2L, false);
            noPrefillKv.setAvailableKvCacheTokens(0L);
            fixture.runtime.applyStatus(fixture.prefillStatus, noPrefillKv);
            var a = fixture.runtime.scheduler().submit(fixture.context(997_001L, 50));
            var b = fixture.runtime.scheduler().submit(fixture.context(997_002L, middlePriority));
            var c = fixture.runtime.scheduler().submit(fixture.context(997_003L, 60));
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 3, 2_000L);
            if (cancelMiddle) {
                fixture.runtime.scheduler().cancelRequest(997_002L, 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(b.get(2, TimeUnit.SECONDS).isSuccess());
            }
            fixture.runtime.applyStatus(fixture.prefillStatus, statusResponse(RoleType.PREFILL, 3L, false));
            awaitCondition(() -> fixture.prefillEndpoint.captureRouteProjectionInputs().queue().admissionBlock() != null,
                    2_000L);
            publishDecode(fixture, "127.0.0.2", 1_000L, false);
            assertTrue(a.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(List.of(997_001L), fixture.submission.requestIds());
            fixture.releaseCapacity();
            if (!cancelMiddle) {
                assertTrue(b.get(2, TimeUnit.SECONDS).isSuccess());
                assertFalse(c.isDone());
                // A and B occupy both Prefill batch slots. A third Decode cannot start C yet.
                publishDecode(fixture, "127.0.0.3", 1_000L, false);
                assertFalse(fixture.submission.awaitCommands(3, 100, TimeUnit.MILLISECONDS));
                var finished = prefillStatusWithFinishedAndUnknownRunning(4L, List.of(997_001L), 0);
                finished.getFinishedTaskInfo().get("997001").setBatchId(fixture.submission.batchIds.get(997_001L));
                fixture.runtime.applyStatus(fixture.prefillStatus, finished);
            }
            assertTrue(c.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(cancelMiddle ? List.of(997_001L, 997_003L) : List.of(997_001L, 997_002L, 997_003L),
                    fixture.submission.requestIds());
        }
    }

    @Test
    @Timeout(20)
    void bothPrioritiesWaitForCollectionWithoutTakingDecodeCapacity() throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(QueueOrderingConfig.priority());
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_RESERVED));
        config.queueScheduler().getOrdering().setPreemption(preemption);
        config.getDispatcher().setType(DispatcherConfig.Type.BATCH);
        config.fixedWindowDecision().setMaxRequests(64);
        config.fixedWindowDecision().setMaxCollectionWaitMs(60_000L);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        try (Fixture fixture = new Fixture(null, config)) {
            var low = fixture.runtime.scheduler().submit(fixture.context(990_001L, 20));
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 1, 2_000L);
            assertNull(fixture.decodeEndpoint.reservationHandle(990_001L));
            var high = fixture.runtime.scheduler().submit(fixture.context(990_002L, 80));
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 2, 2_000L);
            assertNull(fixture.decodeEndpoint.reservationHandle(990_002L));
            assertNull(fixture.decodeEndpoint.reservationHandle(990_001L));
            assertFalse(low.isDone());
            assertFalse(high.isDone());
            assertEquals(0, fixture.decodeEndpoint.routingView().totalLoad());
            assertEquals(0, fixture.decodeEndpoint.routingView().engineCapacityUsed());
            fixture.runtime.scheduler().cancelRequest(990_002L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(high.get(2, TimeUnit.SECONDS).isSuccess());
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 1, 2_000L);
            assertFalse(low.isDone());
            assertNull(fixture.decodeEndpoint.reservationHandle(990_001L));
            assertEquals(List.of(), fixture.submission.requestIds(), "neither queued route was sent to the engine");
            fixture.runtime.scheduler().cancelRequest(990_001L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(low.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(0, fixture.decodeEndpoint.routingView().totalLoad());
            assertEquals(0, fixture.prefillEndpoint.queuedRequestCount());
        }
    }

    @ParameterizedTest
    @CsvSource({"SINGLE,BATCH,false", "FIXED_WINDOW,BATCH,false",
            "SINGLE,NON_BATCH,false", "FIXED_WINDOW,NON_BATCH,false",
            "SINGLE,BATCH,true", "FIXED_WINDOW,BATCH,true"})
    @Timeout(15)
    void cacheLeaderWaitsForKvAtDeliveryWithoutChoosingColderPrefill(
            String decision, DispatcherConfig.Type delivery, boolean cancel) throws Exception {
        FlexlbConfig config = config();
        if (decision.equals("SINGLE")) {
            config.queueScheduler().setDecision(org.flexlb.config.DecisionPolicyConfig.single());
        }
        config.getDispatcher().setType(delivery);
        RoutingConfig.CacheAffinityConfig affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(1_000_000L);
        affinity.setMinPrefixHitPercent(0.0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        try (Fixture fixture = new Fixture(null, config)) {
            var colder = (PrefillEndpoint) publishEndpoint(fixture.runtime.endpointRegistry(), RoleType.PREFILL,
                    "127.0.0.2:18080", initializedStatus(RoleType.PREFILL, "127.0.0.2", 18080));
            var exhausted = statusResponse(RoleType.PREFILL, 2L, false);
            exhausted.setAvailableKvCacheTokens(0L);
            fixture.runtime.applyStatus(fixture.prefillStatus, exhausted);
            when(fixture.cache.findMatchingEngines(any(), any(), any()))
                    .thenReturn(Map.of(Fixture.PREFILL_ADDRESS, 9, "127.0.0.2:18080", 1));
            var context = fixture.context(994_001L, 50, 1_000L);
            context.getRequest().setCacheKeyBlockSize(100L);
            context.getRequest().setBlockCacheKeys(List.of(1L, 2L, 3L));
            var response = fixture.runtime.scheduler().submit(context);

            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 1, 2_000L);
            awaitCondition(() -> "Prefill KV capacity exhausted".equals(
                    fixture.prefillEndpoint.queueWaitDiagnostics().get("cause")), 2_000L);
            long enqueueSequence = context.getWorkerEnqueueSequence();
            assertTrue(enqueueSequence > 0L);
            assertFalse(response.isDone(), "KV exhaustion must block delivery on the cache leader");
            assertEquals(0, colder.observedRequestCount());
            assertEquals(List.of(), fixture.submission.requestIds());

            if (cancel) {
                fixture.runtime.scheduler().cancelRequest(994_001L, 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(response.get(2, TimeUnit.SECONDS).isSuccess());
                assertEquals(0, fixture.prefillEndpoint.queuedRequestCount());
                assertEquals(0, fixture.decodeEndpoint.resourceSnapshot().reserved().size());
                assertEquals(List.of(), fixture.submission.requestIds());
            } else {
                fixture.runtime.applyStatus(fixture.prefillStatus,
                        statusResponse(RoleType.PREFILL, 3L, false));
                Response delivered = response.get(2, TimeUnit.SECONDS);
                assertTrue(delivered.isSuccess());
                assertEquals(Fixture.PREFILL_ADDRESS, prefillAddress(delivered));
                assertEquals(0, colder.observedRequestCount());
                assertEquals(enqueueSequence, context.getWorkerEnqueueSequence());
                assertEquals(delivery == DispatcherConfig.Type.BATCH ? List.of(994_001L) : List.of(),
                        fixture.submission.requestIds());
            }
        }
    }

    @ParameterizedTest
    @CsvSource({"false,release", "false,cancel_head", "false,cancel_waiter", "false,expire_waiter", "false,retire_prefill", "false,other_prefill", "false,cache_changed", "true,priority"})
    @Timeout(20)
    void cacheLeaderRemainsVisibleWhenItsHeadWaitsForAnotherDecode(
            boolean priorityOrdering, String recovery) throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(priorityOrdering
                ? QueueOrderingConfig.priority() : new QueueOrderingConfig());
        RoutingConfig.CacheAffinityConfig affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(1_000_000L);
        affinity.setMinPrefixHitPercent(0.0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        try (Fixture fixture = new Fixture(null, config)) {
            fixture.submission.blockPreparation();
            var a = fixture.runtime.scheduler().submit(fixture.context(995_001L, 20, 4_000L));
            assertTrue(fixture.submission.awaitPreparation(2, TimeUnit.SECONDS));
            fixture.runtime.applyStatus(fixture.decodeStatus, statusResponse(RoleType.DECODE, 2L, true));
            fixture.submission.unblockPreparation();
            awaitCondition(() -> fixture.prefillEndpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock() != null, 2_000L);

            var p2 = (PrefillEndpoint) publishEndpoint(fixture.runtime.endpointRegistry(), RoleType.PREFILL,
                    "127.0.0.2:18080", initializedStatus(RoleType.PREFILL, "127.0.0.2", 18080));
            var d2 = publishDecode(fixture, "127.0.0.2", 3_000L, false);
            when(fixture.cache.findMatchingEngines(any(), any(), any()))
                    .thenReturn(Map.of(Fixture.PREFILL_ADDRESS, 9, "127.0.0.2:18080", 1));
            var bContext = fixture.context(995_002L, priorityOrdering ? 80 : 10, 1_000L);
            bContext.getRequest().setCacheKeyBlockSize(100L);
            bContext.getRequest().setBlockCacheKeys(List.of(1L, 2L, 3L));
            if (recovery.equals("expire_waiter")) {
                bContext.setSchedulingMetadata(SchedulingMetadata.explicit(10, System.currentTimeMillis() + 1_000L));
            }
            var b = fixture.runtime.scheduler().submit(bContext);
            awaitCondition(() -> fixture.metrics.totalPlacementAttempts() >= 2, 2_000L);

            if (priorityOrdering) {
                Response response = b.get(2, TimeUnit.SECONDS);
                assertTrue(response.isSuccess());
                assertEquals(Fixture.PREFILL_ADDRESS, prefillAddress(response),
                        "B precedes A and must not inherit A's D1 blocker or choose a colder P");
                assertEquals("127.0.0.2:18081", decodeAddress(response));
                assertEquals(List.of(995_002L), fixture.submission.requestIds());
                assertFalse(a.isDone());
                return;
            }

            assertFalse(fixture.submission.awaitCommands(1, 100, TimeUnit.MILLISECONDS),
                    "B must wait for its cache leader, not be dispatched to idle P2");
            awaitCapacityWaiters(fixture.runtime.scheduler(), 1);
            assertFalse(b.isDone());
            assertEquals(1, fixture.prefillEndpoint.queuedRequestCount(), "only A is on P1");
            assertEquals(0, p2.observedRequestCount());
            assertEquals(0, d2.routingView().totalLoad(), "global waiting must not reserve D2");
            assertEquals(List.of(), fixture.submission.requestIds());

            if (recovery.equals("other_prefill")) {
                when(fixture.cache.findMatchingEngines(any(), any(), any())).thenAnswer(invocation -> {
                    List<Long> keys = invocation.getArgument(0);
                    return keys != null && !keys.isEmpty() && keys.getFirst() == 2L
                            ? Map.of("127.0.0.2:18080", 9) : Map.of(Fixture.PREFILL_ADDRESS, 9);
                });
                fixture.submission.blockPreparation();
                var cContext = fixture.context(995_003L, 20, 2_000L);
                cContext.getRequest().setCacheKeyBlockSize(100L);
                cContext.getRequest().setBlockCacheKeys(List.of(2L));
                var c = fixture.runtime.scheduler().submit(cContext);
                assertTrue(fixture.submission.awaitPreparation(2, TimeUnit.SECONDS));
                fixture.runtime.applyStatus(d2.getStatus(), statusResponse(RoleType.DECODE, 2L, true));
                fixture.submission.unblockPreparation();
                awaitCondition(() -> p2.captureRouteProjectionInputs().queue().admissionBlock() != null, 2_000L);
                publishDecode(fixture, "127.0.0.3", 1_500L, false);
                var eContext = fixture.context(995_004L, 10, 1_000L);
                eContext.getRequest().setCacheKeyBlockSize(100L);
                eContext.getRequest().setBlockCacheKeys(List.of(2L));
                var e = fixture.runtime.scheduler().submit(eContext);
                awaitCondition(() -> fixture.metrics.context(995_004L) != null
                        && fixture.runtime.scheduler().getQueuedRequestCount() == 4, 2_000L);
                assertFalse(e.isDone());
                fixture.runtime.scheduler().cancelRequest(995_003L, 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(c.get(2, TimeUnit.SECONDS).isSuccess());
                Response response = e.get(2, TimeUnit.SECONDS);
                assertTrue(response.isSuccess());
                assertEquals("127.0.0.2:18080", prefillAddress(response));
                assertFalse(b.isDone(), "P2 progress must not move B away from its cache leader P1");
                assertEquals(List.of(995_004L), fixture.submission.requestIds());
            } else if (recovery.equals("retire_prefill")) {
                EndpointRegistry.DetachedGeneration detached;
                fixture.prefillStatus.lock.lock();
                try {
                    detached = fixture.runtime.endpointRegistry().detachAndBeginRetirement(
                            RoleType.PREFILL, Fixture.PREFILL_ADDRESS, fixture.prefillStatus);
                } finally {
                    fixture.prefillStatus.lock.unlock();
                }
                detached.retireAndAwait();
                Response response = b.get(2, TimeUnit.SECONDS);
                assertTrue(response.isSuccess());
                assertEquals("127.0.0.2:18080", prefillAddress(response), "a retired P is no longer a candidate");
                assertFalse(a.get(2, TimeUnit.SECONDS).isSuccess());
            } else if (recovery.equals("cancel_waiter") || recovery.equals("expire_waiter")) {
                if (recovery.equals("cancel_waiter")) {
                    fixture.runtime.scheduler().cancelRequest(995_002L, 0L, CancelReason.CLIENT_CANCELLED);
                }
                assertFalse(b.get(3, TimeUnit.SECONDS).isSuccess());
                fixture.runtime.applyStatus(fixture.decodeStatus, statusResponse(RoleType.DECODE, 3L, false));
                assertTrue(a.get(2, TimeUnit.SECONDS).isSuccess());
                assertFalse(fixture.submission.awaitCommands(2, 100, TimeUnit.MILLISECONDS));
                assertEquals(List.of(995_001L), fixture.submission.requestIds());
                assertEquals(0, d2.routingView().totalLoad());
            } else {
                boolean cancelHead = recovery.equals("cancel_head") || recovery.equals("cache_changed");
                if (recovery.equals("cache_changed")) {
                    when(fixture.cache.findMatchingEngines(any(), any(), any()))
                            .thenReturn(Map.of(Fixture.PREFILL_ADDRESS, 1, "127.0.0.2:18080", 9));
                }
                if (cancelHead) {
                    fixture.runtime.scheduler().cancelRequest(995_001L, 0L, CancelReason.CLIENT_CANCELLED);
                    assertFalse(a.get(2, TimeUnit.SECONDS).isSuccess());
                } else {
                    fixture.runtime.applyStatus(fixture.decodeStatus, statusResponse(RoleType.DECODE, 3L, false));
                    assertTrue(a.get(2, TimeUnit.SECONDS).isSuccess());
                }
                Response response = b.get(2, TimeUnit.SECONDS);
                assertTrue(response.isSuccess());
                assertEquals(recovery.equals("cache_changed") ? "127.0.0.2:18080" : Fixture.PREFILL_ADDRESS,
                        prefillAddress(response), "a wakeup must recompute cache matches rather than pin B to P1");
                assertEquals(cancelHead ? List.of(995_002L) : List.of(995_001L, 995_002L),
                        fixture.submission.requestIds());
            }
        }
    }

    private static String prefillAddress(Response response) {
        return response.getServerStatus().stream()
                .filter(status -> status.getRole().supportsPrefill())
                .map(status -> status.getServerIp() + ":" + status.getHttpPort())
                .findFirst().orElseThrow();
    }

    private static final int INCIDENT_PENDING_REQUESTS = 10_000;
    private static final int INCIDENT_PREFILL_ENGINES = 750;
    private static final int INCIDENT_DECODE_ENGINES = 750;
    private static final int INCIDENT_DECODE_MAX_CONCURRENCY = 128;
    private static final int INCIDENT_WORKER_STATUS_UPDATES = 1_000;

    @Test
    @Timeout(20)
    void deferredDecodeRetryWaitsForSettlementFenceToExpire()
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        config.setDispatcher(DispatcherConfig.nonBatch());
        verifyRetryWaitsForDecodeSettlementFence(config, 880_001L);
    }

    @Test
    @Timeout(20)
    void capacityCheckedDecodeRetryWaitsForSettlementFenceToExpire()
            throws Exception {
        FlexlbConfig config = config();
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(
                EnumSet.of(VictimStage.DECODE_RESERVED));
        ((QueueOrderingConfig) config.queueScheduler().getOrdering())
                .setPreemption(preemption);
        verifyRetryWaitsForDecodeSettlementFence(config, 880_002L);
    }

    private static void verifyRetryWaitsForDecodeSettlementFence(
            FlexlbConfig config,
            long requestId) throws Exception {
        try (Fixture fixture = new Fixture(null, config)) {
            DecodeEndpoint.ReservationHandle settled;
            try (WorkerEndpoint.GenerationPin pin =
                         fixture.decodeEndpoint.tryPinGeneration()) {
                settled = fixture.decodeEndpoint.reserve(pin, requestId, 128L, 136L, 50);
            }
            assertTrue(fixture.decodeEndpoint.release(settled, DecodeEndpoint.ReleaseReason.COUNTERPART_FINISHED).released());
            assertEquals(0, fixture.totalDecodeReservations());

            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    statusResponse(RoleType.PREFILL, 2L, true));
            CompletableFuture<Response> retried =
                    fixture.runtime.scheduler().submit(
                            fixture.context(requestId, 50, 128_000L));
            assertFalse(retried.isDone());

            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    statusResponse(RoleType.PREFILL, 3L, false));

            Thread.sleep(100L);
            assertFalse(retried.isDone(),
                    "a live Decode settlement fence is a wait, not a failure");
            Thread.sleep(2L);
            fixture.decodeEndpoint.evictExpiredRequests(
                    0L, ignored -> false);

            assertTrue(retried.get(5, TimeUnit.SECONDS).isSuccess());
        }
    }

    @Test
    @Tag("performance-regression")
    @Timeout(90)
    void workerStatusHeartbeatsCannotReplayTenThousandPendingPlacements()
            throws Exception {
        FlexlbConfig config = config();
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests((long) INCIDENT_DECODE_MAX_CONCURRENCY);
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(
                EnumSet.of(VictimStage.DECODE_RESERVED));
        ((QueueOrderingConfig) config.queueScheduler().getOrdering())
                .setPreemption(preemption);

        try (QuietPlacementLogs ignored = new QuietPlacementLogs();
             Fixture fixture = new Fixture(RoleType.DECODE, config, true)) {
            fixture.expandFleet(
                    INCIDENT_PREFILL_ENGINES,
                    INCIDENT_DECODE_ENGINES,
                    INCIDENT_DECODE_MAX_CONCURRENCY);
            assertEquals(INCIDENT_PREFILL_ENGINES,
                    fixture.runtime.endpointRegistry()
                            .getEndpointCount(RoleType.PREFILL));
            assertEquals(INCIDENT_DECODE_ENGINES,
                    fixture.runtime.endpointRegistry()
                            .getEndpointCount(RoleType.DECODE));
            fixture.metrics.resetPlacementWakeups();
            fixture.submission.holdCompletions();
            long firstRequestId = 100_000L;
            long absoluteDeadline = System.currentTimeMillis()
                    + TimeUnit.MINUTES.toMillis(5);
            List<BalanceContext> contexts =
                    new ArrayList<>(INCIDENT_PENDING_REQUESTS);
            List<CompletableFuture<Response>> futures =
                    new ArrayList<>(INCIDENT_PENDING_REQUESTS);

            for (int index = 0; index < INCIDENT_PENDING_REQUESTS; index++) {
                BalanceContext context = fixture.context(
                        firstRequestId + index, 50, 128_000L);
                context.setSchedulingMetadata(SchedulingMetadata.explicit(
                        50, absoluteDeadline));
                contexts.add(context);
                futures.add(fixture.runtime.scheduler().submit(context));
            }

            awaitCondition(() -> fixture.runtime.scheduler().getQueuedRequestCount()
                    == INCIDENT_PENDING_REQUESTS, 10_000L);
            assertEquals(INCIDENT_PENDING_REQUESTS,
                    fixture.runtime.scheduler().getQueuedRequestCount());
            assertTrue(awaitPlacementAttempts(
                    fixture.metrics, INCIDENT_PENDING_REQUESTS, 30_000L),
                    "every queued request must receive an independent route");
            assertTrue(awaitPlacementQuiescence(
                    fixture.metrics, 100L, 5_000L));
            int initialPlacementAttempts =
                    fixture.metrics.totalPlacementAttempts();
            assertTrue(initialPlacementAttempts > 0);
            assertEquals(INCIDENT_PENDING_REQUESTS, initialPlacementAttempts,
                    "each request receives one independent initial route");
            assertEquals(INCIDENT_PENDING_REQUESTS, fixture.totalPrefillQueuedRequests(),
                    "each ranked request waits on its selected Prefill");
            assertEquals(0,
                    fixture.totalPrefillOwnedRequests(),
                    "P-success/D-failure must not retain Prefill ownership");
            assertEquals(0,
                    fixture.totalDecodeReservations(),
                    "P-success/D-failure must not retain a Decode reservation");
            assertCanonicalWaitState(
                    fixture, contexts, futures, absoluteDeadline);

            int attemptsBeforeStatus =
                    fixture.metrics.totalPlacementAttempts();
            fixture.publishDecodeStatusStorm(
                    INCIDENT_WORKER_STATUS_UPDATES,
                    INCIDENT_DECODE_MAX_CONCURRENCY);
            Thread.sleep(100L);

            int statusTriggeredPlacements =
                    fixture.metrics.totalPlacementAttempts()
                            - attemptsBeforeStatus;
            assertEquals(0, statusTriggeredPlacements,
                    "WorkerStatus heartbeat replayed pending placement");
            assertEquals(0L, fixture.metrics.placementWakeups(),
                    "unchanged WorkerStatus must not publish a capacity edge");
            assertEquals(INCIDENT_PENDING_REQUESTS,
                    fixture.runtime.scheduler().getQueuedRequestCount());
            assertCanonicalWaitState(
                    fixture, contexts, futures, absoluteDeadline);

            long capacityReleasedAt = System.nanoTime();
            fixture.runtime.applyStatus(
                    fixture.decodeStatus,
                    decodeRunningStatus(
                            fixture.decodeStatus.appliedStatusCursor()
                                    .statusVersion() + 1L,
                            INCIDENT_DECODE_MAX_CONCURRENCY - 1));

            assertTrue(fixture.submission.awaitCommands(
                    1, 5, TimeUnit.SECONDS));
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
            assertTrue(awaitPlacementQuiescence(
                    fixture.metrics, 100L, 5_000L));
            assertEquals(1, fixture.submission.requestIds().size(),
                    "one Decode slot must not be oversold across Prefill batchers");
            assertEquals(attemptsBeforeStatus, fixture.metrics.totalPlacementAttempts(),
                    "the one available Decode slot wakes Prefill decisions without replaying global placements");
            assertEquals(1, fixture.metrics.maxAttemptsForAnyRequest());
            assertEquals(1L, fixture.metrics.placementWakeups(),
                    "one real capacity release must publish one placement edge");
            assertCanonicalWaitState(
                    fixture, contexts, futures, absoluteDeadline);

            fixture.metrics.print(
                    "worker-status-and-capacity-edge",
                    INCIDENT_PENDING_REQUESTS,
                    fixture.runtime.scheduler().getQueuedRequestCount(),
                    statusTriggeredPlacements,
                    capacityReleasedAt);
        }
    }

    private static void assertCanonicalWaitState(
            Fixture fixture,
            List<BalanceContext> contexts,
            List<CompletableFuture<Response>> futures,
            long absoluteDeadline) {
        for (int index = 0; index < contexts.size(); index++) {
            BalanceContext context = contexts.get(index);
            assertSame(futures.get(index),
                    fixture.runtime.requestFuture(context.getRequestId()),
                    "waiting must retain the canonical future");
            // A bounded planner frontier may not have observed every suffix
            // yet. Any request that was observed must retain its original
            // context and absolute deadline across exact-capacity retries.
            BalanceContext observed =
                    fixture.metrics.context(context.getRequestId());
            if (observed != null) {
                assertSame(context, observed,
                        "placement retry replaced the BalanceContext");
                assertEquals(absoluteDeadline,
                        context.schedulingMetadata().expiresAtMs(),
                        "placement retry changed the absolute deadline");
                assertEquals(absoluteDeadline,
                        fixture.metrics.deadline(context.getRequestId()),
                        "placement observed a different absolute deadline");
            }
        }
    }

    private static boolean awaitPlacementAttempts(
            PlacementMetrics metrics, int expected, long timeoutMs)
            throws InterruptedException {
        long deadline = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (metrics.totalPlacementAttempts() < expected
                && System.nanoTime() < deadline) {
            Thread.sleep(1L);
        }
        return metrics.totalPlacementAttempts() >= expected;
    }

    private static boolean awaitPlacementQuiescence(
            PlacementMetrics metrics,
            long stableForMs,
            long timeoutMs) throws InterruptedException {
        long timeoutAt = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        int observed = metrics.totalPlacementAttempts();
        long unchangedSince = System.nanoTime();
        while (System.nanoTime() < timeoutAt) {
            Thread.sleep(1L);
            int current = metrics.totalPlacementAttempts();
            if (current != observed) {
                observed = current;
                unchangedSince = System.nanoTime();
            } else if (System.nanoTime() - unchangedSince
                    >= TimeUnit.MILLISECONDS.toNanos(stableForMs)) {
                return true;
            }
        }
        return false;
    }

    @Test
    @Timeout(20)
    void prefillPressureRetainsAndLaterDispatchesTheOriginalRequest()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.PREFILL)) {
            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(fixture.context(101L));

            assertFalse(waiting.isDone());
            assertEquals(1, fixture.runtime.scheduler().getQueuedRequestCount());
            assertEquals(List.of(), fixture.submission.requestIds());

            fixture.releaseCapacity();

            assertTrue(waiting.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(List.of(101L), fixture.submission.requestIds());
        }
    }

    @Test
    @Timeout(20)
    void decodePressureRetainsAndLaterDispatchesTheOriginalRequest()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(fixture.context(201L));

            assertFalse(waiting.isDone());
            assertEquals(List.of(), fixture.submission.requestIds());

            fixture.releaseCapacity();

            assertTrue(waiting.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(List.of(201L), fixture.submission.requestIds());
        }
    }

    @Test
    @Timeout(20)
    void decodeWaitersPreserveFifoAtTheSamePriority() throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            fixture.config.fixedWindowDecision().setMaxRequests(2);
            fixture.config.fixedWindowDecision().setMaxCollectionWaitMs(50L);
            fixture.submission.holdCompletions();
            var firstPlanning = new CompletableFuture<Void>();
            var finishFirstPlan = new CompletableFuture<Void>();
            var olderContext = fixture.context(251L, 50);
            olderContext.getRequest().setCacheKeyBlockSize(128L);
            olderContext.getRequest().setBlockCacheKeys(List.of(251L));
            when(fixture.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                if (List.of(251L).equals(call.getArgument(0))) {
                    firstPlanning.complete(null);
                    finishFirstPlan.get(2, TimeUnit.SECONDS);
                }
                return Map.of();
            });
            var older = fixture.runtime.scheduler().submit(olderContext);
            firstPlanning.get(2, TimeUnit.SECONDS);
            var later = fixture.runtime.scheduler().submit(fixture.context(252L, 50));
            try {
                awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 1, 2_000L);
                assertFalse(older.isDone());
            } finally {
                finishFirstPlan.complete(null);
            }
            assertFalse(older.isDone());
            assertFalse(later.isDone());
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 2, 2_000L);
            fixture.releaseCapacity();

            assertTrue(fixture.submission.awaitCommands(
                    1, 2, TimeUnit.SECONDS));
            assertEquals(List.of(251L), fixture.submission.requestIds());
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
        }
    }

    @Test
    @Timeout(20)
    void decodeWaitersUsePriorityOrderingBeforeArrivalOrder()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            fixture.config.fixedWindowDecision().setMaxRequests(2);
            fixture.config.fixedWindowDecision().setMaxCollectionWaitMs(50L);
            fixture.submission.holdCompletions();
            fixture.runtime.scheduler().submit(fixture.context(261L, 50));
            fixture.runtime.scheduler().submit(fixture.context(262L, 80));

            awaitCondition(() -> fixture.runtime.scheduler()
                    .getQueuedRequestCount() >= 2, 2_000L);
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 2, 2_000L);
            fixture.releaseCapacity();

            assertTrue(fixture.submission.awaitCommands(
                    1, 2, TimeUnit.SECONDS));
            assertEquals(List.of(262L), fixture.submission.requestIds());
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
        }
    }

    @Test
    @Timeout(20)
    void releasedDecodeCapacityPreservesFifo() throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            fixture.submission.holdCompletions();
            fixture.runtime.scheduler().submit(fixture.context(271L, 50));
            fixture.runtime.scheduler().submit(fixture.context(272L, 50));

            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 2, 2_000L);
            fixture.releaseCapacity();

            assertTrue(fixture.submission.awaitCommands(
                    1, 2, TimeUnit.SECONDS));
            assertEquals(List.of(271L), fixture.submission.requestIds());
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
        }
    }

    @Test
    @Timeout(20)
    void duplicateCapacitySignalsCannotOversellOneDecodeSlot()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            fixture.submission.holdCompletions();
            fixture.runtime.scheduler().submit(fixture.context(281L, 50));
            fixture.runtime.scheduler().submit(fixture.context(282L, 50));

            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 2, 2_000L);
            fixture.releaseCapacity();
            fixture.runtime.applyStatus(
                    fixture.decodeStatus,
                    statusResponse(RoleType.DECODE, 4L, false));

            assertTrue(fixture.submission.awaitCommands(
                    1, 2, TimeUnit.SECONDS));
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
            assertEquals(List.of(281L), fixture.submission.requestIds());
        }
    }

    @ParameterizedTest
    @CsvSource({"FIFO,BATCH", "FIFO,NON_BATCH", "PRIORITY,BATCH", "PRIORITY,NON_BATCH"})
    @Timeout(20)
    void nonPreemptiveQueueChecksDecodeWhenPrefillCanDispatch(String ordering, DispatcherConfig.Type delivery)
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering("FIFO".equals(ordering)
                ? new QueueOrderingConfig() : QueueOrderingConfig.priority());
        config.queueScheduler().getOrdering().setPreemption(null);
        config.getDispatcher().setType(delivery);
        assertEquals(DecodeMode.WAIT_AT_PLACEMENT, DecodeMode.from(config));

        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            CompletableFuture<Response> waiting = fixture.runtime.scheduler().submit(
                    fixture.context(301L, 50, 128_000L));
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 1, 2_000L);
            assertFalse(waiting.isDone());
            assertEquals(1, fixture.prefillEndpoint.queuedRequestCount());
            assertEquals(0, fixture.decodeEndpoint.resourceSnapshot().queuedCount(),
                    "a blocked placement must not create an unfunded reservation");
            assertEquals(List.of(), fixture.submission.requestIds());

            fixture.releaseCapacity();
            Response response = waiting.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals("127.0.0.1:18081", decodeAddress(response));
            assertEquals(1, fixture.decodeEndpoint.routingView().engineCapacityUsed());
            assertEquals(delivery == DispatcherConfig.Type.BATCH ? List.of(301L) : List.of(),
                    fixture.submission.requestIds());
        }
    }

    @ParameterizedTest
    @CsvSource({"BATCH", "NON_BATCH"})
    @Timeout(20)
    void preemptiveQueueChecksDecodeWhenPrefillCanDispatch(DispatcherConfig.Type delivery) throws Exception {
        FlexlbConfig config = config();
        config.getDispatcher().setType(delivery);
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_RESERVED));
        config.queueScheduler().getOrdering().setPreemption(preemption);
        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            CompletableFuture<Response> waiting = fixture.runtime.scheduler().submit(
                    fixture.context(302L, 50, 128_000L));
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 1, 2_000L);
            assertTrue(fixture.metrics.totalPlacementAttempts() > 0);
            assertFalse(waiting.isDone());
            assertEquals(1, fixture.prefillEndpoint.queuedRequestCount());
            assertEquals(0, fixture.decodeEndpoint.resourceSnapshot().queuedCount());
            assertEquals(List.of(), fixture.submission.requestIds());
            fixture.releaseCapacity();
            Response response = waiting.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals("127.0.0.1:18081", decodeAddress(response));
            assertEquals(1, fixture.decodeEndpoint.routingView().engineCapacityUsed());
            assertEquals(delivery == DispatcherConfig.Type.BATCH ? List.of(302L) : List.of(),
                    fixture.submission.requestIds());
        }
    }

    @ParameterizedTest(name = "{0}: earlier priority {1}, cancel later {2}, next dispatch {3}")
    @CsvSource({"NON_BATCH,50,false,702", "NON_BATCH,70,false,701", "NON_BATCH,50,true,701",
            "BATCH,50,false,702", "BATCH,70,false,701", "BATCH,50,true,701"})
    @Timeout(15)
    void decodeWakeDispatchesTheHighestRankedLiveWaiterOnTheSamePrefill(
            DispatcherConfig.Type delivery, int earlierPriority, boolean cancelLater, long expectedFirst)
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setDecision(org.flexlb.config.DecisionPolicyConfig.single());
        config.priorityOrdering().setPrefillCostOrder(true);
        config.priorityOrdering().setPreemption(null);
        config.getDispatcher().setType(delivery);
        try (Fixture fixture = new Fixture(null, config)) {
            var scheduler = fixture.runtime.scheduler();
            var noPrefillKv = statusResponse(RoleType.PREFILL, 2L, false);
            noPrefillKv.setAvailableKvCacheTokens(0L);
            fixture.runtime.applyStatus(fixture.prefillStatus, noPrefillKv);

            var earlier = fixture.context(701L, earlierPriority, 400L);
            var later = fixture.context(702L, 50, 400L);
            earlier.getRequest().setBlockCacheKeys(List.of(1L, 2L, 3L, 4L));
            later.getRequest().setBlockCacheKeys(List.of(5L, 6L, 7L, 8L));
            earlier.getRequest().setCacheKeyBlockSize(100L);
            later.getRequest().setCacheKeyBlockSize(100L);
            when(fixture.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                List<Long> keys = call.getArgument(0);
                return Map.of(Fixture.PREFILL_ADDRESS, earlier.getRequest().getBlockCacheKeys().equals(keys) ? 1 : 3);
            });
            var first = scheduler.submit(earlier);
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 1, 2_000L);
            var second = scheduler.submit(later);
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 2, 2_000L);

            // P can now serve either request, but D's only slot is occupied by an external stream.
            fixture.runtime.applyStatus(fixture.decodeStatus, statusResponse(RoleType.DECODE, 2L, true));
            fixture.runtime.applyStatus(fixture.prefillStatus, statusResponse(RoleType.PREFILL, 3L, false));
            awaitCondition(() -> fixture.prefillEndpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock() != null, 2_000L);
            assertFalse(first.isDone());
            assertFalse(second.isDone());
            assertTrue(fixture.submission.requestIds().isEmpty());

            if (cancelLater) {
                scheduler.cancelRequest(702L, 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(second.get(2, TimeUnit.SECONDS).isSuccess());
            }
            fixture.runtime.applyStatus(fixture.decodeStatus, statusResponse(RoleType.DECODE, 3L, false));
            var nextDispatch = cancelLater ? first : CompletableFuture.anyOf(first, second);
            nextDispatch.get(2, TimeUnit.SECONDS);
            var winner = expectedFirst == 701L ? first : second;
            var other = expectedFirst == 701L ? second : first;
            assertTrue(winner.isDone(),
                    "the one released Decode slot must go to the highest priority, then highest reuse fraction");
            assertTrue(winner.join().isSuccess());
            assertEquals(cancelLater, other.isDone());
            assertEquals(1L, fixture.decodeEndpoint.routingView().engineCapacityUsed());
            if (delivery == DispatcherConfig.Type.BATCH) {
                assertEquals(List.of(expectedFirst), fixture.submission.requestIds());
            }
        }
    }

    private static void awaitCapacityWaiters(RequestScheduler scheduler, int expected)
            throws InterruptedException {
        Object coordinator = ReflectionTestUtils.getField(scheduler, "globalQueue");
        var lock = (ReentrantLock) ReflectionTestUtils.getField(coordinator, "lock");
        Object waitQueue = ReflectionTestUtils.getField(coordinator, "waitingRequests");
        var waiting = (Map<?, ?>) ReflectionTestUtils.getField(waitQueue, "waiting");
        // Observe completed park registration under its owning lock. Placement
        // counters are updated before park and cannot establish this barrier.
        awaitCondition(() -> {
            lock.lock();
            try {
                return waiting.size() == expected;
            } finally {
                lock.unlock();
            }
        }, 2_000L);
    }

    private static void awaitCondition(
            java.util.function.BooleanSupplier condition,
            long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) {
            Thread.sleep(5L);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true within " + timeoutMs + " ms");
    }

    @Test
    @Timeout(20)
    void batchCreditsLimitDispatchWhileAdditionalRequestsRemainOnTheWorkerQueue()
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        config.fixedWindowDecision().setMaxRequests(2);
        config.fixedWindowDecision().setMaxCollectionWaitMs(60_000L);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(64L);
        assertEquals(DecodeMode.WAIT_AT_PLACEMENT, DecodeMode.from(config));

        try (Fixture fixture = new Fixture(null, config)) {
            fixture.submission.holdCompletions();
            List<CompletableFuture<Response>> waiting = new ArrayList<>(LongStream
                    .rangeClosed(401L, 406L)
                    .mapToObj(requestId -> fixture.runtime.scheduler().submit(
                            fixture.context(requestId, 50)))
                    .toList());

            assertTrue(fixture.submission.awaitCommands(2, 2, TimeUnit.SECONDS));
            awaitCondition(() -> fixture.prefillEndpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock() != null, 2_000L);
            assertTrue(fixture.prefillEndpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock() != null,
                    "the third group must observe both batch credits in use before more requests arrive");
            assertEquals(2, fixture.prefillEndpoint.getInflightBatchCount());
            waiting.addAll(LongStream.rangeClosed(407L, 464L)
                    .mapToObj(requestId -> fixture.runtime.scheduler().submit(
                            fixture.context(requestId, 50)))
                    .toList());

            assertTrue(waiting.stream().noneMatch(CompletableFuture::isDone));
            int capacityRequests = waiting.size();
            awaitCondition(() -> fixture.prefillEndpoint.observedRequestCount() == capacityRequests, 2_000L);
            assertEquals(4, fixture.decodeEndpoint.resourceSnapshot().reserved().size(),
                    () -> "only the two dispatched pairs may own Decode capacity. prefill="
                            + fixture.prefillEndpoint.observedRequestCount()
                            + ", queued=" + fixture.runtime.scheduler().getQueuedRequestCount()
                            + ", done=" + waiting.stream().filter(CompletableFuture::isDone).count()
                            + ", attempts=" + fixture.metrics.totalPlacementAttempts());
            assertFalse(fixture.submission.awaitCommands(1, 100, TimeUnit.MILLISECONDS));
            assertEquals(List.of(2, 2), fixture.submission.submittedItems.stream().map(List::size).toList(),
                    "each dispatched group respects decision.maxRequests");
            assertEquals(capacityRequests, fixture.prefillEndpoint.observedRequestCount());
            assertEquals(capacityRequests - 4, fixture.prefillEndpoint.queuedRequestCount(),
                    "two in-flight batches leave the other requests on the worker queue");
            assertEquals(0,
                    fixture.runtime.scheduler().getQueuedRequestCount()
                            - fixture.prefillEndpoint.queuedRequestCount(),
                    "worker waiting requests are retained without a synthetic batch-times-size queue cap");
        }
    }

    @ParameterizedTest
    @CsvSource({"true", "false"})
    @Timeout(15)
    void losingTheLastDecodeSlotCanUseAnotherDecodeBeforeTheWinnerReleases(
            boolean spareIsReadyOnRetry) throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setDecision(org.flexlb.config.DecisionPolicyConfig.single());
        try (Fixture fixture = new Fixture(null, config, true)) {
            var spare = initializedStatus(RoleType.DECODE, "127.0.0.2", 18082);
            var spareEndpoint = (DecodeEndpoint) publishEndpoint(fixture.runtime.endpointRegistry(),
                    RoleType.DECODE, spare.getIpPort(), spare);
            fixture.runtime.applyStatus(spare, statusResponse(RoleType.DECODE, 2L, true));

            // D0 is the only eligible candidate. Pause before final submission admission,
            // then let an independent caller claim its last real slot.
            fixture.submission.blockPreparation();
            long requestId = 501L;
            var response = fixture.runtime.scheduler().submit(fixture.context(requestId, 50, 128_000L));
            assertTrue(fixture.submission.awaitPreparation(2, TimeUnit.SECONDS));
            var capacity = new DecodeEndpoint.AdmissionCapacity(1L, 100L);
            try (var pin = fixture.decodeEndpoint.tryPinGeneration()) {
                var competitor = takeDecodeSlot(fixture.decodeEndpoint, pin, 502L, capacity);
                assertNotNull(competitor, "D0's last slot must actually belong to the competing request");
                try {
                    if (spareIsReadyOnRetry) {
                        fixture.runtime.applyStatus(spare, statusResponse(RoleType.DECODE, 3L, false));
                    }
                    fixture.submission.unblockPreparation();

                    if (!spareIsReadyOnRetry) {
                        assertFalse(fixture.submission.awaitCommands(1, 200, TimeUnit.MILLISECONDS));
                        assertFalse(response.isDone(), "temporary Decode exhaustion must leave A pending");
                        assertEquals(0, fixture.prefillEndpoint.getInflightBatchCount(),
                                "A cannot hold Prefill execution capacity while every Decode is full");
                        assertEquals(0, spareEndpoint.resourceSnapshot().activeDispatchPermits());
                        fixture.runtime.applyStatus(spare, statusResponse(RoleType.DECODE, 3L, false));
                    }
                    assertTrue(fixture.submission.awaitCommands(1, 2, TimeUnit.SECONDS),
                            "the spare Decode must handle A without any release or heartbeat from D0");
                    Response delivered = response.get(2, TimeUnit.SECONDS);
                    assertTrue(delivered.isSuccess(), delivered.getErrorMessage());
                    assertEquals(Fixture.PREFILL_ADDRESS, prefillAddress(delivered));
                    assertEquals(spare.getIpPort(), decodeAddress(delivered));
                    assertEquals(List.of(requestId), fixture.submission.requestIds());
                    assertEquals(List.of(spare.getIpPort()), fixture.submission.decodeAddresses());
                    assertEquals(1L, fixture.decodeEndpoint.routingView().dispatchUsage().occupiedRequests());
                    assertNull(fixture.decodeEndpoint.reservationHandle(requestId),
                            "the losing candidate must retain no reservation for A");
                } finally {
                    competitor.release();
                }
            }
        }
    }

    @ParameterizedTest
    @CsvSource({"unavailable,false", "insufficientKv,false", "onlyPromptFits,false",
            "exactKvFit,true", "noSlot,false", "shadowOnly,true"})
    @Timeout(10)
    void singleNonBatchSelectsLowestRunningEligibleDecode(String firstWorkerState, boolean chooseFirst) throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setDecision(org.flexlb.config.DecisionPolicyConfig.single());
        config.setDispatcher(DispatcherConfig.nonBatch());
        var decode = config.getRouter().getRoles().getDecode();
        decode.getAvailability().setMaxEngineRequests(3L);
        decode.getCostEstimator().setExpression("running_size");
        try (Fixture fixture = new Fixture(null, config, true)) {
            fixture.expandFleet(1, 2, 1);
            var first = fixture.decodeEndpoint;
            var idleStatus = decodeRunningStatus(3L, 0);
            // Prompt = 128, output budget = 8. Exercise the hard and expected KV checks separately.
            switch (firstWorkerState) {
                case "insufficientKv" -> idleStatus.setAvailableKvCacheTokens(127L);
                case "onlyPromptFits" -> idleStatus.setAvailableKvCacheTokens(128L);
                case "exactKvFit" -> idleStatus.setAvailableKvCacheTokens(136L);
            }
            fixture.runtime.applyStatus(fixture.decodeStatus, idleStatus);
            if (firstWorkerState.equals("unavailable")) {
                fixture.decodeStatus.markRemovedFromDiscovery();
            }
            // Actual running load on the second worker is one in every case.
            // The idle worker can have queued placements or issued permits despite zero running.
            try (var pin = first.tryPinGeneration()) {
                List<DecodeEndpoint.EngineDispatchPermit> permits = new ArrayList<>();
                List<DecodeEndpoint.ReservationHandle> shadows = new ArrayList<>();
                try {
                    for (int i = 0; i < 3; i++) {
                        if (firstWorkerState.equals("noSlot")) {
                            var permit = takeDecodeSlot(first, pin, 7000L + i,
                                    new DecodeEndpoint.AdmissionCapacity(3L, 100L));
                            assertNotNull(permit);
                            permits.add(permit);
                        } else if (firstWorkerState.equals("shadowOnly")) {
                            var shadow = first.reserve(pin, 7000L + i, 128L, 136L, 50);
                            assertNotNull(shadow);
                            shadows.add(shadow);
                        }
                    }
                    var request = fixture.runtime.scheduler().submit(fixture.context(503L));
                    Response result = assertDoesNotThrow(() -> request.get(2, TimeUnit.SECONDS),
                            "a healthy Decode with sufficient capacity is available: " + firstWorkerState);
                    assertTrue(result.isSuccess(), result.getErrorMessage());
                    String expected = chooseFirst ? first.ipPort() : fixture.decodeEndpoints.get(1).ipPort();
                    assertEquals(expected, decodeAddress(result), firstWorkerState);
                } finally {
                    permits.forEach(DecodeEndpoint.EngineDispatchPermit::release);
                    shadows.forEach(handle -> first.release(handle, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK));
                }
            }
        }
    }

    private static DecodeEndpoint.EngineDispatchPermit takeDecodeSlot(
            DecodeEndpoint endpoint, WorkerEndpoint.GenerationPin pin, long requestId,
            DecodeEndpoint.AdmissionCapacity capacity) {
        var reservation = endpoint.reserve(pin, requestId, 128L, 136L, 50);
        assertNotNull(reservation);
        return endpoint.acquireDispatchPermit(reservation, capacity).permit();
    }

    @ParameterizedTest
    @CsvSource({"BATCH,-1", "NON_BATCH,-1", "BATCH,1", "NON_BATCH,1"})
    @Timeout(15)
    void runningLimitWakesAtObservedReleaseWhileEngineOwnedLimitRemainsIndependent(
            DispatcherConfig.Type delivery, long engineLimit) throws Exception {
        FlexlbConfig config = config();
        config.getDispatcher().setType(delivery);
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        var limits = config.getRouter().getRoles().getDecode().getAvailability();
        limits.setMaxEngineRequests(engineLimit);
        limits.setMaxRunningStreamSize(1L);
        try (Fixture fixture = new Fixture(null, config)) {
            var busy = decodeRunningStatus(2L, 1);
            TaskInfo allocated = new TaskInfo();
            long allocatedId = Fixture.EXTERNAL_REQUEST_ID + 1L;
            allocated.setRequestId(allocatedId);
            allocated.setPhase(TaskPhase.KV_ALLOCATED);
            busy.getRunningTaskInfo().put(Long.toString(allocatedId), allocated);
            fixture.runtime.applyStatus(fixture.decodeStatus, busy);

            var result = fixture.runtime.scheduler().submit(fixture.context(604L));
            awaitCondition(() -> fixture.prefillEndpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock() != null, 2_000L);
            assertFalse(result.isDone());

            // One stream finished. The other still owns Decode KV and has not started running.
            var idle = decodeRunningStatus(3L, 0);
            idle.setRunningTaskInfo(Map.of(Long.toString(allocatedId), allocated));
            fixture.runtime.applyStatus(fixture.decodeStatus, idle);
            if (engineLimit > 0) {
                assertThrows(TimeoutException.class, () -> result.get(100, TimeUnit.MILLISECONDS));
                fixture.runtime.applyStatus(fixture.decodeStatus, decodeRunningStatus(4L, 0));
            }
            assertTrue(result.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(0, fixture.prefillEndpoint.queuedRequestCount());
            assertEquals(delivery == DispatcherConfig.Type.BATCH ? List.of(604L) : List.of(),
                    fixture.submission.requestIds());
        }
    }

    @Test
    @Timeout(20)
    void queuedDeliveryWakesWhenItsDecodeCapacityReturns()
            throws Exception {
        verifyCommittedDecodeCapacityWake(config(), 601L);
    }

    @Test
    @Timeout(20)
    void nonBatchDeliveryAlsoWakesWhenItsDecodeCapacityReturns()
            throws Exception {
        FlexlbConfig config = config();
        config.setDispatcher(DispatcherConfig.nonBatch());
        verifyCommittedDecodeCapacityWake(config, 602L);
    }

    @Test
    @Timeout(20)
    void workerReportedUnknownPrefillRequestsConsumeConfiguredRequestLimit()
            throws Exception {
        FlexlbConfig config = config();
        DispatcherConfig dispatcher = DispatcherConfig.nonBatch();
        dispatcher.setMaxInflightPerPrefillWorker(2);
        config.setDispatcher(dispatcher);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(32L);

        try (Fixture fixture = new Fixture(null, config)) {
            List<CompletableFuture<Response>> responses = new ArrayList<>();
            for (long requestId = 700L; requestId < 706L; requestId++) {
                responses.add(fixture.runtime.scheduler().submit(
                        fixture.context(requestId, 50, 128_000L)));
            }

            awaitCompletedResponses(responses, 2, 2, TimeUnit.SECONDS);
            assertEquals(2, completedResponses(responses),
                    "the configured request limit must hold the remaining requests in the global queue");

            List<Long> delivered = LongStream.range(0, responses.size())
                    .filter(index -> responses.get((int) index).isDone())
                    .map(index -> 700L + index)
                    .boxed()
                    .toList();
            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    prefillStatusWithFinishedAndUnknownRunning(
                            2L, delivered, 2));

            Thread.sleep(100L);
            assertEquals(2, completedResponses(responses),
                    "worker-reported unknown work must consume the same hard "
                            + "request-count capacity as local route leases");

            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    prefillStatusWithFinishedAndUnknownRunning(
                            3L, List.of(), 1));
            awaitCompletedResponses(responses, 3, 2, TimeUnit.SECONDS);
            assertEquals(3, completedResponses(responses),
                    "an available request slot permits delivery even when preceding work has no time estimate");
            fixture.runtime.applyStatus(fixture.prefillStatus,
                    prefillStatusWithFinishedAndUnknownRunning(4L, List.of(), 0));
            awaitCompletedResponses(responses, 4, 2, TimeUnit.SECONDS);
            assertEquals(4, completedResponses(responses),
                    "clearing unknown work wakes requests up to the configured request limit");
        }
    }

    @Test
    @Timeout(20)
    void sameVersionHeartbeatChargesEngineWorkAfterLocalTerminal()
            throws Exception {
        FlexlbConfig config = config();
        DispatcherConfig dispatcher = DispatcherConfig.nonBatch();
        dispatcher.setMaxInflightPerPrefillWorker(2);
        config.setDispatcher(dispatcher);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(32L);

        try (Fixture fixture = new Fixture(null, config)) {
            List<CompletableFuture<Response>> admitted = new ArrayList<>();
            for (long requestId = 800L; requestId < 802L; requestId++) {
                admitted.add(fixture.runtime.scheduler().submit(
                        fixture.context(requestId, 50, 128_000L)));
            }

            awaitCompletedResponses(admitted, 2, 2, TimeUnit.SECONDS);
            assertEquals(2, completedResponses(admitted));

            List<RequestExecution> delivered = List.of(
                    fixture.runtime.activeItem(800L),
                    fixture.runtime.activeItem(801L));
            assertEquals(2, delivered.size());
            for (RequestExecution item : delivered) {
                assertTrue(item != null);
                assertTrue(fixture.prefillEndpoint.releaseCommittedItem(item));
            }

            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    prefillRunningStatus(1L, List.of(800L, 801L)));

            List<CompletableFuture<Response>> waiting = new ArrayList<>();
            for (long requestId = 802L; requestId < 806L; requestId++) {
                waiting.add(fixture.runtime.scheduler().submit(
                        fixture.context(requestId, 50, 128_000L)));
            }

            Thread.sleep(250L);
            assertEquals(0, completedResponses(waiting),
                    "a same-version heartbeat must charge work that the "
                            + "Engine still owns after the Master locally "
                            + "terminalized its route");

            int attemptsBeforeHeartbeats =
                    fixture.metrics.totalPlacementAttempts();
            for (int update = 0;
                    update < INCIDENT_WORKER_STATUS_UPDATES;
                    update++) {
                fixture.runtime.applyStatus(
                        fixture.prefillStatus,
                        prefillRunningStatus(1L, List.of(800L, 801L)));
            }
            Thread.sleep(100L);
            assertEquals(attemptsBeforeHeartbeats,
                    fixture.metrics.totalPlacementAttempts(),
                    "unchanged same-version heartbeats must not replay "
                            + "placement");

            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    prefillRunningStatus(1L, List.of(800L)));
            awaitCompletedResponses(waiting, 1, 2, TimeUnit.SECONDS);
            assertEquals(1, completedResponses(waiting),
                    "unknown completion time does not override available request slots");
            fixture.runtime.applyStatus(fixture.prefillStatus,
                    prefillRunningStatus(1L, List.of()));
            awaitCompletedResponses(waiting, 2, 2, TimeUnit.SECONDS);
            assertEquals(2, completedResponses(waiting),
                    "a same-version status clearing unknown work must wake the request-count prefix");
        }
    }

    private static int completedResponses(
            List<CompletableFuture<Response>> responses) {
        return (int) responses.stream().filter(Future::isDone).count();
    }

    private static void awaitCompletedResponses(
            List<CompletableFuture<Response>> responses,
            int expected,
            long timeout,
            TimeUnit unit) throws InterruptedException {
        long deadline = System.nanoTime() + unit.toNanos(timeout);
        while (completedResponses(responses) < expected
                && System.nanoTime() < deadline) {
            Thread.sleep(5L);
        }
    }

    private static void verifyCommittedDecodeCapacityWake(
            FlexlbConfig config,
            long requestId) throws Exception {
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        assertEquals(DecodeMode.WAIT_AT_PLACEMENT, DecodeMode.from(config));

        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            WorkerStatus spareStatus = initializedStatus(
                    RoleType.DECODE, "127.0.0.2", 18_082);
            DecodeEndpoint spareEndpoint = (DecodeEndpoint)
                    publishEndpoint(
                            fixture.runtime.endpointRegistry(),
                            RoleType.DECODE,
                            "127.0.0.2:18082",
                            spareStatus);
            fixture.runtime.applyStatus(
                    spareStatus,
                    statusResponse(RoleType.DECODE, 2L, true));

            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(
                            fixture.context(requestId, 50, 128_000L));
            awaitCondition(() -> fixture.prefillEndpoint.queuedRequestCount() == 1, 2_000L);
            assertFalse(waiting.isDone());
            assertEquals(0, fixture.decodeEndpoint
                    .resourceSnapshot().reserved().size());
            assertEquals(0, spareEndpoint
                    .resourceSnapshot().reserved().size());

            fixture.runtime.applyStatus(
                    fixture.decodeStatus,
                    statusResponse(RoleType.DECODE, 3L, false));

            Response response = waiting.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals("127.0.0.1:18081", decodeAddress(response));
            assertEquals(1, fixture.decodeEndpoint.routingView().engineLoad());
        }
    }

    private static String decodeAddress(Response response) {
        return response.getServerStatus().stream()
                .filter(status -> status.getRole() == RoleType.DECODE)
                .map(status -> status.getServerIp() + ":"
                        + status.getHttpPort())
                .findFirst()
                .orElseThrow();
    }

    private static final class Fixture implements AutoCloseable {
        private static final long EXTERNAL_REQUEST_ID = 9_001L;
        private static final String PREFILL_ADDRESS = "127.0.0.1:18080";
        private static final String DECODE_ADDRESS = "127.0.0.1:18081";

        private final FlexlbConfig config;
        private final ConfigService configService = mock(ConfigService.class);
        private final RecordingSubmissionPort submission =
                new RecordingSubmissionPort();
        private final PlacementMetrics metrics = new PlacementMetrics();
        private final RequestSchedulerTestRuntime runtime;
        private final WorkerStatus prefillStatus;
        private final WorkerStatus decodeStatus;
        private final PrefillEndpoint prefillEndpoint;
        private final DecodeEndpoint decodeEndpoint;
        private final List<PrefillEndpoint> prefillEndpoints =
                new ArrayList<>();
        private final List<DecodeEndpoint> decodeEndpoints =
                new ArrayList<>();
        private final List<WorkerStatus> decodeStatuses =
                new ArrayList<>();
        private final RoleType saturatedRole;
        private final CacheAwareService cache = mock(CacheAwareService.class);

        private Fixture(RoleType saturatedRole) {
            this(saturatedRole, config());
        }

        private Fixture(RoleType saturatedRole, FlexlbConfig config) {
            this(saturatedRole, config, false);
        }

        private Fixture(
                RoleType saturatedRole,
                FlexlbConfig config,
                boolean prefillFirst) {
            this.saturatedRole = saturatedRole;
            this.config = config;
            when(configService.loadBalanceConfig()).thenReturn(config);
            runtime = new RequestSchedulerTestRuntime(
                    configService,
                    submission::tryPrepareSubmission,
                    new BatchSchedulerReporter(new NoOpFlexMonitor()),
                    new RequestSchedulerReporter(new NoOpFlexMonitor()));

            prefillStatus = initializedStatus(
                    RoleType.PREFILL, "127.0.0.1", 18_080);
            decodeStatus = initializedStatus(
                    RoleType.DECODE, "127.0.0.1", 18_081);
            prefillEndpoint = (PrefillEndpoint) publishEndpoint(
                    runtime.endpointRegistry(),
                    RoleType.PREFILL,
                    PREFILL_ADDRESS,
                    prefillStatus);
            decodeEndpoint = (DecodeEndpoint) publishEndpoint(
                    runtime.endpointRegistry(),
                    RoleType.DECODE,
                    DECODE_ADDRESS,
                    decodeStatus);
            prefillEndpoints.add(prefillEndpoint);
            decodeEndpoints.add(decodeEndpoint);
            decodeStatuses.add(decodeStatus);

            if (saturatedRole == RoleType.PREFILL) {
                runtime.applyStatus(
                        prefillStatus,
                        statusResponse(RoleType.PREFILL, 2L, true));
            } else if (saturatedRole == RoleType.DECODE) {
                runtime.applyStatus(
                        decodeStatus,
                        statusResponse(RoleType.DECODE, 2L, true));
            }

            WorkerDirectory workers =
                    new WorkerDirectory(runtime.endpointRegistry());
            when(cache.findMatchingEngines(any(), any(), any()))
                    .thenReturn(Map.of());
            CostBasedPrefillStrategy prefillSelector =
                    new CostBasedPrefillStrategy(
                            workers,
                            cache,
                            mock(EngineHealthReporter.class));
            DecodeSelector decodeSelector =
                    new DecodeSelector(workers);
            runtime.placementAvailability().addListener(
                    event -> metrics.recordPlacementWakeup(event.key()));
            runtime.bindRouter(new RecordingRouter(
                    new DefaultRouter(
                            prefillSelector,
                            decodeSelector,
                            new RandomStrategy(workers),
                            configService,
                            modelMeta(prefillFirst)),
                    metrics));
        }

        private void expandFleet(
                int prefillCount,
                int decodeCount,
                int decodeRunningCount) {
            if (prefillCount < 1 || decodeCount < 1) {
                throw new IllegalArgumentException(
                        "fleet sizes must include the primary endpoints");
            }
            runtime.applyStatus(
                    decodeStatus,
                    decodeRunningStatus(
                            decodeStatus.appliedStatusCursor()
                                    .statusVersion() + 1L,
                            decodeRunningCount));
            for (int index = 1; index < prefillCount; index++) {
                String ip = fleetIp(1, index);
                int port = 20_080;
                WorkerStatus status = initializedStatus(
                        RoleType.PREFILL, ip, port);
                PrefillEndpoint endpoint = (PrefillEndpoint)
                        publishEndpoint(
                                runtime.endpointRegistry(),
                                RoleType.PREFILL,
                                ip + ":" + port,
                                status);
                prefillEndpoints.add(endpoint);
            }
            for (int index = 1; index < decodeCount; index++) {
                String ip = fleetIp(2, index);
                int port = 20_081;
                WorkerStatus status = initializedStatus(
                        RoleType.DECODE, ip, port);
                DecodeEndpoint endpoint = (DecodeEndpoint)
                        publishEndpoint(
                                runtime.endpointRegistry(),
                                RoleType.DECODE,
                                ip + ":" + port,
                                status);
                runtime.applyStatus(
                        status,
                        decodeRunningStatus(2L, decodeRunningCount));
                decodeStatuses.add(status);
                decodeEndpoints.add(endpoint);
            }
        }

        private void publishDecodeStatusStorm(
                int updateCount,
                int runningCount) throws Exception {
            if (updateCount < 0) {
                throw new IllegalArgumentException(
                        "update count cannot be negative");
            }
            int threads = Math.min(32, Math.max(1, decodeStatuses.size()));
            try (ExecutorService executor =
                         Executors.newFixedThreadPool(threads)) {
                int published = 0;
                while (published < updateCount) {
                    int waveSize = Math.min(
                            decodeStatuses.size(), updateCount - published);
                    List<Future<?>> wave = new ArrayList<>(waveSize);
                    for (int index = 0; index < waveSize; index++) {
                        WorkerStatus status = decodeStatuses.get(index);
                        long version = status.appliedStatusCursor()
                                .statusVersion() + 1L;
                        wave.add(executor.submit(() -> runtime.applyStatus(
                                status,
                                decodeRunningStatus(version, runningCount))));
                    }
                    for (Future<?> update : wave) {
                        update.get(10, TimeUnit.SECONDS);
                    }
                    published += waveSize;
                }
            }
        }

        private int totalPrefillQueuedRequests() {
            return prefillEndpoints.stream()
                    .mapToInt(PrefillEndpoint::queuedRequestCount)
                    .sum();
        }

        private int totalPrefillOwnedRequests() {
            return prefillEndpoints.stream()
                    .mapToInt(PrefillEndpoint::getLocallyOwnedRequestCount)
                    .sum();
        }

        private int totalDecodeReservations() {
            return decodeEndpoints.stream()
                    .mapToInt(endpoint -> endpoint.resourceSnapshot()
                            .reserved().size())
                    .sum();
        }

        private static String fleetIp(int roleSubnet, int index) {
            return "127." + roleSubnet + "." + index / 250
                    + "." + (index % 250 + 1);
        }

        private BalanceContext context(long requestId) {
            return context(requestId, 50);
        }

        private BalanceContext context(long requestId, int priority) {
            return context(requestId, priority, 128L);
        }

        private BalanceContext context(
                long requestId, int priority, long sequenceLength) {
            Request request = new Request();
            request.setRequestId(requestId);
            request.setSeqLen(sequenceLength);
            request.setMaxNewTokens(8);
            request.setPriority(priority);
            request.setModel("transient-capacity-contract");
            BalanceContext context = new BalanceContext(config);
            context.setRequest(request);
            return context;
        }

        private void releaseCapacity() {
            if (saturatedRole == RoleType.PREFILL) {
                runtime.applyStatus(
                        prefillStatus,
                        statusResponse(RoleType.PREFILL, 3L, false));
                return;
            }
            runtime.applyStatus(
                    decodeStatus,
                    statusResponse(RoleType.DECODE, 3L, false));
        }

        @Override
        public void close() {
            submission.unblockPreparation();
            runtime.close();
        }
    }

    private static final class RecordingRouter extends DefaultRouter {
        private final DefaultRouter delegate;
        private final PlacementMetrics metrics;

        private RecordingRouter(
                DefaultRouter delegate,
                PlacementMetrics metrics) {
            super(
                    mock(CostBasedPrefillStrategy.class),
                    mock(DecodeSelector.class),
                    mock(RandomStrategy.class),
                    mock(ConfigService.class),
                    modelMeta(false));
            this.delegate = delegate;
            this.metrics = metrics;
        }

        @Override
        boolean hasBetterResidentPrefix(ScheduledRequest item, String group) {
            return delegate.hasBetterResidentPrefix(item, group);
        }

        @Override
        public PlacementResult<RouteAdmission, PlacementKey> select(BalanceContext context) {
            long startedAt = metrics.placementStarted(context);
            try {
                return delegate.select(context);
            } finally {
                metrics.placementFinished(startedAt);
            }
        }

        @Override
        public PlacementResult<RouteAdmission, PlacementKey> select(
                BalanceContext context, String policyGroup) {
            long startedAt = metrics.placementStarted(context);
            try {
                return delegate.select(context, policyGroup);
            } finally {
                metrics.placementFinished(startedAt);
            }
        }

    }

    private static final class PlacementMetrics {
        private final AtomicInteger totalAttempts = new AtomicInteger();
        private final AtomicLong placementWakeups = new AtomicLong();
        private final Map<Long, AtomicInteger> attemptsByRequest =
                new ConcurrentHashMap<>();
        private final Map<Long, BalanceContext> contexts =
                new ConcurrentHashMap<>();
        private final Map<Long, Long> deadlines = new ConcurrentHashMap<>();
        private final List<Long> placementLatenciesNanos =
                new CopyOnWriteArrayList<>();

        private long placementStarted(BalanceContext context) {
            long requestId = context.getRequestId();
            totalAttempts.incrementAndGet();
            attemptsByRequest.computeIfAbsent(
                    requestId, ignored -> new AtomicInteger())
                    .incrementAndGet();
            BalanceContext original = contexts.putIfAbsent(requestId, context);
            if (original != null && original != context) {
                throw new AssertionError(
                        "placement replaced context for request " + requestId);
            }
            SchedulingMetadata metadata = context.schedulingMetadata();
            if (metadata != null) {
                long deadline = metadata.expiresAtMs();
                Long originalDeadline = deadlines.putIfAbsent(
                        requestId, deadline);
                if (originalDeadline != null
                        && originalDeadline != deadline) {
                    throw new AssertionError(
                            "placement changed deadline for request "
                                    + requestId);
                }
            }
            return System.nanoTime();
        }

        private void placementFinished(long startedAt) {
            placementLatenciesNanos.add(System.nanoTime() - startedAt);
        }

        private void recordPlacementWakeup(PlacementKey key) {
            if (key.role() == RoleType.DECODE
                    && "capacity-contract".equals(key.group())) {
                placementWakeups.incrementAndGet();
            }
        }

        private int totalPlacementAttempts() {
            return totalAttempts.get();
        }

        private long placementWakeups() {
            return placementWakeups.get();
        }

        private void resetPlacementWakeups() {
            placementWakeups.set(0L);
        }

        private BalanceContext context(long requestId) {
            return contexts.get(requestId);
        }

        private long deadline(long requestId) {
            return deadlines.getOrDefault(requestId, -1L);
        }

        private int maxAttemptsForAnyRequest() {
            return attemptsByRequest.values().stream()
                    .mapToInt(AtomicInteger::get)
                    .max()
                    .orElse(0);
        }

        private void print(
                String scenario,
                int requests,
                int pendingDepth,
                int statusTriggeredPlacements,
                long capacityReleasedAt) {
            long[] latencies = placementLatenciesNanos.stream()
                    .mapToLong(Long::longValue)
                    .sorted()
                    .toArray();
            long p50 = percentile(latencies, 0.50);
            long p99 = percentile(latencies, 0.99);
            long recoveryLatency = Math.max(
                    0L, System.nanoTime() - capacityReleasedAt);
            System.out.printf(
                    "FlexLB placement contract: scenario=%s requests=%d "
                            + "placement_attempts=%d attempts_per_request=%.4f "
                            + "max_attempts_per_request=%d placement_wakeups=%d "
                            + "pending_depth=%d "
                            + "placement_latency_p50_us=%d "
                            + "placement_latency_p99_us=%d "
                            + "capacity_recovery_latency_us=%d "
                            + "status_triggered_placements=%d%n",
                    scenario,
                    requests,
                    totalPlacementAttempts(),
                    (double) totalPlacementAttempts() / requests,
                    maxAttemptsForAnyRequest(),
                    placementWakeups(),
                    pendingDepth,
                    TimeUnit.NANOSECONDS.toMicros(p50),
                    TimeUnit.NANOSECONDS.toMicros(p99),
                    TimeUnit.NANOSECONDS.toMicros(recoveryLatency),
                    statusTriggeredPlacements);
        }

        private static long percentile(long[] sorted, double quantile) {
            if (sorted.length == 0) {
                return 0L;
            }
            int index = (int) Math.ceil(quantile * sorted.length) - 1;
            return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
        }
    }

    private static final class QuietPlacementLogs implements AutoCloseable {
        private final Logger strategyLogger = (Logger) LoggerFactory.getLogger(
                CostBasedPrefillStrategy.class);
        private final Logger flexlbLogger = (Logger) LoggerFactory.getLogger(
                "flexlbLogger");
        private final Logger syncLogger = (Logger) LoggerFactory.getLogger(
                "syncLogger");
        private final Level previousStrategyLevel = strategyLogger.getLevel();
        private final Level previousFlexlbLevel = flexlbLogger.getLevel();
        private final Level previousSyncLevel = syncLogger.getLevel();

        private QuietPlacementLogs() {
            strategyLogger.setLevel(Level.ERROR);
            flexlbLogger.setLevel(Level.ERROR);
            syncLogger.setLevel(Level.ERROR);
        }

        @Override
        public void close() {
            strategyLogger.setLevel(previousStrategyLevel);
            flexlbLogger.setLevel(previousFlexlbLevel);
            syncLogger.setLevel(previousSyncLevel);
        }
    }

    private static FlexlbConfig config() {
        FlexlbConfig config = org.flexlb.mock.TestFlexlbConfigs.create();
        config.queueScheduler().setOrdering(QueueOrderingConfig.priority());
        config.fixedWindowDecision().setMaxRequests(1);
        config.fixedWindowDecision().setMaxCollectionWaitMs(1L);
        ((DispatcherConfig) config.getDispatcher())
                .setMaxInflightPerPrefillWorker(2);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(1L);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxKvUsagePercent(100L);
        return config;
    }

    private static ModelMetaConfig modelMeta(boolean prefillFirst) {
        ModelMetaConfig meta = mock(
                ModelMetaConfig.class, withSettings().stubOnly());
        when(meta.requiredRoles()).thenReturn(
                prefillFirst
                        ? List.of(RoleType.PREFILL, RoleType.DECODE)
                        : List.of(RoleType.DECODE, RoleType.PREFILL));
        return meta;
    }

    private static WorkerStatus initializedStatus(
            RoleType role, String ip, int port) {
        return WorkerStatus.createDiscovered(
                role, "capacity-contract", ip, port, port + 1, null);
    }

    private static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry,
            RoleType role,
            String address,
            WorkerStatus status) {
        return publishEndpoint(registry, address, status, statusResponse(role, 1L, false));
    }

    private static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry, String address, WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = WorkerStatusTestFixtures.prepare(status, WorkerStatusTestFixtures.observation(status, response));
            WorkerEndpoint endpoint = registry
                    .publishPreparedEndpoint(address, status, prepared)
                    .endpoint();
            return endpoint;
        } finally {
            status.lock.unlock();
        }
    }

    private static DecodeEndpoint publishDecode(Fixture fixture, String ip, long availableKv, boolean busy) {
        WorkerStatusResponse response = statusResponse(RoleType.DECODE, 1L, busy);
        response.setAvailableKvCacheTokens(availableKv);
        return (DecodeEndpoint) publishEndpoint(fixture.runtime.endpointRegistry(), ip + ":18081",
                initializedStatus(RoleType.DECODE, ip, 18081), response);
    }

    private static WorkerStatusResponse statusResponse(
            RoleType role, long version, boolean saturated) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(version);
        response.setLatestFinishedVersion(0L);
        response.setAvailableKvCacheTokens(1_000_000L);
        response.setTotalKvCacheTokens(2_000_000L);
        response.setMaxSeqLen(1_000_000L);
        response.setMaxBatchTokensSize(1_000_000L);
        if (saturated) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(Fixture.EXTERNAL_REQUEST_ID);
            task.setPhase(role == RoleType.PREFILL
                    ? TaskPhase.PENDING : TaskPhase.RUNNING);
            task.setInputLength(128L);
            response.setRunningTaskInfo(Map.of(
                    Long.toString(Fixture.EXTERNAL_REQUEST_ID), task));
            response.setRunningQueryLen(role == RoleType.DECODE ? 1L : 0L);
            response.setWaitingQueryLen(role == RoleType.PREFILL ? 1L : 0L);
        } else {
            response.setRunningTaskInfo(Map.of());
        }
        response.setFinishedTaskInfo(Map.of());
        return response;
    }

    private static WorkerStatusResponse decodeRunningStatus(
            long version, int runningCount) {
        WorkerStatusResponse response =
                statusResponse(RoleType.DECODE, version, false);
        Map<String, TaskInfo> running = new LinkedHashMap<>();
        for (int index = 0; index < runningCount; index++) {
            long requestId = Fixture.EXTERNAL_REQUEST_ID + index;
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setPhase(TaskPhase.RUNNING);
            task.setInputLength(128L);
            running.put(Long.toString(requestId), task);
        }
        response.setRunningTaskInfo(running);
        response.setRunningQueryLen((long) runningCount);
        return response;
    }

    private static WorkerStatusResponse prefillRunningStatus(
            long version, List<Long> requestIds) {
        WorkerStatusResponse response =
                statusResponse(RoleType.PREFILL, version, false);
        Map<String, TaskInfo> running = new LinkedHashMap<>();
        for (long requestId : requestIds) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setPhase(TaskPhase.PENDING);
            task.setInputLength(128_000L);
            running.put(Long.toString(requestId), task);
        }
        response.setRunningTaskInfo(running);
        response.setWaitingQueryLen((long) requestIds.size());
        return response;
    }

    private static WorkerStatusResponse
            prefillStatusWithFinishedAndUnknownRunning(
                    long version,
                    List<Long> finishedRequestIds,
                    int unknownRunningCount) {
        WorkerStatusResponse response =
                statusResponse(RoleType.PREFILL, version, false);
        Map<String, TaskInfo> finished = new LinkedHashMap<>();
        for (long requestId : finishedRequestIds) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setPhase(TaskPhase.RUNNING);
            task.setInputLength(128_000L);
            finished.put(Long.toString(requestId), task);
        }
        Map<String, TaskInfo> running = new LinkedHashMap<>();
        for (int index = 0; index < unknownRunningCount; index++) {
            long requestId = Fixture.EXTERNAL_REQUEST_ID + index;
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setPhase(TaskPhase.RUNNING);
            task.setInputLength(128_000L);
            running.put(Long.toString(requestId), task);
        }
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        response.setRunningQueryLen((long) unknownRunningCount);
        return response;
    }

    private static final class RecordingSubmissionPort {
        private final List<List<ScheduledRequest>> submittedItems =
                new CopyOnWriteArrayList<>();
        private final Map<Long, Long> batchIds = new ConcurrentHashMap<>();
        private final Semaphore commandSignals = new Semaphore(0);
        private final Semaphore preparationSignals = new Semaphore(0);
        private final Semaphore preparationReleases = new Semaphore(0);
        private final AtomicBoolean holdCompletions = new AtomicBoolean();
        private final AtomicBoolean blockPreparation = new AtomicBoolean();

        private CapacityBoundary.Attempt<
                BatchDeliveryStrategy.PreparedSubmission>
                tryPrepareSubmission() {
            if (blockPreparation.get()) {
                preparationSignals.release();
                preparationReleases.acquireUninterruptibly();
            }
            return CapacityBoundary.Attempt.accepted(
                    new BatchDeliveryStrategy.PreparedSubmission() {
                        private boolean submitted;

                        @Override
                        public void submit(BatchDeliveryStrategy.Delivery delivery) {
                            if (submitted) {
                                throw new IllegalStateException(
                                        "prepared submission reused");
                            }
                            submitted = true;
                            delivery.run((exactItems, batchId, predictedMs, decisionReason, observer) -> {
                                for (ScheduledRequest item : exactItems) {
                                    batchIds.put(item.requestId(), batchId);
                                }
                                submittedItems.add(List.copyOf(exactItems));
                                commandSignals.release();
                                if (!holdCompletions.get()) {
                                    for (ScheduledRequest item : exactItems) {
                                        observer.accept(
                                                item,
                                                DeliveryResult
                                                        .delivered());
                                    }
                                }
                            });
                        }

                        @Override
                        public void close() {
                        }
                    });
        }

        private List<Long> requestIds() {
            return submittedItems.stream()
                    .flatMap(List::stream)
                    .map(ScheduledRequest::requestId)
                    .toList();
        }

        private List<String> decodeAddresses() {
            return submittedItems.stream()
                    .flatMap(List::stream)
                    .map(ScheduledRequest.class::cast)
                    .map(item -> item.decode().getServerIp() + ":"
                            + item.decode().getHttpPort())
                    .toList();
        }

        private void blockPreparation() {
            blockPreparation.set(true);
        }

        private boolean awaitPreparation(long timeout, TimeUnit unit)
                throws InterruptedException {
            return preparationSignals.tryAcquire(timeout, unit);
        }

        private void unblockPreparation() {
            blockPreparation.set(false);
            preparationReleases.release();
        }

        private void holdCompletions() {
            holdCompletions.set(true);
        }

        private boolean awaitCommands(
                int count, long timeout, TimeUnit unit)
                throws InterruptedException {
            return commandSignals.tryAcquire(count, timeout, unit);
        }
    }

}
