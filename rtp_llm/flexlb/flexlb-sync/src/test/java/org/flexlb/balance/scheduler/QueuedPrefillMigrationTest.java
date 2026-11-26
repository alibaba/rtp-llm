package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.StatusFailureReporter;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.flexlb.testfixture.WorkerStatusTestFixtures;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class QueuedPrefillMigrationTest {
    private FlexlbConfig config;
    private RequestRegistry registry;
    private EndpointRegistry endpoints;
    private PlacementAvailability placement;
    private RouteDeliveryStrategy dispatch;
    private PrefillEndpoint source;
    private PrefillEndpoint target;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        var window = SchedulingTestConfig.useFixedWindowDecision(config);
        window.setMaxRequests(16);
        window.setMaxCollectionWaitMs(60_000L);
        window.setMaxPredictedExecutionMs(0L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        var reporter = mock(BatchSchedulerReporter.class);
        registry = spy(new RequestRegistry(configService, reporter, mock(RequestSchedulerReporter.class)));
        dispatch = new RouteDeliveryStrategy(registry, new DeliveryMetrics(reporter));
        var projector = new EndpointEventProjector(registry, mock(StatusFailureReporter.class));
        placement = new PlacementAvailability();
        endpoints = new EndpointRegistry(configService, projector, reporter, mock(DeliveryStrategy.class), placement);
        source = endpoint("127.0.0.1");
        target = endpoint("127.0.0.2");
    }

    private PrefillEndpoint endpoint(String ip) {
        var worker = WorkerStatusTestFixtures.readyForScheduling(
                WorkerStatus.createDiscovered(RoleType.PREFILL, "test", ip, 8000, 8001, null));
        var status = new WorkerStatusResponse();
        status.setRole(RoleType.PREFILL);
        status.setAlive(true);
        status.setStatusVersion(1L);
        status.setLatestFinishedVersion(0L);
        worker.lock.lock();
        try {
            return (PrefillEndpoint) endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                    WorkerStatusTestFixtures.prepare(worker, WorkerStatusTestFixtures.observation(worker, status))).endpoint();
        } finally {
            worker.lock.unlock();
        }
    }

    @AfterEach
    void close() {
        registry.closeAdmissionAndAwaitMutations();
        endpoints.close();
        registry.closeOutstandingAndTerminalize();
        registry.closeExpiration();
        registry.closePublisher();
    }

    private ServerStatus status(PrefillEndpoint endpoint, long id, long hits) {
        var result = new ServerStatus();
        result.setRequestId(id);
        result.setSuccess(true);
        result.setRole(RoleType.PREFILL);
        result.setGroup("test");
        result.setServerIp(endpoint.getIp());
        result.setHttpPort(endpoint.getHttpPort());
        var debug = new DebugInfo();
        debug.setHitCacheLen(hits);
        result.setDebugInfo(debug);
        return result;
    }

    private ScheduledRequest queued(long id) {
        return queued(id, true);
    }

    private ScheduledRequest queued(long id, boolean deferDecode) {
        var context = RequestLifecycleTestSupport.context(config, id);
        var future = registry.register(context);
        context.setFuture(future);
        var from = status(source, id, 8L);
        var vit = new ServerStatus();
        vit.setRole(RoleType.VIT);
        vit.setWorkerGeneration(22L);
        var response = new Response();
        response.setServerStatus(List.of(vit, from));
        var decode = ScheduledRequest.DecodeBinding.capture(context);
        if (deferDecode) { decode = decode.defer(mock(DecodeSelector.class), "decode-group"); }
        var plan = source.planTokenWork(8L);
        var item = new ScheduledRequest(context, future, response, from, null, source, null, null,
                System.currentTimeMillis(), decode, plan);
        try (plan; var claim = registry.claimAdmissionHandle(id, future); var pin = source.tryPinGeneration()) {
            assertNotNull(claim);
            assertTrue(registry.commitItemForPublication(item, () -> source.offerPinned(pin, item)));
        }
        return item;
    }

    private SelectedRole select(PrefillEndpoint to, ScheduledRequest item) {
        return SelectedRole.prefill(to.tryPinGeneration(), status(to, item.requestId(), 12L),
                1L, to.placementVersion(), to.planTokenWork(4L));
    }

    @Test
    void transferReplacesOnlyPrefillAndMovesTheWorkOnce() {
        var old = queued(101);
        assertEquals(List.of(new PrefillQueueFlow.Entry(PrefillQueueFlow.Event.ENQUEUE, 1, 8)),
                source.queueMetricsSnapshot().flow().entries());
        assertTrue(registry.tryMigrateQueued(old, 3, () -> select(target, old)));
        var next = registry.requestSlot(101).scheduledItem();
        assertEquals(List.of(), source.captureQueueSnapshot().items());
        assertEquals(List.of(next), target.captureQueueSnapshot().items());
        assertSame(target, next.prefillEp());
        assertSame(old.future(), next.future());
        assertEquals(old.priority(), next.priority());
        assertEquals(old.expiresAtMs(), next.expiresAtMs());
        assertEquals(old.enqueuedAtMs(), next.enqueuedAtMs());
        assertEquals(old.enqueueSeq(), next.enqueueSeq());
        assertEquals(old.decodeBinding(), next.decodeBinding());
        assertEquals(12L, next.hitCache());
        assertEquals(0L, source.tokenWorkloadSnapshot().pendingTokens());
        assertEquals(4L, target.tokenWorkloadSnapshot().pendingTokens());
        assertEquals(0L, target.tokenWorkloadSnapshot().plannedTokens());
        assertEquals(List.of(new PrefillQueueFlow.Entry(PrefillQueueFlow.Event.MIGRATION_OUT, 1, 8)),
                source.queueMetricsSnapshot().flow().entries());
        assertEquals(List.of(new PrefillQueueFlow.Entry(PrefillQueueFlow.Event.MIGRATION_IN, 1, 4)),
                target.queueMetricsSnapshot().flow().entries());
        var roles = next.execution().routeResponse().getServerStatus();
        assertEquals(old.execution().routeResponse().getServerStatus().getFirst(), roles.getFirst());
        assertEquals(target.getIp(), roles.getLast().getServerIp());
        assertEquals(source.getIp(), old.execution().routeResponse().getServerStatus().getLast().getServerIp());
        assertFalse(registry.tryMigrateQueued(next, 3, () -> fail("successful moves are final")));
        assertNull(registry.requestSlot(101).tryBeginCacheReplan(next), "a Decode wake must not bounce it back");
        assertFalse(RequestLifecycleTestSupport.prepareMember(registry, old));
        assertTrue(RequestLifecycleTestSupport.prepareMember(registry, next));
    }

    @Test
    void unavailableCandidateAndBusySourceCanRetryWithoutChargingAnAttempt() {
        var item = queued(102);
        assertFalse(registry.tryMigrateQueued(item, 1, () -> null));
        assertFalse(registry.tryMigrateQueued(item, 1, () -> select(source, item)));
        try (var reservation = source.reserveBatch(item, 21L, 3).reservation()) {
            assertNotNull(reservation);
            assertFalse(registry.tryMigrateQueued(item, 1, () -> select(target, item)));
            assertTrue(registry.canMigrateQueued(item, 1));
            assertEquals(List.of(item), source.captureQueueSnapshot().items());
            assertEquals(0L, target.tokenWorkloadSnapshot().plannedTokens());
        }
        assertTrue(registry.tryMigrateQueued(item, 1, () -> select(target, item)));
    }

    @Test
    void duplicateOnTargetRestoresExactSourceAndLimitsRetries() {
        var item = queued(103);
        var foreign = new ScheduledRequest(item.ctx(), item.future(), item.execution().routeResponse(),
                status(target, 103, 0L), null, target, null, null, item.enqueuedAtMs(), item.decodeBinding());
        try (var pin = target.tryPinGeneration()) { assertTrue(target.offerPinned(pin, foreign)); }
        source.queueMetricsSnapshot();
        target.queueMetricsSnapshot();
        assertFalse(registry.tryMigrateQueued(item, 1, () -> select(target, item)));
        assertEquals(List.of(item), source.captureQueueSnapshot().items());
        assertEquals(List.of(foreign), target.captureQueueSnapshot().items());
        assertSame(item, registry.requestSlot(103).scheduledItem());
        assertEquals(8L, source.tokenWorkloadSnapshot().pendingTokens());
        assertEquals(0L, target.tokenWorkloadSnapshot().plannedTokens());
        assertEquals(List.of(
                new PrefillQueueFlow.Entry(PrefillQueueFlow.Event.MIGRATION_RESTORED, 1, 8),
                new PrefillQueueFlow.Entry(PrefillQueueFlow.Event.MIGRATION_OUT, 1, 8)),
                source.queueMetricsSnapshot().flow().entries());
        assertTrue(target.queueMetricsSnapshot().flow().entries().isEmpty());
        assertFalse(registry.tryMigrateQueued(item, 1, () -> fail("retry limit reached")));
        assertTrue(target.removeQueued(foreign, "test cleanup"));
        assertTrue(registry.tryMigrateQueued(item, 2, () -> select(target, item)));
    }

    private DeliveryStrategy.Transaction prepare(PrefillEndpoint endpoint, List<ScheduledRequest> snapshot) {
        return dispatch.prepare(snapshot, endpoint.getPredictor().evaluator(), OptionalLong.empty());
    }

    @Test
    void sourceDispatchPublishesWhileTargetSelectionIsInProgress() throws Exception {
        var item = queued(104, false);
        var selecting = new CountDownLatch(1);
        var resume = new CountDownLatch(1);
        try (var executor = Executors.newSingleThreadExecutor()) {
            var migration = executor.submit(() -> registry.tryMigrateQueued(item, 2, () -> {
                selecting.countDown();
                RequestLifecycleTestSupport.await(resume);
                return select(target, item);
            }));
            assertTrue(selecting.await(3, TimeUnit.SECONDS));
            try (var transaction = prepare(source, source.captureQueueSnapshot().items())) {
                assertEquals(List.of(item), transaction.items());
                transaction.handoff("test", 0, transaction.commitUnderLock());
                assertEquals(source.getIp(), item.future().get(2, TimeUnit.SECONDS).getServerStatus().getLast().getServerIp());
            } finally {
                resume.countDown();
            }
            assertFalse(migration.get(3, TimeUnit.SECONDS));
        }
        assertEquals(0, target.queuedRequestCount());
        assertEquals(0L, target.tokenWorkloadSnapshot().plannedTokens());
    }

    @Test
    void committedSourceStillPublishesIfMigrationRacesItsHandoff() throws Exception {
        var item = queued(109, false);
        try (var transaction = prepare(source, source.captureQueueSnapshot().items())) {
            var preceding = transaction.commitUnderLock();
            assertTrue(registry.canMigrateQueued(item, 1), "slot still sees QUEUED until the delivery handoff");
            assertFalse(registry.tryMigrateQueued(item, 1, () -> select(target, item)),
                    "the source ledger has already committed this identity");
            transaction.handoff("test", 0, preceding);
        }
        assertEquals(source.getIp(), item.future().get(2, TimeUnit.SECONDS).getServerStatus().getLast().getServerIp());
        assertEquals(0, target.queuedRequestCount());
    }

    // Hold the real queue lock so migration has its slot handle but cannot take the source.
    private ReentrantLock sourceQueueLock() {
        Object runtime = ReflectionTestUtils.getField(source, "runtime");
        return (ReentrantLock) ReflectionTestUtils.getField(runtime, "queueLock");
    }

    private CountDownLatch fireSchedulingDeadlineAfterDeliveryClaim(RequestSlot slot) {
        var deadline = (ExpirationTimer.RequestDeadline) ReflectionTestUtils.getField(slot, "requestDeadline");
        assertNotNull(deadline);
        var fired = new CountDownLatch(1);
        doAnswer(call -> {
            assertEquals(DeliveryClaimKind.ROUTE_DECISION, slot.snapshot().deliveryClaimKind());
            slot.onSchedulingDeadline(deadline);
            fired.countDown();
            return call.callRealMethod();
        }).when(registry).publishRoute(any(), any(), anyLong());
        return fired;
    }

    @Test
    void sourceClaimStillWinsSchedulingDeadlineWhileMigrationHandleIsOpen() throws Exception {
        var item = queued(112, false);
        var fired = fireSchedulingDeadlineAfterDeliveryClaim(registry.requestSlot(112));
        try (var executor = Executors.newFixedThreadPool(2);
                var transaction = prepare(source, source.captureQueueSnapshot().items())) {
            var preceding = transaction.commitUnderLock();
            var lock = sourceQueueLock();
            lock.lock();
            var migration = executor.submit(() -> registry.tryMigrateQueued(item, 1, () -> select(target, item)));
            try {
                RequestLifecycleTestSupport.awaitCondition(() -> !registry.canMigrateQueued(item, 1));
                executor.submit(() -> transaction.handoff("test", 0, preceding));
                assertTrue(fired.await(3, TimeUnit.SECONDS));
            } finally {
                lock.unlock();
            }
            assertFalse(migration.get(3, TimeUnit.SECONDS));
            assertTrue(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        }
        assertEquals(0, target.queuedRequestCount());
    }

    @Test
    void targetClaimStillWinsSchedulingDeadlineBeforeMigrationFinishes() throws Exception {
        var item = queued(113, false);
        var fired = fireSchedulingDeadlineAfterDeliveryClaim(registry.requestSlot(113));
        // Source capacity is announced after target publication, inside the migration handle.
        PlacementAvailability.Listener deliverFromTarget = event -> {
            if (!source.ipPort().equals(event.key().endpoint())) { return; }
            try (var transaction = prepare(target, target.captureQueueSnapshot().items())) {
                transaction.handoff("test", 0, transaction.commitUnderLock());
            }
        };
        placement.addListener(deliverFromTarget);
        try {
            assertTrue(registry.tryMigrateQueued(item, 1, () -> select(target, item)));
            assertTrue(fired.await(2, TimeUnit.SECONDS));
            assertTrue(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        } finally {
            placement.removeListener(deliverFromTarget);
        }
    }

    @Test
    void blockedDecodePreemptionRetriesOnceAfterExactSourceIsRestored() throws Exception {
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_ENGINE_OWNED);
        var item = queued(114);
        assertTrue(item.allowsDecodeEnginePreemption());
        var queue = mock(GlobalQueueCoordinator.class);
        registry.attachGlobalQueue(queue);
        var foreign = new ScheduledRequest(item.ctx(), item.future(), item.execution().routeResponse(),
                status(target, 114, 0L), null, target, null, null, item.enqueuedAtMs(), item.decodeBinding());
        try (var pin = target.tryPinGeneration()) { assertTrue(target.offerPinned(pin, foreign)); }
        try (var executor = Executors.newSingleThreadExecutor()) {
            var lock = sourceQueueLock();
            lock.lock();
            var migration = executor.submit(() -> registry.tryMigrateQueued(item, 2, () -> select(target, item)));
            try {
                RequestLifecycleTestSupport.awaitCondition(() -> !registry.canMigrateQueued(item, 2));
                registry.tryPreemptForDispatch(item);
                registry.tryPreemptForDispatch(item);
                verify(queue, never()).preemptForDispatch(item);
            } finally {
                lock.unlock();
            }
            assertFalse(migration.get(3, TimeUnit.SECONDS));
        }
        assertSame(item, registry.requestSlot(114).scheduledItem());
        queued(115);
        verify(queue).preemptForDispatch(item);
        assertTrue(target.removeQueued(foreign, "test cleanup"));
    }

    @Test
    void staleSourceSnapshotCannotDispatchAfterMigrationAndOnlyTargetPublishes() throws Exception {
        var item = queued(110, false);
        var oldSnapshot = source.captureQueueSnapshot().items();
        assertEquals(List.of(item), oldSnapshot);
        assertTrue(registry.tryMigrateQueued(item, 1, () -> select(target, item)));
        try (var stale = prepare(source, oldSnapshot)) {
            assertTrue(stale.items().isEmpty());
            assertSame(CapacityBoundary.OWNERSHIP_LOST, stale.blockedResult());
        }
        assertFalse(item.future().isDone());
        var targetSnapshot = target.captureQueueSnapshot().items();
        try (var transaction = prepare(target, targetSnapshot)) {
            assertEquals(targetSnapshot, transaction.items());
            transaction.handoff("test", 0, transaction.commitUnderLock());
        }
        assertEquals(target.getIp(), item.future().get(2, TimeUnit.SECONDS).getServerStatus().getLast().getServerIp());
        assertEquals(0, source.observedRequestCount());
        assertEquals(1, target.observedRequestCount());
    }

    @Test
    void expiryWhileChoosingTargetLeavesNoMigratedRoute() throws Exception {
        var item = queued(111);
        assertFalse(registry.tryMigrateQueued(item, 1, () -> {
            registry.onQueuedItemExpired(item);
            return select(target, item);
        }));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(0, source.queuedRequestCount());
        assertEquals(0, target.queuedRequestCount());
        assertEquals(0L, target.tokenWorkloadSnapshot().plannedTokens());
    }

    /** Inject publication races after the exact source item has left its real endpoint. */
    private SelectedRole selectedForPublicationRace(PrefillEndpoint receiver, ScheduledRequest item) {
        var pin = mock(WorkerEndpoint.GenerationPin.class);
        when(pin.endpoint()).thenReturn(receiver);
        var selection = mock(SelectedRole.class);
        when(selection.serverStatus()).thenReturn(status(target, item.requestId(), 12L));
        when(selection.takeGenerationPin()).thenReturn(pin);
        return selection;
    }

    @Test
    void sourceCannotRestoreReturnsTheSameLiveRequestToGlobalPlacement() {
        var item = queued(106);
        var queue = mock(GlobalQueueCoordinator.class);
        registry.attachGlobalQueue(queue);
        when(queue.requeue(item)).thenAnswer(call -> {
            try (var handle = registry.claimAdmissionHandle(106, item.future())) {
                assertNotNull(handle, "normal placement can resume immediately");
            }
            return true;
        });
        var receiver = mock(PrefillEndpoint.class);
        var intervening = new ScheduledRequest(item.ctx(), item.future(), item.execution().routeResponse(),
                item.prefill(), null, source, null, null, item.enqueuedAtMs(), item.decodeBinding());
        when(receiver.offerPinned(any(), any(), eq(PrefillQueueFlow.Event.MIGRATION_IN))).thenAnswer(call -> {
            try (var pin = source.tryPinGeneration()) { assertTrue(source.offerPinned(pin, intervening)); }
            return false;
        });
        assertFalse(registry.tryMigrateQueued(item, 3, () -> selectedForPublicationRace(receiver, item)));
        verify(queue).requeue(item);
        assertFalse(item.future().isDone());
        assertNull(registry.requestSlot(106).scheduledItem());
        assertTrue(source.removeQueued(intervening, "test cleanup"));
    }

    @Test
    void cancellationDuringDestinationPublicationSettlesOriginalFuture() throws Exception {
        var item = queued(107);
        var receiver = mock(PrefillEndpoint.class);
        when(receiver.offerPinned(any(), any(), eq(PrefillQueueFlow.Event.MIGRATION_IN))).thenAnswer(call -> {
            registry.cancelRequest(107, 0L, CancelReason.CLIENT_CANCELLED);
            return false;
        });
        assertFalse(registry.tryMigrateQueued(item, 2, () -> selectedForPublicationRace(receiver, item)));
        assertFalse(item.future().get(2, java.util.concurrent.TimeUnit.SECONDS).isSuccess());
        assertEquals(0, source.queuedRequestCount());
        assertEquals(0L, source.tokenWorkloadSnapshot().pendingTokens());
    }

    @Test
    void boundDecodeDoesNotEnterTargetSelection() {
        var item = queued(108);
        var original = item.decodeBinding();
        item.execution().commitDecodeBinding(original.bind(new ServerStatus(), mock(DecodeEndpoint.class), null));
        try {
            assertFalse(registry.tryMigrateQueued(item, 1, () -> fail("bound Decode remains on its original route")));
        } finally {
            item.execution().commitDecodeBinding(original);
        }
    }

    @Test
    void pendingDecodePreemptionAndCancelledRequestSkipTargetSelection() {
        var item = queued(105);
        var wait = registry.requestSlot(105).beginDispatchPreemption(item);
        assertNotNull(wait);
        assertFalse(registry.tryMigrateQueued(item, 2, () -> fail("pending preemption")));
        item.finishDecodePreemption(wait);
        assertTrue(registry.canMigrateQueued(item, 2));
        registry.cancelRequest(105, 0L, CancelReason.CLIENT_CANCELLED);
        assertFalse(registry.tryMigrateQueued(item, 2, () -> fail("cancelled request")));
    }
}
