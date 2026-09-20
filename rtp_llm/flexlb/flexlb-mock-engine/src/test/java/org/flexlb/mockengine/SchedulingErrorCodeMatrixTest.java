package org.flexlb.mockengine;

import io.grpc.stub.StreamObserver;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.RouteService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.flexlb.dao.loadbalance.AdmissionRejectReason.*;
import static org.flexlb.enums.LoadBalanceStrategyEnum.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * FlexLB UT: real RouteService / Router / resource filters / scheduler / dispatcher;
 * only configuration, cache lookup, metrics and gRPC transport are stand-ins.
 * Transport invokes the in-process Mock Engine; no external service or port binding.
 *
 * <p>Expected values below are FlexLB Response.code, NOT HTTP status or gRPC status.
 * DIRECT and NON_BATCH deliver routes; only BATCH calls Engine.EnqueueBatch.
 * Engine-owned victim ACK/terminal cases are in {@link PreemptionPhasesE2ETest} and
 * {@link MockEngineCancelChannelTest}; they must not be replaced with a mocked 8429 response.
 */
@Timeout(15)
class SchedulingErrorCodeMatrixTest {
    private static final long REQUEST_ID = 71001;
    private static final long BLOCKER_ID = 71000;
    private static final String CAPACITY_MESSAGE = "admission capacity is temporarily exhausted";

    static Stream<String> modes() {
        return Stream.of("DIRECT", "FIFO_NON_BATCH", "PRIORITY_NON_BATCH", "FIFO_BATCH", "PRIORITY_BATCH");
    }

    static Stream<String> queuedModes() {
        return modes().filter(mode -> !mode.equals("DIRECT"));
    }

    static Stream<Arguments> routingFailures() {
        return modes().flatMap(mode -> Stream.of(
                // Unhealthy P/D: 8402/8403 + UNSPECIFIED; not capacity exhaustion.
                Arguments.of(mode, "prefill unhealthy", 8402, UNSPECIFIED, "NOT_ALIVE"),
                Arguments.of(mode, "decode unhealthy", 8403, UNSPECIFIED, "NOT_ALIVE"),
                // Healthy P with full request capacity: 8431 + RESOURCE_EXHAUSTED.
                Arguments.of(mode, "prefill full", 8431, RESOURCE_EXHAUSTED, CAPACITY_MESSAGE),
                // Engine slot owners with trusted higher/same priority: 8430 + corresponding reason.
                Arguments.of(mode, "decode higher", 8430, HIGHER_PRIORITY_AHEAD, "higher-priority requests are ahead"),
                Arguments.of(mode, "decode same", 8430, SAME_PRIORITY_AHEAD, "same-priority requests are ahead"),
                // Only lower-priority owners, preemption disabled: 8431 + RESOURCE_EXHAUSTED.
                Arguments.of(mode, "decode lower", 8431, RESOURCE_EXHAUSTED, CAPACITY_MESSAGE),
                // Unknown owner priority: 8432 + UNSPECIFIED, never invent a priority relationship.
                Arguments.of(mode, "decode unknown", 8432, UNSPECIFIED,
                        "admission unavailable; blocker priority attribution is unavailable")
        ));
    }

    @ParameterizedTest(name = "{0}: {1} -> {2}/{3}")
    @MethodSource("routingFailures")
    void routingFailurePreservesCodeReasonAndEvidence(String mode, String condition, int code,
                                                      AdmissionRejectReason reason, String message) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            switch (condition) {
                case "prefill unhealthy" -> f.h.prefillEndpoint(0).getStatus().setAlive(false);
                case "decode unhealthy" -> f.h.decodeEndpoint(0).getStatus().setAlive(false);
                case "prefill full" -> {
                    // ResourceMeasure takes its threshold at construction; 100 live owners saturate it.
                    for (long id = 1; id <= 100; id++) {
                        assertTrue(f.h.prefillEndpoint(0).tryCommitRequest(id, 60_000, 100));
                    }
                }
                case "decode higher" -> f.h.decodeEndpoint(0).reserve(BLOCKER_ID, 0, 0, 70);
                case "decode same" -> f.h.decodeEndpoint(0).reserve(BLOCKER_ID, 0, 0, 50);
                case "decode lower" -> f.h.decodeEndpoint(0).reserve(BLOCKER_ID, 0, 0, 30);
                case "decode unknown" -> f.h.decodeEndpoint(0).reserve(BLOCKER_ID, 0, 0, 0);
                default -> throw new AssertionError(condition);
            }
            BalanceContext ctx = f.h.context(REQUEST_ID, 50);
            assertFailure(f.route(ctx), code, reason, message);
            assertNotNull(ctx.getSchedulingDiagnostics(), "failure must retain PV evidence");
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID),
                    "failed placement must roll back only the incoming reservation");
            assertTrue(f.h.engineArrivalOrder.isEmpty(), "rejected admission must never reach Engine");
        }
    }

    @ParameterizedTest(name = "{0}: healthy -> 200")
    @MethodSource("modes")
    void healthyPlacementAndDispatch(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            Response response = f.route(f.h.context(REQUEST_ID, 50));
            assertTrue(response.isSuccess(), response.getErrorMessage());
            assertEquals(200, response.getCode());
            assertEquals(UNSPECIFIED, response.getAdmissionRejectReason());
            if (f.h.config.isBatchDispatch()) {
                assertEquals(List.of(REQUEST_ID), f.h.engineArrivalOrder);
                assertEquals(1, f.h.prefillEngines.getFirst().getAcceptedCount());
            } else {
                assertTrue(f.h.engineArrivalOrder.isEmpty(), "frontend owns Engine submission in this mode");
            }
        }
    }

    @ParameterizedTest(name = "{0}: Engine KV pressure with no owner provenance -> 8432")
    @MethodSource("modes")
    void engineReportedKvPressureIsNotMissingWorker(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            var engine = f.h.decodeEngines.getFirst();
            engine.setFaultConfig(FaultInjectionConfig.builder()
                    .kvPressureTokens(engine.getTotalKvTokens()).build());
            f.h.pumpOnce(); // Real Engine WorkerStatus -> DecodeEndpoint, not a mocked rejection.
            assertEquals(0, f.h.decodeEndpoint(0).realKvAvailable());
            assertTrue(f.h.decodeEndpoint(0).getStatus().isAlive());
            // 8432/UNSPECIFIED: KV is exhausted but the owners' priorities are unknown.
            assertFailure(f.route(f.h.context(REQUEST_ID, 50)), 8432, UNSPECIFIED,
                    "admission unavailable; blocker priority attribution is unavailable");
            engine.clearFaultConfig();
            f.h.pumpOnce();
            assertTrue(f.route(f.h.context(REQUEST_ID + 1, 50)).isSuccess(), "capacity recovery must restore admission");
        }
    }

    @ParameterizedTest(name = "{0}: deadline before placement")
    @MethodSource("modes")
    void expiredBeforePlacementDoesNotContactEngine(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            BalanceContext ctx = f.h.context(REQUEST_ID, 50);
            ctx.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() - 1));
            // DIRECT retains 8511/UNSPECIFIED; QUEUE reports 8431/RESOURCE_EXHAUSTED before placement.
            if (f.h.config.isDirect()) {
                assertFailure(f.route(ctx), 8511, UNSPECIFIED, "BATCH_SLO_EXPIRED");
            } else {
                assertFailure(f.route(ctx), 8431, RESOURCE_EXHAUSTED, "context=before placement");
            }
            assertTrue(f.h.engineArrivalOrder.isEmpty());
            assertEquals(0, f.h.decodeEndpoint(0).getInflightCount());
        }
    }

    @ParameterizedTest(name = "{0}: duplicate request_id -> 8406")
    @MethodSource("queuedModes")
    void duplicateDoesNotReplaceOriginalOwner(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.holdPrefillDelivery();
            CompletableFuture<Response> original = f.service.route(f.h.context(REQUEST_ID, 50));
            // 8406/UNSPECIFIED: duplicate rejection must not complete/release the original request.
            assertFailure(f.route(f.h.context(REQUEST_ID, 70)), 8406, UNSPECIFIED,
                    "duplicate request_id: " + REQUEST_ID);
            assertFalse(original.isDone());
            assertTrue(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
        }
    }

    @ParameterizedTest(name = "{0}: global admission full -> 8431")
    @MethodSource("queuedModes")
    void globalCapacityRejectsBeforeRouting(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.holdPrefillDelivery();
            f.h.config.queueScheduler().getCapacity().setMaxOutstandingRequestsGlobal(1);
            var first = f.service.route(f.h.context(REQUEST_ID, 50));
            // 8431/RESOURCE_EXHAUSTED, independent of ordering and dispatch mode.
            assertFailure(f.route(f.h.context(REQUEST_ID + 1, 70)), 8431, RESOURCE_EXHAUSTED,
                    "master outstanding capacity exhausted");
            assertFalse(first.isDone());
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID + 1));
        }
    }

    @ParameterizedTest(name = "{0}: Prefill queue wait at SLO -> 8431")
    @MethodSource("queuedModes")
    void sloUsesQueuePrefillCause(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.holdPrefillDelivery();
            BalanceContext ctx = f.h.context(REQUEST_ID, 50);
            ctx.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 1_000));
            // 8431/RESOURCE_EXHAUSTED; BATCH collection vs NON_BATCH request-slot cause comes from the loop.
            String cause = f.h.config.isBatchDispatch()
                    ? "batch collection window exceeded request scheduling budget" : "prefill request slots exhausted";
            assertFailure(f.route(ctx), 8431, RESOURCE_EXHAUSTED, cause);
            assertTrue(ctx.getSchedulingDiagnostics().get("cause").toString().contains(cause));
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
            assertTrue(f.h.engineArrivalOrder.isEmpty());
        }
    }

    @ParameterizedTest(name = "{0}: Engine enqueue rejection -> 8510")
    @ValueSource(strings = {"FIFO_BATCH", "PRIORITY_BATCH"})
    void engineEnqueueRejectionPreservesDiagnostic(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.h.prefillEngines.getFirst().setFaultConfig(FaultInjectionConfig.builder()
                    .failOnEnqueue(true).enqueueErrorMessage("test enqueue refused").build());
            // 8510/UNSPECIFIED: real EnqueueBatch rejection, not missing-worker or SLO classification.
            assertFailure(f.route(f.h.context(REQUEST_ID, 50)), 8510, UNSPECIFIED, "test enqueue refused");
            assertEquals(List.of(REQUEST_ID), f.h.engineArrivalOrder);
            assertEquals(0, f.h.prefillEngines.getFirst().getAcceptedCount());
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
        }
    }

    @ParameterizedTest(name = "{0}: missing generate_input -> 8406")
    @ValueSource(strings = {"FIFO_BATCH", "PRIORITY_BATCH"})
    void missingBatchInputIsInvalidRequest(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            var ctx = f.h.context(REQUEST_ID, 50);
            ctx.setGenerateInputPbBytes(null);
            // 8406/UNSPECIFIED, before reservation and dispatch.
            assertFailure(f.route(ctx), 8406, UNSPECIFIED, "missing serialized generate_input");
            assertTrue(f.h.engineArrivalOrder.isEmpty());
            assertEquals(0, f.h.decodeEndpoint(0).getInflightCount());
        }
    }

    static Stream<Arguments> fullQueues() {
        return queuedModes().flatMap(mode -> Stream.of(
                // Full queue: higher/same owner -> 8430; lower owner -> 8431 (preemption disabled).
                Arguments.of(mode, 70, 8430, HIGHER_PRIORITY_AHEAD, "higher-priority requests are ahead"),
                Arguments.of(mode, 50, 8430, SAME_PRIORITY_AHEAD, "same-priority requests are ahead"),
                Arguments.of(mode, 30, 8431, RESOURCE_EXHAUSTED, CAPACITY_MESSAGE)
        ));
    }

    @ParameterizedTest(name = "{0}: queue full behind P{1} -> {2}/{3}")
    @MethodSource("fullQueues")
    void fullQueueUsesItsChargedOwners(String mode, int blockerPriority, int code,
                                     AdmissionRejectReason reason, String message) throws Exception {
        try (Fixture f = new Fixture(mode, 10_000)) {
            f.holdPrefillDelivery();
            f.h.config.queueScheduler().getLifecycle().setMaxDeliveredNotAcceptedRequestsGlobal(10_000);
            int capacity;
            if (f.h.config.isBatchDispatch()) {
                f.h.config.batchDispatcher().setMaxWaitingRequestsPerPrefillWorker(1);
                capacity = 1;
            } else {
                // NON_BATCH has an internal fixed bound; exercise it without rewriting final fields.
                capacity = f.h.config.getInternalRuntime().getNonBatchWaitingRequestsPerPrefillWorker();
            }
            var original = f.service.route(f.h.context(REQUEST_ID, blockerPriority));
            for (int i = 1; i < capacity; i++) {
                var queued = f.service.route(f.h.context(REQUEST_ID + i, blockerPriority));
                assertFalse(queued.isDone(), () -> "queue filled early: " + queued.join());
            }
            var incoming = f.h.context(REQUEST_ID + capacity, 50);
            assertFailure(f.route(incoming), code, reason, message);
            assertFalse(original.isDone());
            assertTrue(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(incoming.getRequestId()));
            assertEquals("prefill queue capacity exhausted", incoming.getSchedulingDiagnostics().get("cause"),
                    "terminal publication must not overwrite tryOffer's failure evidence");
            var queue = (Map<?, ?>) ((List<?>) incoming.getSchedulingDiagnostics().get("prefill")).getFirst();
            assertEquals(capacity, queue.get("queueDepth"));
            assertEquals(blockerPriority > 50 ? capacity : 0, queue.get("higherPriorityCount"));
            assertEquals(blockerPriority == 50 ? capacity : 0, queue.get("samePriorityCount"));
            assertEquals(blockerPriority < 50 ? capacity : 0, queue.get("lowerPriorityCount"));
            assertEquals(1, ((List<?>) incoming.getSchedulingDiagnostics().get("decode")).size());
            assertTrue(f.h.engineArrivalOrder.isEmpty());
        }
    }

    @ParameterizedTest(name = "{0}: Decode becomes full after admission; SLO -> 8431")
    @MethodSource("queuedModes")
    void sloUsesQueueDecodeCauseWithoutReclassifyingPriority(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            var realRoute = f.h.routeFn;
            f.h.routeFn = ctx -> {
                var selected = realRoute.apply(ctx);
                assertTrue(selected.isSuccess());
                // Reproduce a concurrent admission taking the slot AFTER successful routing.
                // P70 would yield 8430 on fresh admission; this already-queued request uses its queue cause.
                f.h.decodeEndpoint(0).reserve(BLOCKER_ID, 0, 0, 70);
                return selected;
            };
            var ctx = f.h.context(REQUEST_ID, 50);
            ctx.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 1_000));
            // 8431/RESOURCE_EXHAUSTED + recorded Decode wait; never generic 8511 or a fresh 8430.
            assertFailure(f.route(ctx), 8431, RESOURCE_EXHAUSTED, "decode engine slots exhausted");
            assertTrue(ctx.getSchedulingDiagnostics().get("cause").toString().contains("decode engine slots exhausted"));
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
            assertTrue(f.h.decodeEndpoint(0).reservedView().containsKey(BLOCKER_ID));
            assertTrue(f.h.engineArrivalOrder.isEmpty());
        }
    }

    static Stream<Arguments> defaultPriorities() {
        return modes().flatMap(mode -> Stream.of(0, -1, 101).map(priority -> Arguments.of(mode, priority)));
    }

    @ParameterizedTest(name = "{0}: absent/invalid priority {1} normalizes before classification")
    @MethodSource("defaultPriorities")
    void absentOrInvalidPriorityDoesNotMeanNoWorker(String mode, int rawPriority) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            var ctx = f.h.context(REQUEST_ID, rawPriority);
            ctx.setSchedulingMetadata(SchedulingMetadata.of(rawPriority, null, System.currentTimeMillis() + 30_000, 50));
            assertEquals(50, ctx.getPriority());
            f.h.decodeEndpoint(0).reserve(BLOCKER_ID, 0, 0, 70);
            // 8430/HIGHER_PRIORITY_AHEAD internally, regardless of dash-sc's separate external QoS mapping.
            assertFailure(f.route(ctx), 8430, HIGHER_PRIORITY_AHEAD, "higher-priority requests are ahead");
        }
    }

    @ParameterizedTest(name = "{0}: request cannot fit total Decode KV -> 8431")
    @MethodSource("modes")
    void requestLargerThanTotalKvIsCapacityFailure(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.h.setDecodeKvCapacity(0, 64, 64);
            // 8431/RESOURCE_EXHAUSTED: 128 input tokens cannot fit 64 total tokens, even with no owners.
            assertFailure(f.route(f.h.context(REQUEST_ID, 50)), 8431, RESOURCE_EXHAUSTED, CAPACITY_MESSAGE);
            assertEquals(0, f.h.decodeEndpoint(0).getInflightCount());
        }
    }

    @ParameterizedTest(name = "{0}: client cancellation while queued -> 8504")
    @MethodSource("queuedModes")
    void clientCancelIsNotPriorityPreemption(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.holdPrefillDelivery();
            var future = f.service.route(f.h.context(REQUEST_ID, 50));
            f.service.cancelRequest(REQUEST_ID, 0, CancelReason.CLIENT_CANCELLED);
            // 8504/UNSPECIFIED: explicit client cancellation; no Engine dispatch and no invented 8429.
            assertFailure(future.get(5, TimeUnit.SECONDS), 8504, UNSPECIFIED, "cancel");
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
            assertTrue(f.h.engineArrivalOrder.isEmpty());
        }
    }

    @ParameterizedTest(name = "{0}: malformed protobuf -> 8510")
    @ValueSource(strings = {"FIFO_BATCH", "PRIORITY_BATCH"})
    void malformedBatchInputFailsBeforeRpc(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            var ctx = f.h.context(REQUEST_ID, 50);
            ctx.setGenerateInputPbBytes(new byte[]{(byte) 0xff});
            // Current dispatcher contract: 8510/UNSPECIFIED with build diagnostic (not legacy 8512).
            assertFailure(f.route(ctx), 8510, UNSPECIFIED, "Batch request build failed:");
            assertTrue(f.h.engineArrivalOrder.isEmpty());
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
        }
    }

    @ParameterizedTest(name = "{0}: locally queued victim -> 8429")
    @ValueSource(strings = {"PRIORITY_NON_BATCH", "PRIORITY_BATCH"})
    void decodeReservationPreemptionSettlesVictimAndTransfersKv(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.holdPrefillDelivery();
            f.h.allowPreemption(VictimStage.DECODE_RESERVED);
            f.h.setDecodeKvCapacity(0, 128, 128);
            var victim = f.service.route(f.h.context(REQUEST_ID, 30));
            assertFalse(victim.isDone());
            var incoming = f.service.route(f.h.context(REQUEST_ID + 1, 70));
            // Engine has not seen the victim: still 8429/UNSPECIFIED, never 8400 NO_AVAILABLE_WORKER.
            Response response = victim.get(5, TimeUnit.SECONDS);
            assertFailure(response, 8429, UNSPECIFIED, "preempted by higher-priority request " + (REQUEST_ID + 1));
            assertEquals("preempted by higher-priority request " + (REQUEST_ID + 1), response.getErrorMessage());
            assertFalse(incoming.isDone());
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
            assertTrue(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID + 1));
            assertEquals(128, f.h.decodeEndpoint(0).inflightHardKvReserved());
            assertEquals(0, f.h.prefillEngines.getFirst().getCancelledCount());
            assertTrue(f.h.engineArrivalOrder.isEmpty());
        }
    }

    static Stream<Arguments> terminalsBeforeAck() {
        return Stream.of("FIFO_BATCH", "PRIORITY_BATCH").flatMap(mode -> Stream.of(
                Arguments.of(mode, false, 8513, UNSPECIFIED, "worker error code 2"),
                Arguments.of(mode, true, 8431, RESOURCE_EXHAUSTED, "queue has no recorded wait reason")));
    }

    @ParameterizedTest(name = "{0}: deadline={1} before ACK -> {2}/{3}")
    @MethodSource("terminalsBeforeAck")
    void terminalBeforeEnqueueAckCannotBeOverwrittenByLateSuccess(String mode, boolean deadline,
                                                                 int code, AdmissionRejectReason reason,
                                                                 String message) throws Exception {
        try (Fixture f = new Fixture(mode, 100, 60_000)) {
            var engine = f.h.prefillEngines.getFirst();
            var decodeEngine = f.h.decodeEngines.getFirst();
            CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> engineAck = new CompletableFuture<>();
            CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> deliveredAck = new CompletableFuture<>();
            doAnswer(invocation -> {
                EngineRpcService.EnqueueBatchRequestPB request = invocation.getArgument(2);
                // Run real Engine admission but hold only the transport ACK to reproduce WorkerStatus racing it.
                engine.enqueueBatch(request, new StreamObserver<>() {
                    @Override
                    public void onNext(EngineRpcService.EnqueueBatchResponsePB value) { engineAck.complete(value); }
                    @Override
                    public void onError(Throwable error) { engineAck.completeExceptionally(error); }
                    @Override
                    public void onCompleted() { }
                });
                return deliveredAck;
            }).when(f.h.grpcClient).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
            var responseFuture = f.service.route(f.h.context(REQUEST_ID, 50));
            var ack = engineAck.get(5, TimeUnit.SECONDS);
            assertEquals(1, ack.getSuccessesCount());
            assertFalse(responseFuture.isDone());
            AutoTpmE2EHarness.await(() -> decodeEngine.getRunningCount() == 1, 2_000, "real P->D handoff");
            if (deadline) {
                // Inject the public deadline event after delivery claim, without a wall-clock race.
                // Expected 8431/RESOURCE_EXHAUSTED using queue state; claim alone does not imply 8511.
                f.service.cancelRequest(REQUEST_ID, 0, CancelReason.DEADLINE_EXCEEDED);
            } else {
                assertNotNull(decodeEngine.cancel(REQUEST_ID)); // External cancellation, no Master cancel intent.
                var terminal = AutoTpmE2EHarness.workerStatus(decodeEngine, 0).getFinishedTaskListList().stream()
                        .filter(task -> task.getRequestId() == REQUEST_ID).findFirst().orElseThrow();
                assertEquals(2, terminal.getErrorInfo().getErrorCode()); // Engine CANCELLED=2, not priority 8429.
            }
            f.h.pumpOnce();
            var failure = responseFuture.get(5, TimeUnit.SECONDS);
            assertFailure(failure, code, reason, message);
            deliveredAck.complete(ack);
            assertSame(failure, responseFuture.get(), "late transport success must not replace the terminal");
            assertFalse(f.h.decodeEndpoint(0).reservedView().containsKey(REQUEST_ID));
        }
    }

    @ParameterizedTest(name = "{0}: Engine failure after successful Schedule does not rewrite 200")
    @ValueSource(strings = {"FIFO_BATCH", "PRIORITY_BATCH"})
    void engineTerminalAfterSuccessfulScheduleDoesNotRepublishResponse(String mode) throws Exception {
        try (Fixture f = new Fixture(mode, 100, 60_000)) {
            var future = f.service.route(f.h.context(REQUEST_ID, 50));
            var success = future.get(5, TimeUnit.SECONDS);
            assertTrue(success.isSuccess(), success.getErrorMessage());
            var engine = f.h.decodeEngines.getFirst();
            AutoTpmE2EHarness.await(() -> engine.getRunningCount() == 1, 2_000, "real P->D handoff");
            assertNotNull(engine.cancel(REQUEST_ID));
            f.h.pumpOnce();
            // Schedule already returned 200; Engine stream owns later execution errors. No second 8513 response.
            assertSame(success, future.get());
            assertEquals(200, success.getCode());
            assertEquals(0, f.h.decodeEndpoint(0).getInflightCount());
            assertEquals(0, f.h.scheduler.getInflightSize());
        }
    }

    @ParameterizedTest(name = "{0}: singleton above batch combining limit must not return legacy 8514")
    @ValueSource(strings = {"FIFO_BATCH", "PRIORITY_BATCH"})
    void batchCombiningLimitDoesNotRejectStandaloneRequest(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.h.prefillEndpoint(0).getStatus().setMaxBatchTokensSize(64);
            // A 128-token singleton is accepted: 200, not 8514. This limit governs combining requests.
            var response = f.route(f.h.context(REQUEST_ID, 50));
            assertTrue(response.isSuccess(), response.getErrorMessage());
            assertEquals(200, response.getCode());
            assertEquals(1, f.h.prefillEngines.getFirst().getAcceptedCount());
        }
    }

    @ParameterizedTest(name = "{0}: scheduler shutdown -> existing 8510 contract")
    @MethodSource("queuedModes")
    void schedulerShutdownPreservesExistingContract(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            f.h.scheduler.shutdown();
            // Existing 8510/UNSPECIFIED; this UT does not introduce a gRPC UNAVAILABLE migration.
            assertFailure(f.route(f.h.context(REQUEST_ID, 50)), 8510, UNSPECIFIED, "priority scheduler is shutting down");
            assertTrue(f.h.engineArrivalOrder.isEmpty());
        }
    }

    @ParameterizedTest(name = "{0}: no registered roles -> 8400")
    @MethodSource("modes")
    void noRegisteredWorkerRolesPreservesDiscoveryFailure(String mode) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
            // 8400/UNSPECIFIED is discovery failure only; it must not be used for a preempted victim.
            assertFailure(f.route(f.h.context(REQUEST_ID, 50)), 8400, UNSPECIFIED, "NO_AVAILABLE_WORKER");
            assertTrue(f.h.engineArrivalOrder.isEmpty());
        }
    }

    static Stream<Arguments> fusionFailures() {
        return modes().flatMap(mode -> Stream.of(
                // PDFUSION uses PrefillEndpoint too: absence of a healthy worker is 8404, full capacity is 8431.
                Arguments.of(mode, false, 8404, UNSPECIFIED, "NOT_ALIVE"),
                Arguments.of(mode, true, 8431, RESOURCE_EXHAUSTED, CAPACITY_MESSAGE)));
    }

    @ParameterizedTest(name = "{0}: PDFUSION alive={1} -> {2}")
    @MethodSource("fusionFailures")
    void fusionRoutingFailureUsesItsOwnRole(String mode, boolean alive, int code,
                                           AdmissionRejectReason reason, String message) throws Exception {
        try (Fixture f = new Fixture(mode)) {
            var model = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
            model.getPrefillStatusMap().clear();
            model.getDecodeStatusMap().clear();
            var status = f.h.prefillEndpoint(0).getStatus();
            status.setAlive(alive);
            f.h.endpointRegistry.ensureEndpoint(RoleType.PDFUSION, status.getIpPort(), status);
            model.getPdFusionStatusMap().put(status.getIpPort(), status);
            var endpoint = f.h.endpointRegistry.getPrefillEndpoints(RoleType.PDFUSION).get(status.getIpPort());
            if (alive) {
                for (long id = 1; id <= 100; id++) {
                    assertTrue(endpoint.tryCommitRequest(id, 60_000, 100));
                }
            }
            // This case ends at routing: it does not claim to exercise a PDFusion Engine execution path.
            assertFailure(f.route(f.h.context(REQUEST_ID, 50)), code, reason, message);
            assertTrue(f.h.engineArrivalOrder.isEmpty());
        }
    }

    private static void assertFailure(Response actual, int code, AdmissionRejectReason reason, String message) {
        assertAll(() -> assertFalse(actual.isSuccess()),
                () -> assertEquals(code, actual.getCode(), actual.getErrorMessage()),
                () -> assertEquals(reason, actual.getAdmissionRejectReason()));
        assertNotNull(actual.getErrorMessage());
        assertTrue(actual.getErrorMessage().contains(message), actual.getErrorMessage());
        if (code == 8430 || code == 8432) {
            assertEquals(message, actual.getErrorMessage(), "typed priority messages are a stable contract");
        }
        if (code == 8431) {
            assertTrue(actual.getErrorMessage().startsWith(CAPACITY_MESSAGE));
        }
    }

    /** Per-case resources and restoration of the process-wide model registry. No production test hooks. */
    private static final class Fixture implements AutoCloseable {
        final AutoTpmE2EHarness h;
        final RouteService service;
        final Map<RoleType, Map<String, WorkerStatus>> previousWorkers = new EnumMap<>(RoleType.class);

        Fixture(String mode) {
            this(mode, 100);
        }

        Fixture(String mode, int maxPendingRequests) {
            this(mode, maxPendingRequests, 1);
        }

        Fixture(String mode, int maxPendingRequests, double decodeStepMs) {
            FlexlbConfig config = new FlexlbConfig();
            if (mode.equals("DIRECT")) {
                config.setScheduler(new DirectSchedulerConfig());
            } else if (mode.startsWith("PRIORITY")) {
                config.queueScheduler().setOrdering(new PriorityOrderingConfig());
            }
            if (mode.equals("DIRECT") || mode.endsWith("NON_BATCH")) {
                config.setDispatcher(new NonBatchDispatcherConfig());
                config.nonBatchDispatcher().setMaxInflightRequestsPerPrefillWorker(1);
            } else {
                config.batchDispatcher().setMaxRequests(1);
                config.batchDispatcher().setMaxCollectionWaitMs(0);
            }
            config.getRouter().getRoles().getPrefill().getAvailability().setMaxPendingRequests(maxPendingRequests);
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            h = new AutoTpmE2EHarness(61000, 1, 1, "5", decodeStepMs, true, config);
            for (RoleType role : RoleType.values()) {
                var workers = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role);
                previousWorkers.put(role, new HashMap<>(workers));
                workers.clear();
            }
            registerStatus(RoleType.PREFILL, h.prefillEndpoint(0).getStatus());
            registerStatus(RoleType.DECODE, h.decodeEndpoint(0).getStatus());
            EngineWorkerStatus workers = new EngineWorkerStatus(h.endpointRegistry);
            ResourceMeasureFactory measures = new ResourceMeasureFactory(List.of(
                    new PrefillResourceMeasure(h.configService), new DecodeResourceMeasure(h.configService)));
            DefaultRouter router;
            // Constructors normally register in a global factory. Isolate that registry, while
            // returning REAL strategy instances to DefaultRouter; no select()/classification mocking.
            try (var factory = mockStatic(LoadBalanceStrategyFactory.class)) {
                var prefill = new CostBasedPrefillStrategy(workers, mock(CacheAwareService.class),
                        measures, mock(EngineHealthReporter.class));
                var decode = new CostBasedDecodeStrategy(workers, measures);
                var random = new RandomStrategy(workers, h.configService, measures);
                factory.when(() -> LoadBalanceStrategyFactory.getLoadBalanceStrategy(COST_BASED_PREFILL)).thenReturn(prefill);
                factory.when(() -> LoadBalanceStrategyFactory.getLoadBalanceStrategy(COST_BASED_DECODE)).thenReturn(decode);
                factory.when(() -> LoadBalanceStrategyFactory.getLoadBalanceStrategy(RANDOM)).thenReturn(random);
                router = new DefaultRouter(h.configService, ctx -> GroupRoutingDecision.none(), h.endpointRegistry);
            }
            h.routeFn = router::route;
            service = new RouteService(h.configService, router, h.scheduler, mock(RecentCacheKeyTraceReporter.class));
        }

        Response route(BalanceContext ctx) throws Exception {
            return service.route(ctx).get(5, TimeUnit.SECONDS);
        }

        void holdPrefillDelivery() {
            if (h.config.isBatchDispatch()) {
                h.config.batchDispatcher().setMaxRequests(100);
                h.config.batchDispatcher().setMaxCollectionWaitMs(60_000);
            } else {
                assertTrue(h.prefillEndpoint(0).tryCommitRequest(BLOCKER_ID, 60_000, 1));
            }
        }

        private static void registerStatus(RoleType role, WorkerStatus status) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role).put(status.getIpPort(), status);
        }

        @Override
        public void close() {
            try {
                h.close();
            } finally {
                previousWorkers.forEach((role, previous) -> {
                    var workers = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(role);
                    workers.clear();
                    workers.putAll(previous);
                });
            }
        }
    }
}
