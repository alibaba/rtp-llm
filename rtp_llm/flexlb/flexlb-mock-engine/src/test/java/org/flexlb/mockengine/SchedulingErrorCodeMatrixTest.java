package org.flexlb.mockengine;

import com.google.protobuf.ByteString;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.RouteService;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.flexlb.dao.loadbalance.AdmissionRejectReason.HIGHER_PRIORITY_AHEAD;
import static org.flexlb.dao.loadbalance.AdmissionRejectReason.RESOURCE_EXHAUSTED;
import static org.flexlb.dao.loadbalance.AdmissionRejectReason.SAME_PRIORITY_AHEAD;
import static org.flexlb.dao.loadbalance.AdmissionRejectReason.UNSPECIFIED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Real RouteService, selection, admission, queue, delivery and in-process Mock Engine.
 * Only cache/metrics/config lookup and gRPC transport are stand-ins. Values are
 * FlexLB codes, not HTTP or gRPC statuses. Schema v3 keeps blocked placement in
 * its global queue, and stops the request deadline once delivery is claimed.
 * Engine-owned victim settlement is covered by PreemptionPhasesE2ETest.
 */
@Timeout(15)
class SchedulingErrorCodeMatrixTest {
    private static final long REQUEST_ID = 71_001L;

    static Stream<String> queuedModes() {
        return Stream.of("FIFO", "PRIORITY").flatMap(order -> Stream.of("SINGLE", "FIXED_WINDOW")
                .flatMap(decision -> Stream.of("NON_BATCH", "BATCH").map(dispatch -> order + "/" + decision + "/" + dispatch)));
    }

    static Stream<String> modes() { return Stream.concat(Stream.of("DIRECT"), queuedModes()); }
    static Stream<String> batchModes() { return queuedModes().filter(mode -> mode.endsWith("/BATCH")); }

    @ParameterizedTest(name = "{0}: healthy -> 200")
    @MethodSource("modes")
    void healthyPlacement(String mode) throws Exception {
        try (Fixture fixture = new Fixture(mode)) {
            Response result = fixture.route(fixture.context());
            assertTrue(result.isSuccess(), result.getErrorMessage());
            assertEquals(200, result.getCode());
            assertEquals(UNSPECIFIED, result.getAdmissionRejectReason());
            assertEquals(fixture.harness.config.getDispatcher().requiresGenerateInput() ? List.of(REQUEST_ID) : List.of(),
                    fixture.harness.engineArrivalOrder);
        }
    }

    @ParameterizedTest(name = "{0}: expired before placement")
    @MethodSource("modes")
    void expiredBeforePlacement(String mode) throws Exception {
        try (Fixture fixture = new Fixture(mode)) {
            BalanceContext context = fixture.context();
            context.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() - 1));
            // DIRECT retains 8511; QUEUE reports 8431 and never sends an Engine RPC.
            assertFailure(fixture.route(context), fixture.harness.config.isDirect() ? 8511 : 8431,
                    fixture.harness.config.isDirect() ? UNSPECIFIED : RESOURCE_EXHAUSTED, "expired");
            assertTrue(fixture.harness.engineArrivalOrder.isEmpty());
            assertEquals(0, fixture.harness.decodeEndpoint(0).getInflightCount());
        }
    }

    @ParameterizedTest(name = "{0}: request exceeds every Decode KV budget -> 8431")
    @MethodSource("modes")
    void impossibleDecodeDemand(String mode) throws Exception {
        try (Fixture fixture = new Fixture(mode)) {
            fixture.harness.setDecodeKvCapacity(0, 64, 64);
            assertFailure(fixture.route(fixture.context()), 8431, RESOURCE_EXHAUSTED,
                    "admission capacity is temporarily exhausted");
            assertEquals(0, fixture.harness.decodeEndpoint(0).getInflightCount());
            assertTrue(fixture.harness.engineArrivalOrder.isEmpty());
        }
    }

    static Stream<Arguments> directPriorityCases() {
        return Stream.of(Arguments.of(70, 8430, HIGHER_PRIORITY_AHEAD, "higher-priority requests are ahead"),
                Arguments.of(50, 8430, SAME_PRIORITY_AHEAD, "same-priority requests are ahead"),
                Arguments.of(30, 8431, RESOURCE_EXHAUSTED, "admission capacity is temporarily exhausted"),
                Arguments.of(0, 8432, UNSPECIFIED, "blocker priority attribution is unavailable"));
    }

    static Stream<Arguments> prefillPriorityCases() {
        return directPriorityCases().filter(arguments -> (int) arguments.get()[0] > 0);
    }

    @ParameterizedTest(name = "DIRECT: Prefill owner P{0} -> {1}")
    @MethodSource("prefillPriorityCases")
    void directPrefillCapacityUsesOwnerProvenance(int priority, int code, AdmissionRejectReason reason,
                                                String message) throws Exception {
        try (Fixture fixture = new Fixture("DIRECT", 1)) {
            fixture.harness.config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(0L);
            assertTrue(fixture.route(fixture.harness.context(REQUEST_ID - 1, priority)).isSuccess());
            BalanceContext context = fixture.context();
            Response failure = fixture.route(context);
            assertFailure(failure, code, reason, message);
            assertEquals("PREFILL", context.getSchedulingDiagnostics().get("role"));
            assertEquals(1, context.getSchedulingDiagnostics().get("workers"));
            assertEquals(1, fixture.harness.prefillEndpoint(0).getLocallyOwnedRequestCount());
            assertTrue(fixture.harness.engineArrivalOrder.isEmpty());
        }
    }

    @ParameterizedTest(name = "DIRECT: Decode owner P{0} -> {1}")
    @MethodSource("directPriorityCases")
    void directDecodeCapacityUsesOwnerProvenance(int priority, int code, AdmissionRejectReason reason,
                                                String message) throws Exception {
        try (Fixture fixture = new Fixture("DIRECT")) {
            fixture.reserveDecode(priority);
            BalanceContext context = fixture.context();
            assertFailure(fixture.route(context), code, reason, message);
            assertNotNull(context.getSchedulingDiagnostics());
            assertEquals(1, fixture.harness.decodeEndpoint(0).getInflightCount());
            assertTrue(fixture.harness.engineArrivalOrder.isEmpty());
        }
    }

    @ParameterizedTest(name = "{0}: blocked Decode queue expires -> 8431")
    @MethodSource("queuedModes")
    void queueExpiryReadsRecordedWait(String mode) throws Exception {
        try (Fixture fixture = new Fixture(mode)) {
            fixture.reserveDecode(70);
            BalanceContext context = fixture.context();
            context.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 500));
            // Once QUEUE owns the request, SLO uses the queue cause (8431), not
            // a second admission classification (8430) or generic timeout (8511).
            Response failure = fixture.route(context);
            assertFailure(failure, 8431, RESOURCE_EXHAUSTED, "DECODE");
            assertQueueWaitPv(context, failure, "DECODE");
            assertTrue(fixture.harness.engineArrivalOrder.isEmpty());
            assertEquals(1, fixture.harness.decodeEndpoint(0).getInflightCount());
        }
    }

    @ParameterizedTest(name = "{0}: healthy Prefill has no KV -> queue expiry 8431")
    @MethodSource("queuedModes")
    void queueExpiryRetainsPrefillPressure(String mode) throws Exception {
        try (Fixture fixture = new Fixture(mode)) {
            fixture.harness.prefillEngines.getFirst().setFaultConfig(FaultInjectionConfig.builder()
                    .kvPressureTokens(Long.MAX_VALUE).build());
            fixture.harness.pumpOnce();
            BalanceContext context = fixture.context();
            context.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 500));
            Response failure = fixture.route(context);
            assertFailure(failure, 8431, RESOURCE_EXHAUSTED, "admission capacity is temporarily exhausted");
            assertQueueWaitPv(context, failure, "PREFILL");
            assertTrue(fixture.harness.engineArrivalOrder.isEmpty());
        }
    }

    @ParameterizedTest(name = "{0}: Engine admission rejects -> 8510")
    @MethodSource("batchModes")
    void engineRejectionRetainsItsDiagnostic(String mode) throws Exception {
        try (Fixture fixture = new Fixture(mode)) {
            fixture.harness.config.getRequestLifecycle().getRequest().setTimeoutMs(500L);
            fixture.harness.prefillEngines.getFirst().setFaultConfig(FaultInjectionConfig.builder()
                    .failOnEnqueue(true).enqueueErrorMessage("injected admission rejection").build());
            var future = fixture.service.route(fixture.context());
            assertFailure(future.get(5, TimeUnit.SECONDS), 8510, UNSPECIFIED, "injected admission rejection");
            assertEquals(List.of(REQUEST_ID), fixture.harness.engineArrivalOrder);
            assertEquals(0, fixture.harness.prefillEngines.getFirst().getAcceptedCount());
            // A Prefill rejection does not prove Decode has not observed the request.
            // Preserve that ownership until the existing inactivity reducer settles it.
            AutoTpmE2EHarness.await(() -> fixture.harness.decodeEndpoint(0).getInflightCount() == 0,
                    2_000L, "Decode reservation must settle after inactivity");
            assertFailure(future.join(), 8510, UNSPECIFIED, "injected admission rejection");
        }
    }

    @ParameterizedTest(name = "{0}: duplicate -> 8406; client cancel -> 8504")
    @MethodSource("queuedModes")
    void duplicateAndClientCancellationPreserveOriginalOwnership(String mode) throws Exception {
        try (Fixture fixture = new Fixture(mode)) {
            fixture.reserveDecode(70);
            var original = fixture.service.route(fixture.context());
            assertFailure(fixture.route(fixture.context()), 8406, UNSPECIFIED, "duplicate request_id");
            assertFalse(original.isDone());
            fixture.service.cancelRequest(REQUEST_ID, 0L, CancelReason.CLIENT_CANCELLED);
            assertFailure(original.get(5, TimeUnit.SECONDS), 8504, UNSPECIFIED, "cancel");
            assertEquals(1, fixture.harness.decodeEndpoint(0).getInflightCount());
            assertTrue(fixture.harness.engineArrivalOrder.isEmpty());
        }
    }

    @ParameterizedTest(name = "{0}: missing input -> 8406; malformed protobuf -> 8510")
    @MethodSource("batchModes")
    void invalidBatchInputNeverReachesEngine(String mode) throws Exception {
        try (Fixture fixture = new Fixture(mode)) {
            BalanceContext missing = fixture.context();
            missing.setGenerateInputPb(ByteString.EMPTY);
            assertFailure(fixture.route(missing), 8406, UNSPECIFIED, "missing serialized generate_input");
            BalanceContext malformed = fixture.context();
            malformed.setGenerateInputPb(ByteString.copyFrom(new byte[]{(byte) 0xff}));
            assertFailure(fixture.route(malformed), 8510, UNSPECIFIED, "build");
            assertTrue(fixture.harness.engineArrivalOrder.isEmpty());
        }
    }

    private static void assertQueueWaitPv(BalanceContext context, Response failure, String role) {
        context.setResponse(failure);
        context.setSuccess(failure.isSuccess());
        PvLogData pv = new PvLogData(context, failure.getCode(), RESOURCE_EXHAUSTED.name(),
                "MASTER", 0L, "TIMED_OUT", null, System.currentTimeMillis());
        assertNotNull(pv.getSchedulingDiagnostics());
        assertEquals(context.getSchedulingDiagnostics(), pv.getSchedulingDiagnostics());
        assertTrue(pv.getSchedulingDiagnostics().get("cause").toString()
                .toUpperCase(java.util.Locale.ROOT).contains(role));
    }

    private static void assertFailure(Response response, int code, AdmissionRejectReason reason, String message) {
        assertFalse(response.isSuccess());
        assertEquals(code, response.getCode(), response.getErrorMessage());
        assertEquals(reason, response.getAdmissionRejectReason());
        assertTrue(response.getErrorMessage().contains(message), response.getErrorMessage());
    }

    private static final class Fixture implements AutoCloseable {
        final AutoTpmE2EHarness harness;
        final RouteService service;

        Fixture(String mode) {
            this(mode, 100);
        }

        Fixture(String mode, int prefillCapacity) {
            FlexlbConfig config = new FlexlbConfig();
            DecisionPolicyConfig decision = new DecisionPolicyConfig();
            decision.setMaxRequests(1);
            decision.setMaxCollectionWaitMs(0);
            if (mode.equals("DIRECT")) {
                config.setScheduler(SchedulerConfig.direct());
                config.setDispatcher(DispatcherConfig.nonBatch());
            } else {
                String[] parts = mode.split("/");
                config.queueScheduler().setOrdering(parts[0].equals("PRIORITY")
                        ? QueueOrderingConfig.priority() : new QueueOrderingConfig());
                decision.setType(DecisionPolicyConfig.Type.valueOf(parts[1]));
                config.queueScheduler().setDecision(decision);
                config.getDispatcher().setType(DispatcherConfig.Type.valueOf(parts[2]));
            }
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            config.getDispatcher().setMaxInflightPerPrefillWorker(prefillCapacity);
            harness = new AutoTpmE2EHarness(61_500, 1, 1, "1", 1.0, true,
                    false, decision, true, config);
            service = new RouteService(harness.scheduler, mock(RecentCacheKeyTraceReporter.class));
        }

        BalanceContext context() { return harness.context(REQUEST_ID, 50); }
        Response route(BalanceContext context) throws Exception { return service.route(context).get(5, TimeUnit.SECONDS); }

        void reserveDecode(int priority) {
            var endpoint = harness.decodeEndpoint(0);
            try (var pin = endpoint.tryPinGeneration()) {
                assertNotNull(endpoint.reserveUnqueued(pin, REQUEST_ID - 1, 0L, 0L, priority));
            }
        }

        @Override public void close() throws Exception { harness.close(); }
    }
}
