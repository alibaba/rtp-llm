package org.flexlb.balance.strategy;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;

import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** Regression contracts for Decode capacity projected through Prefill. */
class CostBasedPrefillStrategyBlockerTest {

    @Test
    void failureUsesOneScopedFleetAndAggregatesReasonsIndependentlyOfOrder() {
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            for (AdmissionRejectReason first : AdmissionRejectReason.values()) {
                for (AdmissionRejectReason second : AdmissionRejectReason.values()) {
                    var directory = mock(EndpointRegistry.class);
                    var cache = mock(CacheAwareService.class);
                    var strategy = new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class), org.mockito.Mockito.mock(CacheMetricsReporter.class));
                    var outside = rejectedEndpoint(role, "other", AdmissionRejectReason.UNSPECIFIED);
                    var firstEndpoint = rejectedEndpoint(role, "target", first);
                    var secondEndpoint = rejectedEndpoint(role, "target", second);
                    when(directory.prefillRoutingSnapshot(role)).thenReturn(List.of(
                            new EndpointRegistry.PrefillRoutingEntry("first", firstEndpoint),
                            new EndpointRegistry.PrefillRoutingEntry("outside", outside),
                            new EndpointRegistry.PrefillRoutingEntry("second", secondEndpoint)));
                    var context = new RequestContext(new FlexlbConfig());
                    var request = new Request();
                    request.setRequestId(1L);
                    request.setSeqLen(100L);
                    context.setRequest(request);
                    context.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));

                    var result = strategy.select(freezeInputs(context).getRequirements(), context.getConfig(), role, "target");
                    var failure = result.failure();

                    // Missing provenance dominates; mixed known blockers are capacity, not arbitrary priority.
                    int expectedCode = first == AdmissionRejectReason.UNSPECIFIED
                            || second == AdmissionRejectReason.UNSPECIFIED ? 8432
                            : first != second || first == AdmissionRejectReason.RESOURCE_EXHAUSTED ? 8431 : 8430;
                    assertEquals(expectedCode, failure.getCode(), role + " " + first + " " + second);
                    assertEquals(2, result.diagnostics().get("workers"));
                    verify(directory).prefillRoutingSnapshot(role);
                    verify(outside, never()).admissionSummary(anyInt());
                    verifyNoInteractions(cache);
                }
            }
        }
    }

    private static PrefillEndpoint rejectedEndpoint(RoleType role, String group, AdmissionRejectReason reason) {
        var endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(
                WorkerStatus.createDiscovered(role, group, "worker", 8080, 8090, "test"));
        when(endpoint.admissionSummary(50)).thenReturn(switch (reason) {
            case HIGHER_PRIORITY_AHEAD -> new PrefillState.AdmissionSummary(1, 1, 0, 0, 1, 0);
            case SAME_PRIORITY_AHEAD -> new PrefillState.AdmissionSummary(1, 1, 0, 1, 0, 0);
            case UNSPECIFIED -> new PrefillState.AdmissionSummary(1, 1, 0, 0, 0, 1);
            case RESOURCE_EXHAUSTED -> new PrefillState.AdmissionSummary(0, 0, 0, 0, 0, 0);
        });
        return endpoint;
    }

    @Test
    void everyRegisteredEndpointMustProveTheSameBlocker() {
        assertNull(CostBasedPrefillStrategy.provenPoolWideBlocker(
                Map.of(RoleType.DECODE, 3), 4));
        assertNull(CostBasedPrefillStrategy.provenPoolWideBlocker(
                Map.of(RoleType.DECODE, 2, RoleType.PREFILL, 2), 4));
        assertEquals(
                RoleType.DECODE,
                CostBasedPrefillStrategy.provenPoolWideBlocker(
                        Map.of(RoleType.DECODE, 4), 4));
    }
}
