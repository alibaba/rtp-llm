package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
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
                    var directory = mock(WorkerDirectory.class);
                    var cache = mock(CacheAwareService.class);
                    var strategy = new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class));
                    var outside = rejectedEndpoint(role, "other", AdmissionRejectReason.UNSPECIFIED);
                    var firstEndpoint = rejectedEndpoint(role, "target", first);
                    var secondEndpoint = rejectedEndpoint(role, "target", second);
                    when(directory.prefillRoutingSnapshot(role)).thenReturn(List.of(
                            new EndpointRegistry.PrefillRoutingEntry("first", firstEndpoint),
                            new EndpointRegistry.PrefillRoutingEntry("outside", outside),
                            new EndpointRegistry.PrefillRoutingEntry("second", secondEndpoint)));
                    var context = new BalanceContext(new FlexlbConfig());
                    var request = new Request();
                    request.setRequestId("1");
                    request.setSeqLen(100L);
                    context.setRequest(request);
                    context.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));

                    var result = strategy.select(context, role, "target");
                    var failure = result.failure();

                    // Missing provenance dominates; mixed known blockers are capacity, not arbitrary priority.
                    int expectedCode = first == AdmissionRejectReason.UNSPECIFIED
                            || second == AdmissionRejectReason.UNSPECIFIED ? 8432
                            : first != second || first == AdmissionRejectReason.RESOURCE_EXHAUSTED ? 8431 : 8430;
                    assertEquals(expectedCode, failure.getCode(), role + " " + first + " " + second);
                    assertEquals(2, result.diagnostics().get("workers"));
                    verify(directory).prefillRoutingSnapshot(role);
                    verify(outside, never()).admissionRejectReason(anyInt());
                    verifyNoInteractions(cache);
                }
            }
        }
    }

    private static PrefillEndpoint rejectedEndpoint(RoleType role, String group, AdmissionRejectReason reason) {
        var endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(
                WorkerStatus.createDiscovered(role, group, "worker", 8080, 8090, "test"));
        when(endpoint.admissionRejectReason(50)).thenReturn(reason);
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
