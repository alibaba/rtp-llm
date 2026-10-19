package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Prefill group routing contracts.
 */
class CostBasedPrefillStrategyBlockerTest {

    @Test
    void selectionWithoutMatchingGroupReportsNoWorker() {
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            var directory = mock(WorkerDirectory.class);
            var cache = mock(CacheAwareService.class);
            var strategy = new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class));
            var outside = mock(PrefillEndpoint.class);
            when(outside.getStatus()).thenReturn(
                    WorkerStatus.createDiscovered(role, "other", "worker", 8080, 8090, "test"));
            when(directory.prefillRoutingSnapshot(role)).thenReturn(List.of(
                    new EndpointRegistry.PrefillRoutingEntry("outside", outside)));
            var context = new BalanceContext(new FlexlbConfig());
            var request = new Request();
            request.setRequestId("1");
            request.setSeqLen(100L);
            context.setRequest(request);
            context.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));

            var result = strategy.select(context, role, "target");

            assertEquals(role.getErrorType().getErrorCode(), result.failure().getCode());
            assertEquals(0, result.diagnostics().get("workers"));
            assertEquals("NO_MATCHING_ENDPOINTS", context.selectionReason(role));
            verify(directory).prefillRoutingSnapshot(role);
            verifyNoInteractions(cache);
        }
    }
}
