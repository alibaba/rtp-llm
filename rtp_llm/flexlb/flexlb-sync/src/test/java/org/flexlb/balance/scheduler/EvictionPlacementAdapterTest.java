package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EvictionPlacementAdapterTest {

    @Test
    void decodeFullDeclinesPrefillEvictionBeforeBuildingAnItem() {
        Router router = mock(Router.class);
        BalanceContext context = mock(BalanceContext.class);
        QueueRouteAdmission admission = mock(QueueRouteAdmission.class);
        when(router.routeForQueue(context)).thenReturn(
                new QueueRoutingResult.Admitted(admission));
        when(admission.prepareDecode(context)).thenReturn(
                QueueRouteAdmission.DecodePrepareStatus.CAPACITY_FULL);
        EvictionPlacementAdapter adapter = new EvictionPlacementAdapter(
                router,
                mock(ConfiguredLoadBalanceSelector.class),
                mock(InflightCommitPort.class),
                mock(BatchSchedulerReporter.class));

        assertNull(adapter.preparePrefillEviction(
                context, new CompletableFuture<Response>()));

        verify(admission).close();
        verify(admission, never()).buildItem(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong());
    }
}
