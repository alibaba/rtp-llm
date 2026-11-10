package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.junit.jupiter.api.Test;

import java.util.concurrent.Executors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.doCallRealMethod;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that concurrent queue readers reuse the immutable planning input attached to the exact registration.
 */
class RequestRoutePlanningTest {
    @Test
    void queueReadersReuseTheExactRegisteredInput() throws Exception {
        var input = new GroupPlanner.Item("registered", 50, 3, 100, 200, 10, 2);
        var record = mock(PrefillEndpoint.RequestRecord.class);
        when(record.planningInput()).thenReturn(input);
        var route = mock(RouteResult.class);
        when(route.prefillReservation()).thenReturn(record);
        doCallRealMethod().when(route).planningInput();
        try (var executor = Executors.newFixedThreadPool(4)) {
            var reads =
                    IntStream.range(0, 16)
                            .mapToObj(i -> executor.submit(route::planningInput))
                            .toList();
            for (var read : reads) assertSame(input, read.get());
            verify(route, never()).ctx();
        }
    }
}
