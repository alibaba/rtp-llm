package org.flexlb.service.grace.strategy;

import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class QueryWarmerHookerTest {

    @Test
    void completesWarmUpWithoutCreatingASecondTimer() {
        GracefulLifecycleReporter reporter =
                mock(GracefulLifecycleReporter.class);
        QueryWarmerHooker warmer = new QueryWarmerHooker(reporter, 0L);

        warmer.afterStartUp();

        assertTrue(QueryWarmerHooker.warmUpFinished);
        verify(reporter).reportWarmerComplete(anyLong());
    }
}
