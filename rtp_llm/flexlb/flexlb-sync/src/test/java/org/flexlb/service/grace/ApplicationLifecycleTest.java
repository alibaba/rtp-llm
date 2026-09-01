package org.flexlb.service.grace;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.junit.jupiter.api.Test;
import org.springframework.core.env.Environment;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ApplicationLifecycleTest {

    @Test
    void ownsOnlineHealthAndOfflineState() {
        LBStatusConsistencyService consistency =
                mock(LBStatusConsistencyService.class);
        ActiveRequestCounter requests = new ActiveRequestCounter();
        GracefulLifecycleReporter reporter =
                mock(GracefulLifecycleReporter.class);
        Environment environment = mock(Environment.class);
        when(environment.getActiveProfiles()).thenReturn(new String[0]);
        ApplicationLifecycle lifecycle = new ApplicationLifecycle(
                consistency, requests, reporter, environment,
                0L, 1_000L, 0L);

        assertFalse(lifecycle.isHealthy());
        lifecycle.online();
        assertTrue(lifecycle.isHealthy());
        verify(consistency).start();
        verify(reporter).reportWarmerComplete(anyLong());

        assertTrue(lifecycle.offline());
        assertFalse(lifecycle.isHealthy());
        assertTrue(lifecycle.shutdownCompletedSuccessfully());
        verify(consistency).offline();
        verify(reporter).reportShutdownComplete(anyLong());
    }
}
