package org.flexlb.service.grace;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.env.Environment;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ApplicationLifecycleTest {

    @Test
    void springCanWireProductionConstructor() {
        try (AnnotationConfigApplicationContext context =
                     new AnnotationConfigApplicationContext()) {
            context.registerBean(LBStatusConsistencyService.class,
                    () -> mock(LBStatusConsistencyService.class));
            context.registerBean(ActiveRequestCounter.class,
                    ActiveRequestCounter::new);
            context.registerBean(GracefulLifecycleReporter.class,
                    () -> mock(GracefulLifecycleReporter.class));
            context.registerBean(Environment.class, context::getEnvironment);
            context.register(ApplicationLifecycle.class);
            context.refresh();

            assertNotNull(context.getBean(ApplicationLifecycle.class));
        }
    }

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
