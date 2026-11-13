package org.flexlb.api;

import org.flexlb.api.server.FlexlbGrpcServer;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.master.MasterStatusService;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.event.ContextClosedEvent;
import org.springframework.core.env.Environment;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Verifies Spring lifecycle wiring and that serving shutdown drains transport before releasing resources.
 */
class ApplicationLifecycleTest {
    @Test
    void contextCloseDrainsTransportBeforeDestroyingServingResources() throws Exception {
        var consistency = mock(MasterStatusService.class);
        var grpc = mock(FlexlbGrpcServer.class);
        var draining = new CountDownLatch(1);
        var complete = new CountDownLatch(1);
        doAnswer(
                        invocation -> {
                            verify(consistency).offline();
                            draining.countDown();
                            assertTrue(complete.await(3, TimeUnit.SECONDS));
                            return null;
                        })
                .when(grpc)
                .drain();
        var destroyed = new AtomicBoolean();
        var context = new AnnotationConfigApplicationContext();
        var lifecycle =
                new ApplicationLifecycle(
                        consistency,
                        grpc,
                        mock(ServiceMetricsReporter.class),
                        context.getEnvironment(),
                        new org.flexlb.util.ApplicationWarmupState(),
                        mock(AbstractRequestScheduler.class),
                        0L);
        context.registerBean(ApplicationLifecycle.class, () -> lifecycle);
        context.registerBean(
                "servingResource", DisposableBean.class, () -> () -> destroyed.set(true));
        context.refresh();
        var closing = CompletableFuture.runAsync(context::close);
        try {
            assertTrue(draining.await(2, TimeUnit.SECONDS));
            assertThrows(TimeoutException.class, () -> closing.get(100, TimeUnit.MILLISECONDS));
            assertFalse(destroyed.get());
        } finally {
            complete.countDown();
            closing.get(3, TimeUnit.SECONDS);
        }
        assertTrue(destroyed.get());
        verify(grpc).drain();
    }

    @Test
    void childContextCloseDoesNotStopTheServingContext() {
        var consistency = mock(MasterStatusService.class);
        var grpc = mock(FlexlbGrpcServer.class);
        try (var context = new AnnotationConfigApplicationContext();
                var child = new AnnotationConfigApplicationContext()) {
            var lifecycle =
                    new ApplicationLifecycle(
                            consistency,
                            grpc,
                            mock(ServiceMetricsReporter.class),
                            context.getEnvironment(),
                            new org.flexlb.util.ApplicationWarmupState(),
                            mock(AbstractRequestScheduler.class),
                            0L);
            context.registerBean(ApplicationLifecycle.class, () -> lifecycle);
            context.refresh();
            context.publishEvent(new ContextClosedEvent(child));
            verifyNoInteractions(consistency, grpc);
        }
        verify(consistency).offline();
        verify(grpc).drain();
    }

    @Test
    void springCanWireProductionConstructor() {
        try (var context = new AnnotationConfigApplicationContext()) {
            context.registerBean(MasterStatusService.class, () -> mock(MasterStatusService.class));
            context.registerBean(FlexlbGrpcServer.class, () -> mock(FlexlbGrpcServer.class));
            context.registerBean(
                    ServiceMetricsReporter.class, () -> mock(ServiceMetricsReporter.class));
            context.registerBean(org.flexlb.util.ApplicationWarmupState.class);
            context.register(ApplicationLifecycle.class);
            context.getBeanFactory()
                    .registerSingleton("scheduler", mock(AbstractRequestScheduler.class));
            context.refresh();
            assertNotNull(context.getBean(ApplicationLifecycle.class));
        }
    }

    @Test
    void normalServiceDoesNotDrainAndRepeatedOfflineIsIdempotent() {
        var consistency = mock(MasterStatusService.class);
        var grpc = mock(FlexlbGrpcServer.class);
        var reporter = mock(ServiceMetricsReporter.class);
        var environment = mock(Environment.class);
        when(environment.getActiveProfiles()).thenReturn(new String[0]);
        var lifecycle =
                new ApplicationLifecycle(
                        consistency,
                        grpc,
                        reporter,
                        environment,
                        new org.flexlb.util.ApplicationWarmupState(),
                        mock(AbstractRequestScheduler.class),
                        0L);
        lifecycle.online();
        assertTrue(lifecycle.isHealthy());
        verifyNoInteractions(grpc);
        assertTrue(lifecycle.offline());
        assertTrue(lifecycle.offline());
        assertFalse(lifecycle.isHealthy());
        verify(consistency).offline();
        verify(grpc).drain();
        verify(reporter)
                .reportDuration(
                        org.mockito.ArgumentMatchers.eq(
                                ServiceMetricsReporter.Event.SHUTDOWN_COMPLETE),
                        anyLong());
    }
}
