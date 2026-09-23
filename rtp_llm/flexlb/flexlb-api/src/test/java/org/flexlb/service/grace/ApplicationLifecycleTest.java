package org.flexlb.service.grace;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.httpserver.FlexlbGrpcServer;
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

class ApplicationLifecycleTest {
    @Test
    void contextCloseDrainsTransportBeforeDestroyingServingResources() throws Exception {
        var consistency = mock(LBStatusConsistencyService.class);
        var grpc = mock(FlexlbGrpcServer.class);
        var draining = new CountDownLatch(1);
        var complete = new CountDownLatch(1);
        doAnswer(invocation -> {
            verify(consistency).offline();
            draining.countDown();
            assertTrue(complete.await(3, TimeUnit.SECONDS));
            return null;
        }).when(grpc).drain();
        var destroyed = new AtomicBoolean();
        var context = new AnnotationConfigApplicationContext();
        var lifecycle = new ApplicationLifecycle(consistency, grpc,
                mock(GracefulLifecycleReporter.class), context.getEnvironment(), 0L);
        context.registerBean(ApplicationLifecycle.class, () -> lifecycle);
        context.registerBean("servingResource", DisposableBean.class, () -> () -> destroyed.set(true));
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
        assertTrue(lifecycle.shutdownCompletedSuccessfully());
    }

    @Test
    void childContextCloseDoesNotStopTheServingContext() {
        var consistency = mock(LBStatusConsistencyService.class);
        var grpc = mock(FlexlbGrpcServer.class);
        try (var context = new AnnotationConfigApplicationContext();
             var child = new AnnotationConfigApplicationContext()) {
            var lifecycle = new ApplicationLifecycle(consistency, grpc,
                    mock(GracefulLifecycleReporter.class), context.getEnvironment(), 0L);
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
            context.registerBean(LBStatusConsistencyService.class,
                    () -> mock(LBStatusConsistencyService.class));
            context.registerBean(FlexlbGrpcServer.class, () -> mock(FlexlbGrpcServer.class));
            context.registerBean(GracefulLifecycleReporter.class,
                    () -> mock(GracefulLifecycleReporter.class));
            context.register(ApplicationLifecycle.class);
            context.refresh();
            assertNotNull(context.getBean(ApplicationLifecycle.class));
        }
    }

    @Test
    void normalServiceDoesNotDrainAndRepeatedOfflineIsIdempotent() {
        var consistency = mock(LBStatusConsistencyService.class);
        var grpc = mock(FlexlbGrpcServer.class);
        var reporter = mock(GracefulLifecycleReporter.class);
        var environment = mock(Environment.class);
        when(environment.getActiveProfiles()).thenReturn(new String[0]);
        var lifecycle = new ApplicationLifecycle(consistency, grpc, reporter, environment, 0L);
        lifecycle.online();
        assertTrue(lifecycle.isHealthy());
        verifyNoInteractions(grpc);
        assertTrue(lifecycle.offline());
        assertTrue(lifecycle.offline());
        assertFalse(lifecycle.isHealthy());
        verify(consistency).offline();
        verify(grpc).drain();
        verify(reporter).reportShutdownComplete(anyLong());
    }
}
