package org.flexlb.service.grace;

import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.flexlb.engine.grpc.client.KvcmLeaderResolver;
import org.flexlb.engine.grpc.client.KvcmMetaServiceClient;
import org.flexlb.engine.grpc.client.KvcmWorkerMetadataResolver;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.monitor.KvcmMetricsReporter;
import org.flexlb.httpserver.FlexlbGrpcServer;
import org.flexlb.listener.ApplicationWarmupState;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.event.ContextClosedEvent;
import org.springframework.core.env.Environment;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
                mock(GracefulLifecycleReporter.class), context.getEnvironment(), new ApplicationWarmupState(), 0L);
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
                    mock(GracefulLifecycleReporter.class), context.getEnvironment(), new ApplicationWarmupState(), 0L);
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
            context.setAllowCircularReferences(false);
            context.registerBean(LBStatusConsistencyService.class,
                    () -> mock(LBStatusConsistencyService.class));
            context.registerBean(CacheMatchConfiguration.class, () -> {
                var configuration = mock(CacheMatchConfiguration.class);
                when(configuration.getKvcmRuntimeConfig()).thenReturn(new KvcmCacheMatchingConfig());
                return configuration;
            });
            context.registerBean(KvcmMetaServiceClient.class, () -> mock(KvcmMetaServiceClient.class));
            context.registerBean(KvcmLeaderResolver.class, () -> mock(KvcmLeaderResolver.class));
            context.registerBean(KvcmWorkerMetadataResolver.class, () -> mock(KvcmWorkerMetadataResolver.class));
            context.registerBean(GrpcReporter.class, () -> mock(GrpcReporter.class));
            context.registerBean(KvcmMetricsReporter.class, () -> mock(KvcmMetricsReporter.class));
            context.registerBean(FlexlbGrpcServer.class, () -> {
                // Preserve the server-to-client dependency while mocking the transport.
                context.getBean(KvcmGrpcClient.class);
                return mock(FlexlbGrpcServer.class);
            });
            context.registerBean(GracefulLifecycleReporter.class,
                    () -> mock(GracefulLifecycleReporter.class));
            context.register(ApplicationLifecycle.class, ApplicationWarmupState.class, KvcmGrpcClient.class);
            context.refresh();
            assertNotNull(context.getBean(ApplicationLifecycle.class));
            assertFalse(context.getBean(ApplicationWarmupState.class).isWarmupFinished());

            var client = context.getBean(KvcmGrpcClient.class);
            ReflectionTestUtils.invokeMethod(client, "recordHeartbeat", false);
            assertEquals(0, client.healthSnapshot().consecutiveHeartbeatFailures());

            context.getBean(ApplicationLifecycle.class).online();
            assertTrue(context.getBean(ApplicationWarmupState.class).isWarmupFinished());
            ReflectionTestUtils.invokeMethod(client, "recordHeartbeat", false);
            assertEquals(1, client.healthSnapshot().consecutiveHeartbeatFailures());
        }
    }

    @Test
    void normalServiceDoesNotDrainAndRepeatedOfflineIsIdempotent() {
        var consistency = mock(LBStatusConsistencyService.class);
        var grpc = mock(FlexlbGrpcServer.class);
        var reporter = mock(GracefulLifecycleReporter.class);
        var environment = mock(Environment.class);
        when(environment.getActiveProfiles()).thenReturn(new String[0]);
        var warmupState = new ApplicationWarmupState();
        var lifecycle = new ApplicationLifecycle(consistency, grpc, reporter, environment, warmupState, 0L);
        assertFalse(warmupState.isWarmupFinished());
        assertFalse(lifecycle.isHealthy());
        lifecycle.online();
        assertTrue(warmupState.isWarmupFinished());
        assertTrue(lifecycle.isHealthy());
        verifyNoInteractions(grpc);
        assertTrue(lifecycle.offline());
        assertTrue(lifecycle.offline());
        assertFalse(lifecycle.isHealthy());
        assertTrue(warmupState.isWarmupFinished());
        verify(consistency).offline();
        verify(grpc).drain();
        verify(reporter).reportShutdownComplete(anyLong());
    }

    @Test
    void repeatedOnlineResetsSharedWarmupStateUntilSynchronizationCompletes() {
        var consistency = mock(LBStatusConsistencyService.class);
        var environment = mock(Environment.class);
        when(environment.getActiveProfiles()).thenReturn(new String[0]);
        var warmupState = new ApplicationWarmupState();
        var lifecycle = new ApplicationLifecycle(consistency, mock(FlexlbGrpcServer.class),
                mock(GracefulLifecycleReporter.class), environment, warmupState, 0L);
        doAnswer(invocation -> {
            assertFalse(warmupState.isWarmupFinished());
            assertFalse(lifecycle.isHealthy());
            return null;
        }).when(consistency).start();

        lifecycle.online();
        assertTrue(warmupState.isWarmupFinished());
        lifecycle.online();
        assertTrue(warmupState.isWarmupFinished());
        assertTrue(lifecycle.isHealthy());
    }
}
