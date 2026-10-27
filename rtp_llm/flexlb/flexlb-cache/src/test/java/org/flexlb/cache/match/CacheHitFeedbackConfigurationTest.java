package org.flexlb.cache.match;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Ticker;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.config.ConfigSource;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.ClassPathBeanDefinitionScanner;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import static org.flexlb.cache.CacheMatchTestConfigurations.localSync;
import static org.flexlb.cache.WorkerStatusTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheHitFeedbackConfigurationTest {
    @Test
    void nacosUpdatesResizeAndExpireExistingAssociationsWithoutRebuildingCache() throws Exception {
        ConfigSource nacos = mock(ConfigSource.class, CALLS_REAL_METHODS);
        when(nacos.name()).thenReturn("Nacos");
        when(nacos.load()).thenReturn(document(2, 1000));
        ConfigService.register(nacos);
        ConfigService config = new ConfigService();
        AtomicLong timeNs = new AtomicLong();
        try (var tickers = mockStatic(Ticker.class); var context = new AnnotationConfigApplicationContext()) {
            tickers.when(Ticker::systemTicker).thenReturn((Ticker) timeNs::get);
            ArgumentCaptor<Consumer<String>> updates = ArgumentCaptor.captor();
            verify(nacos).setUpdateListener(updates.capture());
            var comparison = new LocalStandbyComparisonService(localSync(new ModelMetaConfig()), null);
            context.getBeanFactory().registerSingleton("configService", config);
            context.getBeanFactory().registerSingleton("comparisonService", comparison);
            context.getBeanFactory().registerSingleton("metricsReporter", mock(CacheMetricsReporter.class));
            context.getBeanFactory().registerSingleton("queryOrchestrator", mock(CacheMatchQueryOrchestrator.class));
            context.getBeanFactory()
                    .registerSingleton("updateOrchestrator", mock(CacheMetadataUpdateOrchestrator.class));
            var scanner = new ClassPathBeanDefinitionScanner(context);
            scanner.setResourcePattern("CacheHitFeedbackTracker.class");
            scanner.scan(CacheHitFeedbackTracker.class.getPackageName());
            context.register(CacheAwareService.class);
            context.refresh();
            var service = context.getBean(CacheAwareService.class);
            assertSame(context.getBean(CacheHitFeedbackTracker.class),
                    ReflectionTestUtils.getField(service, "feedbackTracker"));
            Cache<?, ?> associations = associations(service);
            assertEquals(2, maximum(associations));
            assertEquals(1000, timeoutMs(associations));

            WorkerStatus worker = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);
            track(service, worker, "before-update");
            timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(750));
            updates.getValue().accept(document(4, 2000));
            track(service, worker, "after-update-1");
            track(service, worker, "after-update-2");
            timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(500));
            assertEquals(4, maximum(associations));
            assertEquals(2000, timeoutMs(associations));
            assertEquals(3, associations.estimatedSize());
            assertSame(associations, associations(service));

            updates.getValue().accept(document(0, 0));
            updates.getValue().accept(document(4, 0));
            assertEquals(4, config.loadBalanceConfig().getObservability().getCacheHit().getFeedbackMaximumEntries());
            assertEquals(4, maximum(associations));
            assertEquals(2000, timeoutMs(associations));

            var telemetry = new WorkerStatus.TaskTelemetry(true, 1, 2, 3, 4, 5, 0, 6, 700, 0, 0, 0, 0, 0, 0);
            var finished = new WorkerStatus.TaskObservation("before-update", 700, 0, 1000, 0, 0, 0, 0, 0, "", 0,
                    TaskPhase.RUNNING, 0, null, telemetry);
            var status = worker.bindStatusObservation(worker.committedEngineObservation(), true, 1L, 1L,
                    Map.of(finished.requestId(), finished));
            assertEquals(700, service.observeCacheHitFeedback(worker, status).getFirst().join().actualHitTokens());
            assertEquals(2, associations.estimatedSize());

            updates.getValue().accept(document(1, 2000));
            assertEquals(1, maximum(associations));
            assertEquals(1, associations.estimatedSize());
            updates.getValue().accept(document(1, 500));
            assertEquals(500, timeoutMs(associations));
            associations.cleanUp();
            assertEquals(0, associations.estimatedSize(), "A shorter timeout must expire already-silent records");
        } finally {
            config.close();
        }
    }

    private static void track(CacheAwareService service, WorkerStatus worker, String requestId) {
        service.trackRoutingPrediction(requestId, RoleType.PREFILL, "default", worker, 1000, 100,
                CacheMatchResult.empty(CacheMatchSource.KVCM), CompletableFuture.completedFuture(null));
    }

    private static Cache<?, ?> associations(CacheAwareService service) {
        Object tracker = ReflectionTestUtils.getField(service, "feedbackTracker");
        return assertInstanceOf(Cache.class, ReflectionTestUtils.getField(tracker, "predictions"));
    }

    private static long maximum(Cache<?, ?> associations) {
        associations.cleanUp();
        return associations.policy().eviction().orElseThrow().getMaximum();
    }

    private static long timeoutMs(Cache<?, ?> associations) {
        return associations.policy().expireAfterAccess().orElseThrow().getExpiresAfter(TimeUnit.MILLISECONDS);
    }

    private static String document(long maximumEntries, long timeoutMs) {
        return """
                {"schemaVersion":3,"observability":{"cacheHit":{"feedbackMaximumEntries":%d,"feedbackTimeoutMs":%d}}}
                """.formatted(maximumEntries, timeoutMs);
    }
}
