package org.flexlb.service;

import org.flexlb.cache.match.theory.TheoryCacheKeyHistory;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.cache.telemetry.TheoryCacheHitStats;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class TheoryCacheHitReporterTest {

    private final List<TheoryCacheHitReporter> reporters = new ArrayList<>();

    private TheoryCacheHitReporter newReporter() {
        TheoryCacheHitReporter reporter = new TheoryCacheHitReporter();
        reporters.add(reporter);
        return reporter;
    }

    @AfterEach
    void closeReporters() {
        reporters.forEach(TheoryCacheHitReporter::closeTheoryLog);
    }

    private static void reportWithConfiguration(TheoryCacheHitReporter reporter,
                                                BalanceContext context) throws Exception {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = context.getConfig();
        Mockito.lenient().when(configService.loadBalanceConfig()).thenReturn(config == null ? new FlexlbConfig() : config);
        inject(reporter, "configService", configService);
        reporter.report(context);
    }

    private static void awaitReports(TheoryCacheHitReporter reporter) throws Exception {
        executor(reporter).submit(() -> {}).get(5L, TimeUnit.SECONDS);
    }

    private static ThreadPoolExecutor executor(TheoryCacheHitReporter reporter) throws Exception {
        Field field = TheoryCacheHitReporter.class.getDeclaredField("theoryHitReportExecutor");
        field.setAccessible(true);
        return (ThreadPoolExecutor) field.get(reporter);
    }

    @Mock
    private CacheMetricsReporter cacheMetricsReporter;

    @Test
    void should_report_request_hits_against_prior_pool() throws Exception {
        TheoryCacheHitReporter reporter = newReporter();
        inject(reporter, "theoryCacheKeyHistory", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        BalanceContext firstContext = mock(BalanceContext.class);
        Request firstRequest = mock(Request.class);
        when(firstContext.getRequest()).thenReturn(firstRequest);
        when(firstRequest.getBlockCacheKeys()).thenReturn(List.of(1L, 2L, 3L));
        when(firstRequest.getSeqLen()).thenReturn(300L);
        when(firstRequest.getCacheKeyBlockSize()).thenReturn(100L);

        BalanceContext secondContext = mock(BalanceContext.class);
        Request secondRequest = mock(Request.class);
        when(secondContext.getRequest()).thenReturn(secondRequest);
        when(secondRequest.getBlockCacheKeys()).thenReturn(List.of(2L, 3L, 4L));
        when(secondRequest.getSeqLen()).thenReturn(300L);
        when(secondRequest.getCacheKeyBlockSize()).thenReturn(100L);

        reportWithConfiguration(reporter, firstContext);
        awaitReports(reporter);
        reportWithConfiguration(reporter, secondContext);
        awaitReports(reporter);

        assertTheoryRequests(0L, 300L, 200L, 300L);
    }

    @Test
    void should_skip_window_write_and_metric_when_window_switch_is_off() throws Exception {
        TheoryCacheHitReporter reporter = newReporter();
        inject(reporter, "theoryCacheKeyHistory", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig disabledConfig = new FlexlbConfig();
        disabledConfig.getObservability().getCacheHit().getRecentKeyWindow()
                .setWriteEnabled(false);
        BalanceContext skippedContext = contextWithConfig(disabledConfig);
        reportWithConfiguration(reporter, skippedContext);
        awaitReports(reporter);

        FlexlbConfig enabledConfig = new FlexlbConfig();
        BalanceContext nextContext = context(enabledConfig, List.of(1L, 2L));
        reportWithConfiguration(reporter, nextContext);
        awaitReports(reporter);

        assertTheoryRequests(0L, 1024L);
    }

    @Test
    void should_write_window_but_skip_metric_when_metric_switch_is_off() throws Exception {
        TheoryCacheHitReporter reporter = newReporter();
        inject(reporter, "theoryCacheKeyHistory", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig metricOffConfig = new FlexlbConfig();
        metricOffConfig.getObservability().getCacheHit().setMetricsEnabled(false);
        BalanceContext firstContext = context(metricOffConfig, List.of(1L, 2L));
        reportWithConfiguration(reporter, firstContext);
        awaitReports(reporter);
        verifyNoInteractions(cacheMetricsReporter);

        FlexlbConfig enabledConfig = new FlexlbConfig();
        BalanceContext secondContext = context(enabledConfig, List.of(2L, 3L));
        reportWithConfiguration(reporter, secondContext);
        awaitReports(reporter);

        assertTheoryRequests(256L, 1024L);
    }

    @Test
    void should_record_zero_theory_hit_for_empty_cache_key_request() throws Exception {
        TheoryCacheHitReporter reporter = newReporter();
        inject(reporter, "theoryCacheKeyHistory", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig config = new FlexlbConfig();
        reportWithConfiguration(reporter, context(config, List.of(), 128L, 64L));
        awaitReports(reporter);
        reportWithConfiguration(reporter, context(config, List.of(1L), 128L, 64L));
        awaitReports(reporter);

        assertTheoryRequests(0L, 128L, 0L, 128L);
        verify(cacheMetricsReporter, Mockito.times(2)).reportTheoryCacheHitMetrics(
                Mockito.any(TheoryCacheHitStats.Snapshot.class));
    }

    @Test
    void should_report_theory_hit_tokens_over_input_tokens() throws Exception {
        TheoryCacheHitReporter reporter = newReporter();
        inject(reporter, "theoryCacheKeyHistory", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig config = new FlexlbConfig();
        reportWithConfiguration(reporter, context(config, List.of(1L, 2L, 3L), 1024L, 256L));
        awaitReports(reporter);
        reportWithConfiguration(reporter, context(config, List.of(2L, 3L, 4L), 1024L, 256L));
        awaitReports(reporter);

        ArgumentCaptor<TheoryCacheHitStats.Snapshot> captor =
                ArgumentCaptor.forClass(TheoryCacheHitStats.Snapshot.class);
        verify(cacheMetricsReporter, Mockito.times(2)).reportTheoryCacheHitMetrics(captor.capture());
        TheoryCacheHitStats.Snapshot second = captor.getAllValues().get(1);
        assertEquals(512L, second.getRequestHitCount());
        assertEquals(1024L, second.getRequestTotalCount());
        assertEquals(512L, second.getAllHitCount());
        assertEquals(2048L, second.getAllTotalCount());
    }

    @Test
    void should_report_theory_hit_tokens_with_page_rr_cache_key_block_size() throws Exception {
        TheoryCacheHitReporter reporter = newReporter();
        inject(reporter, "theoryCacheKeyHistory", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig config = new FlexlbConfig();
        reportWithConfiguration(reporter, context(config, List.of(13L, 17L), 2048L, 1024L));
        awaitReports(reporter);
        reportWithConfiguration(reporter, context(config, List.of(17L, 21L), 2048L, 1024L));
        awaitReports(reporter);

        assertTheoryRequests(0L, 2048L, 1024L, 2048L);
    }

    @Test
    void should_return_without_waiting_and_read_configuration_on_background_thread() throws Exception {
        TheoryCacheHitReporter reporter = newReporter();
        inject(reporter, "theoryCacheKeyHistory", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);
        ThreadPoolExecutor executor = executor(reporter);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        executor.execute(() -> {
            started.countDown();
            awaitLatch(release);
        });
        assertTrue(started.await(5L, TimeUnit.SECONDS));
        try {
            FlexlbConfig config = new FlexlbConfig();
            List<Long> keys = new ArrayList<>(List.of(1L));
            BalanceContext submitted = context(config, keys, 128L, 64L);
            assertTimeoutPreemptively(Duration.ofSeconds(1L), () -> reportWithConfiguration(reporter, submitted));
            verifyNoInteractions(cacheMetricsReporter);
            config.getObservability().getCacheHit().setMetricsEnabled(false);
        } finally {
            release.countDown();
        }
        awaitReports(reporter);
        reportWithConfiguration(reporter, context(new FlexlbConfig(), List.of(1L), 128L, 64L));
        awaitReports(reporter);
        assertTheoryRequests(64L, 128L);
    }

    @Test
    void should_drop_report_when_queue_is_full_without_running_on_caller() throws Exception {
        TheoryCacheHitReporter reporter = newReporter();
        TheoryCacheKeyHistory historyWindow = mock(TheoryCacheKeyHistory.class);
        inject(reporter, "theoryCacheKeyHistory", historyWindow);
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);
        ThreadPoolExecutor executor = executor(reporter);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        executor.execute(() -> {
            started.countDown();
            awaitLatch(release);
        });
        assertTrue(started.await(5L, TimeUnit.SECONDS));
        try {
            while (executor.getQueue().remainingCapacity() > 0) {
                executor.execute(() -> {});
            }
            BalanceContext input = contextWithConfig(new FlexlbConfig());
            assertTimeoutPreemptively(Duration.ofSeconds(1L), () -> reportWithConfiguration(reporter, input));
            verifyNoInteractions(cacheMetricsReporter, historyWindow);
        } finally {
            release.countDown();
        }
        executor.shutdown();
        assertTrue(executor.awaitTermination(5L, TimeUnit.SECONDS));
        verifyNoInteractions(cacheMetricsReporter, historyWindow);
    }

    private static void awaitLatch(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5L, TimeUnit.SECONDS));
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(interrupted);
        }
    }

    private void assertTheoryRequests(long... hitAndInputTokens) {
        ArgumentCaptor<TheoryCacheHitStats.Snapshot> captor =
                ArgumentCaptor.forClass(TheoryCacheHitStats.Snapshot.class);
        verify(cacheMetricsReporter, Mockito.times(hitAndInputTokens.length / 2))
                .reportTheoryCacheHitMetrics(captor.capture());
        List<TheoryCacheHitStats.Snapshot> snapshots = captor.getAllValues();
        for (int i = 0; i < snapshots.size(); i++) {
            assertEquals(hitAndInputTokens[i * 2], snapshots.get(i).getRequestHitCount());
            assertEquals(hitAndInputTokens[i * 2 + 1], snapshots.get(i).getRequestTotalCount());
        }
    }

    private static void inject(Object target, String fieldName, Object value) throws Exception {
        Field field = TheoryCacheHitReporter.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static TheoryCacheKeyHistory smallWindow() {
        FlexlbConfig config = new FlexlbConfig();
        config.getObservability().getCacheHit().getRecentKeyWindow()
                .setDurationMs(60_000L);
        config.getObservability().getCacheHit().getRecentKeyWindow()
                .setMaxKeyOccurrences(3_200L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        WorkerStatusProvider workerStatuses = mock(WorkerStatusProvider.class);
        WorkerStatus worker = mock(WorkerStatus.class);
        WorkerStatus.EngineObservation capacity = mock(WorkerStatus.EngineObservation.class);
        when(workerStatuses.getWorkerStatuses(RoleType.PREFILL, null)).thenReturn(List.of(worker));
        when(worker.committedEngineObservation()).thenReturn(capacity);
        when(capacity.totalKvCacheTokens()).thenReturn(320_000L);
        when(capacity.blockSize()).thenReturn(100L);
        return new TheoryCacheKeyHistory(configService, workerStatuses);
    }

    private static BalanceContext contextWithConfig(FlexlbConfig config) {
        BalanceContext balanceContext = mock(BalanceContext.class);
        when(balanceContext.getConfig()).thenReturn(config);
        return balanceContext;
    }

    private static BalanceContext context(FlexlbConfig config, List<Long> cacheKeys) {
        return context(config, cacheKeys, 1024L, 256L);
    }

    private static BalanceContext context(FlexlbConfig config, List<Long> cacheKeys, long seqLen, long cacheKeyBlockSize) {
        BalanceContext balanceContext = mock(BalanceContext.class);
        Request request = mock(Request.class);
        Mockito.lenient().when(balanceContext.getRequestId()).thenReturn("trace-request");
        when(balanceContext.getConfig()).thenReturn(config);
        when(balanceContext.getRequest()).thenReturn(request);
        when(request.getBlockCacheKeys()).thenReturn(cacheKeys);
        when(request.getSeqLen()).thenReturn(seqLen);
        when(request.getCacheKeyBlockSize()).thenReturn(cacheKeyBlockSize);
        return balanceContext;
    }
}
