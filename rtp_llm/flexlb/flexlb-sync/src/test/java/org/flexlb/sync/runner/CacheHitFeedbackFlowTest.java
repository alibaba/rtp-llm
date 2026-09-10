package org.flexlb.sync.runner;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.hash.RequestBlockHashService;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.CacheMatchFailoverManager;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.cache.match.CacheMetadataUpdateOrchestrator;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
import reactor.netty.resources.LoopResources;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheHitFeedbackFlowTest {
    private final ListAppender<ILoggingEvent> pv = new ListAppender<>();
    private final FlexMonitor monitor = mock(FlexMonitor.class);
    private final LocalStandbyCacheMatchProvider standby = mock(LocalStandbyCacheMatchProvider.class);
    private final CacheMatchFailoverManager failover = mock(CacheMatchFailoverManager.class);
    private EndpointRegistry registry;
    private WorkerDirectory directory;
    private WorkerStatus worker;
    private CacheAwareService cache;
    private CostBasedPrefillStrategy strategy;
    private EngineHealthReporter reporter;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens)");
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setP2pHitDiscount(0.5);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        ConfigService configs = mock(ConfigService.class);
        when(configs.loadBalanceConfig()).thenReturn(config);
        registry = new EndpointRegistry(configs, RunnerTestSupport.eventSink(),
                mock(org.flexlb.service.monitor.BatchSchedulerReporter.class),
                new org.flexlb.balance.scheduler.RouteDeliveryStrategy(
                        mock(org.flexlb.balance.scheduler.RequestRegistry.class),
                        mock(org.flexlb.balance.delivery.DeliveryMetrics.class)),
                new org.flexlb.balance.scheduler.PlacementAvailability());
        directory = new WorkerDirectory(registry);
        worker = RunnerTestSupport.discovered(RoleType.PREFILL, "group", "10.0.0.1", 8080, 8081, "test");
        directory.currentOrDiscover(RoleType.PREFILL, worker.getLogicalIpPort(), () -> worker);
        publish(worker);
        CacheMatchConfiguration cacheConfig = mock(CacheMatchConfiguration.class);
        when(cacheConfig.isKvcmEnabled()).thenReturn(true);
        when(cacheConfig.isLocalStandbyEnabled()).thenReturn(true);
        when(cacheConfig.getLocalStandbyConfig()).thenReturn(new LocalStandbyConfig());
        LocalStandbyComparisonService comparison = new LocalStandbyComparisonService(cacheConfig, standby);
        CacheMetricsReporter cacheMetrics = mock(CacheMetricsReporter.class);
        when(standby.asyncLocalStandbyMatch(any())).thenReturn(CompletableFuture.completedFuture(
                new CacheMatchResult(Map.of(worker.getLogicalIpPort(), HostCacheMatch.local(3)),
                        CacheMatchSource.LOCAL_STANDBY, 1, 100)));
        KvcmCacheMatchProvider kvcm = mock(KvcmCacheMatchProvider.class);
        when(kvcm.findMatchingEngines(anyString(), anyList(), anyLong(), any(), anyString()))
                .thenReturn(Map.of(worker.getLogicalIpPort(), new HostCacheMatch(2, 4, 6)));
        when(failover.activeSource()).thenReturn(CacheMatchSource.KVCM);
        var query = new CacheMatchQueryOrchestrator(mock(LocalSyncCacheMatchProvider.class), kvcm, standby,
                mock(LocalStandbyCacheManager.class), failover, comparison, cacheMetrics, cacheConfig);
        cache = new CacheAwareService(cacheMetrics, query, comparison,
                mock(CacheMetadataUpdateOrchestrator.class), mock(RequestBlockHashService.class));
        EngineGrpcClient engineClient = mock(EngineGrpcClient.class);
        LoopResources loops = mock(LoopResources.class);
        when(engineClient.getEventLoopGroup()).thenReturn(mock(io.netty.channel.EventLoopGroup.class));
        when(loops.onServer(true)).thenReturn(mock(io.netty.channel.EventLoopGroup.class));
        when(loops.onServerSelect(true)).thenReturn(mock(io.netty.channel.EventLoopGroup.class));
        reporter = new EngineHealthReporter(monitor, cacheMetrics, cacheConfig, engineClient, loops, directory);
        strategy = new CostBasedPrefillStrategy(directory, cache, reporter);
        pv.start();
        ((ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger")).addAppender(pv);
    }

    @AfterEach
    void tearDown() {
        ((ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger")).detachAppender(pv);
        pv.stop();
        registry.close();
    }

    @Test
    void realSelectionAndStatusPollingEmitComparisonPvAndMetricsExactlyOnce() throws Exception {
        select("request-1");
        poll(worker, task("request-1", false, 0), false, 2);
        assertTrue(events("cache_hit_comparison").isEmpty(), "missing validity must not mean zero cache hit");
        poll(worker, task("request-1", true, 500), false, 3);
        poll(worker, task("request-1", true, 500), true, 4);
        poll(worker, task("request-1", true, 500), true, 4);
        JsonNode comparison = events("cache_hit_comparison").getFirst();
        assertEquals(1, events("cache_hit_comparison").size());
        assertEquals(500, comparison.path("actual").path("hit").asLong());
        assertEquals(400, comparison.path("kvcm").path("hit").asLong());
        assertEquals(100, comparison.path("kvcm").path("delta").asLong());
        assertEquals(200, comparison.path("kvcm").path("local").path("hit").asLong());
        assertEquals(600, comparison.path("kvcm").path("p2pTotal").path("hit").asLong());
        assertEquals(-100, comparison.path("kvcm").path("p2pTotal").path("delta").asLong());
        assertEquals(300, comparison.path("localStandby").path("hit").asLong());
        assertEquals(200, comparison.path("localStandby").path("delta").asLong());
        verify(monitor, times(1)).report(eq(CACHE_HIT_COMPARISON_DELTA_TOKENS), any(), eq(100.0));
        verify(monitor, times(1)).report(eq(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS), any(), eq(200.0));
        JsonNode status = events("prefill_worker_status").getFirst();
        assertEquals(1, events("prefill_worker_status").size());
        assertEquals(200, status.path("hbmLocalMatchTokens").asLong());
        assertEquals(300, status.path("remoteKvAddedMatchTokens").asLong());
        assertEquals(1200, status.path("firstTokenTimeMs").asLong());
        assertEquals(100, status.path("runningToFirstTokenMs").asLong());
        assertEquals(70, status.path("schedulerWaitMs").asLong());
    }

    @Test
    void truncatedPrefillLengthPreservesLogicalDenominatorAndReportsOnce() throws Exception {
        select("truncated");
        EngineRpcService.TaskInfoPB feedback = task("truncated", true, 500).toBuilder().setInputLength(996).build();
        poll(worker, feedback, false, 2);
        poll(worker, feedback, true, 3);
        assertEquals(1, events("cache_hit_comparison").size());
        JsonNode comparison = events("cache_hit_comparison").getFirst();
        assertEquals(1000, comparison.path("inputTokens").asLong());
        assertEquals(500, comparison.path("actual").path("hit").asLong());
        verify(monitor, times(1)).report(eq(CACHE_HIT_COMPARISON_ACTUAL_RATIO), any(), eq(0.5));
        assertEquals(1, events("prefill_worker_status").size());
        JsonNode status = events("prefill_worker_status").getFirst();
        assertEquals(1000, status.path("inputTokens").asLong());
        assertEquals(996, status.path("engineInputTokens").asLong());
        assertEquals(-4, status.path("inputTokensDelta").asLong());
    }

    @Test
    void fallbackReportsStandbyWithoutPretendingItIsKvcm() throws Exception {
        when(failover.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        select("standby");
        poll(worker, task("standby", true, 0), true, 2);
        JsonNode event = events("cache_hit_comparison").getFirst();
        assertFalse(event.hasNonNull("kvcm"));
        assertEquals("LOCAL_STANDBY", event.path("source").asText());
        assertEquals(0, event.path("actual").path("hit").asLong());
        assertEquals(-300, event.path("localStandby").path("delta").asLong());
        verify(monitor).report(eq(CACHE_HIT_COMPARISON_ACTUAL_TOKENS), any(), eq(0.0));
    }

    @Test
    void failedStandbyDoesNotSuppressKvcmComparison() throws Exception {
        when(standby.asyncLocalStandbyMatch(any())).thenReturn(CompletableFuture.failedFuture(
                new IllegalStateException("standby unavailable")));
        select("failed-standby");
        poll(worker, task("failed-standby", true, 500), true, 2);
        JsonNode event = events("cache_hit_comparison").getFirst();
        assertTrue(event.hasNonNull("kvcm"));
        assertFalse(event.hasNonNull("localStandby"));
        verify(monitor).report(eq(CACHE_HIT_COMPARISON_DELTA_TOKENS), any(), eq(100.0));
    }

    @Test
    void delayedStandbyComparisonDoesNotBlockPollingOrDuplicateFeedback() throws Exception {
        CompletableFuture<CacheMatchResult> delayed = new CompletableFuture<>();
        when(standby.asyncLocalStandbyMatch(any())).thenReturn(delayed);
        select("async");
        poll(worker, task("async", true, 500), true, 2);
        poll(worker, task("async", true, 500), true, 2);
        var lease = worker.tryBeginStatusPoll();
        assertNotNull(lease, "comparison must not retain the poll lease");
        lease.close();
        assertTrue(events("cache_hit_comparison").isEmpty());
        delayed.complete(new CacheMatchResult(Map.of(worker.getLogicalIpPort(), HostCacheMatch.local(3)),
                CacheMatchSource.LOCAL_STANDBY, 1, 100));
        assertEquals(1, events("cache_hit_comparison").size());
        verify(monitor, times(1)).report(eq(CACHE_HIT_COMPARISON_DELTA_TOKENS), any(), eq(100.0));
    }

    @Test
    void unavailableStandbyTimesOutWithoutLosingPrimaryComparison() throws Exception {
        when(standby.asyncLocalStandbyMatch(any())).thenReturn(new CompletableFuture<>());
        select("timeout");
        poll(worker, task("timeout", true, 500), true, 2);
        verify(monitor, timeout(3000).times(1)).report(eq(CACHE_HIT_COMPARISON_DELTA_TOKENS), any(), eq(100.0));
        JsonNode event = events("cache_hit_comparison").getFirst();
        assertEquals(500, event.path("actual").path("hit").asLong());
        assertFalse(event.hasNonNull("localStandby"));
        poll(worker, task("timeout", true, 500), true, 2);
        assertEquals(1, events("cache_hit_comparison").size());
    }

    @Test
    void unknownRequestAndInvalidPrefixNeverFabricateCacheComparison() throws Exception {
        poll(worker, task("untracked", true, 500), true, 2);
        assertTrue(events("cache_hit_comparison").isEmpty());
        select("missing");
        poll(worker, task("missing", false, 0), true, 3);
        assertTrue(events("cache_hit_comparison").isEmpty());
        JsonNode status = events("prefill_worker_status").getFirst();
        assertFalse(status.hasNonNull("actualHitTokens"));
        assertFalse(status.hasNonNull("firstTokenTimeMs"));
        verify(monitor, never()).report(eq(CACHE_HIT_COMPARISON_ACTUAL_TOKENS), any(), anyDouble());
    }

    @Test
    void anotherGenerationCannotConsumeSelectedWorkersPrediction() throws Exception {
        select("generation");
        WorkerStatus replacement = RunnerTestSupport.discovered(RoleType.PREFILL, "group", "10.0.0.1", 8080, 8081, "test");
        var observation = org.flexlb.service.grpc.EngineStatusConverter.convertToStatusObservation(replacement,
                response(task("generation", true, 500), true, 2));
        assertTrue(cache.observeCacheHitFeedback(replacement, observation).isEmpty());
        assertTrue(events("cache_hit_comparison").isEmpty());
        poll(worker, task("generation", true, 500), true, 2);
        assertEquals(1, events("cache_hit_comparison").size());
    }

    @Test
    void feedbackPreservesLogicalEngineIdentityAndExactCachePrediction() throws Exception {
        WorkerStatus engine = WorkerStatus.createDiscovered(RoleType.PREFILL, "group", "10.0.0.1",
                8080, 8081, "test", null, 1, 2);
        CacheMatchResult matches = new CacheMatchResult(Map.of(
                "10.0.0.1:8080@0", HostCacheMatch.local(1),
                "10.0.0.1:8080@1", HostCacheMatch.local(4)), CacheMatchSource.KVCM, 1, 100);
        cache.trackRoutingPrediction("logical-engine", RoleType.PREFILL, "group", engine, 1000, 400, matches);
        var observation = org.flexlb.service.grpc.EngineStatusConverter.convertToStatusObservation(engine,
                response(task("logical-engine", true, 500), true, 2));
        var results = cache.observeCacheHitFeedback(engine, observation);
        assertEquals(1, results.size());
        var comparison = results.getFirst().get();
        assertEquals("10.0.0.1:8080@1", comparison.worker());
        assertEquals(400, comparison.kvcm().local().hit());
        assertEquals(100, comparison.kvcm().local().delta());
        assertTrue(cache.observeCacheHitFeedback(engine, observation).isEmpty());
        JsonNode status = events("prefill_worker_status").getFirst();
        assertEquals("10.0.0.1:8080@1", status.path("worker").asText());
        assertEquals(1, status.path("engineIndex").asInt());
    }

    private void select(String requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1000);
        request.setBlockSize(100);
        request.setBlockCacheKeys(List.of(1L));
        request.setLocalStandbyBlockSize(100);
        request.setLocalStandbyBlockCacheKeys(List.of(1L));
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 60_000));
        var selected = strategy.select(context, RoleType.PREFILL, "group");
        assertEquals(PlacementResult.Status.SUCCESS, selected.status());
        selected.value().close();
    }

    private void publish(WorkerStatus status) {
        var value = RunnerTestSupport.response(status, true, 1, 1_000_000, 1_000_000, 1, Map.of());
        status.lock.lock();
        try {
            registry.publishPreparedEndpoint(status.getLogicalIpPort(), status, status.prepareNewStatus(status.freezeStatusResponse(value)));
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
    }

    private EngineRpcService.TaskInfoPB task(String id, boolean valid, long hit) {
        var task = EngineRpcService.TaskInfoPB.newBuilder().setRequestId(id).setInputLength(1000)
                .setPrefixLength(hit).setPrefixLengthValid(valid).setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING);
        if (valid) {
            task.setRequestReceivedTimeMs(990).setInputQueueEnqueueTimeMs(1000).setInputQueueDrainTimeMs(1010)
                    .setWaitingEnteredTimeMs(1020).setRunningEnteredTimeMs(1100).setRemoteKvWaitMs(10)
                    .setFirstTokenTimeMs(1200).setHbmLocalMatchTokens(hit == 0 ? 0 : 200)
                    .setRemoteKvAddedMatchTokens(hit == 0 ? 0 : 300);
        }
        return task.build();
    }

    private EngineRpcService.WorkerStatusPB response(EngineRpcService.TaskInfoPB task, boolean finished, long version) {
        var result = EngineRpcService.WorkerStatusPB.newBuilder().setRole(RoleType.PREFILL.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL).setAlive(true).setStatusVersion(version)
                .setAvailableKvCache(1_000_000).setTotalKvCache(1_000_000);
        if (finished) {
            result.addFinishedTaskList(task);
        } else {
            result.addRunningTaskInfo(task);
        }
        return result.build();
    }

    private void poll(WorkerStatus status, EngineRpcService.TaskInfoPB task, boolean finished, long version) {
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response(task, finished, version)));
        var lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        new GrpcWorkerStatusRunner("model", status.getLogicalIpPort(), "test", RoleType.PREFILL, "group",
                status, lease, directory, reporter, grpc, 5000, cache, Runnable::run).run();
    }

    private List<JsonNode> events(String event) throws Exception {
        List<JsonNode> result = new java.util.ArrayList<>();
        for (ILoggingEvent item : pv.list) {
            JsonNode node = new ObjectMapper().readTree(item.getFormattedMessage());
            if (event.equals(node.path("event").asText())) {
                result.add(node);
            }
        }
        return result;
    }
}
