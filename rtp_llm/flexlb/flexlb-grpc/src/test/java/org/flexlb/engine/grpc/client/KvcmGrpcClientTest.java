package org.flexlb.engine.grpc.client;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.fasterxml.jackson.databind.JsonNode;
import io.grpc.Deadline;
import io.grpc.Status.Code;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.config.KvcmCacheMatchingConfig.BackendType;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.monitor.KvcmMetricsReporter;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.kvcm.grpc.Status;
import org.flexlb.kvcm.grpc.StorageType;
import org.flexlb.listener.ApplicationWarmupState;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import static io.grpc.Status.DEADLINE_EXCEEDED;
import static io.grpc.Status.UNAVAILABLE;
import static io.grpc.Status.fromThrowable;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class KvcmGrpcClientTest {

    private KvcmGrpcClient client;
    private KvcmMetaServiceClient healthMetaService;
    private ApplicationWarmupState healthWarmup;

    @Test
    void shutdownInterruptsAnActiveServiceStateRefresh() throws InterruptedException {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);
        CountDownLatch refreshStarted = new CountDownLatch(1);
        CountDownLatch refreshInterrupted = new CountDownLatch(1);
        CountDownLatch releaseRefresh = new CountDownLatch(1);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver = mock(KvcmWorkerMetadataResolver.class);
        when(leaderResolver.refresh()).thenAnswer(ignored -> {
            refreshStarted.countDown();
            try {
                releaseRefresh.await(5, TimeUnit.SECONDS);
                return true;
            } catch (InterruptedException error) {
                refreshInterrupted.countDown();
                Thread.currentThread().interrupt();
                return false;
            }
        });
        client = createClient(configuration, mock(KvcmMetaServiceClient.class), leaderResolver,
                metadataResolver, mock(GrpcReporter.class));
        try {
            assertTrue(refreshStarted.await(2, TimeUnit.SECONDS));

            client.shutdown();

            assertTrue(refreshInterrupted.await(2, TimeUnit.SECONDS));
            verify(metadataResolver, never()).refreshNamespacesAndQueryTypes();
        } finally {
            releaseRefresh.countDown();
        }
    }

    @Test
    @DisplayName("真实查询失败后，连续成功查询达到阈值才恢复健康")
    void querySuccessesRecoverUnhealthyClientAtTheExistingRecoveryThreshold() throws Exception {
        createHealthTestClient();
        failQuery();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        succeedQuery();
        succeedQuery();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());

        succeedQuery();

        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
        assertEquals("cache query recovery threshold reached", client.healthSnapshot().lastStateChangeReason());
        assertEquals(0, client.healthSnapshot().consecutiveQueryFailures());
        assertEquals(0, client.healthSnapshot().consecutiveHeartbeatFailures());
    }

    @Test
    void successfulQueriesAndHeartbeatsDoNotCombineIntoEarlyRecovery() throws Exception {
        createHealthTestClient();
        failQuery();
        succeedQuery();
        client.refreshKvcmServiceStateSafely();
        succeedQuery();
        client.refreshKvcmServiceStateSafely();

        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        assertEquals(2, client.healthSnapshot().consecutiveHeartbeatSuccesses());
        client.refreshKvcmServiceStateSafely();
        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
        assertEquals("heartbeat recovery threshold reached", client.healthSnapshot().lastStateChangeReason());
    }

    @Test
    void eitherFailureInterruptsBothRecoverySuccessSequences() throws Exception {
        KvcmLeaderResolver leader = createHealthTestClient();
        failQuery();
        succeedQuery();
        succeedQuery();
        when(leader.refresh()).thenReturn(false);
        client.refreshKvcmServiceStateSafely();
        succeedQuery();
        succeedQuery();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());

        when(leader.refresh()).thenReturn(true);
        client.refreshKvcmServiceStateSafely();
        client.refreshKvcmServiceStateSafely();
        failQuery();
        assertEquals(0, client.healthSnapshot().consecutiveHeartbeatSuccesses());
        succeedQuery();
        succeedQuery();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        succeedQuery();
        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
    }

    @Test
    void queryRecoveryPublishesOneHealthTransitionAndIgnoresWarmupSuccesses() throws Exception {
        createHealthTestClient();
        failQuery();
        List<KvcmHealthSnapshot> updates = new ArrayList<>();
        client.setHealthSnapshotListener(updates::add);
        healthWarmup.setWarmupFinished(false);
        for (int query = 0; query < 5; query++) {
            succeedQuery();
        }
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        healthWarmup.setWarmupFinished(true);
        succeedQuery();
        succeedQuery();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
        succeedQuery();
        succeedQuery();
        assertEquals(1, updates.size());
        assertTrue(updates.getFirst().isHealthy());
    }

    private KvcmLeaderResolver createHealthTestClient() throws InterruptedException {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(0);
        runtimeConfig.setRecoverySuccessThreshold(3);
        runtimeConfig.setQueryFailureThreshold(1);
        runtimeConfig.setHeartbeatFailureThreshold(1);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);
        KvcmLeaderResolver leader = mock(KvcmLeaderResolver.class);
        when(leader.refresh()).thenReturn(true);
        when(leader.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        KvcmWorkerMetadataResolver metadata = mock(KvcmWorkerMetadataResolver.class);
        when(metadata.resolveNamespace(RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadata.resolveQueryType(RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        CountDownLatch initialRefresh = new CountDownLatch(1);
        doAnswer(ignored -> {
            initialRefresh.countDown();
            return null;
        }).when(metadata).refreshNamespacesAndQueryTypes();
        healthMetaService = mock(KvcmMetaServiceClient.class);
        when(healthMetaService.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenReturn(GetHostCacheStateResponse.newBuilder().setHeader(okHeader()).build());
        healthWarmup = new ApplicationWarmupState();
        healthWarmup.setWarmupFinished(true);
        client = new KvcmGrpcClient(configuration, healthMetaService, leader, metadata, healthWarmup,
                mock(GrpcReporter.class), new KvcmMetricsReporter(NoOpFlexMonitor.getInstance()));
        assertTrue(initialRefresh.await(2, TimeUnit.SECONDS));
        return leader;
    }

    private void failQuery() {
        IllegalStateException failure = new IllegalStateException("meta service unavailable");
        when(healthMetaService.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenThrow(failure)
                .thenReturn(GetHostCacheStateResponse.newBuilder().setHeader(okHeader()).build());
        assertSame(failure, assertThrows(IllegalStateException.class, this::succeedQuery));
    }

    private void succeedQuery() {
        client.findMatchingEngines("health-query", List.of(11L), 2192L, RoleType.PREFILL, "default");
    }

    @Test
    @DisplayName("warmup 期间心跳失败不累计，online 后才影响健康状态")
    void warmupHeartbeatDoesNotCountFailures() throws InterruptedException {
        KvcmLeaderResolver leader = createHealthTestClient();
        healthWarmup.setWarmupFinished(false);
        when(leader.refresh()).thenReturn(false);

        client.refreshKvcmServiceStateSafely();

        assertEquals(0, client.healthSnapshot().consecutiveHeartbeatFailures());
        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
        assertTrue(client.healthSnapshot().lastHeartbeatFailureTimeMs() > 0);
        healthWarmup.setWarmupFinished(true);
        client.refreshKvcmServiceStateSafely();
        assertEquals(1, client.healthSnapshot().consecutiveHeartbeatFailures());
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
    }

    @AfterEach
    void tearDown() {
        if (client != null) {
            client.shutdown();
        }
    }

    @Test
    void reportsLogicalWorkerMatchesAndMillisecondCallMetrics() throws Exception {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmConfig config = new KvcmConfig();
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(0);
        runtimeConfig.setMedium(List.of("hbm", "kvs"));
        runtimeConfig.setTopKHostCount(5);
        runtimeConfig.setBackendTypes(List.of(BackendType.ST_TAIRMEMPOOL, BackendType.ST_EVENT_REPORT_L2));
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(config);
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metaServiceClient.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenAnswer(invocation -> {
                    TimeUnit.MILLISECONDS.sleep(5L);
                    return GetHostCacheStateResponse.newBuilder()
                        .setHeader(okHeader())
                        .addHosts(HostCacheMatch.newBuilder()
                                .setHostIpPort("10.0.0.1:8601@1")
                                .setLocal(2)
                                .setGlobal(10))
                        .build();
                });

        GrpcReporter reporter = mock(GrpcReporter.class);
        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                reporter);

        long startedAt = System.nanoTime();
        var result =
                client.findMatchingEngines(
                        "request-1", List.of(11L, 22L), 2192L,
                        RoleType.PREFILL, "default");

        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startedAt) + 1L;
        ArgumentCaptor<Long> duration = ArgumentCaptor.forClass(Long.class);
        verify(reporter).reportCallMetrics(eq("KVCM_GET_HOST_CACHE_STATE"), duration.capture(), anyInt(), eq(false));
        assertTrue(duration.getValue() >= 5L);
        assertTrue(duration.getValue() <= elapsedMs, "Call duration must use milliseconds");

        ArgumentCaptor<GetHostCacheStateRequest> sentRequest =
                ArgumentCaptor.forClass(GetHostCacheStateRequest.class);
        verify(metaServiceClient).getHostCacheState(any(), sentRequest.capture(), any(Deadline.class));
        assertEquals(List.of("hbm", "kvs"), sentRequest.getValue().getMediumList());
        assertEquals(5, sentRequest.getValue().getTopKHostCount());
        assertEquals(List.of(StorageType.ST_TAIRMEMPOOL, StorageType.ST_EVENT_REPORT_L2),
                sentRequest.getValue().getBackendTypesList());

        assertEquals(2, result.get("10.0.0.1:8601@1").localMatchBlocks());
        assertEquals(10, result.get("10.0.0.1:8601@1").globalMatchBlocks());
    }

    @Test
    void logsOnlyRequestIdAndAllHostBlockCountsBeforeHostFiltering() throws Exception {
        createHealthTestClient();
        var errorResponse = GetHostCacheStateResponse.newBuilder()
                .setHeader(CommonResponseHeader.newBuilder().setRequestId("server-request")
                        .setTracerResult("trace-result")
                        .setStatus(Status.newBuilder().setCode(ErrorCode.SERVICE_NOT_READY)
                                .setMessage("not ready\nretry")))
                .build();
        var matchesResponse = GetHostCacheStateResponse.newBuilder().setHeader(okHeader())
                .addHosts(HostCacheMatch.newBuilder().setHostIpPort("worker-1").setLocal(2).setGlobal(98))
                .addHosts(HostCacheMatch.newBuilder().setHostIpPort("worker-2").setGlobal(4))
                .addHosts(HostCacheMatch.newBuilder().setHostIpPort("worker-3"))
                .addHosts(HostCacheMatch.newBuilder().setHostIpPort("worker-1").setGlobal(5))
                .addHosts(HostCacheMatch.newBuilder().setGlobal(6))
                .build();
        when(healthMetaService.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenReturn(errorResponse, matchesResponse);
        Logger logger = (Logger) LoggerFactory.getLogger("kvcmLogger");
        Level originalLevel = logger.getLevel();
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.setLevel(Level.INFO);
        logger.addAppender(appender);
        try {
            assertThrows(KvcmQueryException.class, this::succeedQuery);
            var matches = client.findMatchingEngines("query-log", List.of(11L, 22L), 2192L,
                    RoleType.PREFILL, "default");
            assertEquals(3, matches.size());
            assertEquals(5, matches.get("worker-1").globalMatchBlocks());
            assertEquals(2, appender.list.size());
            var failure = JsonUtils.toObject(appender.list.getFirst().getFormattedMessage(), JsonNode.class);
            assertEquals(2, failure.size());
            assertEquals("health-query", failure.path("requestId").asText());
            assertEquals(0, failure.path("hosts").size());
            var success = JsonUtils.toObject(appender.list.get(1).getFormattedMessage(), JsonNode.class);
            assertEquals("query-log", success.path("requestId").asText());
            assertEquals(2, success.size());
            assertEquals(5, success.path("hosts").size());
            assertEquals(3, success.at("/hosts/0").size());
            assertEquals("worker-1", success.at("/hosts/0/hostIpPort").asText());
            assertEquals(2, success.at("/hosts/0/local").asLong());
            assertEquals(98, success.at("/hosts/0/global").asLong());
            assertEquals(0, success.at("/hosts/2/global").asLong());
            assertEquals("worker-1", success.at("/hosts/3/hostIpPort").asText());
            assertEquals("", success.at("/hosts/4/hostIpPort").asText());
            for (var event : appender.list) {
                assertTrue(!event.getFormattedMessage().contains("\n"));
            }
        } finally {
            logger.detachAppender(appender);
            logger.setLevel(originalLevel);
            appender.stop();
        }
    }

    @Test
    void queriesTopThreeHostsWithNoBackendsByDefault() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(0);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metaServiceClient.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenReturn(GetHostCacheStateResponse.newBuilder()
                        .setHeader(okHeader())
                        .build());

        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        client.findMatchingEngines(
                "request-3", List.of(11L), 2192L, RoleType.PREFILL, "default");

        ArgumentCaptor<GetHostCacheStateRequest> sentRequest =
                ArgumentCaptor.forClass(GetHostCacheStateRequest.class);
        verify(metaServiceClient).getHostCacheState(any(), sentRequest.capture(), any(Deadline.class));
        assertEquals(3, sentRequest.getValue().getTopKHostCount());
        assertTrue(sentRequest.getValue().getBackendTypesList().isEmpty());
        assertTrue(sentRequest.getValue().getMediumList().isEmpty());
    }

    @Test
    void skipsQueriesWhenDisabled() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        when(configuration.isKvcmEnabled()).thenReturn(false);
        client = createClient(
                configuration,
                mock(KvcmMetaServiceClient.class),
                mock(KvcmLeaderResolver.class),
                mock(KvcmWorkerMetadataResolver.class),
                mock(GrpcReporter.class));

        assertTrue(client.findMatchingEngines(
                "request-2", List.of(11L), 2192L,
                RoleType.PREFILL, "default").isEmpty());
    }

    @Test
    @DisplayName("缺少 worker 元数据时查询失败且不发 RPC，并请求后台刷新")
    void missingWorkerMetadataFailsQueriesAndRequestsRefresh() throws Exception {
        KvcmMetaServiceClient metaService = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leader = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadata = mock(KvcmWorkerMetadataResolver.class);
        when(leader.refresh()).thenReturn(true);
        CountDownLatch initialRefreshFinished = new CountDownLatch(1);
        doAnswer(ignored -> {
            initialRefreshFinished.countDown();
            return null;
        }).when(metadata).refreshNamespacesAndQueryTypes();
        client = createQueryTestClient(metaService, leader, metadata, mock(GrpcReporter.class));
        assertTrue(initialRefreshFinished.await(2, TimeUnit.SECONDS));
        CountDownLatch refreshStarted = new CountDownLatch(1);
        CountDownLatch releaseRefresh = new CountDownLatch(1);
        doAnswer(ignored -> {
            refreshStarted.countDown();
            releaseRefresh.await();
            return null;
        }).when(metadata).refreshNamespacesAndQueryTypes();
        try {
            KvcmQueryException first = assertThrows(KvcmQueryException.class,
                    () -> client.findMatchingEngines("missing-1", List.of(11L), 2192L, RoleType.PREFILL, "default"));
            assertTrue(refreshStarted.await(2, TimeUnit.SECONDS));
            KvcmQueryException second = assertThrows(KvcmQueryException.class,
                    () -> client.findMatchingEngines("missing-2", List.of(11L), 2192L, RoleType.PREFILL, "default"));

            assertTrue(first.getMessage().contains("worker metadata unavailable for role=PREFILL, group=default"));
            assertEquals(first.getMessage(), second.getMessage());
            assertEquals(2, client.healthSnapshot().consecutiveQueryFailures());
            verify(metaService, never()).getHostCacheState(any(), any(), any(Deadline.class));
        } finally {
            releaseRefresh.countDown();
        }
    }

    @Test
    void failingGrpcMetricsDoNotChangeSuccessOrMaskRequestFailure() {
        KvcmMetaServiceClient metaService = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leader = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadata = mock(KvcmWorkerMetadataResolver.class);
        when(leader.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metadata.resolveNamespace(RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadata.resolveQueryType(RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(metaService.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenReturn(GetHostCacheStateResponse.newBuilder().setHeader(okHeader()).build())
                .thenThrow(DEADLINE_EXCEEDED.asRuntimeException());
        GrpcReporter reporter = mock(GrpcReporter.class);
        doThrow(new IllegalStateException("monitor unavailable")).when(reporter)
                .reportCallMetrics(eq("KVCM_GET_HOST_CACHE_STATE"), anyLong(), anyInt(), eq(false));
        client = createQueryTestClient(metaService, leader, metadata, reporter);

        var first = client.findMatchingEngines(
                "success", List.of(11L), 2192L, RoleType.PREFILL, "default");
        KvcmQueryException failure = assertThrows(KvcmQueryException.class,
                () -> client.findMatchingEngines("timeout", List.of(11L), 2192L, RoleType.PREFILL, "default"));

        assertTrue(first.isEmpty());
        assertEquals(Code.DEADLINE_EXCEEDED, fromThrowable(failure).getCode());
        assertEquals(1, client.healthSnapshot().consecutiveQueryFailures());
        verify(metaService, times(2)).getHostCacheState(any(), any(), any(Deadline.class));
    }

    @Test
    void usesTheLatestRuntimeConfigForEveryQuery() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig initialConfig = new KvcmCacheMatchingConfig();
        initialConfig.setLeaderRefreshIntervalMs(60_000);
        initialConfig.setMaxQueryRetryCount(0);
        initialConfig.setRequestTimeoutMs(500L);
        AtomicReference<KvcmCacheMatchingConfig> runtimeConfig =
                new AtomicReference<>(initialConfig);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenAnswer(ignored -> runtimeConfig.get());

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metaServiceClient.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenReturn(GetHostCacheStateResponse.newBuilder()
                        .setHeader(okHeader())
                        .build());

        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        client.findMatchingEngines(
                "request-4", List.of(11L), 2192L, RoleType.PREFILL, "default");

        KvcmCacheMatchingConfig updatedConfig = new KvcmCacheMatchingConfig();
        updatedConfig.setLeaderRefreshIntervalMs(60_000);
        updatedConfig.setMaxQueryRetryCount(0);
        updatedConfig.setTopKHostCount(0);
        updatedConfig.setBackendTypes(List.of(BackendType.ST_EVENT_REPORT_L2));
        updatedConfig.setMedium(List.of("kvs"));
        updatedConfig.setRequestTimeoutMs(2000L);
        runtimeConfig.set(updatedConfig);
        AtomicLong updatedBudgetMs = new AtomicLong();
        when(metaServiceClient.getHostCacheState(any(), any(), any(Deadline.class))).thenAnswer(invocation -> {
            Deadline budget = invocation.getArgument(2);
            updatedBudgetMs.set(budget.timeRemaining(TimeUnit.MILLISECONDS));
            return GetHostCacheStateResponse.newBuilder().setHeader(okHeader()).build();
        });
        client.findMatchingEngines(
                "request-5", List.of(11L), 2192L, RoleType.PREFILL, "default");

        ArgumentCaptor<GetHostCacheStateRequest> sentRequest =
                ArgumentCaptor.forClass(GetHostCacheStateRequest.class);
        ArgumentCaptor<Deadline> deadline = ArgumentCaptor.forClass(Deadline.class);
        verify(metaServiceClient, times(2))
                .getHostCacheState(any(), sentRequest.capture(), deadline.capture());
        assertEquals(3, sentRequest.getAllValues().get(0).getTopKHostCount());
        assertEquals(0, sentRequest.getAllValues().get(1).getTopKHostCount());
        assertTrue(sentRequest.getAllValues().get(0).getBackendTypesList().isEmpty());
        assertEquals(List.of(StorageType.ST_EVENT_REPORT_L2),
                sentRequest.getAllValues().get(1).getBackendTypesList());
        assertEquals(List.of("kvs"), sentRequest.getAllValues().get(1).getMediumList());
        assertTrue(updatedBudgetMs.get() <= 2000L);
        assertTrue(updatedBudgetMs.get() > 1000L, "The updated timeout must differ from the original 500 ms");
    }

    @Test
    void reusesRuntimeConfigSnapshotAcrossQueryRetries() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig initialConfig = new KvcmCacheMatchingConfig();
        initialConfig.setLeaderRefreshIntervalMs(60_000);
        initialConfig.setMaxQueryRetryCount(1);
        initialConfig.setTopKHostCount(4);
        initialConfig.setBackendTypes(List.of(BackendType.ST_TAIRMEMPOOL, BackendType.ST_EVENT_REPORT_L2));
        initialConfig.setMedium(List.of("hbm"));
        initialConfig.setRequestTimeoutMs(700L);
        AtomicReference<KvcmCacheMatchingConfig> runtimeConfig =
                new AtomicReference<>(initialConfig);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenAnswer(ignored -> runtimeConfig.get());

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        AtomicInteger attempts = new AtomicInteger();
        when(metaServiceClient.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenAnswer(ignored -> {
                    if (attempts.getAndIncrement() == 0) {
                        KvcmCacheMatchingConfig updatedConfig = new KvcmCacheMatchingConfig();
                        updatedConfig.setLeaderRefreshIntervalMs(60_000);
                        updatedConfig.setMaxQueryRetryCount(0);
                        updatedConfig.setTopKHostCount(9);
                        updatedConfig.setBackendTypes(List.of());
                        updatedConfig.setMedium(List.of("kvs"));
                        updatedConfig.setRequestTimeoutMs(900L);
                        runtimeConfig.set(updatedConfig);
                        throw UNAVAILABLE.asRuntimeException();
                    }
                    return GetHostCacheStateResponse.newBuilder()
                            .setHeader(okHeader())
                            .build();
                });

        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        client.findMatchingEngines(
                "request-6", List.of(11L), 2192L, RoleType.PREFILL, "default");

        ArgumentCaptor<GetHostCacheStateRequest> sentRequest =
                ArgumentCaptor.forClass(GetHostCacheStateRequest.class);
        ArgumentCaptor<Deadline> deadline = ArgumentCaptor.forClass(Deadline.class);
        verify(metaServiceClient, times(2))
                .getHostCacheState(any(), sentRequest.capture(), deadline.capture());
        for (GetHostCacheStateRequest request : sentRequest.getAllValues()) {
            assertEquals(4, request.getTopKHostCount());
            assertEquals(List.of(StorageType.ST_TAIRMEMPOOL, StorageType.ST_EVENT_REPORT_L2),
                    request.getBackendTypesList());
            assertEquals(List.of("hbm"), request.getMediumList());
        }
        assertSame(deadline.getAllValues().get(0), deadline.getAllValues().get(1),
                "Retries must share the original query deadline");
    }

    @Test
    void stopsRetryingWhenTheTotalQueryBudgetExpires() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(5);
        runtimeConfig.setRequestTimeoutMs(300L);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);
        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver = mock(KvcmWorkerMetadataResolver.class);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metadataResolver.resolveNamespace(RoleType.PREFILL, "default", 2192L))
                .thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(RoleType.PREFILL, "default"))
                .thenReturn(QueryType.QT_PREFIX_MATCH);
        when(metaServiceClient.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenAnswer(invocation -> {
                    Deadline deadline = invocation.getArgument(2);
                    TimeUnit.NANOSECONDS.sleep(Math.max(0L,
                            deadline.timeRemaining(TimeUnit.NANOSECONDS)) + 1_000_000L);
                    throw DEADLINE_EXCEEDED.asRuntimeException();
                });
        client = createClient(configuration, metaServiceClient, leaderResolver,
                metadataResolver, mock(GrpcReporter.class));

        assertThrows(KvcmQueryException.class, () -> client.findMatchingEngines(
                "timeout-query", List.of(11L), 2192L, RoleType.PREFILL, "default"));

        verify(metaServiceClient).getHostCacheState(any(), any(), any(Deadline.class));
    }

    @Test
    void appliesLatestHeartbeatThresholdsWithoutRestart() throws Exception {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig initialConfig = new KvcmCacheMatchingConfig();
        initialConfig.setLeaderRefreshIntervalMs(60_000);
        initialConfig.setHeartbeatFailureThreshold(5);
        initialConfig.setRecoverySuccessThreshold(5);
        AtomicReference<KvcmCacheMatchingConfig> runtimeConfig =
                new AtomicReference<>(initialConfig);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenAnswer(ignored -> runtimeConfig.get());

        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        AtomicReference<Boolean> refreshResult = new AtomicReference<>(true);
        CountDownLatch initialRefresh = new CountDownLatch(1);
        when(leaderResolver.refresh()).thenAnswer(ignored -> refreshResult.get());
        doAnswer(ignored -> {
            initialRefresh.countDown();
            return null;
        }).when(metadataResolver).refreshNamespacesAndQueryTypes();
        client = createClient(
                configuration,
                mock(KvcmMetaServiceClient.class),
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));
        assertTrue(initialRefresh.await(2, TimeUnit.SECONDS));

        KvcmCacheMatchingConfig failureConfig = new KvcmCacheMatchingConfig();
        failureConfig.setHeartbeatFailureThreshold(1);
        failureConfig.setRecoverySuccessThreshold(5);
        runtimeConfig.set(failureConfig);
        refreshResult.set(false);
        client.refreshKvcmServiceStateSafely();
        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());

        KvcmCacheMatchingConfig recoveryConfig = new KvcmCacheMatchingConfig();
        recoveryConfig.setHeartbeatFailureThreshold(1);
        recoveryConfig.setRecoverySuccessThreshold(1);
        runtimeConfig.set(recoveryConfig);
        refreshResult.set(true);
        client.refreshKvcmServiceStateSafely();
        assertEquals(KvcmHealthState.HEALTHY, client.healthSnapshot().state());
    }

    @Test
    void appliesLatestQueryFailureThresholdWithoutRestart() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig initialConfig = new KvcmCacheMatchingConfig();
        initialConfig.setLeaderRefreshIntervalMs(60_000);
        initialConfig.setMaxQueryRetryCount(0);
        initialConfig.setQueryFailureThreshold(10);
        AtomicReference<KvcmCacheMatchingConfig> runtimeConfig =
                new AtomicReference<>(initialConfig);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenAnswer(ignored -> runtimeConfig.get());

        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        KvcmWorkerMetadataResolver metadataResolver =
                mock(KvcmWorkerMetadataResolver.class);
        when(leaderResolver.refresh()).thenReturn(true);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        when(metadataResolver.resolveNamespace(
                RoleType.PREFILL, "default", 2192L)).thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(
                RoleType.PREFILL, "default")).thenReturn(QueryType.QT_PREFIX_MATCH);
        when(metaServiceClient.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenThrow(UNAVAILABLE.asRuntimeException());
        client = createClient(
                configuration,
                metaServiceClient,
                leaderResolver,
                metadataResolver,
                mock(GrpcReporter.class));

        KvcmCacheMatchingConfig updatedConfig = new KvcmCacheMatchingConfig();
        updatedConfig.setMaxQueryRetryCount(0);
        updatedConfig.setQueryFailureThreshold(1);
        runtimeConfig.set(updatedConfig);
        assertThrows(KvcmQueryException.class, () -> client.findMatchingEngines(
                "request-7", List.of(11L), 2192L, RoleType.PREFILL, "default"));

        assertEquals(KvcmHealthState.UNHEALTHY, client.healthSnapshot().state());
    }

    @Test
    void reportsFailedGrpcCallMetricsAndLeavesSharedClientLifecycleToSpring() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(0);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);
        KvcmMetaServiceClient metaServiceClient = mock(KvcmMetaServiceClient.class);
        KvcmLeaderResolver leaderResolver = mock(KvcmLeaderResolver.class);
        when(leaderResolver.resolve()).thenReturn(new GrpcTarget("127.0.0.1", 7001));
        KvcmWorkerMetadataResolver metadataResolver = mock(KvcmWorkerMetadataResolver.class);
        when(metadataResolver.resolveNamespace(RoleType.PREFILL, "default", 2192L))
                .thenReturn("deployment_2192");
        when(metadataResolver.resolveQueryType(RoleType.PREFILL, "default"))
                .thenReturn(QueryType.QT_PREFIX_MATCH);
        when(metaServiceClient.getHostCacheState(any(), any(), any(Deadline.class)))
                .thenThrow(UNAVAILABLE.asRuntimeException());
        GrpcReporter reporter = mock(GrpcReporter.class);
        client = createClient(configuration, metaServiceClient, leaderResolver, metadataResolver, reporter);

        assertThrows(KvcmQueryException.class, () -> client.findMatchingEngines(
                "failed-query", List.of(11L), 2192L, RoleType.PREFILL, "default"));

        verify(reporter).reportCallMetrics(eq("KVCM_GET_HOST_CACHE_STATE"), anyLong(), eq(0), eq(false));
        client.shutdown();
        verify(metaServiceClient, never()).shutdown();
    }

    private static KvcmGrpcClient createClient(CacheMatchConfiguration configuration,
                                               KvcmMetaServiceClient metaServiceClient,
                                               KvcmLeaderResolver leaderResolver,
                                               KvcmWorkerMetadataResolver workerMetadataResolver,
                                               GrpcReporter grpcReporter) {
        ApplicationWarmupState warmupState = new ApplicationWarmupState();
        warmupState.setWarmupFinished(true);
        return new KvcmGrpcClient(configuration, metaServiceClient, leaderResolver, workerMetadataResolver,
                warmupState, grpcReporter, new KvcmMetricsReporter(NoOpFlexMonitor.getInstance()));
    }

    private static KvcmGrpcClient createQueryTestClient(KvcmMetaServiceClient metaService,
                                                        KvcmLeaderResolver leader,
                                                        KvcmWorkerMetadataResolver metadata,
                                                        GrpcReporter reporter) {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmCacheMatchingConfig runtimeConfig = new KvcmCacheMatchingConfig();
        runtimeConfig.setLeaderRefreshIntervalMs(60_000);
        runtimeConfig.setMaxQueryRetryCount(0);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(new KvcmConfig());
        when(configuration.getKvcmRuntimeConfig()).thenReturn(runtimeConfig);
        return createClient(configuration, metaService, leader, metadata, reporter);
    }

    private static CommonResponseHeader okHeader() {
        return CommonResponseHeader.newBuilder()
                .setStatus(Status.newBuilder().setCode(ErrorCode.OK))
                .build();
    }
}
