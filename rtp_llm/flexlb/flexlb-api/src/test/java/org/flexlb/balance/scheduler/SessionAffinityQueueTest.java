package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.session.SessionPlacementStore;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Uses canonical scheduler-owned work, not mocked strategy pressure. */
class SessionAffinityQueueTest {
    @Test
    @Timeout(20)
    void committedSessionWorkSpillsTheNextRequestBeyondItsTtftBound() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.fixedWindowDecision().setMaxRequests(1);
        config.fixedWindowDecision().setMaxCollectionWaitMs(1L);
        config.queueScheduler().getCapacity().setMaxWaitingRequestsPerPrefillWorker(16);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        config.getRouter().getRoles().getPrefill().getCandidateChoice()
                .setType(RoutingConfig.CandidateChoiceType.BEST_ONLY);
        RoutingConfig.SessionAffinityConfig affinity = new RoutingConfig.SessionAffinityConfig();
        affinity.setTtlMs(1_800_000L);
        affinity.setMaxExtraTtftMs(5_500L);
        config.getRouter().getRoles().getPrefill().setSessionAffinity(affinity);
        ConfigService configs = mock(ConfigService.class);
        when(configs.loadBalanceConfig()).thenReturn(config);
        SessionPlacementStore store = new SessionPlacementStore();
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());

        try (RequestSchedulerTestRuntime runtime = new RequestSchedulerTestRuntime(
                configs, null, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class), mock(EngineCancelChannel.class))) {
            PrefillEndpoint remembered = publish(runtime, "10.0.0.1");
            WorkerDirectory workers = new WorkerDirectory(runtime.endpointRegistry());
            ModelMetaConfig model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
            CostBasedPrefillStrategy selector = new CostBasedPrefillStrategy(
                    workers, cache, mock(EngineHealthReporter.class), store);
            DefaultRouter router = new DefaultRouter(
                    selector,
                    new CostBasedDecodeStrategy(workers), new RandomStrategy(workers),
                    configs, model);
            runtime.bindRouter(router);

            // A real retained route creates a strict 5,000 ms disadvantage.
            Response seed = runtime.scheduler().submit(context(config, 1L, 5_000L, false))
                    .get(5, TimeUnit.SECONDS);
            assertTrue(seed.isSuccess());
            assertSame(remembered, runtime.activeItem(1L).prefillEp());
            PrefillEndpoint spare = publish(runtime, "10.0.0.3");
            store.record("session-queue", "same-session", "10.0.0.1:8080");
            try (SelectedRole baseline = selector.select(
                    context(config, 2L, 1_000L, false), RoleType.PREFILL, null).value()) {
                assertEquals("10.0.0.3", baseline.serverStatus().getServerIp());
            }
            assertEquals(0L, spare.admissionPendingRequestCount());

            BalanceContext first = context(config, 3L, 1_000L, true);
            Response firstResponse = runtime.scheduler().submit(first).get(5, TimeUnit.SECONDS);
            assertTrue(firstResponse.isSuccess());
            assertEquals("SESSION_AFFINITY", first.getSessionAffinityReason());
            assertSame(remembered, runtime.activeItem(3L).prefillEp());
            assertEquals(2L, remembered.admissionPendingRequestCount());

            // The first session request remains owned while the next is routed.
            // Its additional 1,000 ms pushes the same endpoint over the cap.
            BalanceContext second = context(config, 4L, 1_000L, true);
            Response secondResponse = runtime.scheduler().submit(second).get(5, TimeUnit.SECONDS);
            assertTrue(secondResponse.isSuccess());
            assertEquals("OVER_CAP", second.getSessionAffinityReason());
            assertSame(spare, runtime.activeItem(4L).prefillEp());
            assertEquals("10.0.0.3", secondResponse.getServerStatus().getFirst().getServerIp());
            assertSame(remembered, runtime.activeItem(3L).prefillEp());
            assertEquals(2L, remembered.admissionPendingRequestCount());
            assertEquals(1L, spare.admissionPendingRequestCount());
        }
    }

    private static BalanceContext context(FlexlbConfig config, long id, long tokens, boolean session) {
        Request request = new Request();
        request.setRequestId(id);
        request.setSeqLen(tokens);
        request.setMaxNewTokens(8);
        request.setModel("session-queue");
        if (session) {
            request.setSessionSchemaVersion(Request.SESSION_SCHEMA_VERSION);
            request.setInferenceSessionId("same-session");
            request.setInferenceSessionState(Request.SessionState.ESTABLISHED);
        }
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        return context;
    }

    private static PrefillEndpoint publish(RequestSchedulerTestRuntime runtime, String ip) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "session-queue", ip, 8080, 8081, null);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setAvailableKvCacheTokens(1_000_000L);
        response.setTotalKvCacheTokens(2_000_000L);
        response.setMaxSeqLen(1_000_000L);
        response.setMaxBatchTokensSize(1_000_000L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            WorkerEndpoint endpoint = runtime.endpointRegistry()
                    .publishPreparedEndpoint(ip + ":8080", status, prepared).endpoint();
            status.recordSuccessfulPoll(true);
            return (PrefillEndpoint) endpoint;
        } finally {
            status.lock.unlock();
        }
    }
}
