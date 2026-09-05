package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PdfusionSchedulingTest {
    @ParameterizedTest
    @CsvSource({
            "false,false,false,false", "false,false,false,true", "false,false,true,false", "false,false,true,true",
            "false,true,false,false", "false,true,false,true", "false,true,true,false", "false,true,true,true",
            "true,false,false,false"})
    void fusionNeedsOnlyItsOwnRole(boolean direct, boolean priority, boolean window, boolean batch) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        if (priority) {
            SchedulingTestConfig.usePriorityQueue(config);
        } else {
            SchedulingTestConfig.useFifoQueue(config);
        }
        if (window) {
            SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
            config.queueScheduler().getDecision().setMaxCollectionWaitMs(10L);
        } else {
            SchedulingTestConfig.useSingleDecision(config);
        }
        if (batch) {
            SchedulingTestConfig.useBatchDispatcher(config);
        }
        if (direct) {
            config.setScheduler(SchedulerConfig.direct());
        }
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        RequestSchedulerReporter requestReporter = mock(RequestSchedulerReporter.class);
        RequestRegistry lifecycle = new RequestRegistry(service, reporter, requestReporter, mock(EngineCancelChannel.class));
        PlacementAvailability availability = new PlacementAvailability();
        DeliveryStrategy delivery = batch
                ? new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.accepted(
                        new BatchDeliveryStrategy.PreparedSubmission() {
                            public void submitBatch(List<ScheduledRequest> items, long id, long predicted,
                                    String reason, BiConsumer<ScheduledRequest, DeliveryResult> observer) {
                                items.forEach(item -> observer.accept(item,
                                        DeliveryResult.delivered()));
                            }
                            public void close() { }
                        }), new AtomicLong()::incrementAndGet,
                        lifecycle, new DeliveryMetrics(reporter))
                : new RouteDeliveryStrategy(lifecycle, new DeliveryMetrics(reporter));
        EndpointRegistry endpoints = new EndpointRegistry(service, new EndpointEventProjector(lifecycle), reporter,
                delivery, availability);
        WorkerStatus worker = WorkerStatus.createDiscovered(RoleType.PDFUSION, "g1", "127.0.0.1", 8080, 8081, "test");
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.PDFUSION);
        status.setAlive(true);
        status.setStatusVersion(1L);
        status.setLatestFinishedVersion(0L);
        status.setTotalKvCacheTokens(100000L);
        status.setAvailableKvCacheTokens(100000L);
        status.setMaxSeqLen(10000L);
        status.setMaxBatchTokensSize(10000L);
        worker.lock.lock();
        try {
            endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                    worker.prepareNewStatus(worker.freezeStatusResponse(status)));
        } finally {
            worker.lock.unlock();
        }
        WorkerDirectory directory = new WorkerDirectory(endpoints);
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PDFUSION));
        DefaultRouter router = new DefaultRouter(new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class)),
                new CostBasedDecodeStrategy(directory), new RandomStrategy(directory), service, model);
        RequestScheduler scheduler = direct ? null : new RequestScheduler(service, router, endpoints, reporter,
                mock(EvictionManager.class), lifecycle, availability);
        SchedulerRuntime runtime = direct
                ? new SchedulerRuntime(lifecycle, endpoints, reporter, requestReporter)
                : new SchedulerRuntime(lifecycle, endpoints, reporter, requestReporter, scheduler);
        try {
            long requestId = 920001L;
            var context = RequestLifecycleTestSupport.context(config, requestId);
            Response response = direct ? router.routeDirect(context)
                    : scheduler.submit(context).get(3, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(), "PDFUSION route failed: " + response.getErrorMessage());
            assertEquals(List.of(RoleType.PDFUSION), response.getServerStatus().stream()
                    .map(ServerStatus::getRole).toList());
            assertEquals(0, endpoints.getEndpointCount(RoleType.DECODE));
            assertEquals(0, lifecycle.decodeAcceptanceCount());
            PrefillEndpoint endpoint = (PrefillEndpoint) endpoints.get(RoleType.PDFUSION, worker.getIpPort());
            assertEquals(1, endpoint.admissionPendingRequestCount());
            var committedWork = endpoint.captureRouteProjectionInputs().work();
            assertTrue(committedWork.containsRequest(requestId));
            TaskInfo finished = new TaskInfo();
            finished.setRequestId(requestId);
            finished.setBatchId(direct ? 0L : lifecycle.getRequestState(requestId, 0L).batchId());
            finished.setPhase(TaskPhase.RUNNING);
            finished.setErrorCode(0L);
            status.setStatusVersion(2L);
            status.setLatestFinishedVersion(1L);
            status.setFinishedTaskInfo(Map.of(Long.toString(requestId), finished));
            Runnable projection;
            worker.lock.lock();
            try {
                projection = endpoint.applyPreparedStatus(worker,
                        worker.prepareNewStatus(worker.freezeStatusResponse(status)));
            } finally {
                worker.lock.unlock();
            }
            projection.run();
            assertEquals(0, endpoint.admissionPendingRequestCount());
            assertEquals(0, endpoint.getInflightBatchCount());
            assertTrue(committedWork.containsRequest(requestId), "published snapshots stay immutable");
            assertTrue(!endpoint.captureRouteProjectionInputs().work().containsRequest(requestId));
            if (!direct) {
                assertEquals(RequestState.Phase.COMPLETED, lifecycle.getRequestState(requestId, 0L).state());
                assertEquals(0, lifecycle.liveRequestCount());
            }
        } finally {
            runtime.shutdown();
        }
    }

}
