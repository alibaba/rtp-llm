package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.cache.monitor.CacheMetricsReporter;
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
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PdfusionSchedulingTest {
    @ParameterizedTest
    @CsvSource({ "false,false,false,false", "false,false,false,true", "false,false,true,false", "false,false,true,true", "false,true,false,false", "false,true,false,true", "false,true,true,false", "false,true,true,true", "true,false,false,false" })
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
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        RequestSchedulerReporter requestReporter = mock(RequestSchedulerReporter.class);
        AbstractRequestScheduler lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, reporter, requestReporter,
                mock(RecentCacheKeyTraceReporter.class));
        PlacementAvailability availability = new PlacementAvailability();
        DeliveryStrategy delivery = batch ? new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.accepted(new DefaultBatchDispatcher.PreparedSubmission() {

            public void submit(DefaultBatchDispatcher.Delivery delivery) {
                delivery.run((items, id, predicted, reason, observer) -> items.forEach(item -> observer.accept(item, DeliveryResult.delivered())));
            }

            public void close() {
            }
        }), new AtomicLong()::incrementAndGet, reporter) : new RouteDeliveryStrategy(reporter);
        EndpointRegistry endpoints = new EndpointRegistry(service, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle), reporter, delivery, availability);
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
            endpoints.publishPreparedEndpoint(worker.getIpPort(), worker, worker.prepareNewStatus(worker.freezeStatusResponse(status)));
        } finally {
            worker.lock.unlock();
        }
        EndpointRegistry directory = endpoints;
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PDFUSION));
        RequestWorkerSelector router = new RequestWorkerSelector(new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class), org.mockito.Mockito.mock(CacheMetricsReporter.class)), new DecodeSelector(directory), new VitWorkerSelector(directory), model);
        RequestScheduler scheduler = org.flexlb.balance.scheduler.SchedulerTestSupport.configure(lifecycle, service.loadBalanceConfig(), router, reporter, mock(DecodeCapacityAcquirer.class), availability);
        SchedulerRuntime runtime = new SchedulerRuntime(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle), endpoints, reporter, requestReporter, org.mockito.Mockito.mock(DefaultBatchDispatcher.class), service, org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
        try {
            long requestId = 920001L;
            var context = RequestProtocolTestSupport.context(config, requestId);
            context.setGenerateInputPb(org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB.newBuilder()
                    .setRequestId(requestId).build().toByteString());
            Response response = scheduler.submit(context).get(3, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(), "PDFUSION route failed: " + response.getErrorMessage());
            assertEquals(List.of(RoleType.PDFUSION), response.getServerStatus().stream().map(ServerStatus::getRole).toList());
            assertEquals(0, endpoints.getEndpointCount(RoleType.DECODE));
            assertEquals(1, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
            PrefillEndpoint endpoint = (PrefillEndpoint) endpoints.get(RoleType.PDFUSION, worker.getIpPort());
            assertEquals(1, endpoint.admissionSummary(0).occupiedRequests());
            var committedWork = endpoint.captureRouteProjectionInputs().work();
            assertTrue(committedWork.containsRequest(requestId));
            TaskInfo finished = new TaskInfo();
            finished.setRequestId(requestId);
            finished.setBatchId(direct ? 0L : org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(requestId, 0L).batchId());
            finished.setPhase(TaskPhase.RUNNING);
            finished.setErrorCode(0L);
            status.setStatusVersion(2L);
            status.setLatestFinishedVersion(1L);
            status.setFinishedTaskInfo(Map.of(Long.toString(requestId), finished));
            Runnable projection;
            worker.lock.lock();
            try {
                projection = endpoint.applyPreparedStatus(worker, worker.prepareNewStatus(worker.freezeStatusResponse(status)));
            } finally {
                worker.lock.unlock();
            }
            projection.run();
            lifecycle.runtime.continuations().awaitIdle();
            assertEquals(0, endpoint.admissionSummary(0).occupiedRequests());
            assertEquals(0, endpoint.ownershipStats().batchCount());
            assertTrue(committedWork.containsRequest(requestId), "published snapshots stay immutable");
            assertTrue(!endpoint.captureRouteProjectionInputs().work().containsRequest(requestId));
            assertEquals(RequestState.Phase.COMPLETED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(requestId, 0L).state());
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
        } finally {
            runtime.shutdown();
        }
    }

}
