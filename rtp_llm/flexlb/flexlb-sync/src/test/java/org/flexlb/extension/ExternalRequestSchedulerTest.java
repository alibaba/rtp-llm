package org.flexlb.extension;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.DeliveryExecution;
import org.flexlb.balance.scheduler.ExpirationTimer;
import org.flexlb.balance.scheduler.GlobalQueueEntry;
import org.flexlb.balance.scheduler.OrderedRequestQueue;
import org.flexlb.balance.scheduler.PlacementConfiguration;
import org.flexlb.balance.scheduler.PlacementQueue;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestLifecycle;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.RequestTaskPool;
import org.flexlb.balance.scheduler.RequestWorkerSelector;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor;
import org.flexlb.balance.scheduler.ScheduledTaskTimer;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyInt;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** An extension in another package uses the same execution and settlement as built-in timing. */
class ExternalRequestSchedulerTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void extensionCanSubmitARealWorkerSelectionWithoutTheDefaultScheduler(boolean separateDecode)
            throws Exception {
        try (var fixture = new Fixture(true, separateDecode)) {
            var first = fixture.request(98103L);
            var second = fixture.request(98104L);
            var firstResponse = fixture.entry.submit(first);
            var secondResponse = fixture.entry.submit(second);
            fixture.scheduler.schedule();
            fixture.scheduler.schedule();
            assertFalse(firstResponse.isDone());
            assertFalse(secondResponse.isDone());
            var selected = fixture.endpoint.captureQueue(2).items();
            assertEquals(List.of(first, second), selected.stream().map(RequestRoute::ctx).toList());
            assertNull(
                    fixture.dispatcher.submit(
                            new Delivery(
                                    fixture.endpoint,
                                    true,
                                    selected.size(),
                                    fixture.endpoint.predictor.evaluator(),
                                    "manual_batch"),
                            selected,
                            OptionalLong.of(1L),
                            null,
                            Delivery.PrefixPolicy.ALL_SELECTED,
                            () -> true));
            assertTrue(firstResponse.get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(secondResponse.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(
                    first.snapshot().delivery().identity().batchId(),
                    second.snapshot().delivery().identity().batchId());
            assertEquals(0, fixture.endpoint.queueDepth());
            if (separateDecode) {
                assertTrue(
                        fixture.decode
                                .ownership(first.snapshot().route().decodeReservation())
                                .owned());
                assertTrue(
                        fixture.decode
                                .ownership(second.snapshot().route().decodeReservation())
                                .owned());
                assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            }
        }
    }

    @Test
    void extensionChoosesWhenToDispatchAndUsesTheSharedResourceTransaction() throws Exception {
        try (var fixture = new Fixture()) {
            var request = fixture.request(98101L);
            var response = fixture.entry.submit(request);
            assertFalse(response.isDone());
            assertEquals(0, fixture.endpoint.stats().locallyOwnedRequests());
            verifyNoInteractions(fixture.selector);
            fixture.scheduler.schedule();
            assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(1, fixture.endpoint.stats().locallyOwnedRequests());
            assertNotNull(request.snapshot().delivery());
        }
    }

    @Test
    void cancellationRemovesPendingWorkBeforeTheExtensionRuns() throws Exception {
        try (var fixture = new Fixture()) {
            var request = fixture.request(98102L);
            var response = fixture.entry.submit(request);
            fixture.entry.cancel(98102L, 0L, CancelReason.CLIENT_CANCELLED);
            fixture.scheduler.schedule();
            assertFalse(response.get(2, TimeUnit.SECONDS).isSuccess());
            verifyNoInteractions(fixture.selector);
            assertEquals(0, fixture.endpoint.stats().locallyOwnedRequests());
        }
    }

    private static final class ManualScheduler implements RequestScheduler {
        private final PlacementQueue queue = new OrderedRequestQueue(false);
        private final RequestLifecycle lifecycle;
        private final RequestWorkerSelector selector;
        private final DeliveryDispatcher dispatcher;
        private final QueueExecutionSettings queued;

        ManualScheduler(
                RequestLifecycle lifecycle,
                RequestWorkerSelector selector,
                DeliveryDispatcher dispatcher,
                QueueExecutionSettings queued) {
            this.lifecycle = lifecycle;
            this.selector = selector;
            this.dispatcher = dispatcher;
            this.queued = queued;
        }

        public void start() {}

        public void stop() {}

        public boolean awaitStopped(long deadlineNanos) {
            return true;
        }

        public void close() {
            stop();
        }

        public void onEndpointChanged(WorkerEndpoint endpoint) {}

        public synchronized void onRequest(RequestContext context) {
            queue.add(new GlobalQueueEntry(context, null));
        }

        public synchronized void onRequestChanged(RequestContext context) {
            for (var entry : queue.drain()) if (entry.context() != context) queue.restore(entry);
        }

        public synchronized void schedule() {
            for (var entry : queue.scanForPlanningCandidates(1, 1, candidate -> true)) {
                queue.remove(entry);
                var context = entry.context();
                var admission =
                        lifecycle.claimAdmissionHandle(
                                context.data().getRequestId(), context.data().getFuture());
                if (admission == null) continue;
                try {
                    var selection = selector.select(context, entry.routingGroup());
                    assertEquals(PlacementResult.Status.SUCCESS, selection.status());
                    if (queued != null) {
                        try (var route = selection.value()) {
                            try {
                                assertEquals(
                                        PlacementResult.Status.SUCCESS,
                                        dispatcher.enqueueRoute(route, queued).status());
                            } finally {
                                org.flexlb.util.Failures.rethrow(
                                        DeliveryDispatcher.finishPlacement(route, admission),
                                        "placement cleanup failed");
                            }
                        }
                        continue;
                    }
                    var committed = dispatcher.submitRoute(selection.value(), admission);
                    assertEquals(PlacementResult.Status.SUCCESS, committed.status());
                    assertEquals(
                            null,
                            context.snapshot().placement(),
                            "the complete submission must finish admission before returning");
                } finally {
                    admission.finish();
                }
            }
        }
    }

    private static final class Fixture implements AutoCloseable {
        final org.flexlb.config.FlexlbConfig config = SchedulingTestConfig.newConfig();
        final PlacementConfiguration assembly = new PlacementConfiguration();
        final RequestRepository requests = new RequestRepository();
        final RequestTaskPool tasks = assembly.requestTasks();
        final ScheduledTaskTimer deadlineTimer = assembly.deadlineTaskTimer();
        final ExpirationTimer expiration = assembly.expirationTimer(requests, deadlineTimer);
        final ScheduledTaskTimer timer = assembly.taskTimer();
        final ResponseCompletionExecutor responses;
        final RequestLifecycle lifecycle;
        final RequestWorkerSelector selector = mock(RequestWorkerSelector.class);
        final PrefillEndpoint endpoint;
        final DecodeEndpoint decode;
        final ManualScheduler scheduler;
        final RequestLifecycle entry;
        final DeliveryDispatcher dispatcher;
        final DeliveryExecution deliveries;

        Fixture() {
            this(false);
        }

        Fixture(boolean batch) {
            this(batch, false);
        }

        Fixture(boolean batch, boolean separateDecode) {
            config.setScheduler(
                    batch ? new org.flexlb.config.SchedulerConfig() : ConfigTestSupport.direct());
            config.getDispatcher()
                    .setType(
                            batch
                                    ? org.flexlb.config.DispatcherConfig.Type.BATCH
                                    : org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
            var service = SchedulingTestConfig.configService(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var client = mock(EngineGrpcClient.class);
            when(client.batchEnqueueAsync(anyString(), anyInt(), any()))
                    .thenAnswer(
                            call -> {
                                EngineRpcService.EnqueueBatchRequestPB request =
                                        call.getArgument(2);
                                var reply =
                                        EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                                                .setBatchId(request.getBatchId());
                                for (var slot : request.getDpSlotsList()) {
                                    for (var member : slot.getRequestsList()) {
                                        reply.addSuccesses(
                                                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                                        .setRequestId(
                                                                member.getInput().getRequestId()));
                                    }
                                }
                                return CompletableFuture.completedFuture(reply.build());
                            });
            deliveries = assembly.deliveryExecution(service, tasks, reporter);
            dispatcher =
                    new DeliveryDispatcher(
                            client,
                            service,
                            reporter,
                            timer,
                            new org.flexlb.balance.scheduler.PlacementConfiguration()
                                    .decodeAdmissionCapacity(service),
                            deliveries);
            responses = assembly.responseCompletions(service);
            lifecycle =
                    assembly.requestLifecycle(
                            requests,
                            tasks,
                            expiration,
                            dispatcher,
                            responses,
                            reporter,
                            mock(RecentCacheKeyTraceReporter.class));
            scheduler =
                    new ManualScheduler(
                            lifecycle,
                            selector,
                            dispatcher,
                            batch
                                    ? QueueExecutionSettings.capture(
                                            config, BuiltinGroupPlanner.from(config))
                                    : null);
            lifecycle.bindScheduler(scheduler);
            entry = lifecycle;
            endpoint =
                    EndpointTestSupport.realPrefill(
                            WorkerStatus.createDiscovered(
                                    separateDecode ? RoleType.PREFILL : RoleType.PDFUSION,
                                    "group",
                                    "127.0.0.1",
                                    8080,
                                    8090,
                                    null),
                            config,
                            dispatcher,
                            requests,
                            reporter);
            decode =
                    separateDecode
                            ? EndpointTestSupport.decode(
                                    WorkerStatus.createDiscovered(
                                            RoleType.DECODE,
                                            "group",
                                            "127.0.0.2",
                                            8081,
                                            8091,
                                            null),
                                    requests)
                            : null;
            when(selector.select(any(), isNull()))
                    .thenAnswer(
                            call -> {
                                RequestContext context = call.getArgument(0);
                                return SchedulingTestConfig.prepareRoute(
                                        context,
                                        route -> {
                                            var metadata = new ServerStatus();
                                            metadata.setSuccess(true);
                                            metadata.setRequestId(context.data().getRequestId());
                                            metadata.setRole(endpoint.getStatus().getRole());
                                            metadata.setServerIp(endpoint.getIp());
                                            metadata.setHttpPort(endpoint.getHttpPort());
                                            metadata.setGrpcPort(endpoint.getGrpcPort());
                                            route.add(
                                                    endpoint.tryPinGeneration(), metadata, 1L, 0L);
                                            if (decode != null) {
                                                var decodeMetadata = ServerStatus.copyOf(metadata);
                                                decodeMetadata.setRole(RoleType.DECODE);
                                                decodeMetadata.setServerIp(decode.getIp());
                                                decodeMetadata.setHttpPort(decode.getHttpPort());
                                                decodeMetadata.setGrpcPort(decode.getGrpcPort());
                                                route.add(
                                                        decode.tryPinGeneration(),
                                                        decodeMetadata,
                                                        0L,
                                                        decode.placementVersion());
                                            }
                                            return PlacementResult.success(route);
                                        });
                            });
        }

        RequestContext request(long id) {
            var context = new RequestContext(config);
            context.data().setRequest(Request.builder().requestId(id).seqLen(16L).build());
            context.data()
                    .setGenerateInputPb(
                            EngineRpcService.GenerateInputPB.newBuilder()
                                    .setRequestId(id)
                                    .setGenerateConfig(
                                            EngineRpcService.GenerateConfigPB.newBuilder())
                                    .build()
                                    .toByteString());
            context.data()
                    .setSchedulingMetadata(
                            SchedulingMetadata.explicit(50, System.currentTimeMillis() + 60_000L));
            return context;
        }

        public void close() {
            scheduler.close();
            requests.closeRegistration();
            lifecycle.stopAdmissionsAndAwaitCompletion();
            lifecycle.closeOutstandingAndTerminalize();
            deliveries.awaitDeliveries();
            tasks.awaitIdle();
            expiration.close();
            deadlineTimer.close();
            tasks.close();
            responses.close();
            timer.close();
            endpoint.close();
            EndpointTestSupport.awaitRetirement(endpoint);
            if (decode != null) {
                decode.close();
                EndpointTestSupport.awaitRetirement(decode);
            }
        }
    }
}
