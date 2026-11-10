package org.flexlb.extension;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.DeliveryCoordinator;
import org.flexlb.balance.scheduler.DeliveryExecution;
import org.flexlb.balance.scheduler.ExpirationTimer;
import org.flexlb.balance.scheduler.GlobalQueueEntry;
import org.flexlb.balance.scheduler.OrderedRequestQueue;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestRouteSelector;
import org.flexlb.balance.scheduler.RequestTaskPool;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor;
import org.flexlb.balance.scheduler.RoleSelection;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.balance.scheduler.ScheduledTaskTimer;
import org.flexlb.balance.scheduler.SchedulerConfiguration;
import org.flexlb.balance.scheduler.SchedulingQueue;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.scheduler.SelectionTestSupport;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.TheoryCacheHitReporter;
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

/**
 * An extension in another package uses the same execution and settlement as built-in timing.
 */
class ExternalRequestSchedulerTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void extensionCanSubmitARealWorkerSelectionWithoutTheDefaultScheduler(boolean separateDecode)
            throws Exception {
        try (var fixture = new Fixture(true, separateDecode)) {
            var first = fixture.request(98103L);
            var second = fixture.request(98104L);
            var firstResponse = fixture.scheduler.submit(first);
            var secondResponse = fixture.scheduler.submit(second);
            fixture.advance();
            fixture.advance();
            assertFalse(firstResponse.isDone());
            assertFalse(secondResponse.isDone());
            var selected = fixture.endpoint.captureQueue(2).items();
            assertEquals(List.of(first, second), selected.stream().map(RouteResult::ctx).toList());
            assertNull(
                    fixture.dispatcher.submit(
                            selected,
                            "manual_batch",
                            fixture.endpoint.predictor.evaluator(),
                            OptionalLong.of(1L),
                            Delivery.PrefixPolicy.ALL_SELECTED));
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
            var response = fixture.scheduler.submit(request);
            assertFalse(response.isDone());
            assertEquals(0, fixture.endpoint.stats().locallyOwnedRequests());
            verifyNoInteractions(fixture.selector);
            fixture.advance();
            assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(1, fixture.endpoint.stats().locallyOwnedRequests());
            assertNotNull(request.snapshot().delivery());
        }
    }

    @Test
    void cancellationRemovesPendingWorkBeforeTheExtensionRuns() throws Exception {
        try (var fixture = new Fixture()) {
            var request = fixture.request(98102L);
            var response = fixture.scheduler.submit(request);
            fixture.scheduler.cancel("98102", 0L, CancelReason.CLIENT_CANCELLED);
            fixture.advance();
            assertFalse(response.get(2, TimeUnit.SECONDS).isSuccess());
            verifyNoInteractions(fixture.selector);
            assertEquals(0, fixture.endpoint.stats().locallyOwnedRequests());
        }
    }

    /**
     * Holds requests until an extension explicitly runs a scheduling turn.
     */
    private static final class ManualRequests {
        private final SchedulingQueue queue = new OrderedRequestQueue(false);

        synchronized void add(RequestContext context) {
            queue.add(new GlobalQueueEntry(context, null));
        }

        synchronized void remove(RequestContext context) {
            for (var entry : queue.drain()) if (entry.context() != context) queue.restore(entry);
        }

        synchronized void schedule(java.util.function.Consumer<RequestContext> publish) {
            for (var entry : queue.scanForPlanningCandidates(1, 1, candidate -> true)) {
                queue.remove(entry);
                publish.accept(entry.context());
            }
        }
    }

    /**
     * Changes DIRECT scheduling triggers while retaining its immediate publication policy.
     */
    private static final class ManualDirectScheduler
            extends org.flexlb.balance.scheduler.DirectRequestScheduler {
        private final ManualRequests pending = new ManualRequests();
        private final RequestRouteSelector selector;

        ManualDirectScheduler(Fixture fixture, DeliveryMetricsReporter reporter) {
            super(
                    fixture.selector,
                    fixture.requests,
                    fixture.tasks,
                    fixture.expiration,
                    fixture.dispatcher,
                    fixture.responses,
                    reporter,
                    mock(TheoryCacheHitReporter.class),
                    fixture.endpoints);
            selector = fixture.selector;
        }

        @Override
        public void start() {}

        @Override
        public void scheduleRequest(RequestContext context) {
            pending.add(context);
        }

        @Override
        protected void removeRequest(RequestContext context) {
            pending.remove(context);
        }

        void advance() {
            pending.schedule(
                    context ->
                            assertEquals(
                                    RoutingResult.Status.SUCCESS,
                                    selectAndPublish(context, () -> selector.select(context, null))
                                            .status()));
        }
    }

    /**
     * Changes QUEUE scheduling triggers while retaining Worker queue publication and cleanup.
     */
    private static final class ManualQueuedScheduler
            extends org.flexlb.balance.scheduler.QueuedRequestScheduler {
        private final ManualRequests pending = new ManualRequests();
        private final RequestRouteSelector selector;

        ManualQueuedScheduler(Fixture fixture, DeliveryMetricsReporter reporter) {
            super(
                    fixture.config,
                    fixture.selector,
                    mock(org.flexlb.balance.preemption.DecodePreemption.class),
                    fixture.requests,
                    fixture.tasks,
                    fixture.expiration,
                    fixture.dispatcher,
                    fixture.responses,
                    reporter,
                    mock(TheoryCacheHitReporter.class),
                    new org.flexlb.balance.scheduler.CapacityNotifier(),
                    new OrderedRequestQueue(false),
                    BuiltinGroupPlanner.from(fixture.config),
                    fixture.endpoints);
            selector = fixture.selector;
        }

        @Override
        public void start() {}

        @Override
        public void scheduleRequest(RequestContext context) {
            pending.add(context);
        }

        @Override
        protected void removeRequest(RequestContext context) {
            pending.remove(context);
        }

        void advance() {
            pending.schedule(
                    context -> {
                        assertEquals(
                                RoutingResult.Status.SUCCESS,
                                selectAndPublish(context, () -> selector.select(context, null))
                                        .status());
                        assertNull(
                                context.snapshot().routingContext(),
                                "routing must finish before returning to the extension");
                    });
        }
    }

    /**
     * Wires the manual scheduler to real endpoints and shared delivery services, then drains all accepted work during cleanup.
     */
    private static final class Fixture implements AutoCloseable {
        final org.flexlb.config.FlexlbConfig config = SchedulingTestConfig.newConfig();
        final SchedulerConfiguration assembly = new SchedulerConfiguration();
        final RequestRepository requests = new RequestRepository();
        final RequestTaskPool tasks = assembly.requestTasks();
        final ScheduledTaskTimer deadlineTimer = assembly.deadlineTaskTimer();
        final ExpirationTimer expiration = assembly.expirationTimer(requests, deadlineTimer);
        final ScheduledTaskTimer timer = assembly.taskTimer();
        final ResponseCompletionExecutor responses;
        final RequestRouteSelector selector = mock(RequestRouteSelector.class);
        final PrefillEndpoint endpoint;
        final DecodeEndpoint decode;
        final AbstractRequestScheduler scheduler;
        final DeliveryCoordinator dispatcher;
        final DeliveryExecution deliveries;
        final EndpointRegistry endpoints = mock(EndpointRegistry.class);

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
            when(client.cancelAsync(anyString(), anyInt(), any(), org.mockito.Mockito.anyLong()))
                    .thenAnswer(
                            call ->
                                    CompletableFuture.completedFuture(
                                            EngineGrpcClient.CancelReply.REQUEST_CLEANED));
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
            deliveries = assembly.deliveryExecution(service, reporter);
            dispatcher =
                    new DeliveryCoordinator(
                            requests,
                            client,
                            service,
                            reporter,
                            new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                    .decodeAdmissionCapacity(service),
                            deliveries);
            responses = assembly.responseCompletions(service);
            scheduler =
                    batch
                            ? new ManualQueuedScheduler(this, reporter)
                            : new ManualDirectScheduler(this, reporter);
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
                                var selectedRoles = new java.util.ArrayList<RoleSelection>();

                                var metadata = new ServerStatus();
                                metadata.setSuccess(true);
                                metadata.setRequestId(context.data().getRequestId());
                                metadata.setRole(endpoint.getStatus().getRole());
                                metadata.setServerIp(endpoint.getIp());
                                metadata.setHttpPort(endpoint.getHttpPort());
                                metadata.setGrpcPort(endpoint.getGrpcPort());
                                selectedRoles.add(
                                        SelectionTestSupport.metadata(endpoint, metadata, 1L));
                                if (decode != null) {
                                    var decodeMetadata = ServerStatus.copyOf(metadata);
                                    decodeMetadata.setRole(RoleType.DECODE);
                                    decodeMetadata.setServerIp(decode.getIp());
                                    decodeMetadata.setHttpPort(decode.getHttpPort());
                                    decodeMetadata.setGrpcPort(decode.getGrpcPort());
                                    selectedRoles.add(
                                            SelectionTestSupport.metadata(
                                                    decode, decodeMetadata, 0L));
                                }

                                return org.flexlb.balance.scheduler.SchedulerTestServices
                                        .registerSelection(context, selectedRoles);
                            });
        }

        public void close() {
            org.mockito.Mockito.doAnswer(
                            invocation -> {
                                endpoint.close();
                                EndpointTestSupport.awaitRetirement(endpoint);
                                if (decode != null) {
                                    decode.close();
                                    EndpointTestSupport.awaitRetirement(decode);
                                }
                                return null;
                            })
                    .when(endpoints)
                    .close();
            Throwable failure =
                    org.flexlb.util.Failures.run(null, () -> scheduler.shutdown(System.nanoTime()));
            failure = org.flexlb.util.Failures.run(failure, deadlineTimer::close);
            failure = org.flexlb.util.Failures.run(failure, timer::close);
            org.flexlb.util.Failures.rethrow(failure, "extension fixture failed to close");
        }

        void advance() {
            if (scheduler instanceof ManualQueuedScheduler queued) queued.advance();
            else ((ManualDirectScheduler) scheduler).advance();
        }

        RequestContext request(long id) {
            var context = new RequestContext(config);
            context.data()
                    .setRequest(
                            Request.builder().requestId(String.valueOf(id)).seqLen(16L).build());
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
    }
}
