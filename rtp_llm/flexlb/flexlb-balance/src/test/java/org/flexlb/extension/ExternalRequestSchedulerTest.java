package org.flexlb.extension;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.request.Request;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.CancelReason;
import org.flexlb.balance.schedule.EngineCancellation;
import org.flexlb.balance.schedule.ExpirationTimer;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.balance.schedule.RequestTaskPool;
import org.flexlb.balance.schedule.ResponseCompletionExecutor;
import org.flexlb.balance.schedule.SchedulerConfiguration;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.balance.schedule.SchedulingTestConfig;
import org.flexlb.balance.schedule.SelectionTestSupport;
import org.flexlb.balance.schedule.TaskTimer;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.EndpointTestSupport;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;

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
            assertNotNull(request.snapshot().committedRoute());
        }
    }

    @Test
    void cancellationRemovesPendingWorkBeforeTheExtensionRuns() throws Exception {
        try (var fixture = new Fixture()) {
            var request = fixture.request(98102L);
            var response = fixture.scheduler.submit(request);
            fixture.scheduler.cancel("98102", CancelReason.CLIENT_CANCELLED);
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

        private synchronized void add(RequestContext context) {
            queue.add(context);
        }

        private synchronized void remove(RequestContext context) {
            for (var entry : queue.drain()) if (entry != context) queue.restore(entry);
        }

        private synchronized void schedule(java.util.function.Consumer<RequestContext> publish) {
            for (var entry : queue.scanForPlanningCandidates(1, 1, candidate -> true)) {
                queue.remove(entry);
                publish.accept(entry);
            }
        }
    }

    /**
     * Changes DIRECT scheduling triggers while retaining its immediate publication policy.
     */
    private static final class ManualDirectScheduler
            extends org.flexlb.balance.schedule.DirectRequestScheduler {
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
                    mock(TheoryCacheHitReporter.class));
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
                                    SelectionResult.Status.SUCCESS,
                                    planRoute(context, () -> selector.select(context, null))
                                            .status()));
        }
    }

    /**
     * Changes QUEUE scheduling triggers while retaining Worker queue publication and cleanup.
     */
    private static final class ManualQueuedScheduler
            extends org.flexlb.balance.schedule.QueuedRequestScheduler {
        private final ManualRequests pending = new ManualRequests();
        private final RequestRouteSelector selector;

        ManualQueuedScheduler(Fixture fixture, DeliveryMetricsReporter reporter) {
            super(
                    fixture.config,
                    fixture.selector,
                    org.flexlb.balance.schedule.RequestPreemptionTestSupport.mock(),
                    fixture.requests,
                    fixture.tasks,
                    fixture.expiration,
                    fixture.dispatcher,
                    fixture.responses,
                    reporter,
                    mock(TheoryCacheHitReporter.class),
                    new org.flexlb.worker.endpoint.CapacityNotifier(),
                    new OrderedRequestQueue(false));
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
                                SelectionResult.Status.SUCCESS,
                                planRoute(context, () -> selector.select(context, null)).status());
                        assertNull(
                                context.currentSelectionAttempt(),
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
        final java.util.concurrent.ScheduledThreadPoolExecutor deadlinePool =
                assembly.deadlineTimerExecutor();
        final TaskTimer deadlineTimer = assembly.deadlineTaskTimer(deadlinePool);
        final ExpirationTimer expiration = assembly.expirationTimer(requests, deadlineTimer);
        final java.util.concurrent.ScheduledThreadPoolExecutor cleanupPool =
                assembly.cleanupTimerExecutor();
        final TaskTimer timer = assembly.taskTimer(cleanupPool);
        final ResponseCompletionExecutor responses;
        final RequestRouteSelector selector = mock(RequestRouteSelector.class);
        final PrefillEndpoint endpoint;
        final DecodeEndpoint decode;
        final AbstractRequestScheduler scheduler;
        final EngineCancellation dispatcher;

        final EndpointRegistry endpoints = mock(EndpointRegistry.class);

        Fixture() {
            this(false);
        }

        Fixture(boolean batch) {
            this(batch, false);
        }

        Fixture(boolean batch, boolean separateDecode) {
            config.setScheduler(
                    batch
                            ? new org.flexlb.config.scheduler.SchedulerConfig()
                            : ConfigTestSupport.direct());

            var service = SchedulingTestConfig.configService(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var client = mock(EngineGrpcClient.class);
            when(client.cancelAsync(anyString(), anyInt(), any(), org.mockito.Mockito.anyLong()))
                    .thenAnswer(
                            call ->
                                    CompletableFuture.completedFuture(
                                            EngineGrpcClient.CancelReply.REQUEST_CLEANED));

            dispatcher = new EngineCancellation(client);
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
                                metadata.setRequestId(context.getRequestId());
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

                                return org.flexlb.balance.schedule.SchedulerTestServices
                                        .registerSelection(context, selectedRoles);
                            });
        }

        public void close() {
            org.mockito.Mockito.doAnswer(
                            invocation -> {
                                org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(
                                        endpoint);
                                EndpointTestSupport.awaitClosed(endpoint);
                                if (decode != null) {
                                    EndpointTestSupport.closeAndWait(decode);
                                    EndpointTestSupport.awaitClosed(decode);
                                }
                                return null;
                            })
                    .when(endpoints)
                    .close();
            Throwable failure =
                    org.flexlb.util.Failures.run(
                            null,
                            () ->
                                    org.flexlb.balance.schedule.SchedulerTestServices
                                            .shutdownScheduling(
                                                    scheduler,
                                                    requests,
                                                    tasks,
                                                    expiration,
                                                    dispatcher,
                                                    responses,
                                                    endpoints,
                                                    System.nanoTime()));
            failure = org.flexlb.util.Failures.run(failure, deadlinePool::close);
            failure = org.flexlb.util.Failures.run(failure, cleanupPool::close);
            org.flexlb.util.Failures.rethrow(failure, "extension fixture failed to close");
        }

        void advance() {
            if (scheduler instanceof ManualQueuedScheduler queued) queued.advance();
            else ((ManualDirectScheduler) scheduler).advance();
        }

        RequestContext request(long id) {
            var context = new RequestContext(config);
            context.setRequest(Request.builder().requestId(String.valueOf(id)).seqLen(16L).build());

            context.setSchedulingMetadata(
                    SchedulingMetadata.of(50, null, System.currentTimeMillis() + 60_000L, 50));
            return context;
        }
    }
}
