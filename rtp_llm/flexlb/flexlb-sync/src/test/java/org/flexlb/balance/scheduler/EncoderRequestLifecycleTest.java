package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.Granted;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EncoderRequestLifecycleTest {
    private FlexlbConfig config;
    private AbstractRequestScheduler scheduler;
    private RequestRepository requests;
    private RequestSchedulerReporter requestReporter;
    private PlacementAvailability availability = new PlacementAvailability();

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        requestReporter = mock(RequestSchedulerReporter.class);
        scheduler =
                SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        requestReporter,
                        mock(org.flexlb.service.TheoryCacheHitReporter.class));
        requests = scheduler.runtime.requests();
    }

    @AfterEach
    void tearDown() {
        scheduler.runtime.shutdown();
    }

    @Test
    void distinctStringIdsKeepTheirOriginalLifecycleIdentity() throws Exception {
        Map<String, CompletableFuture<Response>> responses = new HashMap<>();
        for (String requestId : List.of("00123", "123", "req-abc", "9223372036854775808")) {
            RequestContext context = context(requestId);
            CompletableFuture<Response> future = register(context);
            responses.put(requestId, future);
            assertEquals(requestId, requests.getRequestState(requestId, 0L).requestId());
            assertFalse(future.isDone());
        }
        assertEquals(4, requests.liveRequestCount());

        for (var response : responses.entrySet()) {
            String requestId = response.getKey();
            RequestState cancelled = scheduler.cancel(requestId, 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(requestId, cancelled.requestId());
            assertEquals(
                    StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                    response.getValue().get(5, TimeUnit.SECONDS).getCode());
            assertEquals(
                    RequestState.Phase.CANCELLED, requests.getRequestState(requestId, 0L).state());
        }
        assertEquals(0, requests.liveRequestCount());
    }

    @Test
    void lifecycleRetainsOriginalNumericIdAcrossStateAndCancellation() {
        RequestContext context = context("007");
        context.data().setOriginalNumericRequestId(123L);
        register(context);

        assertEquals(123L, requests.getRequestState("007", 0L).originalNumericRequestId());
        RequestState cancelled = scheduler.cancel("007", 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals("007", cancelled.requestId());
        assertEquals(123L, cancelled.originalNumericRequestId());
    }

    @Test
    void reportsUnexpectedRegistrationFailureWithoutRegisteringTheRequest() {
        RequestContext context = context(101L);
        config.getRequestLifecycle().setRequest(null);

        Response response = register(context).join();

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), response.getCode());
        assertEquals(0, requests.liveRequestCount());
        verify(requestReporter).reportLifecycleFailure("registration");
    }

    @Test
    void sameBusinessRequestHasIndependentEncoderAndGenerationRecords() {
        RequestContext encoder = context(106L);
        encoder.data().setRequestPhase(RequestPhase.ENCODER);
        CompletableFuture<Response> encoderFuture = register(encoder);
        CompletableFuture<Response> generationFuture = register(context(106L));

        assertFalse(encoderFuture.isDone());
        assertFalse(generationFuture.isDone());
        assertEquals(2, requests.liveRequestCount());
        assertEquals(1, requests.trackedRequestStats().generationRequests());
        assertEquals(1, requests.trackedRequestStats().encoderRequests());
        assertEquals(1, requests.trackedRequestStats().encoderRequests());
        assertEquals(1, requests.trackedRequestStats().generationRequests());
        assertEquals(
                RequestState.Phase.QUEUED,
                requests.getRequestState("106", 0L, RequestPhase.ENCODER).state());
        assertEquals(RequestState.Phase.QUEUED, requests.getRequestState("106", 0L).state());

        scheduler.cancel("106", 0L, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);

        assertEquals(
                RequestState.Phase.CANCELLED,
                requests.getRequestState("106", 0L, RequestPhase.ENCODER).state());
        assertEquals(RequestState.Phase.QUEUED, requests.getRequestState("106", 0L).state());
        assertFalse(generationFuture.isDone());
        assertEquals(0, requests.trackedRequestStats().encoderRequests());
        assertEquals(0, requests.trackedRequestStats().encoderRequests());
        assertEquals(1, requests.trackedRequestStats().generationRequests());
        assertEquals(1, requests.trackedRequestStats().generationRequests());
    }

    @Test
    void encoderPlacementFailureCompletesItsPhaseWithoutTouchingGeneration() throws Exception {
        RequestContext encoder = context(111L);
        encoder.data().setRequestPhase(RequestPhase.ENCODER);
        CompletableFuture<Response> encoderFuture = register(encoder);
        CompletableFuture<Response> generationFuture = register(context(111L));

        assertTrue(publishFailure("111", Response.error(StrategyErrorType.NO_ENCODER_WORKER)));

        assertEquals(
                StrategyErrorType.NO_ENCODER_WORKER.getErrorCode(),
                encoderFuture.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestState.Phase.FAILED,
                requests.getRequestState("111", 0L, RequestPhase.ENCODER).state());
        assertEquals(RequestState.Phase.QUEUED, requests.getRequestState("111", 0L).state());
        assertFalse(generationFuture.isDone());
    }

    @Test
    void encoderRouteCountsPendingUntilWorkerReportsRunningThenFinished() throws Exception {
        RequestContext context = context(107L);
        context.data().setRequestPhase(RequestPhase.ENCODER);
        CompletableFuture<Response> future = register(context);
        WorkerStatus worker = worker();
        EncoderEndpoint endpoint = new EncoderEndpoint(worker, requests, availability);
        Response route = new Response();
        route.setSuccess(true);

        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(claimEncoder("107", future, pin));
        }
        assertEquals(1, endpoint.pendingEncoderRequestCount());
        assertFalse(future.isDone());
        assertEquals(
                RequestState.Phase.DISPATCHING,
                requests.getRequestState("107", 0L, RequestPhase.ENCODER).state());

        assertTrue(publishEncoder("107", future, route));
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(
                RequestState.Phase.ACKNOWLEDGED,
                requests.getRequestState("107", 0L, RequestPhase.ENCODER).state());

        WorkerStatus.TaskObservation task = mock(WorkerStatus.TaskObservation.class);
        when(task.requestId()).thenReturn("107");
        WorkerStatus.StatusObservation running = mock(WorkerStatus.StatusObservation.class);
        when(running.owner()).thenReturn(worker);
        when(running.activeTasks()).thenReturn(Map.of("107", task));
        endpoint.publishStatus(endpoint.applyStatusHeartbeat(worker, running));

        assertEquals(0, endpoint.pendingEncoderRequestCount());
        assertEquals(
                RequestState.Phase.ACKNOWLEDGED,
                requests.getRequestState("107", 0L, RequestPhase.ENCODER).state());

        WorkerStatus.StatusObservation finished = mock(WorkerStatus.StatusObservation.class);
        when(finished.alive()).thenReturn(true);
        when(finished.activeTasks()).thenReturn(Map.of());
        when(finished.finishedTasks()).thenReturn(Map.of("107", task));
        WorkerStatus.PreparedStatus prepared = mock(WorkerStatus.PreparedStatus.class);
        when(prepared.observation()).thenReturn(finished);
        endpoint.publishStatus(endpoint.applyPreparedStatus(worker, prepared));
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);

        assertEquals(
                RequestState.Phase.COMPLETED,
                requests.getRequestState("107", 0L, RequestPhase.ENCODER).state());
        assertEquals(0, endpoint.pendingEncoderRequestCount());
    }

    @Test
    void encoderLoadUsesActiveWorkerInputLengthWithoutPrefixSubtraction() throws Exception {
        RequestContext context = context(109L);
        context.data().setRequestPhase(RequestPhase.ENCODER);
        context.data().setRequest(context.data().getRequest().toBuilder().seqLen(1000).build());
        context.data()
                .setRequest(
                        context.data().getRequest().toBuilder().encoderCacheHitLen(200L).build());
        CompletableFuture<Response> future = register(context);
        WorkerStatus worker = worker();
        WorkerStatus.EngineObservation engine = mock(WorkerStatus.EngineObservation.class);
        when(worker.committedEngineObservation()).thenReturn(engine);
        when(engine.runningTaskList()).thenReturn(Map.of());
        EncoderEndpoint endpoint = new EncoderEndpoint(worker, requests, availability);

        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(claimEncoder("109", future, pin));
        }
        assertEquals(800L, endpoint.inflightUncachedTokenEstimate());

        WorkerStatus.TaskObservation task = mock(WorkerStatus.TaskObservation.class);
        WorkerStatus.TaskTelemetry telemetry = mock(WorkerStatus.TaskTelemetry.class);
        when(task.requestId()).thenReturn("109");
        when(task.inputLength()).thenReturn(400L);
        when(task.prefixLength()).thenReturn(600L);
        when(task.telemetry()).thenReturn(telemetry);
        when(telemetry.prefixLengthValid()).thenReturn(false);
        when(engine.runningTaskList()).thenReturn(Map.of("109", task));
        WorkerStatus.StatusObservation running = mock(WorkerStatus.StatusObservation.class);
        when(running.owner()).thenReturn(worker);
        when(running.activeTasks()).thenReturn(Map.of("109", task));
        endpoint.publishStatus(endpoint.applyStatusHeartbeat(worker, running));

        assertEquals(400L, endpoint.inflightUncachedTokenEstimate());

        when(telemetry.prefixLengthValid()).thenReturn(true);
        assertEquals(400L, endpoint.inflightUncachedTokenEstimate());

        scheduler.cancel("109", 0L, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);
        when(engine.runningTaskList()).thenReturn(Map.of());
        assertEquals(0L, endpoint.inflightUncachedTokenEstimate());
    }

    @Test
    void encoderCacheOnlyFinishedIgnoresOriginalPromptLength() throws Exception {
        RequestContext context = context(110L);
        context.data().setRequestPhase(RequestPhase.ENCODER);
        context.data().setRequest(context.data().getRequest().toBuilder().seqLen(1000).build());
        context.data()
                .setRequest(context.data().getRequest().toBuilder().encoderCacheHitLen(0L).build());
        CompletableFuture<Response> future = register(context);
        WorkerStatus worker = worker();
        WorkerStatus.EngineObservation engine = mock(WorkerStatus.EngineObservation.class);
        when(worker.committedEngineObservation()).thenReturn(engine);
        when(engine.runningTaskList()).thenReturn(Map.of());
        EncoderEndpoint endpoint = new EncoderEndpoint(worker, requests, availability);

        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(claimEncoder("110", future, pin));
        }
        assertEquals(1000L, endpoint.inflightUncachedTokenEstimate());

        WorkerStatus.TaskObservation task = mock(WorkerStatus.TaskObservation.class);
        when(task.requestId()).thenReturn("110");
        when(task.inputLength()).thenReturn(40L);
        WorkerStatus.StatusObservation finished = mock(WorkerStatus.StatusObservation.class);
        when(finished.alive()).thenReturn(true);
        when(finished.activeTasks()).thenReturn(Map.of());
        when(finished.finishedTasks()).thenReturn(Map.of("110", task));
        WorkerStatus.PreparedStatus prepared = mock(WorkerStatus.PreparedStatus.class);
        when(prepared.observation()).thenReturn(finished);
        endpoint.publishStatus(endpoint.applyPreparedStatus(worker, prepared));
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);

        assertEquals(0L, endpoint.inflightUncachedTokenEstimate());
        assertEquals(
                RequestState.Phase.COMPLETED,
                requests.getRequestState("110", 0L, RequestPhase.ENCODER).state());
    }

    @Test
    void concurrentEncoderSelectionsKeepEveryPendingRequest() throws Exception {
        EncoderEndpoint endpoint = new EncoderEndpoint(worker(), requests, availability);
        int requestCount = 32;
        CountDownLatch start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<Boolean>> selected = new ArrayList<>();
            for (int index = 0; index < requestCount; index++) {
                String requestId = "encoder-" + index;
                selected.add(
                        executor.submit(
                                () -> {
                                    start.await();
                                    return endpoint.trackSelectedRequest(requestId, 0L);
                                }));
            }
            start.countDown();
            for (Future<Boolean> result : selected) {
                assertTrue(result.get(5, TimeUnit.SECONDS));
            }
            assertEquals(requestCount, endpoint.pendingEncoderRequestCount());
            assertFalse(endpoint.trackSelectedRequest("encoder-0", 0L));
        }
        for (int index = 0; index < requestCount; index++) {
            endpoint.forgetRequest("encoder-" + index);
        }
        assertEquals(0, endpoint.pendingEncoderRequestCount());
    }

    @Test
    void encoderStatusAndCleanupRaceCannotDoubleReleasePendingRequest() throws Exception {
        WorkerStatus worker = worker();
        EncoderEndpoint endpoint = new EncoderEndpoint(worker, requests, availability);
        try (var executor = Executors.newFixedThreadPool(2)) {
            for (int index = 0; index < 32; index++) {
                String requestId = "encoder-race-" + index;
                assertTrue(endpoint.trackSelectedRequest(requestId, 0L));
                WorkerStatus.TaskObservation task = mock(WorkerStatus.TaskObservation.class);
                when(task.requestId()).thenReturn(requestId);
                WorkerStatus.StatusObservation status = mock(WorkerStatus.StatusObservation.class);
                when(status.owner()).thenReturn(worker);
                when(status.activeTasks()).thenReturn(Map.of(requestId, task));
                CountDownLatch start = new CountDownLatch(1);
                Future<?> observed =
                        executor.submit(
                                () -> {
                                    start.await();
                                    endpoint.publishStatus(
                                            endpoint.applyStatusHeartbeat(worker, status));
                                    return null;
                                });
                Future<?> forgotten =
                        executor.submit(
                                () -> {
                                    start.await();
                                    endpoint.forgetRequest(requestId);
                                    return null;
                                });
                start.countDown();
                observed.get(5, TimeUnit.SECONDS);
                forgotten.get(5, TimeUnit.SECONDS);
                assertEquals(0, endpoint.pendingEncoderRequestCount());
            }
        }
    }

    @Test
    void encoderRetirementFailsOnlyItsOwnPhaseAndReleasesPendingCount() throws Exception {
        RequestContext context = context(108L);
        context.data().setRequestPhase(RequestPhase.ENCODER);
        CompletableFuture<Response> encoderFuture = register(context);
        CompletableFuture<Response> generationFuture = register(context(108L));
        WorkerStatus worker = worker();
        EncoderEndpoint endpoint = new EncoderEndpoint(worker, requests, availability);
        Response route = new Response();
        route.setSuccess(true);
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(claimEncoder("108", encoderFuture, pin));
        }
        assertTrue(publishEncoder("108", encoderFuture, route));
        assertTrue(encoderFuture.get(5, TimeUnit.SECONDS).isSuccess());
        endpoint.close();
        endpoint.awaitRetirement();

        assertEquals(
                RequestState.Phase.FAILED,
                requests.getRequestState("108", 0L, RequestPhase.ENCODER).state());
        assertEquals(RequestState.Phase.QUEUED, requests.getRequestState("108", 0L).state());
        assertEquals(0, endpoint.pendingEncoderRequestCount());
        assertFalse(generationFuture.isDone());
    }

    @Test
    void encoderWorkerFailureAndInactivityEndOnlyEncoderTracking() throws Exception {
        RequestContext failedContext = context(109L);
        failedContext.data().setRequestPhase(RequestPhase.ENCODER);
        CompletableFuture<Response> failedFuture = register(failedContext);
        WorkerStatus failedWorker = worker();
        EncoderEndpoint failedEndpoint = new EncoderEndpoint(failedWorker, requests, availability);
        Response route = new Response();
        route.setSuccess(true);
        try (var pin = failedEndpoint.tryPinGeneration()) {
            assertTrue(claimEncoder("109", failedFuture, pin));
        }
        assertTrue(publishEncoder("109", failedFuture, route));
        failedFuture.get(5, TimeUnit.SECONDS);
        WorkerStatus.TaskObservation failedTask = mock(WorkerStatus.TaskObservation.class);
        when(failedTask.requestId()).thenReturn("109");
        when(failedTask.errorCode()).thenReturn(7L);
        WorkerStatus.StatusObservation failed = mock(WorkerStatus.StatusObservation.class);
        when(failed.alive()).thenReturn(true);
        when(failed.activeTasks()).thenReturn(Map.of());
        when(failed.finishedTasks()).thenReturn(Map.of("109", failedTask));
        WorkerStatus.PreparedStatus prepared = mock(WorkerStatus.PreparedStatus.class);
        when(prepared.observation()).thenReturn(failed);
        failedEndpoint.publishStatus(failedEndpoint.applyPreparedStatus(failedWorker, prepared));
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);
        assertEquals(
                RequestState.Phase.FAILED,
                requests.getRequestState("109", 0L, RequestPhase.ENCODER).state());
        assertEquals(0, failedEndpoint.pendingEncoderRequestCount());

        RequestContext inactiveContext = context(110L);
        inactiveContext.data().setRequestPhase(RequestPhase.ENCODER);
        CompletableFuture<Response> inactiveFuture = register(inactiveContext);
        WorkerStatus inactiveWorker = worker();
        EncoderEndpoint inactiveEndpoint =
                new EncoderEndpoint(inactiveWorker, requests, availability);
        try (var pin = inactiveEndpoint.tryPinGeneration()) {
            assertTrue(claimEncoder("110", inactiveFuture, pin));
        }
        assertTrue(publishEncoder("110", inactiveFuture, route));
        inactiveFuture.get(5, TimeUnit.SECONDS);
        RequestProtocolTestSupport.expireInactiveRequest(
                scheduler, requests.findActive("110", RequestPhase.ENCODER), Long.MAX_VALUE);
        assertEquals(
                RequestState.Phase.TIMED_OUT,
                requests.getRequestState("110", 0L, RequestPhase.ENCODER).state());
        assertEquals(0, inactiveEndpoint.pendingEncoderRequestCount());
    }

    @Test
    void encoderCancelAfterRouteClearsLocalTracking() throws Exception {
        RequestContext context = context(112L);
        context.data().setRequestPhase(RequestPhase.ENCODER);
        CompletableFuture<Response> future = register(context);
        WorkerStatus worker = worker();
        EncoderEndpoint endpoint = new EncoderEndpoint(worker, requests, availability);
        Response route = new Response();
        route.setSuccess(true);
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(claimEncoder("112", future, pin));
        }
        assertTrue(publishEncoder("112", future, route));
        future.get(5, TimeUnit.SECONDS);
        assertEquals(1, endpoint.pendingEncoderRequestCount());

        scheduler.cancel("112", 0L, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);

        assertEquals(
                RequestState.Phase.CANCELLED,
                requests.getRequestState("112", 0L, RequestPhase.ENCODER).state());
        assertEquals(0, endpoint.pendingEncoderRequestCount());
    }

    @Test
    void encoderCancelBetweenEndpointClaimAndRoutePublicationRejectsSuccess() throws Exception {
        RequestContext context = context(113L);
        context.data().setRequestPhase(RequestPhase.ENCODER);
        CompletableFuture<Response> future = register(context);
        EncoderEndpoint endpoint = new EncoderEndpoint(worker(), requests, availability);
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(claimEncoder("113", future, pin));
        }
        assertEquals(
                RequestState.Phase.DISPATCHING,
                requests.getRequestState("113", 0L, RequestPhase.ENCODER).state());
        assertEquals(1, endpoint.pendingEncoderRequestCount());

        scheduler.cancel("113", 0L, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);

        Response route = new Response();
        route.setSuccess(true);
        assertFalse(publishEncoder("113", future, route));
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestState.Phase.CANCELLED,
                requests.getRequestState("113", 0L, RequestPhase.ENCODER).state());
        assertEquals(0, endpoint.pendingEncoderRequestCount());
    }

    private WorkerStatus worker() {
        var status = mock(WorkerStatus.class);
        when(status.getRole()).thenReturn(org.flexlb.dao.route.RoleType.ENCODER);
        when(status.isActiveGeneration()).thenReturn(true);
        when(status.topologySnapshot())
                .thenReturn(
                        new WorkerStatus.TopologySnapshot("group", "encoder", 8080, 8081, null));
        when(status.getLogicalIpPort()).thenReturn("encoder:8080@0");
        when(status.getGroup()).thenReturn("group");
        when(status.isAvailableForScheduling()).thenReturn(true);
        when(status.getGenerationId()).thenReturn(1L);
        when(status.getMultiEngineNum()).thenReturn(1);
        return status;
    }

    private RequestContext context(long id) {
        return context(Long.toString(id));
    }

    private RequestContext context(String id) {
        var context = new RequestContext(config);
        context.data().setRequest(Request.builder().requestId(id).seqLen(16).build());
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(
                                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    private CompletableFuture<Response> register(RequestContext context) {
        return RequestProtocolTestSupport.register(scheduler, context);
    }

    private boolean claimEncoder(
            String id, CompletableFuture<Response> future, WorkerEndpoint.GenerationPin pin) {
        var context = requests.findActive(id, RequestPhase.ENCODER);
        if (context == null || context.data().getFuture() != future) {
            return false;
        }
        pin.endpoint().requirePinnedGeneration(pin);
        return context.claimEncoderRoute(
                (EncoderEndpoint) pin.endpoint(), System.currentTimeMillis());
    }

    private boolean publishEncoder(
            String id, CompletableFuture<Response> future, Response response) {
        var context = requests.findActive(id, RequestPhase.ENCODER);
        if (context == null || context.data().getFuture() != future) {
            return false;
        }
        var published = new java.util.concurrent.atomic.AtomicBoolean();
        try (var publication = new ResponsePublication(scheduler.runtime, context, false)) {
            publication.completeNow(
                    () -> {
                        var chosen =
                                context.selectResponse(
                                        new ResponseCandidate(
                                                context,
                                                ResponseKind.ENCODER_ACK,
                                                null,
                                                System.currentTimeMillis()));
                        if (!(chosen instanceof Granted<?>)) {
                            return false;
                        }
                        scheduler
                                .runtime
                                .deadlines()
                                .update(context, context.snapshot().deadlines());
                        boolean completed =
                                publication.complete(
                                        RequestFuture.Completion.RESPONSE, response, null);
                        published.set(completed);
                        return completed;
                    });
        }
        RequestProtocolTestSupport.awaitResult(future);
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);
        return published.get();
    }

    private boolean publishFailure(String id, Response failure) {
        var context = requests.findActive(id, RequestPhase.ENCODER);
        if (context == null) {
            return false;
        }
        scheduler.publish(context, failure, false);
        return true;
    }
}
