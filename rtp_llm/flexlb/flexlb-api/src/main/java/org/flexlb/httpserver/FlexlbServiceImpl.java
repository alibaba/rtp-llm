package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import io.opentelemetry.api.trace.Span;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestRouteSelector;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.MasterStatusService;
import org.flexlb.consistency.MasterStatusView;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.RequestId;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.interceptor.GrpcTraceInterceptor;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

import java.net.ConnectException;
import java.net.UnknownHostException;
import java.nio.channels.UnresolvedAddressException;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Receives scheduling and control RPCs, submits requests, and publishes asynchronous responses.
 */
@Component
public class FlexlbServiceImpl extends FlexlbServiceGrpc.FlexlbServiceImplBase {
    private static final org.slf4j.Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    private final AbstractRequestScheduler requestEntry;
    private final FlexlbConfig config;
    private final RequestRepository requestState;
    private final MasterStatusView masterStatusView;
    private final EngineHealthReporter engineHealthReporter;
    private final FlexlbGrpcForwarder grpcForwarder;
    private final DeliveryMetricsReporter deliveryMetricsReporter;
    private final ServerScheduleLatencyRecorder serverLatencyRecorder;
    private final RequestSchedulerReporter requestSchedulerReporter;

    private RequestRouteSelector selector;
    private List<RoleType> modelRoles = List.of();
    private org.flexlb.cache.match.CacheAwareService cacheService;

    @Autowired
    public FlexlbServiceImpl(
            @Lazy AbstractRequestScheduler requestEntry,
            ConfigService configService,
            RequestRepository requestState,
            MasterStatusService masterStatusService,
            EngineHealthReporter engineHealthReporter,
            FlexlbGrpcForwarder grpcForwarder,
            DeliveryMetricsReporter deliveryMetricsReporter,
            ServerScheduleLatencyRecorder serverLatencyRecorder,
            RequestSchedulerReporter requestSchedulerReporter) {
        this(
                requestEntry,
                configService.loadBalanceConfig(),
                requestState,
                (MasterStatusView) masterStatusService,
                engineHealthReporter,
                grpcForwarder,
                deliveryMetricsReporter,
                serverLatencyRecorder,
                requestSchedulerReporter);
    }

    FlexlbServiceImpl(
            AbstractRequestScheduler requestEntry,
            FlexlbConfig config,
            RequestRepository requestState,
            MasterStatusView masterStatusView,
            EngineHealthReporter engineHealthReporter,
            FlexlbGrpcForwarder grpcForwarder,
            DeliveryMetricsReporter deliveryMetricsReporter,
            ServerScheduleLatencyRecorder serverLatencyRecorder,
            RequestSchedulerReporter requestSchedulerReporter) {
        this.requestEntry = requestEntry;
        this.config = config;
        this.requestState = requestState;
        this.masterStatusView = masterStatusView;
        this.engineHealthReporter = engineHealthReporter;
        this.grpcForwarder = grpcForwarder;
        this.deliveryMetricsReporter = deliveryMetricsReporter;
        this.serverLatencyRecorder = serverLatencyRecorder;
        this.requestSchedulerReporter = requestSchedulerReporter;
    }

    @Autowired
    public void configureCache(
            org.flexlb.cache.match.CacheAwareService cacheService,
            org.flexlb.config.ModelMetaConfig model) {
        this.cacheService = cacheService;
        this.modelRoles = model.requiredRoles();
    }

    @Autowired
    public void configureRouting(RequestRouteSelector selector) {
        this.selector = java.util.Objects.requireNonNull(selector);
    }

    @Override
    public void schedule(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver) {
        Long interceptedEntryNanos = GrpcServerTimingInterceptor.getNanos();
        serverLatencyRecorder.recordArrival(
                interceptedEntryNanos != null ? interceptedEntryNanos : System.nanoTime());
        String requestId = RequestId.parse(request);
        ScheduleCompletion completionClaimed = new ScheduleCompletion(responseObserver);
        RequestContext context = null;

        try {
            context = buildContext(request);
            engineHealthReporter.reportArriveDelayTime(
                    context.data().getRequest().getRequestTimeMs(),
                    context.data().getGrpcEntryTime(),
                    context.data().getStartTime());
            boolean shouldForward = masterStatusView.shouldForwardToMaster();
            if (!shouldForward) {
                ScheduleOrigin origin =
                        masterStatusView.isNeedConsistency()
                                ? ScheduleOrigin.LOCAL_MASTER
                                : ScheduleOrigin.LOCAL_STANDALONE;
                scheduleLocally(requestId, context, responseObserver, completionClaimed, origin);
            } else if (shouldForward && request.getForwardHop() == 0) {
                forwardToMaster(request, context, responseObserver, completionClaimed);
            } else {
                Logger.error(
                        "FlexlbService.schedule shouldForward=true have forward multi times, request_id={} ",
                        requestId);
                completeOnce(
                        requestId,
                        context,
                        notMasterResponse(requestId, context),
                        responseObserver,
                        ScheduleOrigin.ENTRY_ERROR,
                        completionClaimed);
            }

        } catch (Exception e) {
            Logger.error("FlexlbService.schedule error, request_id={}", requestId, e);
            completeOnce(
                    requestId,
                    context,
                    buildErrorResponse(e),
                    responseObserver,
                    ScheduleOrigin.ENTRY_ERROR,
                    completionClaimed);
        }
    }

    @Override
    public void getRequestState(
            FlexlbScheduleProtocol.GetRequestStateRequestPB request,
            StreamObserver<FlexlbScheduleProtocol.GetRequestStateResponsePB> responseObserver) {
        FlexlbTrace.setRequestAttributes(
                Span.fromContext(entryTraceContext()), RequestId.parse(request));
        RequestPhase phase;
        try {
            phase = requestPhase(request.getPhase());
        } catch (InvalidScheduleRequestException invalid) {
            responseObserver.onError(
                    Status.INVALID_ARGUMENT
                            .withDescription(invalid.getMessage())
                            .asRuntimeException());
            return;
        }
        RequestState snapshot =
                requestState.getRequestState(RequestId.parse(request), request.getBatchId(), phase);
        if (snapshot == null && shouldForwardToMaster()) {
            grpcForwarder
                    .forwardGetRequestStateToMasterAsync(request)
                    .whenComplete(
                            (forwarded, error) -> {
                                var result =
                                        error == null && forwarded != null && forwarded.getFound()
                                                ? forwarded
                                                : FlexlbScheduleProtocol.GetRequestStateResponsePB
                                                        .newBuilder()
                                                        .setFound(false)
                                                        .build();
                                responseObserver.onNext(result);
                                responseObserver.onCompleted();
                            });
            return;
        }
        FlexlbScheduleProtocol.GetRequestStateResponsePB.Builder response =
                FlexlbScheduleProtocol.GetRequestStateResponsePB.newBuilder()
                        .setFound(snapshot != null);
        if (snapshot != null) {
            response.setLifecycle(toLifecycleProto(snapshot));
        }
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    /**
     * Reduce one request generation into its cancellation lifecycle.
     *
     * <p>Every node first checks its local lifecycle ownership. A found response
     * is authoritative even after leadership changes. Only a hop-zero miss on a
     * follower may be forwarded to the current Master.</p>
     *
     * <p>A found response always carries the reducer's authoritative snapshot:
     * {@code CANCEL_REQUESTED} means accepted but still awaiting engine proof;
     * {@code CANCELLED}/{@code TIMED_OUT} are terminal; another state means the
     * request was found but this cancellation did not replace an earlier
     * terminal or priority-preemption owner.</p>
     */
    @Override
    public void cancel(
            FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
            StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver) {
        FlexlbTrace.setRequestAttributes(
                Span.fromContext(entryTraceContext()), RequestId.parse(request));
        FlexlbScheduleProtocol.FlexlbCancelResponsePB localResponse;
        try {
            localResponse = cancelLocally(request);
        } catch (Exception error) {
            Logger.error(
                    "FlexlbService.cancel error, request_id={}", RequestId.parse(request), error);
            try {
                failCancel(
                        Status.INTERNAL
                                .withDescription("Cancellation reducer failed")
                                .withCause(error),
                        responseObserver);
            } catch (Exception completionError) {
                Logger.warn(
                        "FlexlbService.cancel error completion failed, request_id={}",
                        RequestId.parse(request),
                        completionError);
            }
            return;
        }

        if (localResponse.getFound() || request.getForwardHop() != 0 || !shouldForwardToMaster()) {
            try {
                completeCancel(localResponse, responseObserver);
            } catch (Exception completionError) {
                Logger.warn(
                        "FlexlbService.cancel response completion error, request_id={}",
                        RequestId.parse(request),
                        completionError);
            }
            return;
        }

        AtomicBoolean completionClaimed = new AtomicBoolean(false);
        try {
            grpcForwarder
                    .forwardCancelToMaster(request)
                    .whenComplete(
                            (forwardResult, forwardError) ->
                                    handleCancelForwardCompletion(
                                            request,
                                            localResponse,
                                            responseObserver,
                                            completionClaimed,
                                            forwardResult,
                                            forwardError));
        } catch (Exception error) {
            failCancelOnce(
                    RequestId.parse(request),
                    cancelForwardStatus(failureName(error), "", error),
                    responseObserver,
                    completionClaimed);
        }
    }

    /** Retry locally only when forwarding could not have admitted the request. */
    boolean shouldScheduleLocally(
            RequestContext context, FlexlbGrpcForwarder.MasterForwardResult result) {
        if (result == null || !requestActive(context)) {
            return false;
        }
        var response = result.response();
        if (response != null) {
            return !response.getSuccess()
                    && response.getCode() == StrategyErrorType.NOT_MASTER.getErrorCode()
                    && !response.getEnqueuedByMaster()
                    && !response.hasLifecycle()
                    && response.getServerStatusCount() == 0
                    && response.getAdmissionRejectReasonValue() == 0;
        }
        if (!result.masterFound()) {
            return true;
        }
        if (result.error() != null
                && Status.fromThrowable(result.error()).getCode() == Status.Code.UNAVAILABLE) {
            for (Throwable cause = result.error(); cause != null; cause = cause.getCause()) {
                if (cause instanceof ConnectException
                        || cause instanceof UnknownHostException
                        || cause instanceof UnresolvedAddressException) {
                    return true;
                }
            }
        }
        // Disconnects and RPC timeouts may occur after admission; do not replay them.
        return false;
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB notMasterResponse(
            String requestId, RequestContext context) {
        RequestState owned =
                isAdmissionRequest(context)
                        ? requestState.getRequestState(
                                requestId,
                                0,
                                context == null
                                        ? RequestPhase.GENERATION
                                        : context.data().getPhase())
                        : null;
        if (owned != null) {
            // A repeated request must not be advertised as unaccepted after leadership changes.
            return buildMasterForwardFailureResponse("REQUEST_ALREADY_OWNED", "").toBuilder()
                    .setLifecycle(toLifecycleProto(owned))
                    .build();
        }
        return FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(StrategyErrorType.NOT_MASTER.getErrorCode())
                .setErrorMessage(StrategyErrorType.NOT_MASTER.getErrorMsg())
                .build();
    }

    private static boolean isAdmissionRequest(RequestContext ctx) {
        return ctx == null
                || ctx.data().getRequest() == null
                || !ctx.data().getRequest().isVitRouteOnly();
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildMasterForwardFailureResponse(
            String failure, String masterHost) {
        StrategyErrorType errorType = StrategyErrorType.BATCH_SLO_EXPIRED;
        String detail = "Master scheduling failed (" + failure + "); do not retry or route locally";
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                        .setSuccess(false)
                        .setCode(errorType.getErrorCode())
                        .setErrorMessage(errorType.buildErrorMessage(detail));
        if (masterHost != null && !masterHost.isBlank()) {
            builder.setRealMasterHost(masterHost);
        }
        return builder.build();
    }

    private static FlexlbScheduleProtocol.RequestLifecyclePB toLifecycleProto(
            RequestState snapshot) {
        FlexlbScheduleProtocol.RequestLifecyclePB.Builder lifecycle =
                FlexlbScheduleProtocol.RequestLifecyclePB.newBuilder()
                        .setRequestIdStr(snapshot.requestId())
                        .setRequestId(snapshot.originalNumericRequestId())
                        .setState(
                                switch (snapshot.state()) {
                                    case QUEUED ->
                                            FlexlbScheduleProtocol.RequestStatePB
                                                    .REQUEST_STATE_QUEUED;
                                    case DISPATCHING ->
                                            FlexlbScheduleProtocol.RequestStatePB
                                                    .REQUEST_STATE_DISPATCHING;
                                    case ACKNOWLEDGED ->
                                            FlexlbScheduleProtocol.RequestStatePB
                                                    .REQUEST_STATE_ACKNOWLEDGED;
                                    case CANCEL_REQUESTED ->
                                            FlexlbScheduleProtocol.RequestStatePB
                                                    .REQUEST_STATE_CANCEL_REQUESTED;
                                    case CANCELLED ->
                                            FlexlbScheduleProtocol.RequestStatePB
                                                    .REQUEST_STATE_CANCELLED;
                                    case TIMED_OUT ->
                                            FlexlbScheduleProtocol.RequestStatePB
                                                    .REQUEST_STATE_TIMED_OUT;
                                    case FAILED ->
                                            FlexlbScheduleProtocol.RequestStatePB
                                                    .REQUEST_STATE_FAILED;
                                    case COMPLETED ->
                                            FlexlbScheduleProtocol.RequestStatePB
                                                    .REQUEST_STATE_COMPLETED;
                                });
        if (snapshot.batchId() > 0) {
            lifecycle.setBatchId(snapshot.batchId());
        }
        return lifecycle.build();
    }

    private void forwardToMaster(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
            RequestContext context,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
            ScheduleCompletion completionClaimed) {
        String requestId = RequestId.parse(request);
        Context callerContext = Context.current();
        try {
            grpcForwarder
                    .forwardScheduleToMaster(request)
                    .whenComplete(
                            (result, error) ->
                                    callerContext.run(
                                            () -> {
                                                try {
                                                    handleForwardCompletion(
                                                            requestId,
                                                            context,
                                                            responseObserver,
                                                            completionClaimed,
                                                            result,
                                                            error);
                                                } catch (Exception failure) {
                                                    completeOnce(
                                                            requestId,
                                                            context,
                                                            buildErrorResponse(failure),
                                                            responseObserver,
                                                            ScheduleOrigin.FORWARD_FAILED,
                                                            completionClaimed);
                                                }
                                            }));
        } catch (Exception error) {
            // A callback may have completed before registration throws; completeOnce keeps its
            // result.
            completeOnce(
                    requestId,
                    context,
                    buildErrorResponse(error),
                    responseObserver,
                    ScheduleOrigin.FORWARDED_TO_MASTER,
                    completionClaimed);
        }
    }

    private void handleForwardCompletion(
            String requestId,
            RequestContext context,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
            ScheduleCompletion completionClaimed,
            FlexlbGrpcForwarder.MasterForwardResult forwardResult,
            Throwable forwardError) {
        // Normalize exceptional completion so the routing decision has one input.
        if (forwardError != null) {
            forwardResult = FlexlbGrpcForwarder.MasterForwardResult.failed(forwardError, "");
        } else if (forwardResult == null) {
            forwardResult = FlexlbGrpcForwarder.MasterForwardResult.failed("MISSING_RESULT", "");
        }

        if (shouldScheduleLocally(context, forwardResult)) {
            scheduleLocally(
                    requestId,
                    context,
                    responseObserver,
                    completionClaimed,
                    ScheduleOrigin.LOCAL_FALLBACK);
            return;
        }

        if (isAdmissionRequest(context)
                && forwardResult.masterFound()
                && !forwardResult.masterHost().isBlank()) {
            String originalMaster = forwardResult.masterHost();
            completionClaimed.bindCancellation(
                    () ->
                            reconcileForwardedRoute(
                                    requestId,
                                    context.data().getOriginalNumericRequestId(),
                                    context.data().getPhase(),
                                    originalMaster,
                                    FlexlbScheduleProtocol.CancelReasonPB
                                            .CANCEL_REASON_CLIENT_CANCELLED,
                                    context.data().getTraceContext()));
        }
        var response = forwardResult.response();
        ScheduleOrigin origin = ScheduleOrigin.FORWARDED_TO_MASTER;
        if (response == null) {
            origin = ScheduleOrigin.FORWARD_FAILED;
            response =
                    buildMasterForwardFailureResponse(
                            forwardResult.failure(), forwardResult.masterHost());
            if (!requestActive(context)) {
                Context callerContext = Context.current();
                boolean expired =
                        (context.data().getRequestExpiresAtMs() <= 0L
                                        || System.currentTimeMillis()
                                                >= context.data().getRequestExpiresAtMs())
                                || (callerContext.getDeadline() != null
                                        && callerContext.getDeadline().isExpired());
                if (!expired) {
                    response =
                            response.toBuilder()
                                    .setCode(StrategyErrorType.REQUEST_CANCELLED.getErrorCode())
                                    .build();
                }
            }
        }
        completeOnce(requestId, context, response, responseObserver, origin, completionClaimed);
    }

    private void scheduleLocally(
            String requestId,
            RequestContext context,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
            ScheduleCompletion completionClaimed,
            ScheduleOrigin origin) {
        Context inboundContext = Context.current();
        Context.CancellationListener cancellationListener =
                ignored -> {
                    if (!completionClaimed.transportCancellation
                            || inboundContext.cancellationCause() != null) {
                        completionClaimed.cancel();
                    }
                };
        CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeFuture;
        try {
            if (!isAdmissionRequest(context)) {
                inboundContext.addListener(cancellationListener, Runnable::run);
            }
            // QUEUE installs cancellation after registration and before enqueueing; DIRECT submits
            // immediately.
            routeFuture =
                    routeLocally(
                            context,
                            () -> {
                                completionClaimed.bindCancellation(
                                        () -> cancelUndeliveredRoute(context));
                                inboundContext.addListener(cancellationListener, Runnable::run);
                            });
        } catch (Exception error) {
            inboundContext.removeListener(cancellationListener);
            Logger.warn(
                    "FlexlbService.schedule local route error, request_id={}", requestId, error);
            completeOnce(
                    requestId,
                    context,
                    buildErrorResponse(error),
                    responseObserver,
                    origin,
                    completionClaimed);
            return;
        }
        routeFuture.whenComplete(
                (response, routeError) -> {
                    try {
                        if (routeError != null) {
                            Logger.warn(
                                    "FlexlbService.schedule async error, request_id={}",
                                    requestId,
                                    routeError);
                        }
                        completeOnce(
                                requestId,
                                context,
                                routeError == null ? response : buildErrorResponse(routeError),
                                responseObserver,
                                origin,
                                completionClaimed);
                    } finally {
                        inboundContext.removeListener(cancellationListener);
                    }
                });
    }

    private CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeLocally(
            RequestContext ctx, Runnable onRegistered) {
        var request = ctx.data().getRequest();
        CompletableFuture<Response> result;
        try {
            if (request.isVitRouteOnly()) {
                result = CompletableFuture.completedFuture(selector.routeVit(ctx));
                ctx.data().setFuture(result);
            } else if (request.getSelectedVit() != null
                    && requestState.getRequestState(
                                    ctx.data().getRequestId(), 0L, ctx.data().getPhase())
                            == null
                    && !selector.selectedVitIsValid(ctx)) {
                result =
                        CompletableFuture.completedFuture(
                                Response.error(StrategyErrorType.VIT_ROUTE_STALE));
            } else {
                result = requestEntry.submit(ctx, onRegistered);
            }
        } catch (RuntimeException failure) {
            result =
                    CompletableFuture.completedFuture(
                            Response.buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED, failure.getMessage()));
        }
        return result.thenApply(
                response -> {
                    if (response.isSuccess() && cacheService != null) {
                        try {
                            cacheService.updateFromRoutedRequest(
                                    request, response.getServerStatus());
                        } catch (RuntimeException failure) {
                            Logger.warn("Routed cache metadata update failed", failure);
                        }
                    }
                    FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                            toProtoResponse(response).toBuilder();
                    RequestState lifecycle =
                            isAdmissionRequest(ctx)
                                    ? requestState.getRequestState(
                                            ctx.data().getRequestId(), 0, ctx.data().getPhase())
                                    : null;
                    if (lifecycle != null) {
                        builder.setLifecycle(toLifecycleProto(lifecycle));
                    }
                    return builder.build();
                });
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB toProtoResponse(Response response) {
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder();
        if (response == null) {
            return builder.setSuccess(false)
                    .setCode(StrategyErrorType.DISPATCH_FAILED.getErrorCode())
                    .setErrorMessage("null schedule response")
                    .build();
        }
        builder.setSuccess(response.isSuccess());
        builder.setCode(response.getCode());
        if (response.getErrorMessage() != null) {
            builder.setErrorMessage(response.getErrorMessage());
        }
        if (response.getRealMasterHost() != null) {
            builder.setRealMasterHost(response.getRealMasterHost());
        }
        builder.setQueueLength(response.getQueueLength() != null ? response.getQueueLength() : 0);
        builder.setEnqueuedByMaster(response.isEnqueuedByMaster());
        builder.setAdmissionRejectReason(
                toProtoAdmissionRejectReason(response.getAdmissionRejectReason()));

        if (response.getServerStatus() != null) {
            for (ServerStatus ss : response.getServerStatus()) {
                FlexlbScheduleProtocol.FlexlbServerStatusPB.Builder status =
                        FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                                .setRole(ss.getRole().getCode())
                                .setServerIp(ss.getServerIp() != null ? ss.getServerIp() : "")
                                .setHttpPort(ss.getHttpPort())
                                .setGrpcPort(ss.getGrpcPort())
                                .setGroup(ss.getGroup() != null ? ss.getGroup() : "")
                                .setWorkerGeneration(ss.getWorkerGeneration())
                                .addAllPreemptRequestIds(ss.getPreemptRequestIds());
                if (ss.getEngineIndex() != null) {
                    status.setEngineIndex(ss.getEngineIndex());
                }
                builder.addServerStatus(status.build());
            }
        }
        return builder.build();
    }

    private static FlexlbScheduleProtocol.ScheduleFailureReasonPB toProtoAdmissionRejectReason(
            AdmissionRejectReason reason) {
        if (reason == null) {
            return FlexlbScheduleProtocol.ScheduleFailureReasonPB
                    .SCHEDULE_FAILURE_REASON_UNSPECIFIED;
        }
        return switch (reason) {
            case HIGHER_PRIORITY_AHEAD ->
                    FlexlbScheduleProtocol.ScheduleFailureReasonPB.HIGHER_PRIORITY_AHEAD;
            case SAME_PRIORITY_AHEAD ->
                    FlexlbScheduleProtocol.ScheduleFailureReasonPB.SAME_PRIORITY_AHEAD;
            case RESOURCE_EXHAUSTED ->
                    FlexlbScheduleProtocol.ScheduleFailureReasonPB.RESOURCE_EXHAUSTED;
            case UNSPECIFIED ->
                    FlexlbScheduleProtocol.ScheduleFailureReasonPB
                            .SCHEDULE_FAILURE_REASON_UNSPECIFIED;
        };
    }

    private void cancelUndeliveredRoute(RequestContext exact) {
        if (exact == null || exact.data().getRequestScheduler() == null) {
            return;
        }
        try {
            exact.data()
                    .getRequestScheduler()
                    .cancel(exact, 0L, org.flexlb.balance.scheduler.CancelReason.CLIENT_CANCELLED);
        } catch (Exception error) {
            Logger.warn(
                    "FlexlbService.schedule cancellation failed, request_id={}",
                    exact.data().getRequestId(),
                    error);
        }
    }

    private void completeOnce(
            String requestId,
            RequestContext context,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
            ScheduleOrigin origin,
            ScheduleCompletion completionClaimed) {
        if (!completionClaimed.tryRespond()) {
            return;
        }
        try {
            completeSchedule(context, response, responseObserver, origin, completionClaimed);
        } catch (Exception error) {
            Logger.warn(
                    "FlexlbService.schedule response completion error, request_id={}",
                    requestId,
                    error);
        }
    }

    private void completeSchedule(
            RequestContext ctx,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer,
            ScheduleOrigin origin,
            ScheduleCompletion completion) {
        try {
            try {
                recordScheduleTrace(ctx, response, origin);
                // Report ACK-to-response time for BATCH path (only when engine ACK was received)
                if (ctx != null && ctx.data().getAckAtMs() > 0) {
                    long ackToResponseMs = System.currentTimeMillis() - ctx.data().getAckAtMs();
                    String prefillIp = "";
                    Response published = ctx.data().getResponse();
                    if (published != null && published.getServerStatus() != null) {
                        for (ServerStatus ss : published.getServerStatus()) {
                            if (ss.getRole() == RoleType.PREFILL) {
                                prefillIp = ss.getServerIp() != null ? ss.getServerIp() : "";
                                break;
                            }
                        }
                    }
                    deliveryMetricsReporter.reportLatency(
                            DeliveryMetricsReporter.Latency.ACK_TO_RESPONSE,
                            RoleType.PREFILL.name(),
                            prefillIp,
                            ackToResponseMs);
                }
            } catch (RuntimeException observationFailure) {
                Logger.warn(
                        "Schedule observation failed, request_id={}",
                        ctx == null ? "" : ctx.data().getRequestId(),
                        observationFailure);
            }
            observer.onNext(response);
            observer.onCompleted();
            if (observer instanceof ServerCallStreamObserver<?> transport
                    && transport.isCancelled()) {
                completion.cancel();
            }
        } catch (RuntimeException deliveryError) {
            if (response.getSuccess() && ctx != null && isAdmissionRequest(ctx)) {
                completion.cancel();
            }
            throw deliveryError;
        } finally {
            try {
                serverLatencyRecorder.recordCompletion(ctx, System.nanoTime());
            } finally {
                if (ctx != null) {
                    ctx.data().finishRequestTiming();
                    logPvRecord(ctx, response, origin);
                }
            }
        }
        if (ctx != null) {
            if (origin != ScheduleOrigin.FORWARDED_TO_MASTER && ctx.data().getRequest() != null) {
                engineHealthReporter.reportRequestPayload(
                        response.getSuccess(),
                        ctx.data().getRequest().getSeqLen(),
                        ctx.data().getRequest().getCacheKeyBlockSize(),
                        ctx.data().getRequestMessageBytes());
            }
            var outcome = ctx.data().getResponse();
            if (outcome != null) {
                engineHealthReporter.reportBalancingService(
                        outcome.getCode(), System.currentTimeMillis() - ctx.data().getStartTime());
                if (outcome.getServerStatus() != null) {
                    for (var server : outcome.getServerStatus()) {
                        if (server != null && server.getRole() != null) {
                            engineHealthReporter.reportSelectionResult(
                                    server.getRole(), outcome.isSuccess(), outcome.getCode());
                            engineHealthReporter.reportSelectedWorker(
                                    server.getRole(),
                                    ctx.data().selectionReason(server.getRole()),
                                    server.getMetricIpPort(),
                                    outcome.isSuccess(),
                                    outcome.getCode());
                        }
                    }
                }
            }
            if (isAdmissionRequest(ctx)) {
                reportPrioritySchedule(ctx, response);
            }
        }
    }

    private static void recordScheduleTrace(
            RequestContext ctx,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
            ScheduleOrigin origin) {
        try {
            // Forwarded responses need not populate ctx.response. The wire
            // response is the result actually delivered to this caller.
            io.opentelemetry.context.Context traceContext =
                    ctx == null ? entryTraceContext() : ctx.data().getTraceContext();
            FlexlbTrace.setScheduleAttribute(
                    traceContext, FlexlbTrace.SCHEDULE_CODE, (long) response.getCode());
            FlexlbTrace.setScheduleAttribute(
                    traceContext, FlexlbTrace.ENQUEUED_BY_MASTER, response.getEnqueuedByMaster());
            if (response.getSuccess()) {
                for (var server : response.getServerStatusList()) {
                    String address = server.getServerIp() + ":" + server.getHttpPort();
                    if (RoleType.PREFILL.getCode().equals(server.getRole())
                            || RoleType.PDFUSION.getCode().equals(server.getRole())) {
                        FlexlbTrace.setScheduleAttribute(
                                traceContext, FlexlbTrace.PREFILL_ADDRESS, address);
                    } else if (RoleType.DECODE.getCode().equals(server.getRole())) {
                        FlexlbTrace.setScheduleAttribute(
                                traceContext, FlexlbTrace.DECODE_ADDRESS, address);
                    }
                }
            } else {
                String errorType =
                        switch (origin) {
                            case ENTRY_ERROR -> "FLEXLB_INTERNAL_ERROR";
                            case FORWARD_FAILED -> "FLEXLB_FORWARD_FAILED";
                            default -> FlexlbTrace.scheduleFailureType(response.getCode());
                        };
                FlexlbTrace.markBusinessError(traceContext, response.getCode(), errorType);
            }
            if (ctx != null && ownsLocalRoute(origin)) {
                FlexlbTrace.setScheduleDuration(
                        traceContext,
                        FlexlbTrace.ROUTE_SUBMIT_MS,
                        ctx.data().getServiceStartNanos(),
                        ctx.data().getRouteSubmittedNanos());
                FlexlbTrace.setScheduleDuration(
                        traceContext,
                        FlexlbTrace.BATCH_WAIT_MS,
                        ctx.data().getRouteSubmittedNanos(),
                        ctx.data().getBatchDispatchedNanos());
                FlexlbTrace.setScheduleDuration(
                        traceContext,
                        FlexlbTrace.ACK_TO_RESPONSE_MS,
                        ctx.data().getAckAtNanos(),
                        System.nanoTime());
            }
        } catch (Throwable ignored) {
            // Telemetry must not suppress response publication or cancellation reconciliation.
        }
    }

    private static io.opentelemetry.context.Context entryTraceContext() {
        io.opentelemetry.context.Context context = GrpcTraceInterceptor.getOtelContext();
        return context != null ? context : io.opentelemetry.context.Context.current();
    }

    private static boolean ownsLocalRoute(ScheduleOrigin origin) {
        return origin == ScheduleOrigin.LOCAL_MASTER
                || origin == ScheduleOrigin.LOCAL_FALLBACK
                || origin == ScheduleOrigin.LOCAL_STANDALONE;
    }

    /** Write one PV record on the node that made the scheduling decision. */
    private void logPvRecord(
            RequestContext ctx,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
            ScheduleOrigin origin) {
        if (origin == ScheduleOrigin.FORWARDED_TO_MASTER) {
            return;
        }
        try {
            if (response.getSuccess() ? !pvLogger.isInfoEnabled() : !pvLogger.isErrorEnabled()) {
                return;
            }
            FlexlbScheduleProtocol.RequestLifecyclePB lifecycle =
                    response.hasLifecycle()
                            ? response.getLifecycle()
                            : FlexlbScheduleProtocol.RequestLifecyclePB.getDefaultInstance();
            PvLogData data =
                    new PvLogData(
                            ctx.data(),
                            response.getSuccess(),
                            response.getErrorMessage(),
                            response.getCode(),
                            response.getAdmissionRejectReason().name(),
                            origin.name(),
                            lifecycle.getBatchId(),
                            lifecycle.getState().name(),
                            response.getRealMasterHost(),
                            System.currentTimeMillis());
            String json = JsonUtils.toStringOrEmpty(data);
            if (json.isEmpty()) {
                Logger.warn("Failed to serialize PV log: request_id={}", ctx.data().getRequestId());
            } else if (data.isSuccess()) {
                pvLogger.info(json);
            } else {
                pvLogger.error(json);
            }
        } catch (Exception e) {
            Logger.warn("Failed to write PV log: request_id={}", ctx.data().getRequestId(), e);
        }
    }

    /**
     * Per-request schedule summary plus
     * {@code auto_tpm.schedule.latency_ms} for every scheduler mode.
     */
    private void reportPrioritySchedule(
            RequestContext ctx, FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
        try {
            long now = System.currentTimeMillis();
            long latencyMs = now - ctx.data().getStartTime();
            boolean success = response.getSuccess();
            String result = success ? "success" : "error_" + response.getCode();
            requestSchedulerReporter.reportScheduleLatency(
                    ctx.data().getPriority(), result, latencyMs);
            // Approximate TTFT: schedule-complete minus arrival, as seen by
            // FlexLB. The true TTFT (first token emitted by the engine) is not
            // observable here, so this proxy omits engine-side prefill
            // execution time.
            requestSchedulerReporter.reportTtft(ctx.data().getPriority(), latencyMs);
            String selectedPrefill = "";
            String selectedDecode = "";
            Response published = ctx.data().getResponse();
            if (published != null && published.getServerStatus() != null) {
                for (ServerStatus ss : published.getServerStatus()) {
                    if (ss.getRole() != null && ss.getRole().supportsPrefill()) {
                        selectedPrefill = ss.getServerIp() != null ? ss.getServerIp() : "";
                    } else if (ss.getRole() == RoleType.DECODE) {
                        selectedDecode = ss.getServerIp() != null ? ss.getServerIp() : "";
                    }
                }
            }
            // Keep request summaries at DEBUG while metrics remain always-on.
            String logFormat =
                    "[request-scheduler] request_id={} priority={} seq_len={} max_new_tokens={} "
                            + "request_expires_at_ms={} plan_type={} plan_cost={} "
                            + "victim_count={} selected_prefill={} selected_decode={} failure_reason={} commit_result={}";
            Object[] logArgs = {
                ctx.data().getRequestId(),
                ctx.data().getPriority(),
                ctx.data().getRequest().getSeqLen(),
                ctx.data().getRequest().getMaxNewTokens(),
                ctx.data().getRequestExpiresAtMs(),
                ctx.data().getPlanType(),
                ctx.data().getPlanCost(),
                ctx.data().getPreemptedRequestCount(),
                selectedPrefill,
                selectedDecode,
                success ? "" : response.getErrorMessage(),
                result
            };
            Logger.debug(logFormat, logArgs);
        } catch (Exception e) {
            Logger.debug(
                    "[request-scheduler] schedule observability report failed, request_id={}",
                    ctx.data().getRequestId(),
                    e);
        }
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildErrorResponse(Throwable error) {
        Throwable cause = unwrapCompletion(error);
        if (cause instanceof InvalidScheduleRequestException)
            return buildErrorResponse(
                    StrategyErrorType.INVALID_REQUEST.getErrorCode(), cause.getMessage());
        if (cause instanceof TimeoutException
                || Status.fromThrowable(cause).getCode() == Status.Code.DEADLINE_EXCEEDED) {
            return buildErrorResponse(
                    StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                    "request scheduling deadline exceeded");
        }
        return buildErrorResponse(
                StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                cause.getMessage() != null ? cause.getMessage() : "internal scheduling error");
    }

    private static Throwable unwrapCompletion(Throwable error) {
        while ((error instanceof CompletionException || error instanceof ExecutionException)
                && error.getCause() != null) {
            error = error.getCause();
        }
        return error;
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildErrorResponse(
            int code, String message) {
        return FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setSuccess(false)
                .setCode(code)
                .setErrorMessage(message)
                .build();
    }

    private void reconcileForwardedRoute(
            String requestId,
            long originalNumericRequestId,
            RequestPhase phase,
            String masterHost,
            FlexlbScheduleProtocol.CancelReasonPB reason,
            io.opentelemetry.context.Context traceContext) {
        if (masterHost == null || masterHost.isBlank()) {
            return;
        }
        FlexlbScheduleProtocol.FlexlbCancelRequestPB cancelRequest =
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                        .setRequestIdStr(requestId)
                        .setRequestId(originalNumericRequestId)
                        .setPhase(
                                phase == RequestPhase.ENCODER
                                        ? FlexlbScheduleProtocol.RequestPhasePB
                                                .REQUEST_PHASE_ENCODER
                                        : FlexlbScheduleProtocol.RequestPhasePB
                                                .REQUEST_PHASE_GENERATION)
                        .setReason(reason)
                        .build();
        try {
            Context.ROOT
                    .call(
                            () ->
                                    grpcForwarder.forwardCompensatingCancelToMaster(
                                            cancelRequest, masterHost, traceContext))
                    .whenComplete(
                            (cancelResult, cancelError) -> {
                                if (cancelError != null) {
                                    Logger.warn(
                                            "FlexlbService.schedule cancellation reconciliation failed, request_id={} master={}",
                                            requestId,
                                            masterHost,
                                            cancelError);
                                } else if (cancelResult == null
                                        || cancelResult.response() == null) {
                                    Logger.warn(
                                            "FlexlbService.schedule cancellation reconciliation incomplete, request_id={} master={} failure={}",
                                            requestId,
                                            masterHost,
                                            cancelResult == null
                                                    ? "MISSING_RESULT"
                                                    : cancelResult.failure());
                                }
                            });
        } catch (Exception error) {
            Logger.warn(
                    "FlexlbService.schedule cancellation reconciliation failed to start, request_id={} master={}",
                    requestId,
                    masterHost,
                    error);
        }
    }

    private boolean requestActive(RequestContext context) {
        Context inbound = Context.current();
        return !inbound.isCancelled()
                && (inbound.getDeadline() == null || !inbound.getDeadline().isExpired())
                && !(context.data().getRequestExpiresAtMs() <= 0L
                        || System.currentTimeMillis() >= context.data().getRequestExpiresAtMs());
    }

    private void handleCancelForwardCompletion(
            FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
            FlexlbScheduleProtocol.FlexlbCancelResponsePB localResponse,
            StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver,
            AtomicBoolean completionClaimed,
            FlexlbGrpcForwarder.CancelForwardResult forwardResult,
            Throwable forwardError) {
        try {
            if (forwardError != null) {
                failCancelOnce(
                        RequestId.parse(request),
                        cancelForwardStatus(failureName(forwardError), "", forwardError),
                        responseObserver,
                        completionClaimed);
                return;
            }

            FlexlbScheduleProtocol.FlexlbCancelResponsePB response =
                    forwardResult == null ? null : forwardResult.response();
            if (response != null) {
                completeCancelOnce(
                        RequestId.parse(request), response, responseObserver, completionClaimed);
                return;
            }

            if (forwardResult != null && !forwardResult.masterFound()) {
                // No Master address was selected and no RPC was attempted.
                completeCancelOnce(
                        RequestId.parse(request),
                        localResponse,
                        responseObserver,
                        completionClaimed);
                return;
            }

            // An attempted cancellation may already be committed by the
            // Master. Never run the reducer locally after this point.
            failCancelOnce(
                    RequestId.parse(request),
                    cancelForwardStatus(
                            forwardResult == null ? "MISSING_RESULT" : forwardResult.failure(),
                            forwardResult == null ? "" : forwardResult.masterHost(),
                            null),
                    responseObserver,
                    completionClaimed);
        } catch (Exception error) {
            failCancelOnce(
                    RequestId.parse(request),
                    cancelForwardStatus(failureName(error), "", error),
                    responseObserver,
                    completionClaimed);
        }
    }

    private void failCancelOnce(
            String requestId,
            Status status,
            StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver,
            AtomicBoolean completionClaimed) {
        if (!completionClaimed.compareAndSet(false, true)) {
            return;
        }
        try {
            failCancel(status, responseObserver);
        } catch (Exception error) {
            Logger.warn(
                    "FlexlbService.cancel error completion failed, request_id={}",
                    requestId,
                    error);
        }
    }

    private static void failCancel(
            Status status,
            StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver) {
        responseObserver.onError(status.asRuntimeException());
    }

    private static Status cancelForwardStatus(String failure, String masterHost, Throwable error) {
        Status source = error == null ? Status.UNAVAILABLE : Status.fromThrowable(error);
        Status status =
                source.getCode() == Status.Code.UNKNOWN || source.getCode() == Status.Code.OK
                        ? Status.UNAVAILABLE
                        : source;
        String target = masterHost == null || masterHost.isBlank() ? "" : " at " + masterHost;
        return status.withDescription(
                        "Master cancellation failed"
                                + target
                                + " ("
                                + failure
                                + "); not retried locally")
                .withCause(error);
    }

    private static String failureName(Throwable error) {
        Throwable cause = unwrapCompletion(error);
        Status status = Status.fromThrowable(cause);
        return status.getCode() == Status.Code.UNKNOWN
                ? cause.getClass().getSimpleName()
                : status.getCode().name();
    }

    private void completeCancelOnce(
            String requestId,
            FlexlbScheduleProtocol.FlexlbCancelResponsePB response,
            StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver,
            AtomicBoolean completionClaimed) {
        if (!completionClaimed.compareAndSet(false, true)) {
            return;
        }
        try {
            completeCancel(response, responseObserver);
        } catch (Exception error) {
            Logger.warn(
                    "FlexlbService.cancel response completion error, request_id={}",
                    requestId,
                    error);
        }
    }

    private static void completeCancel(
            FlexlbScheduleProtocol.FlexlbCancelResponsePB response,
            StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver) {
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }

    private FlexlbScheduleProtocol.FlexlbCancelResponsePB cancelLocally(
            FlexlbScheduleProtocol.FlexlbCancelRequestPB request) {
        RequestState snapshot =
                requestEntry.cancel(
                        RequestId.parse(request),
                        request.getBatchId(),
                        toCancelReason(request.getReason()),
                        requestPhase(request.getPhase()));
        FlexlbScheduleProtocol.FlexlbCancelResponsePB.Builder response =
                FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                        .setFound(snapshot != null);
        if (snapshot != null) {
            response.setLifecycle(toLifecycleProto(snapshot));
        }
        return response.build();
    }

    private static CancelReason toCancelReason(FlexlbScheduleProtocol.CancelReasonPB reason) {
        return reason == FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_DEADLINE_EXCEEDED
                ? CancelReason.DEADLINE_EXCEEDED
                : CancelReason.CLIENT_CANCELLED;
    }

    private static RequestPhase requestPhase(FlexlbScheduleProtocol.RequestPhasePB phase) {
        return switch (phase) {
            case REQUEST_PHASE_ENCODER -> RequestPhase.ENCODER;
            case REQUEST_PHASE_GENERATION, REQUEST_PHASE_UNSPECIFIED -> RequestPhase.GENERATION;
            default -> throw new IllegalArgumentException("Unsupported request phase: " + phase);
        };
    }

    private RequestContext buildContext(FlexlbScheduleProtocol.FlexlbScheduleRequestPB pb) {
        RequestContext ctx = new RequestContext(config);
        ctx.data().setRequestMessageBytes(GrpcServerTimingInterceptor.getRequestMessageBytes());
        ctx.data().recordRequestTiming(pb.getRequestTimeMs(), null);
        ctx.data().setTraceContext(entryTraceContext());
        Span span = Span.fromContext(ctx.data().getTraceContext());
        FlexlbTrace.setRequestAttributes(span, RequestId.parse(pb));
        FlexlbTrace.setAttribute(span, FlexlbTrace.SCHEDULE_PRIORITY, (long) pb.getPriority());

        ctx.data().setOriginalNumericRequestId(pb.getRequestId());
        var requested = requestedRoles(pb);
        var effective =
                requested == null
                        ? modelRoles
                        : modelRoles.stream().filter(requested::contains).toList();
        if (requested != null && !modelRoles.containsAll(requested))
            throw new InvalidScheduleRequestException("requested role is not configured");
        ctx.data().setRequestedRoles(effective);
        if (effective.equals(List.of(RoleType.ENCODER))) ctx.data().setPhase(RequestPhase.ENCODER);
        if (config.deliveryType(ctx.data().getPhase())
                        == org.flexlb.config.DispatcherConfig.Type.BATCH
                && !pb.getVitRouteOnly()
                && ctx.data().getPhase() == RequestPhase.GENERATION) {
            try {
                RequestId.toEngineRequestId(RequestId.parse(pb));
            } catch (IllegalArgumentException failure) {
                throw new InvalidScheduleRequestException(failure.getMessage(), failure);
            }
        }
        Request.RequestBuilder request = Request.builder();
        request.requestId(RequestId.parse(pb));
        request.blockCacheKeys(pb.getBlockCacheKeysList());
        request.cacheAffinityKeys(pb.getCacheAffinityKeysList());
        request.vitRouteOnly(pb.getVitRouteOnly());
        if (pb.hasSelectedVit()) {
            var selected = pb.getSelectedVit();
            ServerStatus vit = new ServerStatus();
            vit.setRole(RoleType.VIT.getCode().equals(selected.getRole()) ? RoleType.VIT : null);
            vit.setServerIp(selected.getServerIp());
            vit.setHttpPort(selected.getHttpPort());
            vit.setGrpcPort(selected.getGrpcPort());
            vit.setGroup(selected.getGroup().isEmpty() ? null : selected.getGroup());
            vit.setWorkerGeneration(selected.getWorkerGeneration());
            request.selectedVit(vit);
        }
        request.seqLen(pb.getSeqLen());
        if (pb.hasEncoderCacheHitLen()) {
            if (pb.getEncoderCacheHitLen() < 0 || pb.getEncoderCacheHitLen() > pb.getSeqLen())
                throw new InvalidScheduleRequestException(
                        "encoder_cache_hit_len must be within seq_len");
            request.encoderCacheHitLen(pb.getEncoderCacheHitLen());
        }
        // Keep the wire values for transport compatibility and request
        // observability. FlexLB scheduling expiration is owned by the QUEUE
        // configuration below, not by the caller.
        if (pb.getGenerateTimeout() > 0) {
            request.generateTimeout(pb.getGenerateTimeout());
        }
        request.requestTimeMs(pb.getRequestTimeMs());
        request.maxNewTokens(pb.getMaxNewTokens());
        request.numBeams(pb.getNumBeams());
        request.forceDisableSpRun(pb.getForceDisableSpRun());
        request.model(pb.getModel());
        request.apiKey(pb.getApiKey());
        request.cacheKeyBlockSize(pb.getCacheKeyBlockSize());
        request.blockSize(pb.getCacheKeyBlockSize());
        if (pb.getBlockCacheKeysCount() > 0 && pb.getCacheKeyBlockSize() <= 0)
            throw new InvalidScheduleRequestException(
                    "positive cache_key_block_size required with block_cache_keys");

        // QUEUE owns one absolute scheduling deadline, measured from FlexLB
        // admission through delivery acknowledgement. DIRECT never queues and
        // therefore has no scheduling timeout. The scheduler separately installs
        // inactivity tracking from requestLifecycle.request.timeoutMs in both modes.
        long requestExpiresAtMs = config.resolveExpiresAtMs(ctx.data().getStartTime());
        int defaultPriority = config.defaultPriority();
        SchedulingMetadata schedulingMetadata =
                SchedulingMetadata.of(
                        pb.getPriority(),
                        GrpcQosHeaderInterceptor.get(),
                        requestExpiresAtMs,
                        defaultPriority);
        request.rawPriority(pb.getPriority());
        ctx.data().setRequest(request.build());
        ctx.data().setSchedulingMetadata(schedulingMetadata);
        if (!pb.getVitRouteOnly()) {
            requestSchedulerReporter.reportRequest(schedulingMetadata.priority());
        }

        if (!pb.getGenerateInput().isEmpty()) {
            ctx.data().setGenerateInputPb(pb.getGenerateInput());
        }

        // Capture gRPC server entry time from interceptor context for delay metric splitting
        Long grpcEntryTime = GrpcServerTimingInterceptor.get();
        if (grpcEntryTime != null) {
            ctx.data().setGrpcEntryTime(grpcEntryTime);
        }
        Long grpcEntryNanos = GrpcServerTimingInterceptor.getNanos();
        if (grpcEntryNanos != null) {
            ctx.data().setGrpcEntryNanos(grpcEntryNanos);
        }

        return ctx;
    }

    private static Set<RoleType> requestedRoles(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request) {
        if (request.getScheduleRolesCount() == 0) {
            return null;
        }
        EnumSet<RoleType> roles = EnumSet.noneOf(RoleType.class);
        for (int role : request.getScheduleRolesValueList()) {
            FlexlbScheduleProtocol.ScheduleRolePB scheduleRole =
                    FlexlbScheduleProtocol.ScheduleRolePB.forNumber(role);
            if (scheduleRole == null) {
                throw new InvalidScheduleRequestException("invalid schedule_roles value: " + role);
            }
            roles.add(
                    switch (scheduleRole) {
                        case SCHEDULE_ROLE_ENCODER -> RoleType.ENCODER;
                        case SCHEDULE_ROLE_PREFILL -> RoleType.PREFILL;
                        case SCHEDULE_ROLE_DECODE -> RoleType.DECODE;
                        case SCHEDULE_ROLE_PDFUSION -> RoleType.PDFUSION;
                        default ->
                                throw new InvalidScheduleRequestException(
                                        "invalid schedule_roles value: " + role);
                    });
        }
        return Set.copyOf(roles);
    }

    private boolean shouldForwardToMaster() {
        return masterStatusView.isNeedConsistency() && !masterStatusView.isMaster();
    }

    /**
     * Arbitrates response completion separately from transport cancellation cleanup.
     */
    private static final class ScheduleCompletion {
        private final boolean transportCancellation;
        private final AtomicBoolean responseClaimed = new AtomicBoolean();
        private final AtomicBoolean cancelled = new AtomicBoolean();
        private final AtomicBoolean cancellationSettled = new AtomicBoolean();
        private final AtomicReference<Runnable> cancellation = new AtomicReference<>();

        ScheduleCompletion(StreamObserver<?> observer) {
            transportCancellation = observer instanceof ServerCallStreamObserver<?>;
            // Install during the initial method invocation: gRPC freezes this handler
            // after schedule() returns, before asynchronous forwarding may finish.
            if (observer instanceof ServerCallStreamObserver<?> transport) {
                transport.setOnCancelHandler(this::cancel);
                if (transport.isCancelled()) {
                    cancel();
                }
            }
        }

        boolean tryRespond() {
            return responseClaimed.compareAndSet(false, true);
        }

        void bindCancellation(Runnable action) {
            if (!cancellation.compareAndSet(null, action)) {
                throw new IllegalStateException("schedule cancellation owner already bound");
            }
            settleCancellation();
        }

        void cancel() {
            cancelled.set(true);
            responseClaimed.set(true);
            settleCancellation();
        }

        private void settleCancellation() {
            Runnable action = cancellation.get();
            if (cancelled.get()
                    && action != null
                    && cancellationSettled.compareAndSet(false, true)) {
                action.run();
            }
        }
    }

    /**
     * Reports invalid scheduling input before the request enters admission.
     */
    private static final class InvalidScheduleRequestException extends IllegalArgumentException {
        private InvalidScheduleRequestException(String message) {
            super(message);
        }

        private InvalidScheduleRequestException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    private enum ScheduleOrigin {
        FORWARDED_TO_MASTER,
        FORWARD_FAILED,
        LOCAL_MASTER,
        LOCAL_FALLBACK,
        LOCAL_STANDALONE,
        ENTRY_ERROR
    }
}
