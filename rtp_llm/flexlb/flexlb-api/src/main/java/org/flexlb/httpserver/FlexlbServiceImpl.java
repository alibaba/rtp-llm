package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.consistency.MasterElectService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.loadbalance.TokenIds;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.RequestId;
import org.flexlb.enums.StatusEnum;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.optimizer.OptimizerClient;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

@Component
public class FlexlbServiceImpl extends FlexlbServiceGrpc.FlexlbServiceImplBase {

    private static final org.slf4j.Logger pvLogger =
            LoggerFactory.getLogger("pvLogger");

    private final RouteService routeService;
    private final MasterElectService masterElectService;
    private final EngineHealthReporter engineHealthReporter;
    private final ActiveRequestCounter activeRequestCounter;
    private final FlexlbGrpcForwarder grpcForwarder;
    private final ConfigService configService;
    private final BatchSchedulerReporter batchSchedulerReporter;
    private final ServerScheduleLatencyRecorder serverLatencyRecorder;
    private final RequestSchedulerReporter requestSchedulerReporter;
    private final CacheAwareService cacheAwareService;
    private final OptimizerClient optimizerClient;

    @Autowired
    public FlexlbServiceImpl(RouteService routeService,
                             LBStatusConsistencyService lbStatusConsistencyService,
                             EngineHealthReporter engineHealthReporter,
                             ActiveRequestCounter activeRequestCounter,
                             FlexlbGrpcForwarder grpcForwarder,
                             ConfigService configService,
                             BatchSchedulerReporter batchSchedulerReporter,
                             ServerScheduleLatencyRecorder serverLatencyRecorder,
                             RequestSchedulerReporter requestSchedulerReporter,
                             CacheAwareService cacheAwareService,
                             OptimizerClient optimizerClient) {
        this(routeService, (MasterElectService) lbStatusConsistencyService,
                engineHealthReporter, activeRequestCounter, grpcForwarder,
                configService, batchSchedulerReporter, serverLatencyRecorder,
                requestSchedulerReporter, cacheAwareService, optimizerClient);
    }

    FlexlbServiceImpl(RouteService routeService,
                      MasterElectService masterElectService,
                      EngineHealthReporter engineHealthReporter,
                      ActiveRequestCounter activeRequestCounter,
                      FlexlbGrpcForwarder grpcForwarder,
                      ConfigService configService,
                      BatchSchedulerReporter batchSchedulerReporter,
                      ServerScheduleLatencyRecorder serverLatencyRecorder,
                      RequestSchedulerReporter requestSchedulerReporter) {
        this(routeService, masterElectService, engineHealthReporter,
                activeRequestCounter, grpcForwarder, configService,
                batchSchedulerReporter, serverLatencyRecorder,
                requestSchedulerReporter, null, null);
    }

    FlexlbServiceImpl(RouteService routeService,
                      MasterElectService masterElectService,
                      EngineHealthReporter engineHealthReporter,
                      ActiveRequestCounter activeRequestCounter,
                      FlexlbGrpcForwarder grpcForwarder,
                      ConfigService configService,
                      BatchSchedulerReporter batchSchedulerReporter,
                      ServerScheduleLatencyRecorder serverLatencyRecorder,
                      RequestSchedulerReporter requestSchedulerReporter,
                      CacheAwareService cacheAwareService,
                      OptimizerClient optimizerClient) {
        this.routeService = routeService;
        this.masterElectService = masterElectService;
        this.engineHealthReporter = engineHealthReporter;
        this.activeRequestCounter = activeRequestCounter;
        this.grpcForwarder = grpcForwarder;
        this.configService = configService;
        this.batchSchedulerReporter = batchSchedulerReporter;
        this.serverLatencyRecorder = serverLatencyRecorder;
        this.requestSchedulerReporter = requestSchedulerReporter;
        this.cacheAwareService = cacheAwareService;
        this.optimizerClient = optimizerClient;
    }

    @Override
    public void schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                         StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver) {
        BalanceContext context = createEntryContext(request);
        String requestId;
        try {
            requestId = RequestId.parse(request);
        } catch (IllegalArgumentException error) {
            rejectSchedule(context, responseObserver, error);
            return;
        }
        ActiveRequestCounter.RequestToken token = activeRequestCounter.acquire();
        AtomicBoolean completionClaimed = new AtomicBoolean(false);
        ScheduleOrigin errorOrigin = ScheduleOrigin.ENTRY_ERROR;

        try {
            initializeRequest(context, request, requestId);
            boolean consistencyEnabled = masterElectService.isNeedConsistency();
            boolean masterAtEntry = consistencyEnabled
                    && masterElectService.isMaster();
            boolean forwardToMaster = consistencyEnabled && !masterAtEntry;

            if (forwardToMaster) {
                errorOrigin = ScheduleOrigin.FORWARDED_TO_MASTER;
                grpcForwarder.forwardScheduleToMaster(request).whenComplete(
                        (forwardResult, forwardError) -> handleForwardCompletion(
                                context,
                                responseObserver,
                                token,
                                completionClaimed,
                                forwardResult,
                                forwardError));
                return;
            }

            ScheduleOrigin routeOrigin = consistencyEnabled
                    ? ScheduleOrigin.LOCAL_MASTER
                    : ScheduleOrigin.LOCAL_STANDALONE;
            errorOrigin = routeOrigin;
            routeAndComplete(context, responseObserver, token,
                    completionClaimed, routeOrigin);

        } catch (Exception e) {
            Logger.error("FlexlbService.schedule error, request_id={}", requestId, e);
            completeOnce(requestId, context, buildErrorResponse(e),
                    responseObserver, errorOrigin, token, completionClaimed);
        }
    }

    private void rejectSchedule(BalanceContext context,
                                StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer,
                                IllegalArgumentException error) {
        context.setSuccess(false);
        context.setErrorMessage(error.getMessage());
        try {
            observer.onError(Status.INVALID_ARGUMENT.withDescription(error.getMessage()).asRuntimeException());
        } finally {
            reportScheduleMetrics(context);
            logPvRecord(context, StrategyErrorType.INVALID_REQUEST.getErrorCode(), null,
                    ScheduleOrigin.ENTRY_ERROR, FlexlbScheduleProtocol.RequestLifecyclePB.getDefaultInstance());
        }
    }

    private void handleForwardCompletion(BalanceContext context,
                                         StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
                                         ActiveRequestCounter.RequestToken token,
                                         AtomicBoolean completionClaimed,
                                         FlexlbGrpcForwarder.MasterForwardResult forwardResult,
                                         Throwable forwardError) {
        try {
            if (forwardError != null) {
                Logger.warn("FlexlbService.schedule master forward callback error, request_id={}",
                        context.getRequestId(), forwardError);
                completeOnce(
                        context.getRequestId(),
                        context,
                        buildMasterForwardFailureResponse(
                                failureName(forwardError)),
                        responseObserver,
                        ScheduleOrigin.FORWARD_FAILED,
                        token,
                        completionClaimed);
                return;
            }

            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response =
                    forwardResult == null ? null : forwardResult.response();
            if (response != null) {
                completeOnce(
                        context.getRequestId(),
                        context,
                        response,
                        responseObserver,
                        ScheduleOrigin.FORWARDED_TO_MASTER,
                        token,
                        completionClaimed);
                return;
            }

            if (forwardResult != null && !forwardResult.masterFound()) {
                // No Master address was selected and no RPC was attempted.
                routeAndComplete(context, responseObserver, token,
                        completionClaimed, ScheduleOrigin.LOCAL_FALLBACK);
                return;
            }

            // Once a Master was selected, delivery is ambiguous. A local
            // decision could dispatch the same request twice. The Master may
            // also have committed the route before its response was lost, so
            // reconcile that ownership through the existing cancel reducer.
            reconcileAmbiguousForward(context.getRequestId(), forwardResult);
            completeOnce(
                    context.getRequestId(),
                    context,
                    buildMasterForwardFailureResponse(
                            forwardResult == null
                                    ? "MISSING_RESULT"
                                    : forwardResult.failure()),
                    responseObserver,
                    ScheduleOrigin.FORWARD_FAILED,
                    token,
                    completionClaimed);
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule master forward completion error, request_id={}",
                    context.getRequestId(), error);
            completeOnce(
                    context.getRequestId(),
                    context,
                    buildMasterForwardFailureResponse(failureName(error)),
                    responseObserver,
                    ScheduleOrigin.FORWARD_FAILED,
                    token,
                    completionClaimed);
        }
    }

    private void reconcileAmbiguousForward(String requestId,
                                           FlexlbGrpcForwarder.MasterForwardResult forwardResult) {
        if (forwardResult == null || !forwardResult.masterFound()) {
            return;
        }
        if ("FORWARD_HOP_LIMIT".equals(forwardResult.failure())
                || "SELF_FORWARD_BLOCKED".equals(forwardResult.failure())) {
            return;
        }
        FlexlbScheduleProtocol.CancelReasonPB reason =
                "DEADLINE_EXCEEDED".equals(forwardResult.failure())
                        ? FlexlbScheduleProtocol.CancelReasonPB
                                .CANCEL_REASON_DEADLINE_EXCEEDED
                        : FlexlbScheduleProtocol.CancelReasonPB
                                .CANCEL_REASON_CLIENT_CANCELLED;
        FlexlbScheduleProtocol.FlexlbCancelRequestPB cancelRequest =
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                        .setRequestId(requestId)
                        .setReason(reason)
                        .build();
        try {
            // The Schedule forward inherited the caller's cancelled Context.
            // Start reconciliation from ROOT or the Cancel RPC would be
            // cancelled before it could reach the lifecycle-owning Master.
            Context.ROOT.call(() -> {
                grpcForwarder.forwardCancelToMaster(cancelRequest);
                return null;
            });
        } catch (Exception error) {
            Logger.warn(
                    "FlexlbService.schedule cancellation reconciliation failed to start, request_id={}",
                    requestId, error);
        }
    }

    private void routeAndComplete(BalanceContext context,
                                  StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
                                  ActiveRequestCounter.RequestToken token,
                                  AtomicBoolean completionClaimed,
                                  ScheduleOrigin origin) {
        Context inboundContext = Context.current();
        Context.CancellationListener cancellationListener = ignored -> {
            if (!completionClaimed.get()) {
                cancelUndeliveredRoute(context.getRequestId());
            }
        };
        inboundContext.addListener(cancellationListener, Runnable::run);
        CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeFuture;
        try {
            routeFuture = routeLocally(context, inboundContext);
        } catch (Exception error) {
            routeFuture = CompletableFuture.failedFuture(error);
        }
        // This future completes after hash preparation and scheduler-owned routing have settled.
        routeFuture.whenComplete((response, routeError) -> {
            try {
                boolean cancelled = inboundContext.isCancelled();
                if (routeError != null && !cancelled) {
                    Logger.warn("FlexlbService.schedule async error, request_id={}", context.getRequestId(), routeError);
                }
                completeOnce(context.getRequestId(), context,
                        cancelled ? cancelledScheduleResponse(inboundContext)
                                : routeError == null ? response : buildErrorResponse(routeError),
                        responseObserver, origin, token, completionClaimed, !cancelled);
            } finally {
                inboundContext.removeListener(cancellationListener);
            }
        });
    }

    private void completeOnce(String requestId,
                              BalanceContext context,
                              FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                              StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
                              ScheduleOrigin origin,
                              ActiveRequestCounter.RequestToken token,
                              AtomicBoolean completionClaimed) {
        completeOnce(requestId, context, response, responseObserver, origin,
                token, completionClaimed, true);
    }

    private void completeOnce(String requestId,
                              BalanceContext context,
                              FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                              StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
                              ScheduleOrigin origin,
                              ActiveRequestCounter.RequestToken token,
                              AtomicBoolean completionClaimed,
                              boolean deliverResponse) {
        if (!completionClaimed.compareAndSet(false, true)) {
            return;
        }
        try {
            completeSchedule(context, response, responseObserver, origin, deliverResponse);
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule response completion error, request_id={}",
                    requestId, error);
        } finally {
            reportScheduleMetrics(context);
            logPvRecord(context, response, origin);
            closeRequestToken(requestId, token);
        }
    }

    private void reportScheduleMetrics(BalanceContext context) {
        String requestId = context.getRequest() == null ? null : context.getRequestId();
        try {
            serverLatencyRecorder.recordCompletion(context, System.nanoTime());
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule completion metric failed, request_id={}", requestId, error);
        }
        try {
            engineHealthReporter.reportArriveDelayTime(context);
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule arrival metric failed, request_id={}", requestId, error);
        }
        try {
            engineHealthReporter.reportRequestPayload(context);
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule payload metric failed, request_id={}", requestId, error);
        }
    }

    private void closeRequestToken(String requestId, ActiveRequestCounter.RequestToken token) {
        try {
            token.close();
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule request token close failed, request_id={}",
                    requestId, error);
        }
    }

    private static String failureName(Throwable error) {
        Throwable cause = error;
        while (cause instanceof CompletionException && cause.getCause() != null) {
            cause = cause.getCause();
        }
        return cause.getClass().getSimpleName();
    }

    @Override
    public void getRequestState(FlexlbScheduleProtocol.GetRequestStateRequestPB request,
                                StreamObserver<FlexlbScheduleProtocol.GetRequestStateResponsePB> responseObserver) {
        try {
            RequestId.parse(request);
        } catch (IllegalArgumentException error) {
            responseObserver.onError(Status.INVALID_ARGUMENT.withDescription(error.getMessage()).asRuntimeException());
            return;
        }
        if (shouldForwardToMaster()) {
            FlexlbScheduleProtocol.GetRequestStateResponsePB forwarded =
                    grpcForwarder.forwardGetRequestStateToMaster(request);
            if (forwarded != null && forwarded.getFound()) {
                responseObserver.onNext(forwarded);
                responseObserver.onCompleted();
                return;
            }
        }
        RequestState snapshot = routeService.getRequestState(
                RequestId.parse(request), request.getBatchId());
        FlexlbScheduleProtocol.GetRequestStateResponsePB.Builder response =
                FlexlbScheduleProtocol.GetRequestStateResponsePB.newBuilder().setFound(snapshot != null);
        if (snapshot != null) {
            response.setLifecycle(toLifecycleProto(snapshot));
        }
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    /**
     * Ask the scheduling Master to reduce one request generation into its
     * cancellation lifecycle.
     *
     * <p>A found response always carries the reducer's authoritative snapshot:
     * {@code CANCEL_REQUESTED} means accepted but still awaiting engine proof;
     * {@code CANCELLED}/{@code TIMED_OUT} are terminal; another state means the
     * request was found but this cancellation did not replace an earlier
     * terminal or priority-preemption owner.</p>
     */
    @Override
    public void cancel(FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
                       StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver) {
        try {
            RequestId.parse(request);
        } catch (IllegalArgumentException error) {
            responseObserver.onError(Status.INVALID_ARGUMENT.withDescription(error.getMessage()).asRuntimeException());
            return;
        }
        if (!shouldForwardToMaster()) {
            FlexlbScheduleProtocol.FlexlbCancelResponsePB response;
            try {
                response = cancelLocally(request);
            } catch (Exception error) {
                Logger.error("FlexlbService.cancel error, request_id={}",
                        RequestId.parse(request), error);
                try {
                    failCancel(
                            Status.INTERNAL
                                    .withDescription("Cancellation reducer failed")
                                    .withCause(error),
                            responseObserver);
                } catch (Exception completionError) {
                    Logger.warn("FlexlbService.cancel error completion failed, request_id={}",
                            RequestId.parse(request), completionError);
                }
                return;
            }
            try {
                completeCancel(response, responseObserver);
            } catch (Exception completionError) {
                Logger.warn("FlexlbService.cancel response completion error, request_id={}",
                        RequestId.parse(request), completionError);
            }
            return;
        }

        AtomicBoolean completionClaimed = new AtomicBoolean(false);
        try {
            grpcForwarder.forwardCancelToMaster(request).whenComplete(
                    (forwardResult, forwardError) -> handleCancelForwardCompletion(
                            request,
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

    private void handleCancelForwardCompletion(
            FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
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
                        cancelLocally(request),
                        responseObserver,
                        completionClaimed);
                return;
            }

            // An attempted cancellation may already be committed by the
            // Master. Never run the reducer locally after this point.
            failCancelOnce(
                    RequestId.parse(request),
                    cancelForwardStatus(
                            forwardResult == null
                                    ? "MISSING_RESULT"
                                    : forwardResult.failure(),
                            forwardResult == null
                                    ? ""
                                    : forwardResult.masterHost(),
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

    private FlexlbScheduleProtocol.FlexlbCancelResponsePB cancelLocally(
            FlexlbScheduleProtocol.FlexlbCancelRequestPB request) {
        RequestState snapshot = routeService.cancelRequest(
                RequestId.parse(request),
                request.getBatchId(),
                toCancelReason(request.getReason()));
        FlexlbScheduleProtocol.FlexlbCancelResponsePB.Builder response =
                FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                        .setFound(snapshot != null);
        if (snapshot != null) {
            response.setLifecycle(toLifecycleProto(snapshot));
        }
        return response.build();
    }

    private static CancelReason toCancelReason(
            FlexlbScheduleProtocol.CancelReasonPB reason) {
        return reason == FlexlbScheduleProtocol.CancelReasonPB
                .CANCEL_REASON_DEADLINE_EXCEEDED
                ? CancelReason.DEADLINE_EXCEEDED
                : CancelReason.CLIENT_CANCELLED;
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
            Logger.warn("FlexlbService.cancel response completion error, request_id={}",
                    requestId, error);
        }
    }

    private static void completeCancel(
            FlexlbScheduleProtocol.FlexlbCancelResponsePB response,
            StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver) {
        responseObserver.onNext(response);
        responseObserver.onCompleted();
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
            Logger.warn("FlexlbService.cancel error completion failed, request_id={}",
                    requestId, error);
        }
    }

    private static void failCancel(
            Status status,
            StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver) {
        responseObserver.onError(status.asRuntimeException());
    }

    private static Status cancelForwardStatus(
            String failure,
            String masterHost,
            Throwable error) {
        Status source = error == null ? Status.UNAVAILABLE : Status.fromThrowable(error);
        Status status = source.getCode() == Status.Code.UNKNOWN
                || source.getCode() == Status.Code.OK
                ? Status.UNAVAILABLE
                : source;
        String target = masterHost == null || masterHost.isBlank()
                ? ""
                : " at " + masterHost;
        return status
                .withDescription("Master cancellation failed" + target
                        + " (" + failure + "); not retried locally")
                .withCause(error);
    }

    private CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeLocally(BalanceContext ctx,
                                                                                            Context inboundContext) {
        if (inboundContext.isCancelled()) {
            return CompletableFuture.completedFuture(cancelledScheduleResponse(inboundContext));
        }
        return prepareBlockCacheKeys(ctx).thenCompose(ignored -> {
            if (inboundContext.isCancelled()) {
                StrategyErrorType error = inboundContext.getDeadline() != null && inboundContext.getDeadline().isExpired()
                        ? StrategyErrorType.BATCH_SLO_EXPIRED : StrategyErrorType.REQUEST_CANCELLED;
                return CompletableFuture.completedFuture(Response.error(error));
            }
            CompletableFuture<Response> route = routeService.route(ctx);
            // Cancellation can arrive between the pre-submit check and scheduler registration.
            if (inboundContext.isCancelled()) {
                cancelUndeliveredRoute(ctx.getRequestId());
            }
            return route;
        }).thenApply(response -> {
            ctx.setResponse(response);
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                    toProtoResponse(response).toBuilder();
            RequestState lifecycle = routeService.getRequestState(ctx.getRequestId(), 0);
            if (lifecycle != null) {
                builder.setLifecycle(toLifecycleProto(lifecycle));
            }
            return builder.build();
        });
    }

    private CompletableFuture<Void> prepareBlockCacheKeys(BalanceContext context) {
        return cacheAwareService.prepareBlockCacheKeys(context);
    }

    private void completeSchedule(BalanceContext ctx,
                                  FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                                  StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer,
                                  ScheduleOrigin origin,
                                  boolean deliverResponse) {
        ctx.setSuccess(response.getSuccess());
        if (!response.getSuccess()) {
            ctx.setErrorMessage(response.getErrorMessage());
        }
        // Delivery confirmation is an Engine ACK for BATCH and route publication for NON_BATCH.
        if (ctx.getAckAtMs() > 0 && ctx.getResponse() != null && ctx.getResponse().getServerStatus() != null) {
            for (ServerStatus worker : ctx.getResponse().getServerStatus()) {
                if (worker.getRole() == RoleType.PREFILL || worker.getRole() == RoleType.PDFUSION) {
                    batchSchedulerReporter.reportAckToResponseTimeMs(worker.getRole().name(), worker.getMetricIpPort(),
                            Math.max(0L, System.currentTimeMillis() - ctx.getAckAtMs()));
                    break;
                }
            }
        }
        if (isLocalSuccessfulDecision(ctx, response, origin)) {
            updateRequestCacheMetadata(ctx);
            fireOptimizerTraceQuery(ctx);
        }
        try {
            if (deliverResponse) {
                observer.onNext(response);
                observer.onCompleted();
            }
        } catch (RuntimeException deliveryError) {
            ctx.setSuccess(false);
            ctx.setErrorMessage("Schedule response delivery failed: " + deliveryError.getMessage());
            if (response.getSuccess() && ownsLocalRoute(origin)) {
                cancelUndeliveredRoute(ctx.getRequestId());
            }
            throw deliveryError;
        }
        engineHealthReporter.reportBalancingService(ctx);
        reportPrioritySchedule(ctx, response);
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleResponsePB cancelledScheduleResponse(Context inboundContext) {
        boolean expired = inboundContext.getDeadline() != null && inboundContext.getDeadline().isExpired();
        StrategyErrorType errorType = expired ? StrategyErrorType.BATCH_SLO_EXPIRED
                : StrategyErrorType.REQUEST_CANCELLED;
        return FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setSuccess(false)
                .setCode(errorType.getErrorCode())
                .setErrorMessage(expired ? "Schedule RPC deadline exceeded" : "Schedule RPC cancelled")
                .setLifecycle(FlexlbScheduleProtocol.RequestLifecyclePB.newBuilder()
                        .setState(expired ? FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_TIMED_OUT
                                : FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_CANCELLED))
                .build();
    }

    private void cancelUndeliveredRoute(String requestId) {
        try {
            routeService.cancelRequest(requestId, 0L, CancelReason.CLIENT_CANCELLED);
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule cancellation failed, request_id={}",
                    requestId, error);
        }
    }

    private static boolean ownsLocalRoute(ScheduleOrigin origin) {
        return origin == ScheduleOrigin.LOCAL_MASTER
                || origin == ScheduleOrigin.LOCAL_FALLBACK
                || origin == ScheduleOrigin.LOCAL_STANDALONE;
    }

    private boolean isLocalSuccessfulDecision(
            BalanceContext ctx,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
            ScheduleOrigin origin) {
        return ownsLocalRoute(origin)
                && response.getSuccess()
                && ctx.getResponse() != null
                && ctx.getResponse().isSuccess();
    }

    private void updateRequestCacheMetadata(BalanceContext ctx) {
        try {
            if (cacheAwareService == null
                    || ctx.getResponse().getServerStatus() == null
                    || ctx.getResponse().getServerStatus().isEmpty()) {
                return;
            }
            cacheAwareService.updateFromRoutedRequest(
                    ctx.getRequest(), ctx.getResponse().getServerStatus());
        } catch (RuntimeException error) {
            Logger.warn("Failed to update request cache metadata, request_id={}",
                    ctx.getRequestId(), error);
        }
    }

    private void fireOptimizerTraceQuery(BalanceContext ctx) {
        try {
            if (optimizerClient == null) {
                return;
            }
            ServerStatus selectedWorker =
                    ctx.getResponse().getServerStatus() == null
                            || ctx.getResponse().getServerStatus().isEmpty()
                            ? null
                            : ctx.getResponse().getServerStatus().get(0);
            optimizerClient.traceQuery(ctx.getRequest(), selectedWorker);
        } catch (RuntimeException error) {
            Logger.warn("Failed to dispatch optimizer trace query, request_id={}",
                    ctx.getRequestId(), error);
        }
    }

    /** Write one PV record on the node that made the scheduling decision. */
    private void logPvRecord(BalanceContext ctx,
                             FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                             ScheduleOrigin origin) {
        logPvRecord(ctx, response.getCode(),
                response.getSuccess() || response.getAdmissionRejectReason()
                        == FlexlbScheduleProtocol.ScheduleFailureReasonPB.SCHEDULE_FAILURE_REASON_UNSPECIFIED
                        ? null : response.getAdmissionRejectReason().name(),
                origin, response.getLifecycle());
    }

    private void logPvRecord(BalanceContext ctx,
                             int code,
                             String admissionRejectReason,
                             ScheduleOrigin origin,
                             FlexlbScheduleProtocol.RequestLifecyclePB lifecycle) {
        if (origin == ScheduleOrigin.FORWARDED_TO_MASTER) {
            return;
        }
        try {
            ctx.finishRequestTiming();
            PvLogData data = new PvLogData(
                    ctx,
                    code,
                    admissionRejectReason,
                    origin.name(),
                    lifecycle.getBatchId(),
                    lifecycle.getState().name(),
                    System.currentTimeMillis());
            String json = JsonUtils.toStringOrEmpty(data);
            if (json.isEmpty()) {
                Logger.warn("Failed to serialize PV log: request_id={}", ctx.getRequestId());
            } else if (data.isSuccess()) {
                pvLogger.info(json);
            } else {
                pvLogger.error(json);
            }
        } catch (Exception e) {
            Logger.warn("Failed to write PV log: request_id={}", ctx.getRequestId(), e);
        }
    }

    /**
     * Per-request schedule summary plus
     * {@code auto_tpm.schedule.latency_ms} for every scheduler mode.
     */
    private void reportPrioritySchedule(BalanceContext ctx,
                                       FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
        try {
            long now = System.currentTimeMillis();
            long latencyMs = now - ctx.getStartTime();
            boolean success = response.getSuccess();
            String result = success ? "success" : "error_" + response.getCode();
            requestSchedulerReporter.reportScheduleLatency(ctx.getPriority(), result, latencyMs);
            // Approximate TTFT: schedule-complete minus arrival, as seen by
            // FlexLB. The true TTFT (first token emitted by the engine) is not
            // observable here, so this proxy omits engine-side prefill
            // execution time.
            requestSchedulerReporter.reportTtft(ctx.getPriority(), latencyMs);
            String selectedPrefill = "";
            String selectedDecode = "";
            if (ctx.getResponse() != null && ctx.getResponse().getServerStatus() != null) {
                for (ServerStatus ss : ctx.getResponse().getServerStatus()) {
                    if (ss.getRole() == RoleType.PREFILL || ss.getRole() == RoleType.PDFUSION) {
                        selectedPrefill = ss.getServerIp() != null ? ss.getServerIp() : "";
                    } else if (ss.getRole() == RoleType.DECODE) {
                        selectedDecode = ss.getServerIp() != null ? ss.getServerIp() : "";
                    }
                }
            }
            // Keep request summaries at DEBUG while metrics remain always-on.
            String logFormat = "[request-scheduler] request_id={} priority={} seq_len={} max_new_tokens={} "
                    + "request_expires_at_ms={} plan_type={} plan_cost={} "
                    + "victim_count={} selected_prefill={} selected_decode={} failure_reason={} commit_result={}";
            Logger.debug(
                    logFormat,
                    ctx.getRequestId(),
                    ctx.getPriority(),
                    ctx.getRequest().getSeqLen(),
                    ctx.getRequest().getMaxNewTokens(),
                    ctx.getRequestExpiresAtMs(),
                    ctx.getPlanType(), ctx.getPlanCost(), ctx.getVictimCount(),
                    selectedPrefill, selectedDecode,
                    success ? "" : response.getErrorMessage(),
                    result);
        } catch (Exception e) {
            Logger.debug("[request-scheduler] schedule observability report failed, request_id={}",
                    ctx.getRequestId(), e);
        }
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildErrorResponse(Throwable error) {
        Throwable cause = error;
        while (cause instanceof CompletionException && cause.getCause() != null) {
            cause = cause.getCause();
        }
        if (cause instanceof TimeoutException) {
            return buildErrorResponse(
                    StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                    "NO_AVAILABLE_WORKER: schedule timeout");
        }
        if (cause instanceof IllegalArgumentException) {
            return buildErrorResponse(
                    StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                    cause.getMessage() != null
                            ? cause.getMessage()
                            : StrategyErrorType.INVALID_REQUEST.getErrorMsg());
        }
        return buildErrorResponse(StatusEnum.INTERNAL_ERROR.getCode(),
                error.getMessage() != null ? error.getMessage() : "internal error");
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildErrorResponse(int code, String message) {
        return FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setSuccess(false)
                .setCode(code)
                .setErrorMessage(message)
                .build();
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB
    buildMasterForwardFailureResponse(String failure) {
        StrategyErrorType errorType = StrategyErrorType.BATCH_SLO_EXPIRED;
        String detail = "Master scheduling failed (" + failure
                + "); do not retry or route locally";
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                        .setSuccess(false)
                        .setCode(errorType.getErrorCode())
                        .setErrorMessage(errorType.buildErrorMessage(detail));
        return builder.build();
    }

    private BalanceContext createEntryContext(FlexlbScheduleProtocol.FlexlbScheduleRequestPB pb) {
        BalanceContext ctx = new BalanceContext();
        ctx.setInputIdsCount((long) pb.getInputIdsCount());
        ctx.setRequestMessageBytes((long) pb.getSerializedSize());
        ctx.recordRequestTiming(pb.getRequestTimeMs(), null);
        Long grpcEntryTime = GrpcServerTimingInterceptor.get();
        if (grpcEntryTime != null) {
            ctx.setGrpcEntryTime(grpcEntryTime);
        }
        Long grpcEntryNanos = GrpcServerTimingInterceptor.getNanos();
        if (grpcEntryNanos != null) {
            ctx.setGrpcEntryNanos(grpcEntryNanos);
        }
        serverLatencyRecorder.recordArrival(grpcEntryNanos != null ? grpcEntryNanos : ctx.getServiceStartNanos());
        return ctx;
    }

    private void initializeRequest(BalanceContext ctx,
                                   FlexlbScheduleProtocol.FlexlbScheduleRequestPB pb,
                                   String requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        ctx.setRequest(request);
        request.setBlockCacheKeys(pb.getBlockCacheKeysList());
        request.setSeqLen(pb.getSeqLen());
        // Keep the wire values for transport compatibility and request
        // observability. FlexLB scheduling expiration is owned by the QUEUE
        // configuration below, not by the caller.
        if (pb.getGenerateTimeout() > 0) {
            request.setGenerateTimeout(pb.getGenerateTimeout());
        }
        request.setRequestTimeMs(pb.getRequestTimeMs());
        request.setMaxNewTokens(pb.getMaxNewTokens());
        request.setNumBeams(pb.getNumBeams());
        request.setForceDisableSpRun(pb.getForceDisableSpRun());
        request.setModel(pb.getModel());
        request.setApiKey(pb.getApiKey());
        request.setCacheKeyBlockSize(pb.getCacheKeyBlockSize());
        if (pb.getInputIdsCount() > 0) {
            request.setInputIds(TokenIds.wrap(pb.getInputIdsCount(), pb::getInputIds));
        }
        // KVCM matching uses the generic block-size field. The dsv4 wire
        // protocol still names the same value cache_key_block_size.
        request.setBlockSize(pb.getCacheKeyBlockSize());

        var config = configService.loadBalanceConfig();
        // QUEUE owns one absolute scheduling deadline, measured from FlexLB
        // admission through delivery acknowledgement. DIRECT never queues and
        // therefore has no scheduling timeout.
        long requestExpiresAtMs = config.isQueue()
                ? config.queueScheduler().resolveExpiresAtMs(ctx.getStartTime())
                : Long.MAX_VALUE;
        int defaultPriority = config.isPriorityOrdering()
                ? config.priorityOrdering().getDefaultPriority()
                : PriorityNormalizer.DEFAULT_PRIORITY;
        SchedulingMetadata schedulingMetadata = SchedulingMetadata.of(
                pb.getPriority(),
                GrpcQosHeaderInterceptor.get(),
                requestExpiresAtMs,
                defaultPriority);
        request.setPriority(schedulingMetadata.priority());
        ctx.setSchedulingMetadata(schedulingMetadata);
        requestSchedulerReporter.reportRequest(schedulingMetadata.priority());

        if (!pb.getGenerateInput().isEmpty()) {
            ctx.setGenerateInputPb(pb.getGenerateInput());
        }
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB toProtoResponse(Response response) {
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder();
        if (response == null) {
            return builder.setSuccess(false)
                    .setCode(StatusEnum.INTERNAL_ERROR.getCode())
                    .setErrorMessage("null response")
                    .build();
        }
        builder.setSuccess(response.isSuccess());
        builder.setCode(response.getCode());
        if (response.getErrorMessage() != null) {
            builder.setErrorMessage(response.getErrorMessage());
        }
        builder.setQueueLength(response.getQueueLength() != null ? response.getQueueLength() : 0);
        builder.setEnqueuedByMaster(response.isEnqueuedByMaster());
        builder.setAdmissionRejectReason(toProtoAdmissionRejectReason(
                response.getAdmissionRejectReason()));

        if (response.getServerStatus() != null) {
            for (ServerStatus ss : response.getServerStatus()) {
                FlexlbScheduleProtocol.FlexlbServerStatusPB.Builder status =
                        FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                                .setRole(ss.getRole().getCode())
                                .setServerIp(ss.getServerIp() != null ? ss.getServerIp() : "")
                                .setHttpPort(ss.getHttpPort())
                                .setGrpcPort(ss.getGrpcPort());
                if (ss.getEngineIndex() != null) {
                    status.setEngineIndex(ss.getEngineIndex());
                }
                builder.addServerStatus(status);
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
            case UNSPECIFIED -> FlexlbScheduleProtocol.ScheduleFailureReasonPB
                    .SCHEDULE_FAILURE_REASON_UNSPECIFIED;
        };
    }

    private boolean shouldForwardToMaster() {
        return masterElectService.isNeedConsistency()
                && !masterElectService.isMaster();
    }

    private enum ScheduleOrigin {
        FORWARDED_TO_MASTER,
        FORWARD_FAILED,
        CONFIGURED_FALLBACK,
        LOCAL_MASTER,
        LOCAL_FALLBACK,
        LOCAL_STANDALONE,
        ENTRY_ERROR
    }

    private static FlexlbScheduleProtocol.RequestLifecyclePB toLifecycleProto(
            RequestState snapshot) {
        FlexlbScheduleProtocol.RequestLifecyclePB.Builder lifecycle =
                FlexlbScheduleProtocol.RequestLifecyclePB.newBuilder()
                        .setRequestId(snapshot.requestId())
                        .setState(switch (snapshot.state()) {
                            case QUEUED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_QUEUED;
                            case DISPATCHING -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_DISPATCHING;
                            case ACKNOWLEDGED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_ACKNOWLEDGED;
                            case CANCEL_REQUESTED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_CANCEL_REQUESTED;
                            case CANCELLED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_CANCELLED;
                            case TIMED_OUT -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_TIMED_OUT;
                            case FAILED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_FAILED;
                            case COMPLETED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_COMPLETED;
                        });
        if (snapshot.batchId() > 0) {
            lifecycle.setBatchId(snapshot.batchId());
        }
        return lifecycle.build();
    }
}
