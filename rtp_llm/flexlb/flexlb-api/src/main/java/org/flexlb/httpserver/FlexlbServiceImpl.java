package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.consistency.MasterElectService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.config.ConfigService;
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

    @Autowired
    public FlexlbServiceImpl(RouteService routeService,
                             LBStatusConsistencyService lbStatusConsistencyService,
                             EngineHealthReporter engineHealthReporter,
                             ActiveRequestCounter activeRequestCounter,
                             FlexlbGrpcForwarder grpcForwarder,
                             ConfigService configService,
                             BatchSchedulerReporter batchSchedulerReporter,
                             ServerScheduleLatencyRecorder serverLatencyRecorder,
                             RequestSchedulerReporter requestSchedulerReporter) {
        this(routeService, (MasterElectService) lbStatusConsistencyService,
                engineHealthReporter, activeRequestCounter, grpcForwarder,
                configService, batchSchedulerReporter, serverLatencyRecorder,
                requestSchedulerReporter);
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
        this.routeService = routeService;
        this.masterElectService = masterElectService;
        this.engineHealthReporter = engineHealthReporter;
        this.activeRequestCounter = activeRequestCounter;
        this.grpcForwarder = grpcForwarder;
        this.configService = configService;
        this.batchSchedulerReporter = batchSchedulerReporter;
        this.serverLatencyRecorder = serverLatencyRecorder;
        this.requestSchedulerReporter = requestSchedulerReporter;
    }

    @Override
    public void schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                         StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver) {
        Long interceptedEntryNanos = GrpcServerTimingInterceptor.getNanos();
        serverLatencyRecorder.recordArrival(
                interceptedEntryNanos != null ? interceptedEntryNanos : System.nanoTime());
        ActiveRequestCounter.RequestToken token = activeRequestCounter.acquire();
        AtomicBoolean completionClaimed = new AtomicBoolean(false);
        BalanceContext context = null;
        ScheduleOrigin errorOrigin = ScheduleOrigin.ENTRY_ERROR;

        try {
            context = buildContext(request);
            BalanceContext requestContext = context;
            boolean consistencyEnabled = masterElectService.isNeedConsistency();
            boolean masterAtEntry = consistencyEnabled
                    && masterElectService.isMaster();
            boolean forwardToMaster = consistencyEnabled && !masterAtEntry;
            engineHealthReporter.reportArriveDelayTime(requestContext);

            if (forwardToMaster) {
                errorOrigin = ScheduleOrigin.FORWARDED_TO_MASTER;
                grpcForwarder.forwardScheduleToMaster(request).whenComplete(
                        (forwardResult, forwardError) -> handleForwardCompletion(
                                request,
                                requestContext,
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
            routeAndComplete(request, requestContext, responseObserver, token,
                    completionClaimed, routeOrigin);

        } catch (Exception e) {
            Logger.error("FlexlbService.schedule error, request_id={}", request.getRequestId(), e);
            completeOnce(request.getRequestId(), context, buildErrorResponse(e),
                    responseObserver, errorOrigin, token, completionClaimed);
        }
    }

    private void handleForwardCompletion(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
            BalanceContext context,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
            ActiveRequestCounter.RequestToken token,
            AtomicBoolean completionClaimed,
            FlexlbGrpcForwarder.MasterForwardResult forwardResult,
            Throwable forwardError) {
        try {
            if (forwardError != null) {
                Logger.warn("FlexlbService.schedule master forward callback error, request_id={}",
                        request.getRequestId(), forwardError);
                completeOnce(
                        request.getRequestId(),
                        context,
                        buildMasterForwardFailureResponse(
                                failureName(forwardError), ""),
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
                        request.getRequestId(),
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
                routeAndComplete(request, context, responseObserver, token,
                        completionClaimed, ScheduleOrigin.LOCAL_FALLBACK);
                return;
            }

            // Once a Master was selected, delivery is ambiguous. A local
            // decision could dispatch the same request twice. The Master may
            // also have committed the route before its response was lost, so
            // reconcile that ownership through the existing cancel reducer.
            reconcileAmbiguousForward(request, forwardResult);
            completeOnce(
                    request.getRequestId(),
                    context,
                    buildMasterForwardFailureResponse(
                            forwardResult == null
                                    ? "MISSING_RESULT"
                                    : forwardResult.failure(),
                            forwardResult == null
                                    ? ""
                                    : forwardResult.masterHost()),
                    responseObserver,
                    ScheduleOrigin.FORWARD_FAILED,
                    token,
                    completionClaimed);
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule master forward completion error, request_id={}",
                    request.getRequestId(), error);
            completeOnce(
                    request.getRequestId(),
                    context,
                    buildMasterForwardFailureResponse(failureName(error), ""),
                    responseObserver,
                    ScheduleOrigin.FORWARD_FAILED,
                    token,
                    completionClaimed);
        }
    }

    private void reconcileAmbiguousForward(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
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
                        .setRequestId(request.getRequestId())
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
                    request.getRequestId(), error);
        }
    }

    private void routeAndComplete(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
            BalanceContext context,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
            ActiveRequestCounter.RequestToken token,
            AtomicBoolean completionClaimed,
            ScheduleOrigin origin) {
        CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeFuture;
        try {
            // route() registers the scheduler owner synchronously. Install the
            // cancellation listener only after that owner exists, so an
            // already-cancelled Context cannot race ahead of registration.
            routeFuture = routeLocally(context);
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule local route error, request_id={}",
                    request.getRequestId(), error);
            completeOnce(
                    request.getRequestId(),
                    context,
                    buildErrorResponse(error),
                    responseObserver,
                    origin,
                    token,
                    completionClaimed);
            return;
        }
        Context inboundContext = Context.current();
        Context.CancellationListener cancellationListener = ignored -> {
            if (!completionClaimed.compareAndSet(false, true)) {
                return;
            }
            try {
                cancelUndeliveredRoute(request.getRequestId());
            } finally {
                closeRequestToken(request.getRequestId(), token);
            }
        };
        inboundContext.addListener(cancellationListener, Runnable::run);
        Runnable removeCancellationListener =
                () -> inboundContext.removeListener(cancellationListener);
        routeFuture.whenComplete((response, routeError) -> {
            if (routeError != null) {
                Logger.warn("FlexlbService.schedule async error, request_id={}",
                        request.getRequestId(), routeError);
            }
            completeOnce(
                    request.getRequestId(),
                    context,
                    routeError == null ? response : buildErrorResponse(routeError),
                    responseObserver,
                    origin,
                    token,
                    completionClaimed,
                    removeCancellationListener);
        });
    }

    private void completeOnce(
            long requestId,
            BalanceContext context,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
            ScheduleOrigin origin,
            ActiveRequestCounter.RequestToken token,
            AtomicBoolean completionClaimed) {
        completeOnce(requestId, context, response, responseObserver, origin,
                token, completionClaimed, () -> { });
    }

    private void completeOnce(
            long requestId,
            BalanceContext context,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver,
            ScheduleOrigin origin,
            ActiveRequestCounter.RequestToken token,
            AtomicBoolean completionClaimed,
            Runnable completionCleanup) {
        if (!completionClaimed.compareAndSet(false, true)) {
            completionCleanup.run();
            return;
        }
        try {
            completeSchedule(context, response, responseObserver, origin);
        } catch (Exception error) {
            Logger.warn("FlexlbService.schedule response completion error, request_id={}",
                    requestId, error);
        } finally {
            completionCleanup.run();
            closeRequestToken(requestId, token);
        }
    }

    private void closeRequestToken(
            long requestId, ActiveRequestCounter.RequestToken token) {
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
        if (shouldForwardToMaster()) {
            FlexlbScheduleProtocol.GetRequestStateResponsePB forwarded =
                    grpcForwarder.forwardGetRequestStateToMaster(request);
            if (forwarded != null && forwarded.getFound()) {
                responseObserver.onNext(forwarded);
                responseObserver.onCompleted();
                return;
            }
        }
        RequestState.Snapshot snapshot = routeService.getRequestState(
                request.getRequestId(), request.getBatchId());
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
        if (!shouldForwardToMaster()) {
            FlexlbScheduleProtocol.FlexlbCancelResponsePB response;
            try {
                response = cancelLocally(request);
            } catch (Exception error) {
                Logger.error("FlexlbService.cancel error, request_id={}",
                        request.getRequestId(), error);
                try {
                    failCancel(
                            Status.INTERNAL
                                    .withDescription("Cancellation reducer failed")
                                    .withCause(error),
                            responseObserver);
                } catch (Exception completionError) {
                    Logger.warn("FlexlbService.cancel error completion failed, request_id={}",
                            request.getRequestId(), completionError);
                }
                return;
            }
            try {
                completeCancel(response, responseObserver);
            } catch (Exception completionError) {
                Logger.warn("FlexlbService.cancel response completion error, request_id={}",
                        request.getRequestId(), completionError);
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
                    request.getRequestId(),
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
                        request.getRequestId(),
                        cancelForwardStatus(failureName(forwardError), "", forwardError),
                        responseObserver,
                        completionClaimed);
                return;
            }

            FlexlbScheduleProtocol.FlexlbCancelResponsePB response =
                    forwardResult == null ? null : forwardResult.response();
            if (response != null) {
                completeCancelOnce(
                        request.getRequestId(), response, responseObserver, completionClaimed);
                return;
            }

            if (forwardResult != null && !forwardResult.masterFound()) {
                // No Master address was selected and no RPC was attempted.
                completeCancelOnce(
                        request.getRequestId(),
                        cancelLocally(request),
                        responseObserver,
                        completionClaimed);
                return;
            }

            // An attempted cancellation may already be committed by the
            // Master. Never run the reducer locally after this point.
            failCancelOnce(
                    request.getRequestId(),
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
                    request.getRequestId(),
                    cancelForwardStatus(failureName(error), "", error),
                    responseObserver,
                    completionClaimed);
        }
    }

    private FlexlbScheduleProtocol.FlexlbCancelResponsePB cancelLocally(
            FlexlbScheduleProtocol.FlexlbCancelRequestPB request) {
        RequestState.Snapshot snapshot = routeService.cancelRequest(
                request.getRequestId(),
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
            long requestId,
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
            long requestId,
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

    private CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeLocally(BalanceContext ctx) {
        return routeService.route(ctx).thenApply(response -> {
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                    toProtoResponse(response).toBuilder();
            RequestState.Snapshot lifecycle = routeService.getRequestState(ctx.getRequestId(), 0);
            if (lifecycle != null) {
                builder.setLifecycle(toLifecycleProto(lifecycle));
            }
            return builder.build();
        });
    }

    private void completeSchedule(BalanceContext ctx,
                                  FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                                  StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer,
                                  ScheduleOrigin origin) {
        // Report ACK-to-response time for BATCH path (only when engine ACK was received)
        if (ctx != null && ctx.getAckAtMs() > 0) {
            long ackToResponseMs = System.currentTimeMillis() - ctx.getAckAtMs();
            String prefillIp = "";
            if (ctx.getResponse() != null && ctx.getResponse().getServerStatus() != null) {
                for (ServerStatus ss : ctx.getResponse().getServerStatus()) {
                    if (ss.getRole() == RoleType.PREFILL) {
                        prefillIp = ss.getServerIp() != null ? ss.getServerIp() : "";
                        break;
                    }
                }
            }
            batchSchedulerReporter.reportAckToResponseTimeMs(
                    RoleType.PREFILL.name(), prefillIp, ackToResponseMs);
        }
        if (ctx != null) {
            ctx.setSuccess(response.getSuccess());
            if (!response.getSuccess()) {
                ctx.setErrorMessage(response.getErrorMessage());
            }
        }
        try {
            observer.onNext(response);
            observer.onCompleted();
        } catch (RuntimeException deliveryError) {
            if (response.getSuccess() && ownsLocalRoute(origin) && ctx != null) {
                cancelUndeliveredRoute(ctx.getRequestId());
            }
            throw deliveryError;
        } finally {
            try {
                serverLatencyRecorder.recordCompletion(ctx, System.nanoTime());
            } finally {
                if (ctx != null) {
                    logPvRecord(ctx, response, origin);
                }
            }
        }
        if (ctx != null) {
            engineHealthReporter.reportBalancingService(ctx);
            reportPrioritySchedule(ctx, response);
        }
    }

    private void cancelUndeliveredRoute(long requestId) {
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

    /** Write one PV record on the node that made the scheduling decision. */
    private void logPvRecord(BalanceContext ctx,
                             FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                             ScheduleOrigin origin) {
        if (origin == ScheduleOrigin.FORWARDED_TO_MASTER) {
            return;
        }
        try {
            FlexlbScheduleProtocol.RequestLifecyclePB lifecycle = response.hasLifecycle()
                    ? response.getLifecycle()
                    : FlexlbScheduleProtocol.RequestLifecyclePB.getDefaultInstance();
            PvLogData data = new PvLogData(
                    ctx,
                    response.getCode(),
                    response.getAdmissionRejectReason().name(),
                    origin.name(),
                    lifecycle.getBatchId(),
                    lifecycle.getState().name(),
                    response.getRealMasterHost(),
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
     * priority scheduling Phase 0 observability: per-request one-line schedule log plus
     * {@code auto_tpm.schedule.latency_ms}. Shared by the legacy path and the
     * priority scheduler path (both funnel through completeSchedule).
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
            // execution time (task10 P2-8).
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
            // Metrics above stay always-on; the per-request log line drops to
            // DEBUG when priority scheduling is disabled to avoid INFO noise on the
            // legacy path (task10 P2-7).
            String logFormat = "[priority-scheduler] request_id={} priority={} seq_len={} max_new_tokens={} "
                    + "request_expires_at_ms={} schedule_attempt={} plan_type={} plan_cost={} "
                    + "victim_count={} selected_prefill={} selected_decode={} failure_reason={} commit_result={}";
            Object[] logArgs = {
                    ctx.getRequestId(), ctx.getPriority(), ctx.getRequest().getSeqLen(),
                    ctx.getRequest().getMaxNewTokens(),
                    ctx.getRequestExpiresAtMs(),
                    ctx.getScheduleAttempt(), ctx.getPlanType(), ctx.getPlanCost(), ctx.getVictimCount(),
                    selectedPrefill, selectedDecode,
                    success ? "" : response.getErrorMessage(),
                    result};
            Logger.debug(logFormat, logArgs);
        } catch (Exception e) {
            Logger.debug("[priority-scheduler] schedule observability report failed, request_id={}",
                    ctx.getRequestId(), e);
        }
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildErrorResponse(Throwable error) {
        Throwable cause = error;
        while (cause instanceof CompletionException && cause.getCause() != null) {
            cause = cause.getCause();
        }
        if (cause instanceof TimeoutException) {
            return buildErrorResponse(8402, "NO_AVAILABLE_WORKER: schedule timeout");
        }
        return buildErrorResponse(500,
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
    buildMasterForwardFailureResponse(String failure, String masterHost) {
        StrategyErrorType errorType = StrategyErrorType.BATCH_SLO_EXPIRED;
        String detail = "Master scheduling failed (" + failure
                + "); do not retry or route locally";
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

    private BalanceContext buildContext(FlexlbScheduleProtocol.FlexlbScheduleRequestPB pb) {
        BalanceContext ctx = new BalanceContext();

        Request request = new Request();
        request.setRequestId(pb.getRequestId());
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
        ctx.setRequest(request);
        ctx.setSchedulingMetadata(schedulingMetadata);
        requestSchedulerReporter.reportRequest(schedulingMetadata.priority());

        if (!pb.getGenerateInput().isEmpty()) {
            ctx.setGenerateInputPb(pb.getGenerateInput());
        }

        // Capture gRPC server entry time from interceptor context for delay metric splitting
        Long grpcEntryTime = GrpcServerTimingInterceptor.get();
        if (grpcEntryTime != null) {
            ctx.setGrpcEntryTime(grpcEntryTime);
        }
        Long grpcEntryNanos = GrpcServerTimingInterceptor.getNanos();
        if (grpcEntryNanos != null) {
            ctx.setGrpcEntryNanos(grpcEntryNanos);
        }

        return ctx;
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB toProtoResponse(Response response) {
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder();
        if (response == null) {
            return builder.setSuccess(false).setCode(500).setErrorMessage("null response").build();
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
        builder.setAdmissionRejectReason(toProtoAdmissionRejectReason(
                response.getAdmissionRejectReason()));

        if (response.getServerStatus() != null) {
            for (ServerStatus ss : response.getServerStatus()) {
                builder.addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                        .setRole(ss.getRole().getCode())
                        .setServerIp(ss.getServerIp() != null ? ss.getServerIp() : "")
                        .setHttpPort(ss.getHttpPort())
                        .setGrpcPort(ss.getGrpcPort())
                        .build());
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
        LOCAL_MASTER,
        LOCAL_FALLBACK,
        LOCAL_STANDALONE,
        ENTRY_ERROR
    }

    private static FlexlbScheduleProtocol.RequestLifecyclePB toLifecycleProto(
            RequestState.Snapshot snapshot) {
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
