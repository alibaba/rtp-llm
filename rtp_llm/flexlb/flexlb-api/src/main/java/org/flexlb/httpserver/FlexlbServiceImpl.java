package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
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
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.slf4j.LoggerFactory;
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
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final EngineHealthReporter engineHealthReporter;
    private final ActiveRequestCounter activeRequestCounter;
    private final FlexlbGrpcForwarder grpcForwarder;
    private final ConfigService configService;
    private final BatchSchedulerReporter batchSchedulerReporter;
    private final ServerScheduleLatencyRecorder serverLatencyRecorder;
    private final PrioritySloPolicy prioritySloPolicy;
    private final PrioritySchedulerReporter prioritySchedulerReporter;
    public FlexlbServiceImpl(RouteService routeService,
                             LBStatusConsistencyService lbStatusConsistencyService,
                             EngineHealthReporter engineHealthReporter,
                             ActiveRequestCounter activeRequestCounter,
                             FlexlbGrpcForwarder grpcForwarder,
                             ConfigService configService,
                             BatchSchedulerReporter batchSchedulerReporter,
                             ServerScheduleLatencyRecorder serverLatencyRecorder,
                             PrioritySloPolicy prioritySloPolicy,
                             PrioritySchedulerReporter prioritySchedulerReporter) {
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        this.activeRequestCounter = activeRequestCounter;
        this.grpcForwarder = grpcForwarder;
        this.configService = configService;
        this.batchSchedulerReporter = batchSchedulerReporter;
        this.serverLatencyRecorder = serverLatencyRecorder;
        this.prioritySloPolicy = prioritySloPolicy;
        this.prioritySchedulerReporter = prioritySchedulerReporter;
    }

    @Override
    public void schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                         StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver) {
        Long interceptedEntryNanos = GrpcServerTimingInterceptor.getNanos();
        serverLatencyRecorder.recordArrival(
                interceptedEntryNanos != null ? interceptedEntryNanos : System.nanoTime());
        ActiveRequestCounter.RequestToken token = activeRequestCounter.acquire();
        AtomicBoolean responded = new AtomicBoolean(false);
        BalanceContext ctx = null;
        ScheduleOrigin errorOrigin = ScheduleOrigin.ENTRY_ERROR;

        try {
            ctx = buildContext(request);
            BalanceContext requestContext = ctx;
            boolean consistencyEnabled = lbStatusConsistencyService.isNeedConsistency();
            boolean masterAtEntry = consistencyEnabled
                    && lbStatusConsistencyService.isMaster();
            boolean forwardToMaster = consistencyEnabled && !masterAtEntry;
            boolean localFallback = false;
            engineHealthReporter.reportArriveDelayTime(requestContext);

            if (forwardToMaster) {
                errorOrigin = ScheduleOrigin.FORWARDED_TO_MASTER;
                FlexlbGrpcForwarder.MasterForwardResult forwardResult =
                        grpcForwarder.forwardToMaster(request);
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB forwardResponse =
                        forwardResult == null ? null : forwardResult.response();
                if (forwardResponse != null) {
                    responded.set(true);
                    completeSchedule(requestContext, forwardResponse, responseObserver,
                            ScheduleOrigin.FORWARDED_TO_MASTER);
                    token.close();
                    return;
                }

                if (forwardResult != null && !forwardResult.masterFound()) {
                    // No Master address was selected and no RPC was attempted.
                    localFallback = true;
                    errorOrigin = ScheduleOrigin.LOCAL_FALLBACK;
                } else {
                    // Once a Master was selected, delivery is ambiguous. A
                    // local decision could dispatch the same request twice.
                    errorOrigin = ScheduleOrigin.FORWARD_FAILED;
                    FlexlbScheduleProtocol.FlexlbScheduleResponsePB unavailable =
                            buildMasterForwardFailureResponse(
                                    forwardResult == null
                                            ? "MISSING_RESULT"
                                            : forwardResult.failure(),
                                    forwardResult == null
                                            ? ""
                                            : forwardResult.masterHost());
                    responded.set(true);
                    completeSchedule(requestContext, unavailable, responseObserver,
                            ScheduleOrigin.FORWARD_FAILED);
                    token.close();
                    return;
                }
            }

            ScheduleOrigin routeOrigin = localFallback
                    ? ScheduleOrigin.LOCAL_FALLBACK
                    : (consistencyEnabled
                            ? ScheduleOrigin.LOCAL_MASTER
                            : ScheduleOrigin.LOCAL_STANDALONE);
            errorOrigin = routeOrigin;

            CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeFuture =
                    routeLocally(requestContext);

            routeFuture.whenComplete((response, ex) -> {
                try {
                    if (responded.compareAndSet(false, true)) {
                        if (ex != null) {
                            Logger.warn("FlexlbService.schedule async error, request_id={}",
                                    request.getRequestId(), ex);
                            completeSchedule(requestContext, buildErrorResponse(ex), responseObserver,
                                    routeOrigin);
                        } else {
                            completeSchedule(requestContext, response, responseObserver, routeOrigin);
                        }
                    }
                } catch (Exception e) {
                    Logger.warn("FlexlbService.schedule callback error, request_id={}",
                            request.getRequestId(), e);
                } finally {
                    token.close();
                }
            });

        } catch (Exception e) {
            Logger.error("FlexlbService.schedule error, request_id={}", request.getRequestId(), e);
            try {
                if (responded.compareAndSet(false, true)) {
                    completeSchedule(ctx, buildErrorResponse(e), responseObserver, errorOrigin);
                }
            } catch (Exception inner) {
                Logger.warn("FlexlbService.schedule error-response send failed, request_id={}",
                        request.getRequestId(), inner);
            } finally {
                token.close();
            }
        }
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
        RequestLifecycleSnapshot snapshot = routeService.getRequestState(
                request.getRequestId(), request.getBatchId());
        FlexlbScheduleProtocol.GetRequestStateResponsePB.Builder response =
                FlexlbScheduleProtocol.GetRequestStateResponsePB.newBuilder().setFound(snapshot != null);
        if (snapshot != null) {
            response.setLifecycle(toLifecycleProto(snapshot));
        }
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    private CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeLocally(BalanceContext ctx) {
        return routeService.route(ctx).thenApply(response -> {
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                    toProtoResponse(response).toBuilder();
            RequestLifecycleSnapshot lifecycle = routeService.getRequestState(ctx.getRequestId(), 0);
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
            reportAutoTpmSchedule(ctx, response);
        }
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
     * Auto-TPM schedule metrics shared by the legacy path and the priority
     * scheduler path (both funnel through completeSchedule).
     */
    private void reportAutoTpmSchedule(BalanceContext ctx,
                                       FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
        try {
            long now = System.currentTimeMillis();
            long latencyMs = now - ctx.getStartTime();
            boolean success = response.getSuccess();
            String result = success ? "success" : "error_" + response.getCode();
            prioritySchedulerReporter.reportScheduleLatency(ctx.getPriority(), result, latencyMs);
            // Approximate TTFT: schedule-complete minus arrival, as seen by
            // FlexLB. The true TTFT (first token emitted by the engine) is not
            // observable here, so this proxy omits engine-side prefill
            // execution time (task10 P2-8).
            prioritySchedulerReporter.reportTtft(ctx.getPriority(), latencyMs);
            if (ctx.getDeadlineMs() > 0 && now > ctx.getDeadlineMs()) {
                prioritySchedulerReporter.reportDeadlineMiss(ctx.getPriority());
            }
        } catch (Exception ignored) {
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
        request.setGenerateTimeout(pb.getGenerateTimeout());
        request.setRequestTimeMs(pb.getRequestTimeMs());
        request.setMaxNewTokens(pb.getMaxNewTokens());
        request.setNumBeams(pb.getNumBeams());
        request.setForceDisableSpRun(pb.getForceDisableSpRun());
        request.setModel(pb.getModel());
        request.setApiKey(pb.getApiKey());
        request.setCacheKeyBlockSize(pb.getCacheKeyBlockSize());

        // Auto-TPM: construct the immutable ScheduleBudget in one shot —
        // PriorityNormalizer.normalize + PrioritySloPolicy.requestSloMs +
        // coarse deadline — and wire it onto the context.  normalize()
        // always returns 1-100 (never NO_PRIORITY/0): a request carrying no
        // priority gets the default (50) and participates in Auto-TPM at the
        // normal level, so the budget is always constructed.
        ScheduleBudget budget = ScheduleBudget.of(
                pb.getPriority(),
                GrpcQosHeaderInterceptor.get(),
                pb.getSeqLen(),
                ctx.getStartTime(),
                configService.loadBalanceConfig().getAutoTpmDefaultPriority(),
                prioritySloPolicy);
        request.setPriority(budget.priority());
        ctx.setRequest(request);
        ctx.setBudget(budget);
        prioritySchedulerReporter.reportRequest(budget.priority(),
                prioritySloPolicy.bucketLabel(pb.getSeqLen()), budget.requestSloMs());

        if (!pb.getGenerateInput().isEmpty()) {
            ctx.setGenerateInputPbBytes(pb.getGenerateInput().toByteArray());
        }

        ctx.setScheduleMode(configService.loadBalanceConfig().getDefaultScheduleModeEnum());

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
        return lbStatusConsistencyService.isNeedConsistency()
                && !lbStatusConsistencyService.isMaster();
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
            RequestLifecycleSnapshot snapshot) {
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
