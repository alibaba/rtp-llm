package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import io.grpc.Deadline;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

@Component
public class FlexlbServiceImpl extends FlexlbServiceGrpc.FlexlbServiceImplBase {

    private final RouteService routeService;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final EngineHealthReporter engineHealthReporter;
    private final ActiveRequestCounter activeRequestCounter;
    private final FlexlbGrpcForwarder grpcForwarder;
    private final ConfigService configService;
    private final BatchSchedulerReporter batchSchedulerReporter;
    private final ServerScheduleLatencyRecorder serverLatencyRecorder;
    public FlexlbServiceImpl(RouteService routeService,
                             LBStatusConsistencyService lbStatusConsistencyService,
                             EngineHealthReporter engineHealthReporter,
                             ActiveRequestCounter activeRequestCounter,
                             FlexlbGrpcForwarder grpcForwarder,
                             ConfigService configService,
                             BatchSchedulerReporter batchSchedulerReporter,
                             ServerScheduleLatencyRecorder serverLatencyRecorder) {
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        this.activeRequestCounter = activeRequestCounter;
        this.grpcForwarder = grpcForwarder;
        this.configService = configService;
        this.batchSchedulerReporter = batchSchedulerReporter;
        this.serverLatencyRecorder = serverLatencyRecorder;
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

        try {
            ctx = buildContext(request);
            BalanceContext requestContext = ctx;
            boolean forwardToMaster = shouldForwardToMaster();
            AtomicBoolean routedLocally = new AtomicBoolean(!forwardToMaster);
            bindClientCancellation(responseObserver, requestContext, forwardToMaster, routedLocally);
            engineHealthReporter.reportArriveDelayTime(requestContext);

            if (forwardToMaster) {
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB forwardResponse =
                        grpcForwarder.forwardToMaster(request);
                if (forwardResponse != null) {
                    responded.set(true);
                    completeSchedule(requestContext, forwardResponse, responseObserver);
                    token.close();
                    return;
                }
                if (requestContext.isCancelled()) {
                    responded.set(true);
                    completeSchedule(requestContext,
                            buildErrorResponse(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                                    "request cancelled before local fallback"),
                            responseObserver);
                    token.close();
                    return;
                }
                routedLocally.set(true);
            }

            CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeFuture =
                    routeLocally(requestContext);

            Context grpcContext = Context.current();
            Context.CancellationListener cancellationListener = context -> {
                    requestContext.cancel();
                    routeFuture.completeExceptionally(
                            Status.CANCELLED.withDescription("gRPC context cancelled").asRuntimeException());
            };
            grpcContext.addListener(cancellationListener, Runnable::run);

            routeFuture.whenComplete((response, ex) -> {
                grpcContext.removeListener(cancellationListener);
                try {
                    if (responded.compareAndSet(false, true)) {
                        if (ex != null) {
                            Status.Code grpcCode = grpcStatusCode(ex);
                            if (grpcCode == Status.Code.CANCELLED
                                    || grpcCode == Status.Code.DEADLINE_EXCEEDED) {
                                // Client-initiated cancellation or deadline expiry is an expected
                                // operational outcome, not a server fault. Downgrade to WARN (no full
                                // stack at WARN; stack available at DEBUG) so genuine errors are not
                                // masked by this high-volume client-driven noise.
                                // Deadline expiry is a timeout, not an active client cancel; keep the
                                // response code unified (REQUEST_CANCELLED) but differentiate the
                                // message so logs/responses describe the actual outcome.
                                String cancelMsg = (grpcCode == Status.Code.DEADLINE_EXCEEDED)
                                        ? "request deadline exceeded"
                                        : "request cancelled by client";
                                Logger.warn("FlexlbService.schedule client cancelled/timeout (gRPC {}), request_id={}",
                                        grpcCode, request.getRequestId());
                                Logger.debug("FlexlbService.schedule client cancelled/timeout detail, request_id={}",
                                        request.getRequestId(), ex);
                                completeSchedule(requestContext,
                                        buildErrorResponse(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                                                cancelMsg),
                                        responseObserver);
                            } else {
                                Logger.error("FlexlbService.schedule async error, request_id={}", request.getRequestId(), ex);
                                completeSchedule(requestContext, buildErrorResponse(ex), responseObserver);
                            }
                        } else {
                            completeSchedule(requestContext, response, responseObserver);
                        }
                    }
                } catch (Exception e) {
                    Logger.error("FlexlbService.schedule callback error, request_id={}", request.getRequestId(), e);
                } finally {
                    token.close();
                }
            });

        } catch (Exception e) {
            Logger.error("FlexlbService.schedule error, request_id={}", request.getRequestId(), e);
            try {
                if (responded.compareAndSet(false, true)) {
                    completeSchedule(ctx, buildErrorResponse(e), responseObserver);
                }
            } catch (Exception inner) {
                Logger.error("FlexlbService.schedule error-response send failed, request_id={}",
                             request.getRequestId(), inner);
            } finally {
                token.close();
            }
        }
    }

    @Override
    public void cancel(FlexlbScheduleProtocol.FlexlbCancelRequestPB request,
                       StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseObserver) {
        try {
            FlexlbScheduleProtocol.FlexlbCancelResponsePB response = null;
            if (shouldForwardToMaster()) {
                response = grpcForwarder.forwardCancelToMaster(request);
            }
            RequestLifecycleSnapshot snapshot = routeService.cancelByRequestId(
                    request.getRequestId(), toCancelReason(request.getReason()),
                    request.getBatchId());
            if (snapshot != null || response == null) {
                response = toCancelResponse(snapshot);
            }
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (Exception e) {
            Logger.error("FlexlbService.cancel error, request_id={}", request.getRequestId(), e);
            responseObserver.onError(io.grpc.Status.INTERNAL
                    .withDescription(e.getMessage())
                    .asRuntimeException());
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
                                  StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
        // Report ACK-to-response time for BATCH path (only when engine ACK was received)
        if (ctx != null && ctx.getAckAtMs() > 0) {
            long ackToResponseMs = System.currentTimeMillis() - ctx.getAckAtMs();
            String prefillIp = "";
            String prefillIpPort = "";
            if (ctx.getResponse() != null && ctx.getResponse().getServerStatus() != null) {
                for (ServerStatus ss : ctx.getResponse().getServerStatus()) {
                    if (ss.getRole() == RoleType.PREFILL) {
                        prefillIp = ss.getServerIp() != null ? ss.getServerIp() : "";
                        prefillIpPort = ss.getServerIp() != null
                                ? ss.getServerIp() + ":" + ss.getHttpPort() : "";
                        break;
                    }
                }
            }
            batchSchedulerReporter.reportAckToResponseTimeMs(
                    RoleType.PREFILL.name(), prefillIp, prefillIpPort, ackToResponseMs);
        }
        try {
            observer.onNext(response);
            observer.onCompleted();
        } finally {
            serverLatencyRecorder.recordCompletion(ctx, System.nanoTime());
        }
        if (ctx != null) {
            ctx.setSuccess(response.getSuccess());
            if (!response.getSuccess()) {
                ctx.setErrorMessage(response.getErrorMessage());
            }
            engineHealthReporter.reportBalancingService(ctx);
        }
    }

    /**
     * Extracts the gRPC {@link Status.Code} from {@code throwable}, unwrapping the cause
     * chain to handle wrappers such as {@link java.util.concurrent.CompletionException} or
     * {@link java.util.concurrent.ExecutionException}. Returns {@code null} when no gRPC
     * status is present in the chain.
     *
     * <p>The explicit cause walk avoids relying on {@link Status#fromThrowable(Throwable)}
     * whose traversal semantics vary across gRPC versions, keeping the classification
     * deterministic. Client cancellations ({@link Status.Code#CANCELLED}) and deadline
     * expiries ({@link Status.Code#DEADLINE_EXCEEDED}) are expected operational outcomes;
     * callers use them to downgrade logging from ERROR to WARN so genuine server faults are
     * not masked by client-driven noise.</p>
     */
    static Status.Code grpcStatusCode(Throwable throwable) {
        Throwable cur = throwable;
        int depth = 0;
        // Bounded cause walk: a self-referential cause chain would otherwise loop forever.
        while (cur != null && depth++ < 64) {
            if (cur instanceof StatusRuntimeException sre) {
                return sre.getStatus().getCode();
            } else if (cur instanceof StatusException se) {
                return se.getStatus().getCode();
            }
            cur = cur.getCause();
        }
        return null;
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildErrorResponse(Throwable error) {
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
        ctx.setRequest(request);

        if (!pb.getGenerateInput().isEmpty()) {
            ctx.setGenerateInputPbBytes(pb.getGenerateInput().toByteArray());
        }

        ctx.setScheduleMode(resolveScheduleMode(pb.getScheduleMode(), configService.loadBalanceConfig()));

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

    private static ScheduleModeEnum resolveScheduleMode(FlexlbScheduleProtocol.FlexlbScheduleModePB mode,
                                                        FlexlbConfig config) {
        return switch (mode) {
            case FLEXLB_SCHEDULE_BATCH -> ScheduleModeEnum.BATCH;
            case FLEXLB_SCHEDULE_DIRECT -> ScheduleModeEnum.DIRECT;
            case FLEXLB_SCHEDULE_QUEUE -> ScheduleModeEnum.QUEUE;
            default -> config.getDefaultScheduleModeEnum();
        };
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

    private void bindClientCancellation(StreamObserver<?> observer,
                                        BalanceContext ctx,
                                        boolean forwardToMaster,
                                        AtomicBoolean routedLocally) {
        if (observer instanceof ServerCallStreamObserver<?> serverObserver) {
            Deadline deadline = Context.current().getDeadline();
            serverObserver.setOnCancelHandler(() -> {
                ctx.cancel();
                CancelReason reason = deadline != null && deadline.isExpired()
                        ? CancelReason.DEADLINE_EXCEEDED
                        : CancelReason.CLIENT_CANCELLED;
                if (forwardToMaster) {
                    FlexlbScheduleProtocol.FlexlbCancelRequestPB cancelRequest =
                            FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                                    .setRequestId(ctx.getRequestId())
                                    .setReason(toProtoCancelReason(reason))
                                    .build();
                    FlexlbScheduleProtocol.FlexlbCancelResponsePB forwarded =
                            grpcForwarder.forwardCancelToMaster(cancelRequest);
                    if (!routedLocally.get() && forwarded != null && forwarded.getFound()) {
                        return;
                    }
                }
                routeService.cancel(ctx, reason);
            });
        }
    }

    private boolean shouldForwardToMaster() {
        return lbStatusConsistencyService.isNeedConsistency()
                && !lbStatusConsistencyService.isMaster();
    }

    private static CancelReason toCancelReason(FlexlbScheduleProtocol.CancelReasonPB reason) {
        return switch (reason) {
            case CANCEL_REASON_DEADLINE_EXCEEDED -> CancelReason.DEADLINE_EXCEEDED;
            default -> CancelReason.CLIENT_CANCELLED;
        };
    }

    private static FlexlbScheduleProtocol.CancelReasonPB toProtoCancelReason(CancelReason reason) {
        return switch (reason) {
            case DEADLINE_EXCEEDED ->
                    FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_DEADLINE_EXCEEDED;
            default -> FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED;
        };
    }

    private static FlexlbScheduleProtocol.FlexlbCancelResponsePB toCancelResponse(
            RequestLifecycleSnapshot snapshot) {
        FlexlbScheduleProtocol.FlexlbCancelResponsePB.Builder response =
                FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder().setFound(snapshot != null);
        if (snapshot != null) {
            response.setLifecycle(toLifecycleProto(snapshot));
        }
        return response.build();
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
